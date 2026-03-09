//! gRPC server implementation using tonic.

use std::net::SocketAddr;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

use rayon::prelude::*;
use tonic::{Request, Response, Status};
use tonic::transport::Server;
use tracing::{info, instrument, warn};

use forge_db::{
    config::ForgeConfig,
    distance::DistanceMetric,
    vector::Vector,

    WalOperation,
};

use crate::auth::{extract_api_key, verify_api_key};
use crate::collections::{
    Collection, CollectionIndex, CollectionMeta,
    validate_batch_size, validate_collection_name, validate_query_dimension,
    validate_top_k, validate_vector_dimensions,
};
use crate::state::AppState;

// Include generated protobuf code
pub mod proto {
    tonic::include_proto!("forge");
}

use proto::forge_service_server::{ForgeService, ForgeServiceServer};
use proto::*;

/// Implementation of the ForgeService gRPC service.
pub struct ForgeServiceImpl {
    state: Arc<AppState>,
    config: ForgeConfig,
}

impl ForgeServiceImpl {
    pub fn new(state: Arc<AppState>, config: ForgeConfig) -> Self {
        Self { state, config }
    }

    #[allow(clippy::result_large_err)]
    fn authenticate<T>(&self, req: &Request<T>) -> Result<(), Status> {
        let auth_config = match &self.config.server.auth {
            Some(auth) if auth.required => auth,
            _ => return Ok(()),
        };

        let metadata = req.metadata();
        let authorization = metadata
            .get("authorization")
            .and_then(|v| v.to_str().ok());
        let x_api_key = metadata
            .get("x-api-key")
            .and_then(|v| v.to_str().ok());

        match extract_api_key(authorization, x_api_key) {
            Some(key) if verify_api_key(&key, &auth_config.api_keys) => Ok(()),
            _ => Err(Status::unauthenticated("invalid or missing API key")),
        }
    }

    fn parse_distance_metric(s: &str) -> DistanceMetric {
        match s.to_lowercase().as_str() {
            "cosine" => DistanceMetric::Cosine,
            "dot_product" | "inner_product" => DistanceMetric::DotProduct,
            "euclidean_squared" => DistanceMetric::EuclideanSquared,
            _ => DistanceMetric::Euclidean,
        }
    }
}

#[tonic::async_trait]
impl ForgeService for ForgeServiceImpl {
    #[instrument(skip_all, fields(collection))]
    async fn upsert_vectors(
        &self,
        request: Request<UpsertRequest>,
    ) -> Result<Response<UpsertResponse>, Status> {
        self.authenticate(&request)?;
        let req = request.into_inner();
        tracing::Span::current().record("collection", &req.collection);

        // Validate collection name
        validate_collection_name(&req.collection)
            .map_err(Status::invalid_argument)?;

        // Validate batch size
        validate_batch_size(req.vectors.len())
            .map_err(Status::invalid_argument)?;

        let coll = self
            .state
            .get_collection(&req.collection)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", req.collection)))?;

        // Validate vector dimensions
        {
            let guard = coll.read();
            let expected_dim = guard.effective_dimension();
            let vecs: Vec<Vec<f32>> = req.vectors.iter().map(|r| r.vector.clone()).collect();
            validate_vector_dimensions(&vecs, expected_dim)
                .map_err(Status::invalid_argument)?;
        }

        let count = req.vectors.len() as u64;

        // WAL: record inserts before applying to index
        {
            let mut wal = self.state.wal.lock();
            for record in &req.vectors {
                if let Err(e) = wal.append(&WalOperation::Insert {
                    id: record.id,
                    vector: record.vector.clone(),
                    metadata: if record.metadata_json.is_empty() {
                        None
                    } else {
                        Some(record.metadata_json.clone())
                    },
                }) {
                    tracing::error!(error = %e, "WAL append failed for upsert");
                    return Err(Status::internal(format!("WAL write failed: {e}")));
                }
            }
        }

        // Insert into the in-memory index (CPU-bound for HNSW/IVF-PQ)
        {
            let vectors: Vec<Vector> = req.vectors
                .iter()
                .map(|r| Vector::new(r.id, r.vector.clone()))
                .collect();
            let metadata_jsons: Vec<Option<String>> = req.vectors
                .iter()
                .map(|r| if r.metadata_json.is_empty() { None } else { Some(r.metadata_json.clone()) })
                .collect();
            tokio::task::spawn_blocking(move || {
                coll.write().upsert_batch_with_metadata(vectors, metadata_jsons);
            })
            .await
            .map_err(|e| Status::internal(format!("upsert task panicked: {e}")))?;
        }

        self.state
            .stats
            .total_upserts
            .fetch_add(count, Ordering::Relaxed);

        metrics::counter!("forge_upsert_total", "collection" => req.collection.clone()).increment(count);

        Ok(Response::new(UpsertResponse {
            upserted_count: count,
            failed_ids: vec![],
        }))
    }

    async fn delete_vectors(
        &self,
        request: Request<DeleteRequest>,
    ) -> Result<Response<DeleteResponse>, Status> {
        self.authenticate(&request)?;
        let req = request.into_inner();

        let coll = self
            .state
            .get_collection(&req.collection)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", req.collection)))?;

        // WAL: record deletes before applying to index
        {
            let mut wal = self.state.wal.lock();
            for &id in &req.ids {
                if let Err(e) = wal.append(&WalOperation::Delete { id }) {
                    tracing::error!(error = %e, "WAL append failed for delete");
                    return Err(Status::internal(format!("WAL write failed: {e}")));
                }
            }
        }

        let mut deleted = 0u64;
        {
            let mut guard = coll.write();
            for &id in &req.ids {
                if guard.delete(id) {
                    deleted += 1;
                }
            }
        }

        self.state.stats.total_deletes.fetch_add(deleted, Ordering::Relaxed);

        metrics::counter!("forge_delete_total", "collection" => req.collection.clone()).increment(deleted);

        Ok(Response::new(DeleteResponse { deleted_count: deleted }))
    }

    #[instrument(skip_all, fields(collection, k = request.get_ref().top_k))]
    async fn search(
        &self,
        request: Request<SearchRequest>,
    ) -> Result<Response<SearchResponse>, Status> {
        self.authenticate(&request)?;
        let req = request.into_inner();
        tracing::Span::current().record("collection", &req.collection);

        // Validate top_k
        validate_top_k(req.top_k as usize)
            .map_err(Status::invalid_argument)?;

        let coll = self
            .state
            .get_collection(&req.collection)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", req.collection)))?;

        // Validate query dimension
        {
            let guard = coll.read();
            let expected_dim = guard.effective_dimension();
            validate_query_dimension(&req.query, expected_dim)
                .map_err(Status::invalid_argument)?;
        }

        let query = req.query.clone();
        let k = req.top_k as usize;
        let start = Instant::now();
        let (raw_results, _elapsed_us) = tokio::task::spawn_blocking(move || {
            coll.read().search(&query, k)
        })
        .await
        .map_err(|e| Status::internal(format!("search task panicked: {e}")))?;
        let elapsed_us = start.elapsed().as_micros() as u64;

        let results: Vec<SearchResultItem> = raw_results
            .into_iter()
            .filter(|&(_, dist)| req.max_distance == 0.0 || dist <= req.max_distance)
            .map(|(id, distance)| SearchResultItem {
                id,
                distance,
                metadata_json: String::new(),
            })
            .collect();

        self.state.stats.total_searches.fetch_add(1, Ordering::Relaxed);
        self.state.stats.search_latency_sum_us.fetch_add(elapsed_us, Ordering::Relaxed);

        metrics::histogram!("forge_search_latency_seconds", "collection" => req.collection.clone()).record(start.elapsed().as_secs_f64());

        Ok(Response::new(SearchResponse {
            results,
            stats: Some(SearchStats {
                latency_ms: elapsed_us as f32 / 1000.0,
                vectors_scanned: 0,
                partitions_probed: 0,
            }),
        }))
    }

    async fn batch_search(
        &self,
        request: Request<BatchSearchRequest>,
    ) -> Result<Response<BatchSearchResponse>, Status> {
        self.authenticate(&request)?;
        let req = request.into_inner();

        // Validate top_k for each query
        for (i, q) in req.queries.iter().enumerate() {
            validate_top_k(q.top_k as usize)
                .map_err(|e| Status::invalid_argument(format!("query[{}]: {}", i, e)))?;
        }

        let coll = self
            .state
            .get_collection(&req.collection)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", req.collection)))?;

        // Validate query dimensions
        {
            let guard = coll.read();
            let expected_dim = guard.effective_dimension();
            for (i, q) in req.queries.iter().enumerate() {
                validate_query_dimension(&q.query, expected_dim)
                    .map_err(|e| Status::invalid_argument(format!("query[{}]: {}", i, e)))?;
            }
        }

        let start = Instant::now();
        let queries = req.queries.clone();
        let all_results = tokio::task::spawn_blocking(move || {
            queries.par_iter().map(|query| {
                let (raw, latency_us) = coll.read().search(&query.query, query.top_k as usize);
                let items: Vec<SearchResultItem> = raw
                    .into_iter()
                    .map(|(id, distance)| SearchResultItem { id, distance, metadata_json: String::new() })
                    .collect();
                SearchResponse {
                    results: items,
                    stats: Some(SearchStats {
                        latency_ms: latency_us as f32 / 1000.0,
                        vectors_scanned: 0,
                        partitions_probed: 0,
                    }),
                }
            }).collect()
        })
        .await
        .map_err(|e| Status::internal(format!("batch_search task panicked: {e}")))?;

        let total_latency_ms = start.elapsed().as_millis() as f32;
        self.state.stats.total_searches.fetch_add(req.queries.len() as u64, Ordering::Relaxed);

        Ok(Response::new(BatchSearchResponse {
            results: all_results,
            total_latency_ms,
        }))
    }

    async fn build_index(
        &self,
        request: Request<BuildIndexRequest>,
    ) -> Result<Response<BuildIndexResponse>, Status> {
        self.authenticate(&request)?;
        let req = request.into_inner();

        let coll = self
            .state
            .get_collection(&req.collection)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", req.collection)))?;

        let vectors: Vec<Vector> = req.vectors
            .iter()
            .map(|r| Vector::new(r.id, r.vector.clone()))
            .collect();
        let n_clusters = req.n_clusters as usize;
        let n_subvectors = req.n_subvectors as usize;
        let auto_tune = req.auto_tune;

        let count = tokio::task::spawn_blocking(move || {
            coll.write().build_ivfpq(vectors, n_clusters, n_subvectors, auto_tune)
        })
        .await
        .map_err(|e| Status::internal(format!("build_index task panicked: {e}")))?
        .map_err(|e| Status::internal(format!("build_index failed: {e}")))?;

        Ok(Response::new(BuildIndexResponse {
            success: true,
            message: format!("IVF-PQ index built with {} vectors", count),
            vectors_indexed: count as u64,
        }))
    }

    async fn create_collection(
        &self,
        request: Request<CreateCollectionRequest>,
    ) -> Result<Response<CreateCollectionResponse>, Status> {
        self.authenticate(&request)?;
        let req = request.into_inner();

        // Validate collection name
        validate_collection_name(&req.name)
            .map_err(Status::invalid_argument)?;

        let cfg = req.config.unwrap_or_default();
        let metric = Self::parse_distance_metric(&cfg.distance_metric);

        let meta = CollectionMeta {
            name: req.name.clone(),
            dimension: cfg.dimension as usize,
            index_type: format!("{:?}", cfg.index_type),
            distance_metric: cfg.distance_metric.clone(),
            n_clusters: cfg.n_clusters as usize,
            n_subvectors: cfg.n_subvectors as usize,
            nprobe: cfg.nprobe as usize,
            m: if cfg.m == 0 { 16 } else { cfg.m as usize },
            ef_construction: if cfg.ef_construction == 0 { 200 } else { cfg.ef_construction as usize },
            ef_search: if cfg.ef_search == 0 { 50 } else { cfg.ef_search as usize },
            enable_reranking: cfg.enable_reranking,
            rerank_factor: if cfg.rerank_factor == 0 { 4 } else { cfg.rerank_factor as usize },
        };

        let collection = match cfg.index_type {
            1 => Collection::build_hnsw(req.name.clone(), meta, metric),
            _ => Collection::build_brute_force(meta, metric),
        };

        match self.state.create_collection(&req.name, collection) {
            Ok(_) => {
                info!(name = %req.name, "collection created");
                Ok(Response::new(CreateCollectionResponse {
                    created: true,
                    message: format!("collection '{}' created", req.name),
                }))
            }
            Err(e) => Err(Status::already_exists(e)),
        }
    }

    async fn drop_collection(
        &self,
        request: Request<DropCollectionRequest>,
    ) -> Result<Response<DropCollectionResponse>, Status> {
        self.authenticate(&request)?;
        let req = request.into_inner();
        let dropped = self.state.drop_collection(&req.name);
        Ok(Response::new(DropCollectionResponse { dropped }))
    }

    async fn list_collections(
        &self,
        request: Request<ListCollectionsRequest>,
    ) -> Result<Response<ListCollectionsResponse>, Status> {
        self.authenticate(&request)?;
        Ok(Response::new(ListCollectionsResponse {
            names: self.state.list_collections(),
        }))
    }

    async fn get_collection_info(
        &self,
        request: Request<GetCollectionInfoRequest>,
    ) -> Result<Response<GetCollectionInfoResponse>, Status> {
        self.authenticate(&request)?;
        let req = request.into_inner();
        let coll = self
            .state
            .get_collection(&req.name)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", req.name)))?;

        let guard = coll.read();
        Ok(Response::new(GetCollectionInfoResponse {
            name: req.name,
            config: None,
            vector_count: guard.len() as u64,
            memory_bytes: guard.memory_bytes() as u64,
            health_status: guard.index.health(),
        }))
    }

    async fn checkpoint(
        &self,
        request: Request<CheckpointRequest>,
    ) -> Result<Response<CheckpointResponse>, Status> {
        self.authenticate(&request)?;
        let req = request.into_inner();

        let data_dir = self.config.data_dir.clone();

        let target: Vec<String> = if req.collection.is_empty() {
            self.state.collections.iter().map(|r| r.key().clone()).collect()
        } else {
            vec![req.collection.clone()]
        };

        for name in &target {
            if let Some(coll) = self.state.collections.get(name) {
                if let Err(e) = coll.value().read().save_to_disk(&data_dir) {
                    warn!(collection = %name, error = %e, "failed to save collection");
                }
            }
        }

        let seq = {
            let mut wal = self.state.wal.lock();
            let seq = wal.next_sequence();
            if let Err(e) = wal.checkpoint(seq.saturating_sub(1)) {
                tracing::error!(error = %e, "WAL checkpoint failed");
            }
            seq
        };

        Ok(Response::new(CheckpointResponse {
            success: true,
            checkpoint_seq: seq,
        }))
    }

    async fn compact(
        &self,
        request: Request<CompactRequest>,
    ) -> Result<Response<CompactResponse>, Status> {
        self.authenticate(&request)?;
        let req = request.into_inner();

        let coll = self
            .state
            .get_collection(&req.collection)
            .ok_or_else(|| Status::not_found(format!("collection '{}' not found", req.collection)))?;

        let removed = {
            let mut guard = coll.write();
            match &mut guard.index {
                CollectionIndex::IvfPq(idx) => {
                    let before = idx.num_tombstones();
                    idx.compact();
                    before as u64
                }
                _ => 0,
            }
        };

        {
            let mut wal = self.state.wal.lock();
            if let Err(e) = wal.append(&WalOperation::Compact) {
                tracing::error!(error = %e, "WAL append failed for compact");
            }
        }

        Ok(Response::new(CompactResponse {
            success: true,
            vectors_removed: removed,
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            status: "healthy".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            uptime_seconds: self.state.stats.uptime_seconds(),
        }))
    }

    async fn get_stats(
        &self,
        request: Request<GetStatsRequest>,
    ) -> Result<Response<GetStatsResponse>, Status> {
        self.authenticate(&request)?;
        Ok(Response::new(GetStatsResponse {
            total_vectors: self.state.total_vectors(),
            total_collections: self.state.list_collections().len() as u64,
            total_memory_bytes: self.state.total_memory_bytes(),
            avg_search_latency_ms: self.state.stats.avg_search_latency_ms(),
            total_searches: self.state.stats.total_searches.load(Ordering::Relaxed),
            uptime_seconds: self.state.stats.uptime_seconds(),
        }))
    }
}

/// Start the gRPC server with graceful shutdown support.
pub async fn serve(
    addr: SocketAddr,
    state: Arc<AppState>,
    config: ForgeConfig,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let service = ForgeServiceImpl::new(Arc::clone(&state), config.clone());
    let svc = ForgeServiceServer::new(service);

    info!(%addr, "gRPC server listening");

    // Enable TLS if configured.
    if let Some(tls_cfg) = &config.server.tls {
        let cert = tokio::fs::read(&tls_cfg.cert_path).await
            .map_err(|e| anyhow::anyhow!("reading TLS cert {}: {e}", tls_cfg.cert_path.display()))?;
        let key = tokio::fs::read(&tls_cfg.key_path).await
            .map_err(|e| anyhow::anyhow!("reading TLS key {}: {e}", tls_cfg.key_path.display()))?;

        let identity = tonic::transport::Identity::from_pem(cert, key);
        let mut tls = tonic::transport::ServerTlsConfig::new().identity(identity);

        if let Some(ca_path) = &tls_cfg.ca_cert_path {
            let ca_cert = tokio::fs::read(ca_path).await
                .map_err(|e| anyhow::anyhow!("reading CA cert {}: {e}", ca_path.display()))?;
            let ca = tonic::transport::Certificate::from_pem(ca_cert);
            tls = tls.client_ca_root(ca);
        }

        let mut tls_builder = Server::builder()
            .tls_config(tls)
            .map_err(|e| anyhow::anyhow!("TLS config error: {e}"))?
            .tcp_keepalive(Some(Duration::from_secs(60)))
            .tcp_nodelay(true)
            .http2_keepalive_interval(Some(Duration::from_secs(30)))
            .http2_keepalive_timeout(Some(Duration::from_secs(10)))
            .concurrency_limit_per_connection(config.server.max_concurrency)
            .initial_connection_window_size(Some(4 * 1024 * 1024))
            .initial_stream_window_size(Some(2 * 1024 * 1024));

        // Apply rate limiting if configured. Uses ConcurrencyLimitLayer because
        // tower's RateLimitLayer does not produce a Clone service, which tonic requires.
        if config.server.requests_per_second > 0 {
            return tls_builder
                .layer(tower::limit::ConcurrencyLimitLayer::new(
                    config.server.requests_per_second as usize,
                ))
                .add_service(svc)
                .serve_with_shutdown(addr, shutdown)
                .await
                .map_err(|e| anyhow::anyhow!("gRPC server error: {e}"));
        }

        return tls_builder
            .add_service(svc)
            .serve_with_shutdown(addr, shutdown)
            .await
            .map_err(|e| anyhow::anyhow!("gRPC server error: {e}"));
    }

    let mut builder = Server::builder()
        .tcp_keepalive(Some(Duration::from_secs(60)))
        .tcp_nodelay(true)
        .http2_keepalive_interval(Some(Duration::from_secs(30)))
        .http2_keepalive_timeout(Some(Duration::from_secs(10)))
        .concurrency_limit_per_connection(config.server.max_concurrency)
        .initial_connection_window_size(Some(4 * 1024 * 1024))
        .initial_stream_window_size(Some(2 * 1024 * 1024));

    // Apply rate limiting if configured. Uses ConcurrencyLimitLayer because
    // tower's RateLimitLayer does not produce a Clone service, which tonic requires.
    if config.server.requests_per_second > 0 {
        return builder
            .layer(tower::limit::ConcurrencyLimitLayer::new(
                config.server.requests_per_second as usize,
            ))
            .add_service(svc)
            .serve_with_shutdown(addr, shutdown)
            .await
            .map_err(|e| anyhow::anyhow!("gRPC server error: {e}"));
    }

    builder
        .add_service(svc)
        .serve_with_shutdown(addr, shutdown)
        .await
        .map_err(|e| anyhow::anyhow!("gRPC server error: {e}"))
}
