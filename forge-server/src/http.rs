//! HTTP/REST API server using axum.
//!
//! All gRPC operations are also available over JSON/HTTP for ease of integration.
//!
//! # Endpoints
//!
//! | Method | Path                                    | Description             |
//! |--------|-----------------------------------------|-------------------------|
//! | GET    | /health                                 | Health check            |
//! | GET    | /metrics                                | Prometheus metrics      |
//! | GET    | /v1/collections                         | List collections        |
//! | POST   | /v1/collections                         | Create collection       |
//! | DELETE | /v1/collections/{name}                  | Drop collection         |
//! | GET    | /v1/collections/{name}                  | Get collection info     |
//! | POST   | /v1/collections/{name}/vectors          | Upsert vectors          |
//! | POST   | /v1/collections/{name}/search           | Search                  |
//! | POST   | /v1/collections/{name}/batch-search     | Batch search            |
//! | DELETE | /v1/collections/{name}/vectors          | Delete vectors by IDs   |
//! | POST   | /v1/collections/{name}/checkpoint       | Checkpoint collection   |
//! | POST   | /v1/collections/{name}/compact          | Compact collection      |
//! | GET    | /v1/stats                               | Global stats            |

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Json},
    routing::{delete, get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use tower_http::{
    compression::CompressionLayer,
    cors::CorsLayer,
    limit::RequestBodyLimitLayer,
    timeout::TimeoutLayer as HttpTimeoutLayer,
    trace::TraceLayer,
};
use tracing::{info, instrument};

use forge_db::{
    config::ForgeConfig,
    distance::DistanceMetric,
    vector::Vector,
};

use crate::auth::{extract_api_key, verify_api_key};
use crate::collections::{
    Collection, CollectionIndex, CollectionMeta,
    validate_batch_size, validate_collection_name, validate_query_dimension,
    validate_top_k, validate_vector_dimensions,
};
use crate::state::AppState;

// ─────────────────────────────────────────────────────────────────────────────
// JSON request / response DTOs
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct CreateCollectionBody {
    pub name: Option<String>,
    pub dimension: usize,
    #[serde(default = "default_index_type")]
    pub index_type: String,
    #[serde(default = "default_metric")]
    pub distance_metric: String,
    pub n_clusters: Option<usize>,
    pub n_subvectors: Option<usize>,
    pub nprobe: Option<usize>,
    pub m: Option<usize>,
    pub ef_construction: Option<usize>,
    pub ef_search: Option<usize>,
    pub enable_reranking: Option<bool>,
    pub rerank_factor: Option<usize>,
}

fn default_index_type() -> String {
    "ivf_pq".to_string()
}
fn default_metric() -> String {
    "euclidean".to_string()
}

#[derive(Debug, Serialize, Deserialize)]
pub struct VectorRecord {
    pub id: u64,
    pub vector: Vec<f32>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct UpsertBody {
    pub vectors: Vec<VectorRecord>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchBody {
    pub query: Vec<f32>,
    pub top_k: usize,
    pub filter: Option<serde_json::Value>,
    pub max_distance: Option<f32>,
    pub nprobe: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SingleQueryBody {
    pub query: Vec<f32>,
    pub top_k: usize,
    pub filter: Option<serde_json::Value>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BatchSearchBody {
    pub queries: Vec<SingleQueryBody>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DeleteVectorsBody {
    pub ids: Vec<u64>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BuildIndexBody {
    pub vectors: Vec<VectorRecord>,
    #[serde(default)]
    pub n_clusters: usize,
    #[serde(default)]
    pub n_subvectors: usize,
    #[serde(default = "default_auto_tune")]
    pub auto_tune: bool,
}

fn default_auto_tune() -> bool {
    true
}

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub id: u64,
    pub distance: f32,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub latency_ms: f32,
    pub vectors_scanned: usize,
}

#[derive(Debug, Serialize)]
pub struct CollectionInfo {
    pub name: String,
    pub dimension: usize,
    pub index_type: String,
    pub vector_count: usize,
    pub memory_mb: f32,
    pub health: String,
}

#[derive(Debug, Serialize)]
pub struct StatsResponse {
    pub total_vectors: u64,
    pub total_collections: usize,
    pub total_memory_mb: f32,
    pub avg_search_latency_ms: f32,
    pub total_searches: u64,
    pub uptime_seconds: u64,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
    pub error: String,
    pub code: u16,
}

// ─────────────────────────────────────────────────────────────────────────────
// Handler helpers
// ─────────────────────────────────────────────────────────────────────────────

fn parse_metric(s: &str) -> DistanceMetric {
    match s.to_lowercase().as_str() {
        "cosine" => DistanceMetric::Cosine,
        "dot_product" | "inner_product" => DistanceMetric::DotProduct,
        "euclidean_squared" => DistanceMetric::EuclideanSquared,
        _ => DistanceMetric::Euclidean,
    }
}

fn api_error(code: StatusCode, message: impl Into<String>) -> impl IntoResponse {
    let msg = message.into();
    let status = code.as_u16();
    (code, Json(ErrorResponse { error: msg, code: status }))
}

/// Authenticate via X-API-Key or Authorization: Bearer header.
fn check_auth(headers: &HeaderMap, config: &ForgeConfig) -> Result<(), StatusCode> {
    let auth_cfg = match &config.server.auth {
        Some(a) if a.required => a,
        _ => return Ok(()),
    };

    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok());
    let x_api_key = headers
        .get("x-api-key")
        .and_then(|v| v.to_str().ok());

    match extract_api_key(auth, x_api_key) {
        Some(key) if verify_api_key(&key, &auth_cfg.api_keys) => Ok(()),
        _ => Err(StatusCode::UNAUTHORIZED),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Route handlers
// ─────────────────────────────────────────────────────────────────────────────

type AppStateArc = Arc<crate::state::AppState>;

async fn health_handler() -> impl IntoResponse {
    Json(serde_json::json!({
        "status": "healthy",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn list_collections_handler(
    State(state): State<AppStateArc>,
) -> impl IntoResponse {
    let names = state.list_collections();
    Json(serde_json::json!({ "collections": names }))
}

async fn create_collection_handler(
    State(state): State<AppStateArc>,
    headers: HeaderMap,
    axum::extract::Extension(config): axum::extract::Extension<ForgeConfig>,
    Json(body): Json<CreateCollectionBody>,
) -> impl IntoResponse {
    if check_auth(&headers, &config).is_err() {
        return api_error(StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }

    let name = body.name.clone().unwrap_or_else(|| "default".to_string());

    // Validate collection name
    if let Err(e) = validate_collection_name(&name) {
        return api_error(StatusCode::BAD_REQUEST, e).into_response();
    }

    let metric = parse_metric(&body.distance_metric);

    let meta = CollectionMeta {
        name: name.clone(),
        dimension: body.dimension,
        index_type: body.index_type.clone(),
        distance_metric: body.distance_metric.clone(),
        n_clusters: body.n_clusters.unwrap_or(1024),
        n_subvectors: body.n_subvectors.unwrap_or(32),
        nprobe: body.nprobe.unwrap_or(16),
        m: body.m.unwrap_or(16),
        ef_construction: body.ef_construction.unwrap_or(200),
        ef_search: body.ef_search.unwrap_or(50),
        enable_reranking: body.enable_reranking.unwrap_or(false),
        rerank_factor: body.rerank_factor.unwrap_or(4),
    };

    let collection = match body.index_type.as_str() {
        "hnsw" => Collection::build_hnsw(name.clone(), meta, metric),
        _ => Collection::build_brute_force(meta, metric),
    };

    match state.create_collection(&name, collection) {
        Ok(_) => Json(serde_json::json!({ "created": true, "name": name })).into_response(),
        Err(e) => api_error(StatusCode::CONFLICT, e).into_response(),
    }
}

#[instrument(skip(state), fields(collection = %name))]
async fn search_handler(
    State(state): State<AppStateArc>,
    headers: HeaderMap,
    axum::extract::Extension(config): axum::extract::Extension<ForgeConfig>,
    Path(name): Path<String>,
    Json(body): Json<SearchBody>,
) -> impl IntoResponse {
    if check_auth(&headers, &config).is_err() {
        return api_error(StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }

    // Validate top_k
    if let Err(e) = validate_top_k(body.top_k) {
        return api_error(StatusCode::BAD_REQUEST, e).into_response();
    }

    let coll = match state.get_collection(&name) {
        Some(c) => c,
        None => return api_error(StatusCode::NOT_FOUND, format!("collection '{name}' not found")).into_response(),
    };

    // Validate query dimension
    {
        let guard = coll.read();
        let expected_dim = guard.effective_dimension();
        if let Err(e) = validate_query_dimension(&body.query, expected_dim) {
            return api_error(StatusCode::BAD_REQUEST, e).into_response();
        }
    }

    let query = body.query.clone();
    let top_k = body.top_k;
    let start = Instant::now();
    let (raw_results, _latency_us) = match tokio::task::spawn_blocking(move || {
        coll.read().search(&query, top_k)
    })
    .await
    {
        Ok(r) => r,
        Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("search task panicked: {e}")).into_response(),
    };
    let latency_ms = start.elapsed().as_millis() as f32;

    let max_dist = body.max_distance.unwrap_or(f32::INFINITY);
    let results: Vec<SearchResult> = raw_results
        .into_iter()
        .filter(|&(_, d)| d <= max_dist)
        .map(|(id, distance)| SearchResult { id, distance, metadata: None })
        .collect();

    state
        .stats
        .total_searches
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    metrics::histogram!("forge_search_latency_seconds", "collection" => name.clone()).record(start.elapsed().as_secs_f64());

    Json(SearchResponse {
        results,
        latency_ms,
        vectors_scanned: 0,
    })
    .into_response()
}

async fn get_collection_info_handler(
    State(state): State<AppStateArc>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    let coll = match state.get_collection(&name) {
        Some(c) => c,
        None => return api_error(StatusCode::NOT_FOUND, format!("collection '{name}' not found")).into_response(),
    };

    let guard = coll.read();
    Json(CollectionInfo {
        name: name.clone(),
        dimension: guard.meta.dimension,
        index_type: guard.meta.index_type.clone(),
        vector_count: guard.len(),
        memory_mb: guard.memory_bytes() as f32 / (1024.0 * 1024.0),
        health: guard.index.health(),
    })
    .into_response()
}

async fn drop_collection_handler(
    State(state): State<AppStateArc>,
    headers: HeaderMap,
    axum::extract::Extension(config): axum::extract::Extension<ForgeConfig>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if check_auth(&headers, &config).is_err() {
        return api_error(StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }

    let dropped = state.drop_collection(&name);
    Json(serde_json::json!({ "dropped": dropped })).into_response()
}

async fn upsert_handler(
    State(state): State<AppStateArc>,
    headers: HeaderMap,
    axum::extract::Extension(config): axum::extract::Extension<ForgeConfig>,
    Path(name): Path<String>,
    Json(body): Json<UpsertBody>,
) -> impl IntoResponse {
    if check_auth(&headers, &config).is_err() {
        return api_error(StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }

    // Validate batch size
    if let Err(e) = validate_batch_size(body.vectors.len()) {
        return api_error(StatusCode::BAD_REQUEST, e).into_response();
    }

    let coll = match state.get_collection(&name) {
        Some(c) => c,
        None => return api_error(StatusCode::NOT_FOUND, format!("collection '{name}' not found")).into_response(),
    };

    // Validate vector dimensions
    {
        let guard = coll.read();
        let expected_dim = guard.effective_dimension();
        let vecs: Vec<Vec<f32>> = body.vectors.iter().map(|r| r.vector.clone()).collect();
        if let Err(e) = validate_vector_dimensions(&vecs, expected_dim) {
            return api_error(StatusCode::BAD_REQUEST, e).into_response();
        }
    }

    let count = body.vectors.len();

    // WAL: record upserts before applying
    {
        let mut wal = state.wal.lock();
        for record in &body.vectors {
            if let Err(e) = wal.append(&forge_db::WalOperation::Insert {
                id: record.id,
                vector: record.vector.clone(),
                metadata: record.metadata.as_ref().map(|m| m.to_string()),
            }) {
                tracing::error!(error = %e, "WAL append failed for upsert");
                return api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("WAL write failed: {e}")).into_response();
            }
        }
    }

    // Insert into the in-memory index (CPU-bound for HNSW/IVF-PQ)
    {
        let vectors: Vec<Vector> = body.vectors
            .iter()
            .map(|r| Vector::new(r.id, r.vector.clone()))
            .collect();
        let metadata_jsons: Vec<Option<String>> = body.vectors
            .iter()
            .map(|r| r.metadata.as_ref().map(|m| m.to_string()))
            .collect();
        if let Err(e) = tokio::task::spawn_blocking(move || {
            coll.write().upsert_batch_with_metadata(vectors, metadata_jsons);
        })
        .await
        {
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("upsert task panicked: {e}")).into_response();
        }
    }

    state
        .stats
        .total_upserts
        .fetch_add(count as u64, std::sync::atomic::Ordering::Relaxed);

    metrics::counter!("forge_upsert_total", "collection" => name.clone()).increment(count as u64);

    Json(serde_json::json!({ "upserted": count })).into_response()
}

async fn delete_vectors_handler(
    State(state): State<AppStateArc>,
    headers: HeaderMap,
    axum::extract::Extension(config): axum::extract::Extension<ForgeConfig>,
    Path(name): Path<String>,
    Json(body): Json<DeleteVectorsBody>,
) -> impl IntoResponse {
    if check_auth(&headers, &config).is_err() {
        return api_error(StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }

    let coll = match state.get_collection(&name) {
        Some(c) => c,
        None => return api_error(StatusCode::NOT_FOUND, format!("collection '{name}' not found")).into_response(),
    };

    // WAL: record deletes before applying to index
    {
        let mut wal = state.wal.lock();
        for &id in &body.ids {
            if let Err(e) = wal.append(&forge_db::WalOperation::Delete { id }) {
                tracing::error!(error = %e, "WAL append failed for delete");
                return api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("WAL write failed: {e}")).into_response();
            }
        }
    }

    let mut deleted = 0u64;
    {
        let mut guard = coll.write();
        for &id in &body.ids {
            if guard.delete(id) {
                deleted += 1;
            }
        }
    }

    metrics::counter!("forge_delete_total", "collection" => name.clone()).increment(deleted);

    Json(serde_json::json!({ "deleted": deleted })).into_response()
}

async fn batch_search_handler(
    State(state): State<AppStateArc>,
    headers: HeaderMap,
    axum::extract::Extension(config): axum::extract::Extension<ForgeConfig>,
    Path(name): Path<String>,
    Json(body): Json<BatchSearchBody>,
) -> impl IntoResponse {
    if check_auth(&headers, &config).is_err() {
        return api_error(StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }

    // Validate top_k for each query
    for (i, q) in body.queries.iter().enumerate() {
        if let Err(e) = validate_top_k(q.top_k) {
            return api_error(StatusCode::BAD_REQUEST, format!("query[{}]: {}", i, e)).into_response();
        }
    }

    let coll = match state.get_collection(&name) {
        Some(c) => c,
        None => return api_error(StatusCode::NOT_FOUND, format!("collection '{name}' not found")).into_response(),
    };

    // Validate query dimensions
    {
        let guard = coll.read();
        let expected_dim = guard.effective_dimension();
        for (i, q) in body.queries.iter().enumerate() {
            if let Err(e) = validate_query_dimension(&q.query, expected_dim) {
                return api_error(StatusCode::BAD_REQUEST, format!("query[{}]: {}", i, e)).into_response();
            }
        }
    }

    let start = Instant::now();
    let queries = body.queries.clone();
    let all_results = match tokio::task::spawn_blocking(move || {
        queries.par_iter().map(|q| {
            let (raw, _) = coll.read().search(&q.query, q.top_k);
            let items: Vec<SearchResult> = raw
                .into_iter()
                .map(|(id, distance)| SearchResult { id, distance, metadata: None })
                .collect();
            SearchResponse {
                results: items,
                latency_ms: 0.0,
                vectors_scanned: 0,
            }
        }).collect::<Vec<_>>()
    })
    .await
    {
        Ok(r) => r,
        Err(e) => return api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("batch_search task panicked: {e}")).into_response(),
    };

    let total_ms = start.elapsed().as_millis() as f32;
    state
        .stats
        .total_searches
        .fetch_add(body.queries.len() as u64, std::sync::atomic::Ordering::Relaxed);

    Json(serde_json::json!({
        "results": all_results,
        "total_latency_ms": total_ms,
    }))
    .into_response()
}

async fn build_index_handler(
    State(state): State<AppStateArc>,
    headers: HeaderMap,
    axum::extract::Extension(config): axum::extract::Extension<ForgeConfig>,
    Path(name): Path<String>,
    Json(body): Json<BuildIndexBody>,
) -> impl IntoResponse {
    if check_auth(&headers, &config).is_err() {
        return api_error(StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }

    let coll = match state.get_collection(&name) {
        Some(c) => c,
        None => return api_error(StatusCode::NOT_FOUND, format!("collection '{name}' not found")).into_response(),
    };

    let vectors: Vec<Vector> = body.vectors
        .iter()
        .map(|r| Vector::new(r.id, r.vector.clone()))
        .collect();
    let n_clusters = body.n_clusters;
    let n_subvectors = body.n_subvectors;
    let auto_tune = body.auto_tune;

    let result = tokio::task::spawn_blocking(move || {
        coll.write().build_ivfpq(vectors, n_clusters, n_subvectors, auto_tune)
    })
    .await;

    match result {
        Ok(Ok(count)) => Json(serde_json::json!({
            "success": true,
            "vectors_indexed": count,
        })).into_response(),
        Ok(Err(e)) => api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        Err(e) => api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("build_index task panicked: {e}")).into_response(),
    }
}

async fn checkpoint_handler(
    State(state): State<AppStateArc>,
    headers: HeaderMap,
    axum::extract::Extension(config): axum::extract::Extension<ForgeConfig>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if check_auth(&headers, &config).is_err() {
        return api_error(StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }

    let data_dir = &config.data_dir;
    if let Some(coll) = state.get_collection(&name) {
        if let Err(e) = coll.read().save_to_disk(data_dir) {
            return api_error(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response();
        }
    }

    let seq = {
        let mut wal = state.wal.lock();
        let s = wal.next_sequence();
        if let Err(e) = wal.checkpoint(s.saturating_sub(1)) {
            tracing::error!(error = %e, "WAL checkpoint failed");
        }
        s
    };

    Json(serde_json::json!({ "success": true, "checkpoint_seq": seq })).into_response()
}

async fn compact_handler(
    State(state): State<AppStateArc>,
    headers: HeaderMap,
    axum::extract::Extension(config): axum::extract::Extension<ForgeConfig>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    if check_auth(&headers, &config).is_err() {
        return api_error(StatusCode::UNAUTHORIZED, "unauthorized").into_response();
    }

    let coll = match state.get_collection(&name) {
        Some(c) => c,
        None => return api_error(StatusCode::NOT_FOUND, format!("collection '{name}' not found")).into_response(),
    };

    let removed = {
        let mut guard = coll.write();
        match &mut guard.index {
            CollectionIndex::IvfPq(idx) => {
                let before = idx.num_tombstones();
                idx.compact();
                before
            }
            _ => 0,
        }
    };

    Json(serde_json::json!({ "success": true, "vectors_removed": removed })).into_response()
}

async fn stats_handler(State(state): State<AppStateArc>) -> impl IntoResponse {
    Json(StatsResponse {
        total_vectors: state.total_vectors(),
        total_collections: state.list_collections().len(),
        total_memory_mb: state.total_memory_bytes() as f32 / (1024.0 * 1024.0),
        avg_search_latency_ms: state.stats.avg_search_latency_ms(),
        total_searches: state
            .stats
            .total_searches
            .load(std::sync::atomic::Ordering::Relaxed),
        uptime_seconds: state.stats.uptime_seconds(),
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Router assembly
// ─────────────────────────────────────────────────────────────────────────────

pub fn build_router(state: Arc<AppState>, config: ForgeConfig) -> Router {
    let rps = config.server.requests_per_second;

    let mut app = Router::new()
        .route("/health", get(health_handler))
        .route("/v1/stats", get(stats_handler))
        .route("/v1/collections", get(list_collections_handler))
        .route("/v1/collections", post(create_collection_handler))
        .route("/v1/collections/:name", get(get_collection_info_handler))
        .route("/v1/collections/:name", delete(drop_collection_handler))
        .route("/v1/collections/:name/vectors", post(upsert_handler))
        .route("/v1/collections/:name/vectors", delete(delete_vectors_handler))
        .route("/v1/collections/:name/search", post(search_handler))
        .route("/v1/collections/:name/batch-search", post(batch_search_handler))
        .route("/v1/collections/:name/build", post(build_index_handler))
        .route("/v1/collections/:name/checkpoint", post(checkpoint_handler))
        .route("/v1/collections/:name/compact", post(compact_handler))
        .layer(axum::extract::Extension(config))
        .layer(TraceLayer::new_for_http())
        .layer(CompressionLayer::new())
        .layer(CorsLayer::permissive())
        .layer(HttpTimeoutLayer::new(std::time::Duration::from_secs(30)))
        .layer(RequestBodyLimitLayer::new(256 * 1024 * 1024)) // 256 MiB
        .with_state(state);

    // Apply rate limiting if configured (requests_per_second > 0).
    // Uses ConcurrencyLimitLayer because tower's RateLimitLayer does not
    // produce a Clone service, which axum requires.
    if rps > 0 {
        app = app.layer(tower::limit::ConcurrencyLimitLayer::new(rps as usize));
    }

    app
}

/// Start the HTTP/REST server with graceful shutdown support.
pub async fn serve(
    addr: SocketAddr,
    state: Arc<AppState>,
    config: ForgeConfig,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let app = build_router(state, config);

    info!(%addr, "HTTP server listening");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown)
        .await
        .map_err(|e| anyhow::anyhow!("HTTP server error: {e}"))
}
