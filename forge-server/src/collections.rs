//! Collection abstraction — a named, typed vector index with config.

use std::path::Path;
use std::time::Instant;

use anyhow::anyhow;
use forge_db::{
    distance::DistanceMetric,
    metadata::MetadataStore,
    vector::Vector,
    BruteForceIndex, HNSWIndex, IVFPQIndex, IVFPQIndexBuilder, Persistable,
    persistence::IndexType,
};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Input validation constants and helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Maximum allowed value for top_k in search queries.
pub const MAX_TOP_K: usize = 10_000;

/// Maximum number of vectors in a single batch insert.
pub const MAX_BATCH_SIZE: usize = 100_000;

/// Maximum length for a collection name.
pub const MAX_COLLECTION_NAME_LEN: usize = 128;

/// Validate a collection name: non-empty, <= 128 chars, matches `[a-zA-Z0-9_-]+`.
pub fn validate_collection_name(name: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err("collection name must not be empty".to_string());
    }
    if name.len() > MAX_COLLECTION_NAME_LEN {
        return Err(format!(
            "collection name exceeds maximum length of {} characters",
            MAX_COLLECTION_NAME_LEN
        ));
    }
    if !name
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err(
            "collection name must match pattern [a-zA-Z0-9_-]+".to_string(),
        );
    }
    Ok(())
}

/// Validate that top_k is within bounds.
pub fn validate_top_k(k: usize) -> Result<(), String> {
    if k == 0 {
        return Err("top_k must be greater than 0".to_string());
    }
    if k > MAX_TOP_K {
        return Err(format!("top_k must not exceed {}", MAX_TOP_K));
    }
    Ok(())
}

/// Validate batch size.
pub fn validate_batch_size(count: usize) -> Result<(), String> {
    if count > MAX_BATCH_SIZE {
        return Err(format!(
            "batch size {} exceeds maximum of {} vectors",
            count, MAX_BATCH_SIZE
        ));
    }
    Ok(())
}

/// Validate that all vectors have the expected dimension. If `expected_dim` is 0
/// (unconfigured), infer from the first vector and return the inferred dimension.
pub fn validate_vector_dimensions(
    vectors: &[Vec<f32>],
    expected_dim: usize,
) -> Result<usize, String> {
    if vectors.is_empty() {
        return Ok(expected_dim);
    }
    let dim = if expected_dim == 0 {
        vectors[0].len()
    } else {
        expected_dim
    };
    for (i, v) in vectors.iter().enumerate() {
        if v.len() != dim {
            return Err(format!(
                "vector at index {} has dimension {} but expected {}",
                i,
                v.len(),
                dim
            ));
        }
    }
    Ok(dim)
}

/// Validate that a query vector matches the expected dimension.
pub fn validate_query_dimension(query: &[f32], expected_dim: usize) -> Result<(), String> {
    if expected_dim == 0 {
        return Ok(());
    }
    if query.len() != expected_dim {
        return Err(format!(
            "query vector has dimension {} but collection expects {}",
            query.len(),
            expected_dim
        ));
    }
    Ok(())
}

/// Convert a serde_json::Value to a MetadataValue.
fn json_value_to_metadata(val: &serde_json::Value) -> forge_db::MetadataValue {
    match val {
        serde_json::Value::Bool(b) => forge_db::MetadataValue::Boolean(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                forge_db::MetadataValue::Integer(i)
            } else if let Some(f) = n.as_f64() {
                forge_db::MetadataValue::Float(f)
            } else {
                forge_db::MetadataValue::Null
            }
        }
        serde_json::Value::String(s) => forge_db::MetadataValue::String(s.clone()),
        serde_json::Value::Array(arr) => {
            let items: Vec<forge_db::MetadataValue> = arr.iter().map(json_value_to_metadata).collect();
            forge_db::MetadataValue::Array(items)
        }
        _ => forge_db::MetadataValue::Null,
    }
}

/// Configuration stored alongside an index for reload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionMeta {
    pub name: String,
    pub dimension: usize,
    pub index_type: String,
    pub distance_metric: String,
    pub n_clusters: usize,
    pub n_subvectors: usize,
    pub nprobe: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub enable_reranking: bool,
    pub rerank_factor: usize,
}

/// A named vector collection.
pub enum CollectionIndex {
    IvfPq(IVFPQIndex),
    Hnsw(HNSWIndex),
    BruteForce(BruteForceIndex),
}

impl CollectionIndex {
    pub fn len(&self) -> usize {
        match self {
            Self::IvfPq(idx) => idx.len(),
            Self::Hnsw(idx) => idx.len(),
            Self::BruteForce(idx) => idx.len(),
        }
    }

    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[allow(dead_code)]
    pub fn dimension(&self) -> usize {
        match self {
            Self::IvfPq(idx) => idx.dim(),
            Self::Hnsw(idx) => idx.dimension(),
            Self::BruteForce(idx) => idx.dimension(),
        }
    }

    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        match self {
            Self::IvfPq(idx) => idx.search(query, k),
            Self::Hnsw(idx) => idx.search(query, k),
            Self::BruteForce(idx) => idx.search(query, k),
        }
    }

    /// Insert a batch of vectors.
    pub fn upsert_batch(&mut self, vectors: Vec<Vector>) {
        match self {
            Self::IvfPq(idx) => idx.insert_batch(vectors),
            Self::Hnsw(idx) => {
                for v in vectors {
                    idx.add(v);
                }
                // Rebuild the flat graph for lock-free search after each batch.
                idx.finalize();
            }
            Self::BruteForce(idx) => {
                for v in vectors {
                    idx.add(v);
                }
            }
        }
    }

    pub fn delete(&mut self, id: u64) -> bool {
        match self {
            Self::IvfPq(idx) => idx.delete(id),
            Self::Hnsw(_) | Self::BruteForce(_) => false,
        }
    }

    pub fn memory_bytes(&self) -> usize {
        match self {
            Self::IvfPq(idx) => idx.statistics().memory_bytes,
            Self::Hnsw(idx) => idx.len() * idx.dimension() * 4,
            Self::BruteForce(idx) => idx.len() * idx.dimension() * 4,
        }
    }

    pub fn health(&self) -> String {
        match self {
            Self::IvfPq(idx) => match idx.health_check() {
                forge_db::HealthStatus::Healthy => "healthy".to_string(),
                forge_db::HealthStatus::Warning(_) => "degraded".to_string(),
                forge_db::HealthStatus::Unhealthy(_) => "unhealthy".to_string(),
            },
            _ => "healthy".to_string(),
        }
    }
}

/// A named collection wrapping an index with metadata.
pub struct Collection {
    pub meta: CollectionMeta,
    pub index: CollectionIndex,
    /// Per-vector metadata store for filtered search.
    pub metadata: MetadataStore,
}

impl Collection {
    /// Create a new empty HNSW collection.
    pub fn build_hnsw(name: String, meta: CollectionMeta, metric: DistanceMetric) -> Self {
        let index = HNSWIndex::new(meta.m, meta.ef_construction, metric);
        let mut final_meta = meta;
        final_meta.name = name;
        Self {
            meta: final_meta,
            index: CollectionIndex::Hnsw(index),
            metadata: MetadataStore::new(),
        }
    }

    /// Create a new empty BruteForce collection.
    pub fn build_brute_force(meta: CollectionMeta, metric: DistanceMetric) -> Self {
        let index = BruteForceIndex::new(metric);
        Self {
            meta,
            index: CollectionIndex::BruteForce(index),
            metadata: MetadataStore::new(),
        }
    }

    /// Insert a batch of vectors.
    pub fn upsert_batch(&mut self, vectors: Vec<Vector>) {
        self.index.upsert_batch(vectors);
    }

    /// Insert a batch of vectors with optional JSON metadata.
    pub fn upsert_batch_with_metadata(&mut self, vectors: Vec<Vector>, metadata_jsons: Vec<Option<String>>) {
        // Store metadata for each vector that has it.
        for (v, meta_json) in vectors.iter().zip(metadata_jsons.iter()) {
            if let Some(json) = meta_json {
                if let Ok(serde_json::Value::Object(map)) = serde_json::from_str(json.as_str()) {
                    for (key, val) in map {
                        let mv = json_value_to_metadata(&val);
                        self.metadata.insert(v.id, key, mv);
                    }
                }
            }
        }
        self.index.upsert_batch(vectors);
    }

    /// Search the collection.
    pub fn search(&self, query: &[f32], k: usize) -> (Vec<(u64, f32)>, u64) {
        let start = Instant::now();
        let results = self.index.search(query, k);
        let elapsed_us = start.elapsed().as_micros() as u64;
        (results, elapsed_us)
    }

    /// Build an IVF-PQ index from the given vectors, replacing the current index.
    /// Optionally specify n_clusters and n_subvectors; if 0, auto-tune is used.
    pub fn build_ivfpq(
        &mut self,
        vectors: Vec<Vector>,
        n_clusters: usize,
        n_subvectors: usize,
        auto_tune: bool,
    ) -> anyhow::Result<usize> {
        let count = vectors.len();
        let mut builder = IVFPQIndexBuilder::new()
            .vectors(vectors)
            .auto_tune(auto_tune);

        if n_clusters > 0 {
            builder = builder.num_clusters(n_clusters);
        }
        if n_subvectors > 0 {
            builder = builder.num_subvectors(n_subvectors);
        }

        let new_index = builder.build().map_err(|e| anyhow!("IVF-PQ build failed: {e}"))?;
        self.index = CollectionIndex::IvfPq(new_index);
        Ok(count)
    }

    /// Delete a vector by ID.
    pub fn delete(&mut self, id: u64) -> bool {
        self.index.delete(id)
    }

    /// Return the effective dimension for this collection.
    /// Uses the meta dimension if set, otherwise queries the index.
    pub fn effective_dimension(&self) -> usize {
        if self.meta.dimension > 0 {
            self.meta.dimension
        } else if !self.index.is_empty() {
            self.index.dimension()
        } else {
            0
        }
    }

    /// Number of vectors in the collection.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Returns true if the collection contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.index.len() == 0
    }

    /// Estimated memory usage in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.index.memory_bytes()
    }

    /// Detect the index type from the file header and load accordingly.
    pub fn load_from_disk(name: &str, path: &Path) -> anyhow::Result<Self> {
        // Peek at the file header to determine the index type.
        let raw = std::fs::read(path)
            .map_err(|e| anyhow!("reading {}: {e}", path.display()))?;

        if raw.len() < forge_db::persistence::FileHeader::SIZE {
            return Err(anyhow!("file too small: {}", path.display()));
        }

        let header = forge_db::persistence::FileHeader::from_bytes(
            &raw[..forge_db::persistence::FileHeader::SIZE],
        )
        .map_err(|e| anyhow!("invalid header in {}: {e}", path.display()))?;

        match header.index_type {
            IndexType::IvfPq => {
                let index = IVFPQIndex::load(path)
                    .map_err(|e| anyhow!("loading IVF-PQ index from {}: {e}", path.display()))?;
                let dim = index.dim();
                let n_parts = index.n_partitions();
                let nprobe = index.nprobe();
                Ok(Self {
                    meta: CollectionMeta {
                        name: name.to_string(),
                        dimension: dim,
                        index_type: "ivf_pq".to_string(),
                        distance_metric: "euclidean".to_string(),
                        n_clusters: n_parts,
                        n_subvectors: 0,
                        nprobe,
                        m: 16,
                        ef_construction: 200,
                        ef_search: 50,
                        enable_reranking: false,
                        rerank_factor: 4,
                    },
                    index: CollectionIndex::IvfPq(index),
                    metadata: MetadataStore::new(),
                })
            }
            IndexType::BruteForce => {
                let index = BruteForceIndex::load(path)
                    .map_err(|e| anyhow!("loading BruteForce index from {}: {e}", path.display()))?;
                let dim = index.dimension();
                Ok(Self {
                    meta: CollectionMeta {
                        name: name.to_string(),
                        dimension: dim,
                        index_type: "brute_force".to_string(),
                        distance_metric: "euclidean".to_string(),
                        n_clusters: 0,
                        n_subvectors: 0,
                        nprobe: 0,
                        m: 16,
                        ef_construction: 200,
                        ef_search: 50,
                        enable_reranking: false,
                        rerank_factor: 4,
                    },
                    index: CollectionIndex::BruteForce(index),
                    metadata: MetadataStore::new(),
                })
            }
            other => Err(anyhow!("unsupported index type {:?} in {}", other, path.display())),
        }
    }

    /// Save the collection to disk.
    pub fn save_to_disk(&self, dir: &Path) -> anyhow::Result<()> {
        let path = dir.join(format!("{}.fdb", self.meta.name));
        match &self.index {
            CollectionIndex::IvfPq(idx) => idx
                .save(&path)
                .map_err(|e| anyhow!("saving IVF-PQ index: {e}")),
            CollectionIndex::BruteForce(idx) => idx
                .save(&path)
                .map_err(|e| anyhow!("saving BruteForce index: {e}")),
            CollectionIndex::Hnsw(_) => {
                tracing::warn!(
                    collection = %self.meta.name,
                    "HNSW persistence not yet implemented, skipping"
                );
                Ok(())
            }
        }
    }
}
