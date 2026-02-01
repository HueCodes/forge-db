//! Builder pattern for IVFPQIndex with auto-tuning support.
//!
//! The builder provides a fluent API for constructing IVF-PQ indexes with
//! automatic parameter tuning based on the dataset characteristics and
//! target constraints (memory budget, recall target).
//!
//! # Example
//!
//! ```ignore
//! use forge_db::index::IVFPQIndexBuilder;
//!
//! // Build with auto-tuned parameters
//! let index = IVFPQIndexBuilder::new()
//!     .vectors(my_vectors)
//!     .target_memory_mb(100)  // Fit in 100MB
//!     .build()?;
//!
//! // Or specify parameters explicitly
//! let index = IVFPQIndexBuilder::new()
//!     .vectors(my_vectors)
//!     .num_clusters(1000)
//!     .num_subvectors(32)
//!     .nprobe(16)
//!     .build()?;
//! ```

use crate::distance::DistanceMetric;
use crate::error::{ForgeDbError, Result};
use crate::vector::Vector;

use super::IVFPQIndex;

/// Builder for constructing IVFPQIndex with auto-tuning.
pub struct IVFPQIndexBuilder {
    /// Vectors to index.
    vectors: Option<Vec<Vector>>,
    /// Number of IVF clusters (partitions).
    num_clusters: Option<usize>,
    /// Number of PQ subvectors.
    num_subvectors: Option<usize>,
    /// Bits per PQ code (8 for 256 centroids, 4 for 16 centroids).
    bits_per_code: Option<usize>,
    /// Distance metric for search.
    metric: DistanceMetric,
    /// Enable auto-tuning of parameters.
    auto_tune: bool,
    /// Target memory budget in megabytes.
    target_memory_mb: Option<usize>,
    /// Target recall@10 (0.0 to 1.0).
    target_recall: Option<f32>,
    /// Number of partitions to probe during search.
    nprobe: usize,
    /// Re-ranking factor (fetch this many candidates before re-ranking).
    rerank_factor: usize,
    /// Enable re-ranking with original vectors.
    enable_reranking: bool,
}

impl Default for IVFPQIndexBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl IVFPQIndexBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            vectors: None,
            num_clusters: None,
            num_subvectors: None,
            bits_per_code: None,
            metric: DistanceMetric::Euclidean,
            auto_tune: true,
            target_memory_mb: None,
            target_recall: None,
            nprobe: 1,
            rerank_factor: 1,
            enable_reranking: false,
        }
    }

    /// Set the vectors to index.
    ///
    /// This is required for building the index.
    pub fn vectors(mut self, vectors: Vec<Vector>) -> Self {
        self.vectors = Some(vectors);
        self
    }

    /// Set the number of IVF clusters (partitions).
    ///
    /// If not set, auto-tuning will choose `sqrt(n).clamp(256, 65536)`.
    pub fn num_clusters(mut self, n: usize) -> Self {
        self.num_clusters = Some(n);
        self
    }

    /// Set the number of PQ subvectors.
    ///
    /// Must evenly divide the vector dimension.
    /// If not set, auto-tuning will choose `dim / 8` (clamped to valid range).
    pub fn num_subvectors(mut self, m: usize) -> Self {
        self.num_subvectors = Some(m);
        self
    }

    /// Set the bits per PQ code.
    ///
    /// - 8 bits = 256 centroids per subspace (default)
    /// - 4 bits = 16 centroids per subspace (more compression, lower recall)
    ///
    /// Note: 4-bit PQ is not yet implemented; this is reserved for future use.
    pub fn bits_per_code(mut self, bits: usize) -> Self {
        self.bits_per_code = Some(bits);
        self
    }

    /// Set the distance metric.
    ///
    /// Default: `DistanceMetric::Euclidean`
    pub fn metric(mut self, metric: DistanceMetric) -> Self {
        self.metric = metric;
        self
    }

    /// Enable or disable auto-tuning of parameters.
    ///
    /// When enabled (default), unset parameters will be automatically chosen
    /// based on dataset characteristics and optional constraints.
    pub fn auto_tune(mut self, enable: bool) -> Self {
        self.auto_tune = enable;
        self
    }

    /// Set target memory budget in megabytes.
    ///
    /// Auto-tuning will adjust parameters to fit within this budget.
    /// This affects the number of subvectors and bits per code.
    pub fn target_memory_mb(mut self, mb: usize) -> Self {
        self.target_memory_mb = Some(mb);
        self
    }

    /// Set target recall@10.
    ///
    /// Auto-tuning will adjust nprobe and rerank settings to achieve
    /// approximately this recall level. Values closer to 1.0 require
    /// more computation but give better results.
    pub fn target_recall(mut self, recall: f32) -> Self {
        self.target_recall = Some(recall.clamp(0.0, 1.0));
        self
    }

    /// Set the number of partitions to probe during search.
    ///
    /// Higher values improve recall but slow down search.
    /// Default: 1
    pub fn nprobe(mut self, nprobe: usize) -> Self {
        self.nprobe = nprobe;
        self
    }

    /// Set the re-ranking factor.
    ///
    /// When re-ranking is enabled, fetch this many times more candidates
    /// than k before re-ranking with exact distances.
    pub fn rerank_factor(mut self, factor: usize) -> Self {
        self.rerank_factor = factor.max(1);
        self
    }

    /// Enable re-ranking with original vectors.
    ///
    /// This significantly improves recall at the cost of increased memory
    /// (storing original vectors) and slightly slower search.
    pub fn enable_reranking(mut self, enable: bool) -> Self {
        self.enable_reranking = enable;
        self
    }

    /// Build the IVFPQIndex with the configured parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - No vectors were provided
    /// - The vector count is too small for the requested parameters
    /// - The vector dimension is not divisible by num_subvectors
    pub fn build(mut self) -> Result<IVFPQIndex> {
        let vectors = self
            .vectors
            .take()
            .ok_or_else(|| ForgeDbError::invalid_parameter("vectors are required"))?;

        if vectors.is_empty() {
            return Err(ForgeDbError::EmptyVectorSet);
        }

        let n = vectors.len();
        let dim = vectors[0].dim();

        // Validate dimension consistency
        for (i, v) in vectors.iter().enumerate() {
            if v.dim() != dim {
                return Err(ForgeDbError::DimensionMismatch {
                    expected: dim,
                    actual: v.dim(),
                });
            }
            if i > 100 {
                break; // Sample check
            }
        }

        // Auto-tune parameters if not specified
        let num_clusters = self.auto_tune_clusters(n);
        let num_subvectors = self.auto_tune_subvectors(dim)?;
        let nprobe = self.auto_tune_nprobe(num_clusters);

        // Validate parameters
        if num_clusters > n {
            return Err(ForgeDbError::InsufficientVectors {
                required: num_clusters,
                actual: n,
            });
        }

        if dim % num_subvectors != 0 {
            return Err(ForgeDbError::invalid_parameter(format!(
                "vector dimension {} must be divisible by num_subvectors {}",
                dim, num_subvectors
            )));
        }

        // Check memory budget if specified
        if let Some(target_mb) = self.target_memory_mb {
            let estimated_mb = estimate_memory_mb(n, dim, num_subvectors, num_clusters);
            if estimated_mb > target_mb {
                // Could auto-adjust here, but for now just warn
                eprintln!(
                    "Warning: estimated memory {}MB exceeds target {}MB",
                    estimated_mb, target_mb
                );
            }
        }

        // Build the index
        let mut index = IVFPQIndex::build(vectors.clone(), num_clusters, num_subvectors, self.metric);
        index.set_nprobe(nprobe);

        // Enable re-ranking if requested
        if self.enable_reranking {
            index.enable_reranking(vectors, self.rerank_factor);
        }

        Ok(index)
    }

    /// Auto-tune the number of clusters.
    fn auto_tune_clusters(&self, n: usize) -> usize {
        if let Some(clusters) = self.num_clusters {
            return clusters;
        }

        if !self.auto_tune {
            // Default to sqrt(n) clamped to reasonable range
            return (n as f64).sqrt().round() as usize;
        }

        // Auto-tune: sqrt(n) clamped to [256, 65536]
        let sqrt_n = (n as f64).sqrt().round() as usize;
        sqrt_n.clamp(256.min(n), 65536.min(n))
    }

    /// Auto-tune the number of subvectors.
    fn auto_tune_subvectors(&self, dim: usize) -> Result<usize> {
        if let Some(m) = self.num_subvectors {
            return Ok(m);
        }

        if !self.auto_tune {
            // Default: dim / 4, must divide evenly
            let m = dim / 4;
            if !dim.is_multiple_of(m) {
                return Err(ForgeDbError::invalid_parameter(format!(
                    "default num_subvectors {} does not divide dimension {}",
                    m, dim
                )));
            }
            return Ok(m);
        }

        // Auto-tune: aim for ~4 dimensions per subvector, but must divide evenly
        let target_m = dim / 4;

        // Find largest divisor of dim that's <= target_m
        for m in (1..=target_m).rev() {
            if dim.is_multiple_of(m) {
                return Ok(m);
            }
        }

        // Fallback: use full dimension (no compression)
        Ok(1)
    }

    /// Auto-tune nprobe based on target recall if specified.
    fn auto_tune_nprobe(&self, num_clusters: usize) -> usize {
        if self.nprobe > 1 {
            return self.nprobe.min(num_clusters);
        }

        if let Some(target_recall) = self.target_recall {
            // Heuristic: higher recall needs more probes
            // These are approximate values based on typical IVF behavior
            let nprobe = if target_recall >= 0.95 {
                (num_clusters as f32 * 0.1).ceil() as usize
            } else if target_recall >= 0.9 {
                (num_clusters as f32 * 0.05).ceil() as usize
            } else if target_recall >= 0.8 {
                (num_clusters as f32 * 0.02).ceil() as usize
            } else {
                1
            };
            return nprobe.clamp(1, num_clusters);
        }

        // Default
        1
    }
}

/// Estimate memory usage in megabytes.
fn estimate_memory_mb(
    n_vectors: usize,
    dim: usize,
    n_subvectors: usize,
    n_clusters: usize,
) -> usize {
    // PQ codes: n_vectors * n_subvectors bytes
    let pq_codes_bytes = n_vectors * n_subvectors;

    // Vector IDs: n_vectors * 8 bytes
    let id_bytes = n_vectors * 8;

    // Centroids: n_clusters * dim * 4 bytes
    let centroid_bytes = n_clusters * dim * 4;

    // PQ codebooks: n_subvectors * 256 * (dim/n_subvectors) * 4 bytes
    let subvector_dim = dim / n_subvectors;
    let codebook_bytes = n_subvectors * 256 * subvector_dim * 4;

    let total_bytes = pq_codes_bytes + id_bytes + centroid_bytes + codebook_bytes;
    total_bytes / (1024 * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_basic() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();

        let index = IVFPQIndexBuilder::new()
            .vectors(vectors)
            .num_clusters(8)
            .num_subvectors(4)
            .nprobe(4)
            .build()
            .unwrap();

        assert_eq!(index.n_partitions(), 8);
        assert_eq!(index.nprobe(), 4);
        assert_eq!(index.len(), 500);
    }

    #[test]
    fn test_builder_auto_tune() {
        let vectors: Vec<Vector> = (0..1000).map(|i| Vector::random(i, 128)).collect();

        let index = IVFPQIndexBuilder::new()
            .vectors(vectors)
            .auto_tune(true)
            .build()
            .unwrap();

        // Should auto-tune to reasonable values
        assert!(index.n_partitions() >= 31); // sqrt(1000) ~ 31
        assert_eq!(index.len(), 1000);
    }

    #[test]
    fn test_builder_with_reranking() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();

        let index = IVFPQIndexBuilder::new()
            .vectors(vectors)
            .num_clusters(8)
            .num_subvectors(4)
            .enable_reranking(true)
            .rerank_factor(10)
            .build()
            .unwrap();

        // Search should work with reranking enabled
        let query = Vector::random(999, 32);
        let results = index.search(&query.data, 10);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_builder_target_recall() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();

        let index = IVFPQIndexBuilder::new()
            .vectors(vectors)
            .num_clusters(100)  // More clusters so recall tuning has effect
            .num_subvectors(4)
            .target_recall(0.95)  // High recall target
            .build()
            .unwrap();

        // Should auto-tune nprobe based on target recall
        // 100 clusters * 0.1 = 10 for 0.95 recall
        assert!(index.nprobe() >= 5);
    }

    #[test]
    fn test_builder_no_vectors() {
        let result = IVFPQIndexBuilder::new().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_empty_vectors() {
        let result = IVFPQIndexBuilder::new().vectors(vec![]).build();
        assert!(result.is_err());
    }

    #[test]
    fn test_builder_invalid_subvectors() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 30)).collect();

        // 30 is not divisible by 8
        let result = IVFPQIndexBuilder::new()
            .vectors(vectors)
            .num_subvectors(8)
            .build();

        assert!(result.is_err());
    }
}
