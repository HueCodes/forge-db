//! Named constants for configuration values.
//!
//! This module centralizes magic numbers and default values used throughout
//! the codebase, making them easier to find, document, and tune.

/// Constants for Product Quantization (PQ).
pub mod pq {
    /// Number of centroids for 8-bit PQ codebooks.
    /// Each subvector is quantized to one of 256 centroids.
    pub const CENTROIDS_8BIT: usize = 256;

    /// Number of centroids for 4-bit PQ codebooks.
    /// Each subvector is quantized to one of 16 centroids.
    pub const CENTROIDS_4BIT: usize = 16;

    /// Default number of k-means iterations for PQ codebook training.
    /// 25 iterations typically provides good convergence for codebooks.
    pub const KMEANS_ITERATIONS: usize = 25;
}

/// Constants for k-means clustering.
pub mod kmeans {
    /// Threshold for switching from k-means++ to random initialization.
    /// k-means++ has O(k²n) complexity, which becomes expensive for large k.
    pub const KMEANSPP_THRESHOLD: usize = 64;

    /// Convergence threshold for early stopping.
    /// Training stops when centroid movement falls below this value.
    pub const CONVERGENCE_THRESHOLD: f32 = 0.001;

    /// Default number of iterations for IVF centroid training.
    pub const DEFAULT_MAX_ITERATIONS: usize = 100;
}

/// Constants for HNSW index.
pub mod hnsw {
    /// Number of cached entry points for faster search convergence.
    pub const ENTRY_CACHE_SIZE: usize = 8;

    /// Default M parameter (max connections per layer).
    pub const DEFAULT_M: usize = 16;

    /// Default ef_construction (beam width during build).
    pub const DEFAULT_EF_CONSTRUCTION: usize = 200;
}

/// Constants for cache optimization.
pub mod cache {
    /// CPU cache line size in bytes (typical for modern x86_64/ARM).
    pub const CACHE_LINE_SIZE: usize = 64;

    /// Number of iterations to prefetch ahead.
    /// Tuned for typical memory latency hiding.
    pub const PREFETCH_DISTANCE: usize = 3;

    /// Chunk size for parallel brute force search.
    /// Sized to fit multiple vectors in L2 cache.
    pub const BRUTE_FORCE_CHUNK_SIZE: usize = 1000;
}

/// Constants for IVF-PQ index.
pub mod ivf_pq {
    /// Threshold below which to use scalar distance computation.
    /// SIMD overhead is not worth it for very small partition sizes.
    pub const SCALAR_THRESHOLD: usize = 8;

    /// Threshold for parallel partition processing.
    /// Small partitions are processed sequentially to avoid overhead.
    pub const PARALLEL_THRESHOLD: usize = 100;

    /// Batch size for asymmetric distance computation.
    /// Sized for efficient L1 cache utilization.
    pub const BATCH_SIZE: usize = 8;

    /// Threshold for SIMD distance computation.
    /// Below this, scalar is faster due to reduced setup overhead.
    pub const SIMD_THRESHOLD: usize = 16;
}

/// Constants for resource limits.
pub mod limits {
    /// Default maximum vectors per index.
    pub const DEFAULT_MAX_VECTORS: usize = 100_000_000;

    /// Default memory limit in bytes (8 GB).
    pub const DEFAULT_MEMORY_LIMIT: usize = 8 * 1024 * 1024 * 1024;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_constants() {
        assert_eq!(pq::CENTROIDS_8BIT, 256);
        assert_eq!(pq::CENTROIDS_4BIT, 16);
    }

    #[test]
    fn test_cache_alignment() {
        assert!(cache::CACHE_LINE_SIZE.is_power_of_two());
    }
}
