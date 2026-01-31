//! Metrics and statistics for index monitoring.
//!
//! Provides detailed statistics about index state and search operations
//! for performance monitoring, capacity planning, and debugging.
//!
//! # Index Statistics
//!
//! Use [`IndexStatistics`] to understand the current state of your index:
//!
//! ```ignore
//! let index = IVFPQIndex::build(vectors, 32, 8, DistanceMetric::Euclidean);
//! let stats = index.statistics();
//!
//! println!("{}", stats.summary());
//! // Output:
//! // IndexStatistics:
//! //   Vectors: 1000 (0 tombstones, 0.0% fragmented)
//! //   Partitions: 32 (min=25, max=38, mean=31.3, std=3.2)
//! //   Memory: 0.02 MB (compression ratio: 64.0x)
//! ```
//!
//! # Search Statistics
//!
//! Use [`SearchStatistics`] to profile query performance:
//!
//! ```ignore
//! let (results, stats) = index.search_with_stats(&query, 10);
//! println!("Query took {:.3}ms, scanned {} vectors", stats.query_time_ms(), stats.vectors_scanned);
//! ```
//!
//! # Resource Limits
//!
//! Use [`ResourceLimits`] to enforce capacity constraints:
//!
//! ```ignore
//! let limits = ResourceLimits::none()
//!     .with_max_memory_bytes(100 * 1024 * 1024)  // 100 MB
//!     .with_max_vectors(1_000_000)
//!     .with_query_timeout_ms(50);
//!
//! index.check_resource_limits(&limits)?;  // Returns error if exceeded
//! ```
//!
//! # Health Checks
//!
//! Use [`HealthStatus`] to monitor index health:
//!
//! ```ignore
//! match index.health_check() {
//!     HealthStatus::Healthy => println!("Index is healthy"),
//!     HealthStatus::Warning(issues) => println!("Warnings: {:?}", issues),
//!     HealthStatus::Unhealthy(issues) => println!("Critical: {:?}", issues),
//! }
//! ```

use std::time::Duration;

/// Resource limits for index operations.
///
/// Used to prevent unbounded resource usage during index operations.
#[derive(Clone, Debug, Default)]
pub struct ResourceLimits {
    /// Maximum memory usage in bytes (for checking, not enforcement).
    pub max_memory_bytes: Option<usize>,
    /// Query timeout in milliseconds.
    pub query_timeout_ms: Option<u64>,
    /// Maximum number of vectors allowed in the index.
    pub max_vectors: Option<usize>,
}

impl ResourceLimits {
    /// Create new resource limits with no restrictions.
    pub fn none() -> Self {
        Self::default()
    }

    /// Set the query timeout.
    pub fn with_query_timeout_ms(mut self, timeout_ms: u64) -> Self {
        self.query_timeout_ms = Some(timeout_ms);
        self
    }

    /// Set the maximum memory limit.
    pub fn with_max_memory_bytes(mut self, max_bytes: usize) -> Self {
        self.max_memory_bytes = Some(max_bytes);
        self
    }

    /// Set the maximum vector limit.
    pub fn with_max_vectors(mut self, max_vectors: usize) -> Self {
        self.max_vectors = Some(max_vectors);
        self
    }

    /// Get the query timeout as a Duration.
    pub fn query_timeout(&self) -> Option<Duration> {
        self.query_timeout_ms.map(Duration::from_millis)
    }
}

/// Health status of an index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HealthStatus {
    /// Index is healthy and operating normally.
    Healthy,
    /// Index has warnings but is still functional.
    Warning(Vec<String>),
    /// Index has critical issues that may affect functionality.
    Unhealthy(Vec<String>),
}

impl HealthStatus {
    /// Check if the status is healthy.
    pub fn is_healthy(&self) -> bool {
        matches!(self, HealthStatus::Healthy)
    }

    /// Check if the status has warnings (but is still functional).
    pub fn is_warning(&self) -> bool {
        matches!(self, HealthStatus::Warning(_))
    }

    /// Check if the status is unhealthy.
    pub fn is_unhealthy(&self) -> bool {
        matches!(self, HealthStatus::Unhealthy(_))
    }

    /// Get the issues if any.
    pub fn issues(&self) -> Option<&[String]> {
        match self {
            HealthStatus::Healthy => None,
            HealthStatus::Warning(issues) | HealthStatus::Unhealthy(issues) => Some(issues),
        }
    }
}

/// Statistics about the current state of an IVF-PQ index.
#[derive(Clone, Debug, Default)]
pub struct IndexStatistics {
    /// Total number of indexed vectors (excluding tombstones).
    pub num_vectors: usize,
    /// Total number of IVF partitions.
    pub num_partitions: usize,
    /// Vector dimension.
    pub dimension: usize,
    /// Estimated memory usage in bytes.
    pub memory_bytes: usize,
    /// Compression ratio (original size / compressed size).
    pub compression_ratio: f32,
    /// Number of PQ subvectors.
    pub num_subvectors: usize,
    /// Minimum partition size (vectors per partition).
    pub partition_size_min: usize,
    /// Maximum partition size (vectors per partition).
    pub partition_size_max: usize,
    /// Mean partition size.
    pub partition_size_mean: f32,
    /// Standard deviation of partition sizes.
    pub partition_size_std: f32,
    /// Number of tombstoned (deleted) vectors.
    pub num_tombstones: usize,
    /// Fragmentation ratio (tombstones / (vectors + tombstones)).
    pub fragmentation_ratio: f32,
    /// Whether reranking is enabled.
    pub reranking_enabled: bool,
    /// Current nprobe setting.
    pub nprobe: usize,
    /// Rerank factor (if reranking enabled).
    pub rerank_factor: usize,
}

impl IndexStatistics {
    /// Create a human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "IndexStatistics:\n  \
             Vectors: {} ({} tombstones, {:.1}% fragmented)\n  \
             Partitions: {} (min={}, max={}, mean={:.1}, std={:.1})\n  \
             Dimension: {}, Subvectors: {}\n  \
             Memory: {:.2} MB (compression ratio: {:.1}x)\n  \
             Reranking: {} (factor: {})\n  \
             nprobe: {}",
            self.num_vectors,
            self.num_tombstones,
            self.fragmentation_ratio * 100.0,
            self.num_partitions,
            self.partition_size_min,
            self.partition_size_max,
            self.partition_size_mean,
            self.partition_size_std,
            self.dimension,
            self.num_subvectors,
            self.memory_bytes as f64 / (1024.0 * 1024.0),
            self.compression_ratio,
            if self.reranking_enabled { "yes" } else { "no" },
            self.rerank_factor,
            self.nprobe
        )
    }
}

/// Statistics about a single search operation.
#[derive(Clone, Debug, Default)]
pub struct SearchStatistics {
    /// Total query execution time.
    pub query_time: Duration,
    /// Number of partitions probed.
    pub partitions_probed: usize,
    /// Number of vectors scanned (after filter pushdown if applicable).
    pub vectors_scanned: usize,
    /// Number of PQ distance computations performed.
    pub pq_distances_computed: usize,
    /// Whether reranking was performed.
    pub reranking_performed: bool,
    /// Number of candidates reranked (if reranking performed).
    pub candidates_reranked: usize,
    /// Number of vectors filtered out by metadata filter.
    pub vectors_filtered: usize,
}

impl SearchStatistics {
    /// Get query time in milliseconds.
    pub fn query_time_ms(&self) -> f32 {
        self.query_time.as_secs_f32() * 1000.0
    }

    /// Create a human-readable summary.
    pub fn summary(&self) -> String {
        format!(
            "SearchStatistics:\n  \
             Time: {:.3}ms\n  \
             Partitions probed: {}\n  \
             Vectors scanned: {}\n  \
             PQ distances computed: {}\n  \
             Reranking: {} (candidates: {})\n  \
             Vectors filtered: {}",
            self.query_time_ms(),
            self.partitions_probed,
            self.vectors_scanned,
            self.pq_distances_computed,
            if self.reranking_performed { "yes" } else { "no" },
            self.candidates_reranked,
            self.vectors_filtered
        )
    }
}

/// Builder for collecting search statistics during a search operation.
#[derive(Default)]
pub struct SearchStatsBuilder {
    stats: SearchStatistics,
}

impl SearchStatsBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the number of partitions probed.
    pub fn partitions_probed(&mut self, count: usize) {
        self.stats.partitions_probed = count;
    }

    /// Add to the count of vectors scanned.
    pub fn add_vectors_scanned(&mut self, count: usize) {
        self.stats.vectors_scanned += count;
    }

    /// Add to the count of PQ distances computed.
    pub fn add_pq_distances(&mut self, count: usize) {
        self.stats.pq_distances_computed += count;
    }

    /// Record that reranking was performed.
    pub fn reranking_performed(&mut self, candidates: usize) {
        self.stats.reranking_performed = true;
        self.stats.candidates_reranked = candidates;
    }

    /// Add to the count of filtered vectors.
    pub fn add_filtered(&mut self, count: usize) {
        self.stats.vectors_filtered += count;
    }

    /// Set the query execution time.
    pub fn set_query_time(&mut self, duration: Duration) {
        self.stats.query_time = duration;
    }

    /// Build the final statistics.
    pub fn build(self) -> SearchStatistics {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_statistics_summary() {
        let stats = IndexStatistics {
            num_vectors: 1000,
            num_partitions: 10,
            dimension: 128,
            memory_bytes: 1024 * 1024,
            compression_ratio: 32.0,
            num_subvectors: 8,
            partition_size_min: 80,
            partition_size_max: 120,
            partition_size_mean: 100.0,
            partition_size_std: 10.0,
            num_tombstones: 50,
            fragmentation_ratio: 0.05,
            reranking_enabled: true,
            nprobe: 4,
            rerank_factor: 10,
        };

        let summary = stats.summary();
        assert!(summary.contains("1000"));
        assert!(summary.contains("50 tombstones"));
        assert!(summary.contains("128"));
    }

    #[test]
    fn test_search_statistics_summary() {
        let stats = SearchStatistics {
            query_time: Duration::from_micros(1500),
            partitions_probed: 4,
            vectors_scanned: 1000,
            pq_distances_computed: 800,
            reranking_performed: true,
            candidates_reranked: 100,
            vectors_filtered: 200,
        };

        let summary = stats.summary();
        assert!(summary.contains("1.5"));
        assert!(summary.contains("4"));
        assert!(summary.contains("1000"));
    }

    #[test]
    fn test_search_stats_builder() {
        let mut builder = SearchStatsBuilder::new();
        builder.partitions_probed(4);
        builder.add_vectors_scanned(500);
        builder.add_vectors_scanned(300);
        builder.add_pq_distances(700);
        builder.reranking_performed(100);
        builder.set_query_time(Duration::from_millis(5));

        let stats = builder.build();
        assert_eq!(stats.partitions_probed, 4);
        assert_eq!(stats.vectors_scanned, 800);
        assert_eq!(stats.pq_distances_computed, 700);
        assert!(stats.reranking_performed);
        assert_eq!(stats.candidates_reranked, 100);
        assert_eq!(stats.query_time, Duration::from_millis(5));
    }
}
