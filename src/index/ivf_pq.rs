//! IVF-PQ Index: Inverted File with Product Quantization.
//!
//! Combines IVF (coarse quantization for partitioning) with PQ (fine
//! quantization for compression). This achieves both fast search through
//! reduced search space and low memory through vector compression.
//!
//! # Search Process
//!
//! 1. Find nprobe nearest IVF centroids to query
//! 2. Build PQ lookup table for the query
//! 3. Scan vectors in selected partitions using asymmetric PQ distance
//! 4. Return top-k results
//!
//! # Example
//!
//! ```
//! use forge_db::{Vector, IVFPQIndex, DistanceMetric};
//!
//! // Create random vectors (in practice, load your embeddings)
//! let vectors: Vec<Vector> = (0..1000)
//!     .map(|i| Vector::random(i, 128))
//!     .collect();
//!
//! // Build the index with 32 partitions and 8 subvectors
//! let mut index = IVFPQIndex::build(
//!     vectors,
//!     32,  // n_clusters (IVF partitions)
//!     8,   // n_subvectors (PQ compression)
//!     DistanceMetric::Euclidean
//! );
//!
//! // Configure search: probe 4 partitions for better recall
//! index.set_nprobe(4);
//!
//! // Search for 10 nearest neighbors
//! let query = Vector::random(9999, 128);
//! let results = index.search(&query.data, 10);
//!
//! // Results are (id, distance) pairs, sorted by distance
//! for (id, distance) in results {
//!     println!("Vector {} at distance {}", id, distance);
//! }
//! ```
//!
//! # Builder Pattern
//!
//! For auto-tuned parameters, use the builder:
//!
//! ```
//! use forge_db::{Vector, IVFPQIndexBuilder, DistanceMetric};
//!
//! let vectors: Vec<Vector> = (0..1000)
//!     .map(|i| Vector::random(i, 128))
//!     .collect();
//!
//! let index = IVFPQIndexBuilder::new()
//!     .vectors(vectors)
//!     .target_recall(0.9)
//!     .build()
//!     .expect("Failed to build index");
//! ```
//!
//! # Performance Tips
//!
//! - Use higher nprobe for better recall (at cost of latency)
//! - Enable reranking for highest recall when memory permits
//! - Use batch_search() for multiple queries to amortize overhead
//! - Consider HNSW if memory is not constrained and you need lowest latency

use crate::distance::{euclidean_distance_squared, DistanceMetric};
use crate::error::Result;
use crate::kmeans::KMeans;
use crate::persistence::{verify_header, write_with_header, IndexType, Persistable};
use crate::pq::ProductQuantizer;
use crate::vector::Vector;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
use std::sync::RwLock;

// =============================================================================
// Software Prefetching for x86_64
// =============================================================================
// Prefetching hides memory latency by loading data into cache before it's needed.
// We prefetch 2-3 cache lines ahead to overlap memory access with computation.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

/// Prefetch data into L1 cache (temporal - expected to be reused).
/// Call 2-3 iterations ahead of where you're reading.
#[inline(always)]
fn prefetch_read<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
    }
    // No-op on non-x86_64 platforms
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}

/// Data stored in each IVF partition.
struct PartitionData {
    /// Vector IDs in this partition
    ids: Vec<u64>,
    /// PQ codes for each vector (M bytes per vector)
    codes: Vec<Vec<u8>>,
    /// Tombstones for deleted vectors (IDs that should be skipped during search)
    tombstones: std::collections::HashSet<u64>,
}

// =============================================================================
// Serialization Support
// =============================================================================

/// Serializable representation of a vector (for persistence).
#[derive(Serialize, Deserialize)]
struct SerializableVector {
    id: u64,
    data: Vec<f32>,
}

impl From<&Vector> for SerializableVector {
    fn from(v: &Vector) -> Self {
        Self {
            id: v.id,
            data: v.data.to_vec(),
        }
    }
}

impl From<SerializableVector> for Vector {
    fn from(v: SerializableVector) -> Self {
        Vector::new(v.id, v.data)
    }
}

/// Serializable representation of partition data.
#[derive(Serialize, Deserialize)]
struct SerializablePartition {
    ids: Vec<u64>,
    codes: Vec<Vec<u8>>,
}

impl From<&PartitionData> for SerializablePartition {
    fn from(p: &PartitionData) -> Self {
        Self {
            ids: p.ids.clone(),
            codes: p.codes.clone(),
        }
    }
}

impl From<SerializablePartition> for PartitionData {
    fn from(p: SerializablePartition) -> Self {
        Self {
            ids: p.ids,
            codes: p.codes,
            tombstones: std::collections::HashSet::new(),
        }
    }
}

/// Serializable representation of ProductQuantizer.
#[derive(Serialize, Deserialize)]
struct SerializablePQ {
    codebooks: Vec<Vec<SerializableVector>>,
    n_subvectors: usize,
    subvector_dim: usize,
    dim: usize,
}

impl From<&ProductQuantizer> for SerializablePQ {
    fn from(pq: &ProductQuantizer) -> Self {
        Self {
            codebooks: pq
                .codebooks
                .iter()
                .map(|cb| cb.iter().map(SerializableVector::from).collect())
                .collect(),
            n_subvectors: pq.n_subvectors,
            subvector_dim: pq.subvector_dim,
            dim: pq.dim,
        }
    }
}

impl From<SerializablePQ> for ProductQuantizer {
    fn from(pq: SerializablePQ) -> Self {
        Self {
            codebooks: pq
                .codebooks
                .into_iter()
                .map(|cb| cb.into_iter().map(Vector::from).collect())
                .collect(),
            n_subvectors: pq.n_subvectors,
            subvector_dim: pq.subvector_dim,
            dim: pq.dim,
        }
    }
}

/// Serializable representation of DistanceMetric.
#[derive(Serialize, Deserialize)]
enum SerializableMetric {
    Euclidean,
    EuclideanSquared,
    Cosine,
    DotProduct,
    Manhattan,
}

impl From<DistanceMetric> for SerializableMetric {
    fn from(m: DistanceMetric) -> Self {
        match m {
            DistanceMetric::Euclidean => Self::Euclidean,
            DistanceMetric::EuclideanSquared => Self::EuclideanSquared,
            DistanceMetric::Cosine => Self::Cosine,
            DistanceMetric::DotProduct => Self::DotProduct,
            DistanceMetric::Manhattan => Self::Manhattan,
        }
    }
}

impl From<SerializableMetric> for DistanceMetric {
    fn from(m: SerializableMetric) -> Self {
        match m {
            SerializableMetric::Euclidean => Self::Euclidean,
            SerializableMetric::EuclideanSquared => Self::EuclideanSquared,
            SerializableMetric::Cosine => Self::Cosine,
            SerializableMetric::DotProduct => Self::DotProduct,
            SerializableMetric::Manhattan => Self::Manhattan,
        }
    }
}

/// Serializable representation of IVFPQIndex.
#[derive(Serialize, Deserialize)]
struct SerializableIVFPQ {
    centroids: Vec<SerializableVector>,
    pq: SerializablePQ,
    partitions: Vec<SerializablePartition>,
    nprobe: usize,
    metric: SerializableMetric,
    original_vectors: Option<Vec<SerializableVector>>,
    rerank_factor: usize,
}

/// IVF-PQ Index for scalable approximate nearest neighbor search.
///
/// Uses inverted file indexing for coarse partitioning and product
/// quantization for memory-efficient storage within partitions.
///
/// # Thread Safety
///
/// This index is designed for concurrent read access:
/// - `search()`, `batch_search()`, and `search_filtered()` can be called concurrently
/// - `nprobe` can be modified during concurrent searches via atomic operations
/// - Incremental updates (`insert_batch`, `delete`, `compact`) require exclusive access
///
/// For concurrent read/write access, wrap the index in a `RwLock<IVFPQIndex>`.
pub struct IVFPQIndex {
    /// IVF cluster centroids
    centroids: Vec<Vector>,
    /// Product quantizer for compression
    pq: ProductQuantizer,
    /// Vectors organized by partition, wrapped in RwLock for thread-safe updates
    partitions: Vec<RwLock<PartitionData>>,
    /// Number of partitions to probe during search (atomic for thread-safe modification)
    nprobe: AtomicUsize,
    /// Distance metric (used for IVF assignment)
    _metric: DistanceMetric,
    /// Original vectors for re-ranking (optional, improves recall)
    /// Stored as HashMap for O(1) lookup, wrapped in RwLock for thread-safe updates
    original_vectors: Option<RwLock<HashMap<u64, Vector>>>,
    /// Re-ranking factor: fetch this many candidates, then re-rank
    rerank_factor: usize,
}

// Thread-safety assertions: IVFPQIndex is Send + Sync
// This enables concurrent searches from multiple threads
static_assertions::assert_impl_all!(IVFPQIndex: Send, Sync);

impl IVFPQIndex {
    /// Build an IVF-PQ index from vectors.
    ///
    /// # Arguments
    /// * `vectors` - Vectors to index
    /// * `n_clusters` - Number of IVF partitions (typically sqrt(n) to 4*sqrt(n))
    /// * `n_subvectors` - Number of PQ subvectors (typically 8 for 128-dim)
    /// * `metric` - Distance metric for search
    pub fn build(
        vectors: Vec<Vector>,
        n_clusters: usize,
        n_subvectors: usize,
        metric: DistanceMetric,
    ) -> Self {
        use rand::seq::SliceRandom;

        // Step 1: Train IVF centroids on a sample (for speed)
        let train_sample_size = vectors.len().min(30_000);

        let training_sample: Vec<Vector> = if vectors.len() <= train_sample_size {
            vectors.clone()
        } else {
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..vectors.len()).collect();
            indices.shuffle(&mut rng);
            indices
                .into_iter()
                .take(train_sample_size)
                .map(|i| vectors[i].clone())
                .collect()
        };

        let mut kmeans = KMeans::new(n_clusters, 25); // Fewer iterations, converges fast
        kmeans.fit(&training_sample);
        let centroids = kmeans.centroids;

        // Step 2: Assign vectors to partitions (parallel)
        let assignments: Vec<usize> = vectors
            .par_iter()
            .map(|vector| {
                centroids
                    .iter()
                    .enumerate()
                    .map(|(idx, c)| (idx, euclidean_distance_squared(&vector.data, &c.data)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap()
                    .0
            })
            .collect();

        let mut partition_vectors: Vec<Vec<Vector>> = vec![Vec::new(); n_clusters];
        for (vector, &partition_id) in vectors.iter().zip(assignments.iter()) {
            partition_vectors[partition_id].push(vector.clone());
        }

        // Step 3: Train PQ on residuals (vector - centroid) for better recall
        let pq_sample_size = vectors.len().min(100_000);

        // Compute residuals for PQ training
        let residual_sample: Vec<Vector> = {
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..vectors.len()).collect();
            indices.shuffle(&mut rng);
            indices
                .into_iter()
                .take(pq_sample_size)
                .map(|i| {
                    let v = &vectors[i];
                    let centroid_idx = assignments[i];
                    let centroid = &centroids[centroid_idx];
                    // Compute residual: vector - centroid
                    let residual: Vec<f32> = v
                        .data
                        .iter()
                        .zip(centroid.data.iter())
                        .map(|(a, b)| a - b)
                        .collect();
                    Vector::new(v.id, residual)
                })
                .collect()
        };
        let pq = ProductQuantizer::train(&residual_sample, n_subvectors);

        // Step 4: Encode residuals in each partition
        let partitions: Vec<PartitionData> = partition_vectors
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(|(centroid_idx, vectors)| {
                let centroid = &centroids[centroid_idx];
                let ids: Vec<u64> = vectors.iter().map(|v| v.id).collect();
                let codes: Vec<Vec<u8>> = vectors
                    .iter()
                    .map(|v| {
                        // Encode residual instead of original vector
                        let residual: Vec<f32> = v
                            .data
                            .iter()
                            .zip(centroid.data.iter())
                            .map(|(a, b)| a - b)
                            .collect();
                        let residual_vec = Vector::new(v.id, residual);
                        pq.encode(&residual_vec)
                    })
                    .collect();
                PartitionData {
                    ids,
                    codes,
                    tombstones: std::collections::HashSet::new(),
                }
            })
            .collect();

        // Print statistics
        let total_vectors: usize = partitions.iter().map(|p| p.ids.len()).sum();
        let total_bytes: usize = partitions
            .iter()
            .map(|p| p.codes.len() * n_subvectors)
            .sum();

        println!(
            "Index built: {} vectors, {} MB compressed",
            total_vectors,
            total_bytes / (1024 * 1024)
        );

        // Wrap partitions in RwLock for thread-safe access
        let partitions: Vec<RwLock<PartitionData>> =
            partitions.into_iter().map(RwLock::new).collect();

        Self {
            centroids,
            pq,
            partitions,
            nprobe: AtomicUsize::new(1),
            metric,
            original_vectors: None,
            rerank_factor: 1,
        }
    }

    /// Enable re-ranking with original vectors for higher recall.
    ///
    /// Stores original vectors in a HashMap for O(1) lookup and re-ranks
    /// PQ candidates using exact distance. Increases memory but improves
    /// recall by 20-30%.
    pub fn enable_reranking(&mut self, vectors: Vec<Vector>, rerank_factor: usize) {
        let map: HashMap<u64, Vector> = vectors.into_iter().map(|v| (v.id, v)).collect();
        self.original_vectors = Some(RwLock::new(map));
        self.rerank_factor = rerank_factor.max(1);
    }

    /// Set the number of partitions to probe during search.
    ///
    /// This can be safely called while searches are in progress (thread-safe).
    /// Higher nprobe = better recall but slower search.
    /// Typically use 1-16 depending on desired recall/speed tradeoff.
    pub fn set_nprobe(&self, nprobe: usize) {
        self.nprobe
            .store(nprobe.min(self.centroids.len()), AtomicOrdering::Relaxed);
    }

    /// Get the current nprobe value atomically.
    #[inline(always)]
    fn get_nprobe(&self) -> usize {
        self.nprobe.load(AtomicOrdering::Relaxed)
    }

    /// Search for k nearest neighbors to the query.
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        // If re-ranking enabled, fetch more candidates
        let fetch_k = if self.original_vectors.is_some() {
            k * self.rerank_factor
        } else {
            k
        };

        // Step 1: Find nprobe nearest centroids using partial sort
        let nprobe = self.nprobe();
        let mut centroid_distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, euclidean_distance_squared(query, &c.data)))
            .collect();

        // Partial sort - only need top nprobe, not full sort
        let nth = nprobe.min(centroid_distances.len()).saturating_sub(1);
        centroid_distances.select_nth_unstable_by(nth, |a, b| a.1.partial_cmp(&b.1).unwrap());

        // Step 2-3: For each partition, compute residual and scan
        // Use a bounded heap to avoid storing all results
        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(PartialEq)]
        struct HeapItem(u64, f32);

        impl Eq for HeapItem {}
        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                // Max heap - we want largest distances at top for removal
                self.1.partial_cmp(&other.1)
            }
        }
        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(fetch_k + 1);

        for i in 0..nprobe.min(centroid_distances.len()) {
            let partition_id = centroid_distances[i].0;
            let partition = self.partitions[partition_id].read().unwrap();
            let centroid = &self.centroids[partition_id];

            // Compute query residual for this partition (reuse allocation)
            let query_residual: Vec<f32> = query
                .iter()
                .zip(centroid.data.iter())
                .map(|(q, c)| q - c)
                .collect();

            // Build lookup table for the residual
            let lookup_table = self.pq.build_lookup_table(&query_residual);

            // Get current threshold for early termination
            let threshold = if heap.len() >= fetch_k {
                heap.peek().map(|h| h.1).unwrap_or(f32::MAX)
            } else {
                f32::MAX
            };

            // Use indexed loop for prefetching
            let codes_slice = &partition.codes;
            let ids_slice = &partition.ids;
            let len = codes_slice.len();

            for j in 0..len {
                // Prefetch 3 vectors ahead to hide memory latency
                if j + 3 < len {
                    prefetch_read(codes_slice[j + 3].as_ptr());
                }

                let codes = &codes_slice[j];
                let id = ids_slice[j];

                // Skip tombstoned vectors
                if partition.tombstones.contains(&id) {
                    continue;
                }

                let dist = self.pq.asymmetric_distance_fast(&lookup_table, codes);

                // Only insert if better than current k-th best
                if dist < threshold || heap.len() < fetch_k {
                    heap.push(HeapItem(id, dist));
                    if heap.len() > fetch_k {
                        heap.pop();
                    }
                }
            }
        }

        // Convert heap to sorted vector
        let mut results: Vec<(u64, f32)> = heap.into_iter().map(|h| (h.0, h.1)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Re-rank with original vectors if enabled
        if let Some(ref id_to_vec_lock) = self.original_vectors {
            let id_to_vec = id_to_vec_lock.read().unwrap();
            // Re-compute exact distances for candidates (O(1) lookup per candidate)
            let mut reranked: Vec<(u64, f32)> = results
                .iter()
                .filter_map(|(id, _)| {
                    id_to_vec.get(id).map(|v| {
                        let dist = euclidean_distance_squared(query, &v.data).sqrt();
                        (*id, dist)
                    })
                })
                .collect();

            reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            reranked.truncate(k);
            return reranked;
        }

        results.truncate(k);
        results
    }

    /// Search for k nearest neighbors with metadata filtering.
    ///
    /// This performs filter-pushdown: candidates are filtered during the search
    /// process rather than after, which is more efficient when filters are selective.
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    /// * `metadata_store` - Store containing vector metadata
    /// * `filter` - Filter condition to apply
    ///
    /// # Returns
    /// Vector of (id, distance) pairs for matching vectors, sorted by distance.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        metadata_store: &crate::metadata::MetadataStore,
        filter: &crate::metadata::FilterCondition,
    ) -> Vec<(u64, f32)> {
        // Get bitmap of matching IDs for efficient checking
        let matching_bitmap = metadata_store.filter_bitmap(filter);

        // If no matches, return empty
        if matching_bitmap.is_empty() {
            return Vec::new();
        }

        // If re-ranking enabled, fetch more candidates
        let fetch_k = if self.original_vectors.is_some() {
            k * self.rerank_factor
        } else {
            k
        };

        // Use a larger fetch factor for filtered search since many candidates may be filtered
        let filter_fetch_k = fetch_k * 4;

        // Step 1: Find nprobe nearest centroids
        let nprobe = self.nprobe();
        let mut centroid_distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, euclidean_distance_squared(query, &c.data)))
            .collect();

        let nth = nprobe.min(centroid_distances.len()).saturating_sub(1);
        centroid_distances.select_nth_unstable_by(nth, |a, b| a.1.partial_cmp(&b.1).unwrap());

        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(PartialEq)]
        struct HeapItem(u64, f32);

        impl Eq for HeapItem {}
        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.1.partial_cmp(&other.1)
            }
        }
        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(filter_fetch_k + 1);

        for i in 0..nprobe.min(centroid_distances.len()) {
            let partition_id = centroid_distances[i].0;
            let partition = self.partitions[partition_id].read().unwrap();
            let centroid = &self.centroids[partition_id];

            let query_residual: Vec<f32> = query
                .iter()
                .zip(centroid.data.iter())
                .map(|(q, c)| q - c)
                .collect();

            let lookup_table = self.pq.build_lookup_table(&query_residual);

            let threshold = if heap.len() >= filter_fetch_k {
                heap.peek().map(|h| h.1).unwrap_or(f32::MAX)
            } else {
                f32::MAX
            };

            let codes_slice = &partition.codes;
            let ids_slice = &partition.ids;
            let len = codes_slice.len();

            for j in 0..len {
                let id = ids_slice[j];

                // Skip tombstoned vectors
                if partition.tombstones.contains(&id) {
                    continue;
                }

                // Filter pushdown: skip vectors that don't match the filter
                // Use bitmap index if available, otherwise check metadata
                if let Some(idx) = metadata_store.get_index(id) {
                    if !matching_bitmap.contains(idx) {
                        continue;
                    }
                } else {
                    // ID not in metadata store - skip it
                    continue;
                }

                // Prefetch 3 vectors ahead
                if j + 3 < len {
                    prefetch_read(codes_slice[j + 3].as_ptr());
                }

                let codes = &codes_slice[j];
                let dist = self.pq.asymmetric_distance_fast(&lookup_table, codes);

                if dist < threshold || heap.len() < filter_fetch_k {
                    heap.push(HeapItem(id, dist));
                    if heap.len() > filter_fetch_k {
                        heap.pop();
                    }
                }
            }
        }

        // Convert heap to sorted vector
        let mut results: Vec<(u64, f32)> = heap.into_iter().map(|h| (h.0, h.1)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Re-rank with original vectors if enabled
        if let Some(ref id_to_vec_lock) = self.original_vectors {
            let id_to_vec = id_to_vec_lock.read().unwrap();
            let mut reranked: Vec<(u64, f32)> = results
                .iter()
                .filter_map(|(id, _)| {
                    id_to_vec.get(id).map(|v| {
                        let dist = euclidean_distance_squared(query, &v.data).sqrt();
                        (*id, dist)
                    })
                })
                .collect();

            reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            reranked.truncate(k);
            return reranked;
        }

        results.truncate(k);
        results
    }

    /// Batch search optimized for cache efficiency and SIMD throughput.
    ///
    /// Instead of processing queries one-by-one, this groups queries by their
    /// nearest partitions and processes partition-by-partition. This keeps
    /// partition data hot in cache while scoring multiple queries.
    ///
    /// Optimizations applied:
    /// 1. Partition-centric processing: keeps partition data hot in cache
    /// 2. Batch SIMD: processes multiple queries against same vector using AVX2
    /// 3. Prefetching: hides memory latency by loading data ahead of computation
    ///
    /// # Arguments
    /// * `queries` - Slice of query vectors (anything that can be dereferenced to &[f32])
    /// * `k` - Number of neighbors to return per query
    ///
    /// # Returns
    /// Vector of results for each query, in the same order as input.
    pub fn batch_search<Q: AsRef<[f32]>>(&self, queries: &[Q], k: usize) -> Vec<Vec<(u64, f32)>> {
        use crate::pq::batch_asymmetric_distance_dispatch;

        if queries.is_empty() {
            return Vec::new();
        }

        let fetch_k = if self.original_vectors.is_some() {
            k * self.rerank_factor
        } else {
            k
        };

        // Step 1: Find nearest partitions for all queries
        let nprobe = self.nprobe();
        let query_partitions: Vec<(usize, Vec<usize>)> = queries
            .iter()
            .enumerate()
            .map(|(qi, q)| {
                let q = q.as_ref();
                let mut centroid_distances: Vec<(usize, f32)> = self
                    .centroids
                    .iter()
                    .enumerate()
                    .map(|(idx, c)| (idx, euclidean_distance_squared(q, &c.data)))
                    .collect();

                let nth = nprobe.min(centroid_distances.len()).saturating_sub(1);
                centroid_distances
                    .select_nth_unstable_by(nth, |a, b| a.1.partial_cmp(&b.1).unwrap());

                let partitions: Vec<usize> = centroid_distances[..=nth]
                    .iter()
                    .map(|(idx, _)| *idx)
                    .collect();

                (qi, partitions)
            })
            .collect();

        // Step 2: Build partition -> queries mapping
        let mut partition_queries: HashMap<usize, Vec<usize>> = HashMap::new();
        for (qi, partitions) in &query_partitions {
            for &p in partitions {
                partition_queries.entry(p).or_default().push(*qi);
            }
        }

        // Step 3: Initialize result heaps for each query
        use std::cmp::Ordering;

        #[derive(PartialEq)]
        struct HeapItem(u64, f32);

        impl Eq for HeapItem {}
        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.1.partial_cmp(&other.1)
            }
        }
        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        let mut heaps: Vec<std::collections::BinaryHeap<HeapItem>> = (0..queries.len())
            .map(|_| std::collections::BinaryHeap::with_capacity(fetch_k + 1))
            .collect();

        // Step 4: Process partition by partition (cache-friendly)
        for (partition_id, query_indices) in partition_queries {
            let partition = self.partitions[partition_id].read().unwrap();
            let centroid = &self.centroids[partition_id];

            // Build lookup tables for all queries hitting this partition
            let tables: Vec<Vec<f32>> = query_indices
                .iter()
                .map(|&qi| {
                    let q = queries[qi].as_ref();
                    // Compute query residual
                    let residual: Vec<f32> = q
                        .iter()
                        .zip(centroid.data.iter())
                        .map(|(qv, cv)| qv - cv)
                        .collect();
                    self.pq.build_lookup_table_flat(&residual)
                })
                .collect();

            // Convert tables to slices for batch distance function
            let table_refs: Vec<&[f32]> = tables.iter().map(|t| t.as_slice()).collect();

            // Score all vectors in partition against all relevant queries
            let codes_slice = &partition.codes;
            let ids_slice = &partition.ids;
            let len = codes_slice.len();

            // Use batch SIMD when we have multiple queries (>= 4 for best efficiency)
            let use_batch_simd = query_indices.len() >= 4;

            for i in 0..len {
                // Prefetch ahead
                if i + 3 < len {
                    prefetch_read(codes_slice[i + 3].as_ptr());
                }

                let codes = &codes_slice[i];
                let id = ids_slice[i];

                // Skip tombstoned vectors
                if partition.tombstones.contains(&id) {
                    continue;
                }

                if use_batch_simd {
                    // Compute distances to all queries at once using batch SIMD
                    let distances = batch_asymmetric_distance_dispatch(&table_refs, codes);

                    for (table_idx, dist) in distances.into_iter().enumerate() {
                        let qi = query_indices[table_idx];
                        let heap = &mut heaps[qi];

                        let threshold = if heap.len() >= fetch_k {
                            heap.peek().map(|h| h.1).unwrap_or(f32::MAX)
                        } else {
                            f32::MAX
                        };

                        if dist < threshold || heap.len() < fetch_k {
                            heap.push(HeapItem(id, dist));
                            if heap.len() > fetch_k {
                                heap.pop();
                            }
                        }
                    }
                } else {
                    // Fall back to per-query distance computation for small batches
                    for (table_idx, table) in tables.iter().enumerate() {
                        let qi = query_indices[table_idx];
                        let heap = &mut heaps[qi];

                        let dist = self.pq.asymmetric_distance_flat(table, codes);

                        let threshold = if heap.len() >= fetch_k {
                            heap.peek().map(|h| h.1).unwrap_or(f32::MAX)
                        } else {
                            f32::MAX
                        };

                        if dist < threshold || heap.len() < fetch_k {
                            heap.push(HeapItem(id, dist));
                            if heap.len() > fetch_k {
                                heap.pop();
                            }
                        }
                    }
                }
            }
        }

        // Step 5: Convert heaps to sorted results
        let mut results: Vec<Vec<(u64, f32)>> = heaps
            .into_iter()
            .map(|heap| {
                let mut r: Vec<(u64, f32)> = heap.into_iter().map(|h| (h.0, h.1)).collect();
                r.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                r
            })
            .collect();

        // Step 6: Re-rank with original vectors if enabled
        if let Some(ref id_to_vec_lock) = self.original_vectors {
            let id_to_vec = id_to_vec_lock.read().unwrap();
            for (qi, result) in results.iter_mut().enumerate() {
                let q = queries[qi].as_ref();
                let mut reranked: Vec<(u64, f32)> = result
                    .iter()
                    .filter_map(|(id, _)| {
                        id_to_vec.get(id).map(|v| {
                            let dist = euclidean_distance_squared(q, &v.data).sqrt();
                            (*id, dist)
                        })
                    })
                    .collect();

                reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                reranked.truncate(k);
                *result = reranked;
            }
        } else {
            for result in results.iter_mut() {
                result.truncate(k);
            }
        }

        results
    }

    /// Parallel batch search using Rayon.
    ///
    /// Divides queries into chunks and processes each chunk in parallel.
    /// Within each chunk, uses the cache-optimized partition-centric approach.
    ///
    /// # Arguments
    /// * `queries` - Slice of query vectors
    /// * `k` - Number of neighbors to return per query
    ///
    /// # Returns
    /// Vector of results for each query, in the same order as input.
    pub fn batch_search_parallel<Q: AsRef<[f32]> + Sync>(
        &self,
        queries: &[Q],
        k: usize,
    ) -> Vec<Vec<(u64, f32)>> {
        if queries.is_empty() {
            return Vec::new();
        }

        // Chunk size tuned for L2 cache (~256KB per core)
        // Each query needs: nprobe * lookup_table_size bytes
        // lookup_table = n_subvectors * 256 * 4 bytes
        let nprobe = self.nprobe();
        let table_bytes = self.pq.n_subvectors * 256 * 4;
        let chunk_size = (256 * 1024 / (nprobe.max(1) * table_bytes)).clamp(4, 64);

        // Process chunks in parallel
        let chunk_results: Vec<Vec<Vec<(u64, f32)>>> = queries
            .par_chunks(chunk_size)
            .map(|chunk| self.batch_search(chunk, k))
            .collect();

        // Flatten results
        chunk_results.into_iter().flatten().collect()
    }

    /// Return the total number of indexed vectors (excluding deleted).
    pub fn len(&self) -> usize {
        self.partitions
            .iter()
            .map(|p| {
                let partition = p.read().unwrap();
                partition.ids.len() - partition.tombstones.len()
            })
            .sum()
    }

    /// Return the total number of stored vectors (including deleted).
    pub fn len_with_tombstones(&self) -> usize {
        self.partitions
            .iter()
            .map(|p| p.read().unwrap().ids.len())
            .sum()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of partitions.
    pub fn n_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// Get the current nprobe setting (thread-safe).
    pub fn nprobe(&self) -> usize {
        self.get_nprobe()
    }

    /// Get the vector dimension.
    pub fn dim(&self) -> usize {
        self.pq.dim
    }

    // =========================================================================
    // Metrics and Statistics
    // =========================================================================

    /// Get comprehensive statistics about the index state.
    pub fn statistics(&self) -> crate::metrics::IndexStatistics {
        // Collect partition sizes
        let partition_sizes: Vec<usize> = self
            .partitions
            .iter()
            .map(|p| {
                let partition = p.read().unwrap();
                partition.ids.len() - partition.tombstones.len()
            })
            .collect();

        let num_vectors: usize = partition_sizes.iter().sum();
        let num_tombstones = self.num_tombstones();
        let total_with_tombstones = num_vectors + num_tombstones;

        // Calculate partition size statistics
        let (partition_size_min, partition_size_max, partition_size_mean, partition_size_std) =
            if partition_sizes.is_empty() {
                (0, 0, 0.0, 0.0)
            } else {
                let min = *partition_sizes.iter().min().unwrap();
                let max = *partition_sizes.iter().max().unwrap();
                let mean = num_vectors as f32 / partition_sizes.len() as f32;
                let variance: f32 = partition_sizes
                    .iter()
                    .map(|&s| {
                        let diff = s as f32 - mean;
                        diff * diff
                    })
                    .sum::<f32>()
                    / partition_sizes.len() as f32;
                let std = variance.sqrt();
                (min, max, mean, std)
            };

        // Calculate memory usage estimate
        // PQ codes: num_vectors * n_subvectors bytes
        // IDs: num_vectors * 8 bytes
        // Centroids: n_partitions * dim * 4 bytes
        // PQ codebooks: n_subvectors * 256 * subvector_dim * 4 bytes
        // Original vectors (if reranking): num_vectors * dim * 4 bytes
        let pq_code_bytes = total_with_tombstones * self.pq.n_subvectors;
        let id_bytes = total_with_tombstones * 8;
        let centroid_bytes = self.centroids.len() * self.pq.dim * 4;
        let codebook_bytes = self.pq.n_subvectors * 256 * self.pq.subvector_dim * 4;
        let original_vec_bytes = if self.original_vectors.is_some() {
            num_vectors * self.pq.dim * 4
        } else {
            0
        };
        let memory_bytes = pq_code_bytes + id_bytes + centroid_bytes + codebook_bytes + original_vec_bytes;

        // Compression ratio: original size / compressed size
        let original_size = num_vectors * self.pq.dim * 4;
        let compressed_size = pq_code_bytes;
        let compression_ratio = if compressed_size > 0 {
            original_size as f32 / compressed_size as f32
        } else {
            0.0
        };

        crate::metrics::IndexStatistics {
            num_vectors,
            num_partitions: self.partitions.len(),
            dimension: self.pq.dim,
            memory_bytes,
            compression_ratio,
            num_subvectors: self.pq.n_subvectors,
            partition_size_min,
            partition_size_max,
            partition_size_mean,
            partition_size_std,
            num_tombstones,
            fragmentation_ratio: if total_with_tombstones > 0 {
                num_tombstones as f32 / total_with_tombstones as f32
            } else {
                0.0
            },
            reranking_enabled: self.original_vectors.is_some(),
            nprobe: self.get_nprobe(),
            rerank_factor: self.rerank_factor,
        }
    }

    /// Search for k nearest neighbors and return statistics.
    ///
    /// This is the same as `search()` but also returns statistics about
    /// the search operation for monitoring and debugging.
    pub fn search_with_stats(
        &self,
        query: &[f32],
        k: usize,
    ) -> (Vec<(u64, f32)>, crate::metrics::SearchStatistics) {
        use std::time::Instant;

        let start = Instant::now();
        let mut stats_builder = crate::metrics::SearchStatsBuilder::new();

        let nprobe = self.get_nprobe();
        let fetch_k = if self.original_vectors.is_some() {
            k * self.rerank_factor
        } else {
            k
        };

        // Step 1: Find nprobe nearest centroids
        let mut centroid_distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, euclidean_distance_squared(query, &c.data)))
            .collect();

        let nth = nprobe.min(centroid_distances.len()).saturating_sub(1);
        centroid_distances.select_nth_unstable_by(nth, |a, b| a.1.partial_cmp(&b.1).unwrap());

        stats_builder.partitions_probed(nprobe.min(centroid_distances.len()));

        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(PartialEq)]
        struct HeapItem(u64, f32);

        impl Eq for HeapItem {}
        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.1.partial_cmp(&other.1)
            }
        }
        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(fetch_k + 1);

        for i in 0..nprobe.min(centroid_distances.len()) {
            let partition_id = centroid_distances[i].0;
            let partition = self.partitions[partition_id].read().unwrap();
            let centroid = &self.centroids[partition_id];

            let query_residual: Vec<f32> = query
                .iter()
                .zip(centroid.data.iter())
                .map(|(q, c)| q - c)
                .collect();

            let lookup_table = self.pq.build_lookup_table(&query_residual);

            let threshold = if heap.len() >= fetch_k {
                heap.peek().map(|h| h.1).unwrap_or(f32::MAX)
            } else {
                f32::MAX
            };

            let codes_slice = &partition.codes;
            let ids_slice = &partition.ids;
            let len = codes_slice.len();

            let mut vectors_scanned = 0;
            let mut pq_distances = 0;

            for j in 0..len {
                let id = ids_slice[j];

                if partition.tombstones.contains(&id) {
                    continue;
                }

                vectors_scanned += 1;
                pq_distances += 1;

                let codes = &codes_slice[j];
                let dist = self.pq.asymmetric_distance_fast(&lookup_table, codes);

                if dist < threshold || heap.len() < fetch_k {
                    heap.push(HeapItem(id, dist));
                    if heap.len() > fetch_k {
                        heap.pop();
                    }
                }
            }

            stats_builder.add_vectors_scanned(vectors_scanned);
            stats_builder.add_pq_distances(pq_distances);
        }

        let mut results: Vec<(u64, f32)> = heap.into_iter().map(|h| (h.0, h.1)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Re-rank if enabled
        if let Some(ref id_to_vec_lock) = self.original_vectors {
            let id_to_vec = id_to_vec_lock.read().unwrap();
            let candidates = results.len();

            let mut reranked: Vec<(u64, f32)> = results
                .iter()
                .filter_map(|(id, _)| {
                    id_to_vec.get(id).map(|v| {
                        let dist = euclidean_distance_squared(query, &v.data).sqrt();
                        (*id, dist)
                    })
                })
                .collect();

            reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            reranked.truncate(k);

            stats_builder.reranking_performed(candidates);
            stats_builder.set_query_time(start.elapsed());

            return (reranked, stats_builder.build());
        }

        results.truncate(k);
        stats_builder.set_query_time(start.elapsed());

        (results, stats_builder.build())
    }

    /// Search for k nearest neighbors with a timeout.
    ///
    /// This method checks for timeout during the search loop and returns
    /// early with a `Timeout` error if the deadline is exceeded. Use this
    /// when you need to guarantee bounded query latency.
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `k` - Number of neighbors to return
    /// * `timeout` - Maximum time allowed for the search
    ///
    /// # Returns
    /// - `Ok((results, actual_time))` if search completes within timeout
    /// - `Err(Timeout)` if the deadline is exceeded
    ///
    /// # Note
    /// Timeout checking adds some overhead. Use regular `search()` if you
    /// don't need timeout guarantees.
    pub fn search_with_timeout(
        &self,
        query: &[f32],
        k: usize,
        timeout: std::time::Duration,
    ) -> Result<(Vec<(u64, f32)>, std::time::Duration)> {
        use std::time::Instant;

        let start = Instant::now();
        let deadline = start + timeout;

        // If re-ranking enabled, fetch more candidates
        let fetch_k = if self.original_vectors.is_some() {
            k * self.rerank_factor
        } else {
            k
        };

        // Step 1: Find nprobe nearest centroids
        let nprobe = self.nprobe();
        let mut centroid_distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, euclidean_distance_squared(query, &c.data)))
            .collect();

        let nth = nprobe.min(centroid_distances.len()).saturating_sub(1);
        centroid_distances.select_nth_unstable_by(nth, |a, b| a.1.partial_cmp(&b.1).unwrap());

        // Check timeout after centroid selection
        if Instant::now() > deadline {
            return Err(crate::error::ForgeDbError::Timeout);
        }

        use std::cmp::Ordering;
        use std::collections::BinaryHeap;

        #[derive(PartialEq)]
        struct HeapItem(u64, f32);

        impl Eq for HeapItem {}
        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.1.partial_cmp(&other.1)
            }
        }
        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> Ordering {
                self.partial_cmp(other).unwrap_or(Ordering::Equal)
            }
        }

        let mut heap: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(fetch_k + 1);

        // Process partitions with periodic timeout checks
        for i in 0..nprobe.min(centroid_distances.len()) {
            let partition_id = centroid_distances[i].0;
            let partition = self.partitions[partition_id].read().unwrap();
            let centroid = &self.centroids[partition_id];

            let query_residual: Vec<f32> = query
                .iter()
                .zip(centroid.data.iter())
                .map(|(q, c)| q - c)
                .collect();

            let lookup_table = self.pq.build_lookup_table(&query_residual);

            let threshold = if heap.len() >= fetch_k {
                heap.peek().map(|h| h.1).unwrap_or(f32::MAX)
            } else {
                f32::MAX
            };

            let codes_slice = &partition.codes;
            let ids_slice = &partition.ids;
            let len = codes_slice.len();

            // Check timeout every 1000 vectors to avoid overhead
            const CHECK_INTERVAL: usize = 1000;

            for j in 0..len {
                // Periodic timeout check
                if j % CHECK_INTERVAL == 0 && Instant::now() > deadline {
                    return Err(crate::error::ForgeDbError::Timeout);
                }

                // Prefetch 3 vectors ahead
                if j + 3 < len {
                    prefetch_read(codes_slice[j + 3].as_ptr());
                }

                let codes = &codes_slice[j];
                let id = ids_slice[j];

                // Skip tombstoned vectors
                if partition.tombstones.contains(&id) {
                    continue;
                }

                let dist = self.pq.asymmetric_distance_fast(&lookup_table, codes);

                if dist < threshold || heap.len() < fetch_k {
                    heap.push(HeapItem(id, dist));
                    if heap.len() > fetch_k {
                        heap.pop();
                    }
                }
            }

            // Check timeout after each partition
            if Instant::now() > deadline {
                return Err(crate::error::ForgeDbError::Timeout);
            }
        }

        // Convert heap to sorted vector
        let mut results: Vec<(u64, f32)> = heap.into_iter().map(|h| (h.0, h.1)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Re-rank with original vectors if enabled
        if let Some(ref id_to_vec_lock) = self.original_vectors {
            // Check timeout before reranking
            if Instant::now() > deadline {
                return Err(crate::error::ForgeDbError::Timeout);
            }

            let id_to_vec = id_to_vec_lock.read().unwrap();
            let mut reranked: Vec<(u64, f32)> = results
                .iter()
                .filter_map(|(id, _)| {
                    id_to_vec.get(id).map(|v| {
                        let dist = euclidean_distance_squared(query, &v.data).sqrt();
                        (*id, dist)
                    })
                })
                .collect();

            reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            reranked.truncate(k);

            return Ok((reranked, start.elapsed()));
        }

        results.truncate(k);
        Ok((results, start.elapsed()))
    }

    /// Check if the index satisfies resource limits.
    ///
    /// Validates the current index state against the provided limits.
    /// This is useful for capacity planning and preventing resource exhaustion.
    ///
    /// # Arguments
    /// * `limits` - Resource limits to check against
    ///
    /// # Returns
    /// - `Ok(())` if all limits are satisfied
    /// - `Err(MemoryLimitExceeded)` if memory limit is exceeded
    /// - `Err(VectorLimitExceeded)` if vector count limit is exceeded
    pub fn check_resource_limits(&self, limits: &crate::metrics::ResourceLimits) -> Result<()> {
        // Check memory limit
        if let Some(max_memory) = limits.max_memory_bytes {
            let stats = self.statistics();
            if stats.memory_bytes > max_memory {
                return Err(crate::error::ForgeDbError::memory_limit_exceeded(
                    stats.memory_bytes,
                    max_memory,
                ));
            }
        }

        // Check vector count limit
        if let Some(max_vectors) = limits.max_vectors {
            let num_vectors = self.len();
            if num_vectors > max_vectors {
                return Err(crate::error::ForgeDbError::vector_limit_exceeded(
                    num_vectors,
                    max_vectors,
                ));
            }
        }

        Ok(())
    }

    /// Validate the index integrity.
    ///
    /// Checks that the index is internally consistent:
    /// - All partitions have matching ids and codes arrays
    /// - PQ dimension matches centroid dimension
    /// - All codes have the correct length
    /// - Original vectors (if present) have correct dimension
    ///
    /// # Errors
    /// Returns an error if any integrity issue is found.
    pub fn validate(&self) -> Result<()> {
        // Check PQ dimension matches centroids
        if !self.centroids.is_empty() && self.centroids[0].dim() != self.pq.dim {
            return Err(crate::error::ForgeDbError::invalid_parameter(format!(
                "PQ dimension {} does not match centroid dimension {}",
                self.pq.dim,
                self.centroids[0].dim()
            )));
        }

        // Check each partition
        for (i, partition_lock) in self.partitions.iter().enumerate() {
            let partition = partition_lock.read().unwrap();

            // ids and codes should have same length
            if partition.ids.len() != partition.codes.len() {
                return Err(crate::error::ForgeDbError::invalid_parameter(format!(
                    "Partition {} has mismatched ids ({}) and codes ({}) lengths",
                    i,
                    partition.ids.len(),
                    partition.codes.len()
                )));
            }

            // Each code should have n_subvectors bytes
            for (j, codes) in partition.codes.iter().enumerate() {
                if codes.len() != self.pq.n_subvectors {
                    return Err(crate::error::ForgeDbError::invalid_parameter(format!(
                        "Partition {} vector {} has incorrect code length {} (expected {})",
                        i,
                        j,
                        codes.len(),
                        self.pq.n_subvectors
                    )));
                }
            }

            // Tombstones should only contain IDs that exist in the partition
            for &tombstone_id in &partition.tombstones {
                if !partition.ids.contains(&tombstone_id) {
                    return Err(crate::error::ForgeDbError::invalid_parameter(format!(
                        "Partition {} has tombstone for non-existent ID {}",
                        i, tombstone_id
                    )));
                }
            }
        }

        // Check original vectors if present
        if let Some(ref original_lock) = self.original_vectors {
            let original = original_lock.read().unwrap();
            for (id, vec) in original.iter() {
                if vec.dim() != self.pq.dim {
                    return Err(crate::error::ForgeDbError::invalid_parameter(format!(
                        "Original vector {} has incorrect dimension {} (expected {})",
                        id,
                        vec.dim(),
                        self.pq.dim
                    )));
                }
            }
        }

        Ok(())
    }

    /// Perform a comprehensive health check on the index.
    ///
    /// Returns a health status indicating whether the index is:
    /// - Healthy: All checks pass
    /// - Warning: Some non-critical issues found (e.g., high fragmentation)
    /// - Unhealthy: Critical issues found
    pub fn health_check(&self) -> crate::metrics::HealthStatus {
        use crate::metrics::HealthStatus;

        let mut warnings: Vec<String> = Vec::new();
        let mut errors: Vec<String> = Vec::new();

        // Run validation
        if let Err(e) = self.validate() {
            errors.push(format!("Validation failed: {}", e));
        }

        // Check fragmentation
        let frag_ratio = self.fragmentation_ratio();
        if frag_ratio > 0.5 {
            errors.push(format!(
                "High fragmentation: {:.1}% (consider calling compact())",
                frag_ratio * 100.0
            ));
        } else if frag_ratio > 0.2 {
            warnings.push(format!(
                "Moderate fragmentation: {:.1}% (compact() recommended)",
                frag_ratio * 100.0
            ));
        }

        // Check if index is empty
        if self.len() == 0 {
            warnings.push("Index is empty".to_string());
        }

        // Check partition balance
        let partition_sizes: Vec<usize> = self
            .partitions
            .iter()
            .map(|p| p.read().unwrap().ids.len())
            .collect();

        if !partition_sizes.is_empty() {
            let max_size = *partition_sizes.iter().max().unwrap();
            let min_size = *partition_sizes.iter().min().unwrap();
            let mean_size = partition_sizes.iter().sum::<usize>() as f32 / partition_sizes.len() as f32;

            // Check for severely imbalanced partitions
            if min_size > 0 && max_size > min_size * 10 {
                warnings.push(format!(
                    "Severely imbalanced partitions: min={}, max={}, mean={:.1}",
                    min_size, max_size, mean_size
                ));
            }

            // Check for empty partitions
            let empty_count = partition_sizes.iter().filter(|&&s| s == 0).count();
            if empty_count > 0 {
                warnings.push(format!("{} empty partitions", empty_count));
            }
        }

        // Check nprobe setting
        let nprobe = self.get_nprobe();
        if nprobe > self.partitions.len() / 2 {
            warnings.push(format!(
                "High nprobe ({}) relative to partitions ({}), may be slow",
                nprobe,
                self.partitions.len()
            ));
        }

        // Determine overall status
        if !errors.is_empty() {
            HealthStatus::Unhealthy(errors)
        } else if !warnings.is_empty() {
            HealthStatus::Warning(warnings)
        } else {
            HealthStatus::Healthy
        }
    }

    // =========================================================================
    // Incremental Updates
    // =========================================================================

    /// Insert a batch of vectors into the index.
    ///
    /// New vectors are assigned to their nearest centroids and encoded using
    /// the existing PQ codebook. This is faster than rebuilding but may result
    /// in slightly lower recall if the data distribution has changed.
    ///
    /// # Arguments
    /// * `vectors` - Vectors to insert
    ///
    /// # Note
    /// If re-ranking is enabled, you should call `enable_reranking()` again
    /// with the updated vector set after inserting.
    pub fn insert_batch(&mut self, vectors: Vec<Vector>) {
        if vectors.is_empty() {
            return;
        }

        for vector in vectors {
            // Find nearest centroid
            let (partition_id, _) = self
                .centroids
                .iter()
                .enumerate()
                .map(|(idx, c)| (idx, euclidean_distance_squared(&vector.data, &c.data)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();

            // Compute residual
            let centroid = &self.centroids[partition_id];
            let residual: Vec<f32> = vector
                .data
                .iter()
                .zip(centroid.data.iter())
                .map(|(a, b)| a - b)
                .collect();
            let residual_vec = Vector::new(vector.id, residual);

            // Encode and add to partition
            let codes = self.pq.encode(&residual_vec);
            {
                let mut partition = self.partitions[partition_id].write().unwrap();
                partition.ids.push(vector.id);
                partition.codes.push(codes);

                // If this ID was tombstoned, remove from tombstones
                partition.tombstones.remove(&vector.id);
            }

            // Add to original vectors if re-ranking is enabled
            if let Some(ref id_to_vec_lock) = self.original_vectors {
                let mut id_to_vec = id_to_vec_lock.write().unwrap();
                id_to_vec.insert(vector.id, vector);
            }
        }
    }

    /// Mark a vector as deleted.
    ///
    /// The vector's data is not immediately removed but is skipped during search.
    /// Call `compact()` to physically remove deleted vectors and reclaim memory.
    ///
    /// # Arguments
    /// * `id` - ID of the vector to delete
    ///
    /// # Returns
    /// `true` if the vector was found and deleted, `false` if not found.
    pub fn delete(&mut self, id: u64) -> bool {
        // Find which partition contains this ID
        for partition_lock in &self.partitions {
            let mut partition = partition_lock.write().unwrap();
            if partition.ids.contains(&id) && !partition.tombstones.contains(&id) {
                partition.tombstones.insert(id);
                drop(partition); // Release lock before accessing original_vectors

                // Remove from original vectors if present
                if let Some(ref id_to_vec_lock) = self.original_vectors {
                    let mut id_to_vec = id_to_vec_lock.write().unwrap();
                    id_to_vec.remove(&id);
                }

                return true;
            }
        }
        false
    }

    /// Get the number of tombstoned (deleted) vectors.
    pub fn num_tombstones(&self) -> usize {
        self.partitions
            .iter()
            .map(|p| p.read().unwrap().tombstones.len())
            .sum()
    }

    /// Get the fragmentation ratio (tombstones / total vectors).
    pub fn fragmentation_ratio(&self) -> f32 {
        let total = self.len();
        if total == 0 {
            return 0.0;
        }
        let tombstones = self.num_tombstones();
        tombstones as f32 / total as f32
    }

    /// Remove tombstoned entries and reclaim memory.
    ///
    /// This physically removes deleted vectors from the index, reducing memory
    /// usage. Should be called periodically when fragmentation is high.
    pub fn compact(&mut self) {
        for partition_lock in &self.partitions {
            let mut partition = partition_lock.write().unwrap();
            if partition.tombstones.is_empty() {
                continue;
            }

            // Filter out tombstoned entries
            let mut new_ids = Vec::with_capacity(partition.ids.len() - partition.tombstones.len());
            let mut new_codes = Vec::with_capacity(partition.codes.len() - partition.tombstones.len());

            for (id, codes) in partition.ids.iter().zip(partition.codes.iter()) {
                if !partition.tombstones.contains(id) {
                    new_ids.push(*id);
                    new_codes.push(codes.clone());
                }
            }

            partition.ids = new_ids;
            partition.codes = new_codes;
            partition.tombstones.clear();
        }
    }

    /// Rebuild the entire index from the vectors stored in original_vectors.
    ///
    /// This re-clusters all vectors and rebuilds the PQ codebooks. Use this
    /// when the data distribution has changed significantly or after many
    /// insertions/deletions.
    ///
    /// # Errors
    /// Returns an error if re-ranking is not enabled (no original vectors stored).
    pub fn rebuild(&mut self) -> Result<()> {
        let vectors: Vec<Vector> = self
            .original_vectors
            .as_ref()
            .ok_or_else(|| {
                crate::error::ForgeDbError::not_supported(
                    "rebuild requires original vectors (enable re-ranking)",
                )
            })?
            .read()
            .unwrap()
            .values()
            .cloned()
            .collect();

        if vectors.is_empty() {
            return Err(crate::error::ForgeDbError::EmptyVectorSet);
        }

        let n_clusters = self.centroids.len();
        let n_subvectors = self.pq.n_subvectors;
        let metric = self.metric;
        let nprobe = self.nprobe();
        let rerank_factor = self.rerank_factor;

        // Rebuild the index
        *self = IVFPQIndex::build(vectors.clone(), n_clusters, n_subvectors, metric);
        self.set_nprobe(nprobe);
        self.enable_reranking(vectors, rerank_factor);

        Ok(())
    }

    /// Convert to serializable format.
    fn to_serializable(&self) -> SerializableIVFPQ {
        SerializableIVFPQ {
            centroids: self.centroids.iter().map(SerializableVector::from).collect(),
            pq: SerializablePQ::from(&self.pq),
            partitions: self
                .partitions
                .iter()
                .map(|p| SerializablePartition::from(&*p.read().unwrap()))
                .collect(),
            nprobe: self.nprobe(),
            metric: SerializableMetric::from(self.metric),
            original_vectors: self.original_vectors.as_ref().map(|vecs| {
                vecs.read()
                    .unwrap()
                    .values()
                    .map(SerializableVector::from)
                    .collect()
            }),
            rerank_factor: self.rerank_factor,
        }
    }

    /// Construct from serializable format.
    fn from_serializable(s: SerializableIVFPQ) -> Self {
        let original_vectors = s.original_vectors.map(|vecs| {
            RwLock::new(
                vecs.into_iter()
                    .map(|v| {
                        let vec = Vector::from(v);
                        (vec.id, vec)
                    })
                    .collect(),
            )
        });

        Self {
            centroids: s.centroids.into_iter().map(Vector::from).collect(),
            pq: ProductQuantizer::from(s.pq),
            partitions: s
                .partitions
                .into_iter()
                .map(|p| RwLock::new(PartitionData::from(p)))
                .collect(),
            nprobe: AtomicUsize::new(s.nprobe),
            metric: DistanceMetric::from(s.metric),
            original_vectors,
            rerank_factor: s.rerank_factor,
        }
    }
}

impl Persistable for IVFPQIndex {
    fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let serializable = self.to_serializable();
        let data = bincode::serialize(&serializable)?;
        write_with_header(path, IndexType::IvfPq, &data)
    }

    fn load(path: impl AsRef<Path>) -> Result<Self> {
        let file_data = std::fs::read(path)?;
        let data = verify_header(&file_data, IndexType::IvfPq)?;
        let serializable: SerializableIVFPQ = bincode::deserialize(data)?;
        Ok(Self::from_serializable(serializable))
    }

    fn load_mmap(path: impl AsRef<Path>) -> Result<Self> {
        use memmap2::Mmap;
        use std::fs::File;

        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        let data = verify_header(&mmap, IndexType::IvfPq)?;
        let serializable: SerializableIVFPQ = bincode::deserialize(data)?;
        Ok(Self::from_serializable(serializable))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivf_pq_build() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();

        let index = IVFPQIndex::build(vectors, 16, 4, DistanceMetric::Euclidean);

        assert_eq!(index.len(), 500);
        assert_eq!(index.n_partitions(), 16);
    }

    #[test]
    fn test_ivf_pq_search() {
        // Need 300+ vectors for PQ k-means (256 centroids per codebook)
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();

        let mut index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(4);

        let query = vectors[0].data.to_vec();
        let results = index.search(&query, 10);

        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    }

    #[test]
    fn test_nprobe_setting() {
        // Need 300+ vectors for PQ k-means (256 centroids per codebook)
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();

        let mut index = IVFPQIndex::build(vectors, 10, 4, DistanceMetric::Euclidean);

        index.set_nprobe(5);
        assert_eq!(index.nprobe(), 5);

        // nprobe should be capped at number of partitions
        index.set_nprobe(100);
        assert_eq!(index.nprobe(), 10);
    }

    #[test]
    fn test_batch_search() {
        // Need 300+ vectors for PQ k-means (256 centroids per codebook)
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();

        let mut index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(4);

        // Create query vectors as Vec<f32>
        let queries: Vec<Vec<f32>> = (0..5).map(|i| vectors[i].data.to_vec()).collect();

        let results = index.batch_search(&queries, 10);

        // Should return results for all queries
        assert_eq!(results.len(), 5);

        // Each result should have k neighbors, sorted by distance
        for result in &results {
            assert_eq!(result.len(), 10);
            for i in 1..result.len() {
                assert!(result[i - 1].1 <= result[i].1);
            }
        }
    }

    #[test]
    fn test_batch_search_parallel() {
        // Need 300+ vectors for PQ k-means (256 centroids per codebook)
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();

        let mut index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(4);

        let queries: Vec<Vec<f32>> = (0..10)
            .map(|i| vectors[i % vectors.len()].data.to_vec())
            .collect();

        let results = index.batch_search_parallel(&queries, 10);

        assert_eq!(results.len(), 10);
        for result in &results {
            assert_eq!(result.len(), 10);
        }
    }

    #[test]
    fn test_batch_search_consistency() {
        // Test that batch_search gives same results as individual search calls
        // Need 300+ vectors for PQ k-means (256 centroids per codebook)
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();

        let mut index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(8); // Search all partitions for determinism

        let queries: Vec<Vec<f32>> = (0..3).map(|i| vectors[i * 10].data.to_vec()).collect();

        // Get batch results
        let batch_results = index.batch_search(&queries, 5);

        // Get individual results
        let individual_results: Vec<Vec<(u64, f32)>> =
            queries.iter().map(|q| index.search(q, 5)).collect();

        // Results should match (same IDs, similar distances)
        assert_eq!(batch_results.len(), individual_results.len());
        for (batch, indiv) in batch_results.iter().zip(individual_results.iter()) {
            assert_eq!(batch.len(), indiv.len());
            // Check that we get the same IDs (order may vary slightly due to ties)
            let batch_ids: std::collections::HashSet<u64> =
                batch.iter().map(|(id, _)| *id).collect();
            let indiv_ids: std::collections::HashSet<u64> =
                indiv.iter().map(|(id, _)| *id).collect();
            assert_eq!(
                batch_ids, indiv_ids,
                "Batch and individual search should find same IDs"
            );
        }
    }

    #[test]
    fn test_batch_search_empty() {
        // Need 300+ vectors for PQ k-means (256 centroids per codebook)
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();

        let index = IVFPQIndex::build(vectors, 8, 4, DistanceMetric::Euclidean);

        let empty_queries: Vec<Vec<f32>> = vec![];
        let results = index.batch_search(&empty_queries, 10);

        assert!(results.is_empty());
    }

    #[test]
    fn test_persistence_roundtrip() {
        use std::fs;
        use tempfile::NamedTempFile;

        // Build an index
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let mut index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(4);

        // Create a temp file
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Save and load
        index.save(path).expect("save should succeed");
        let loaded = IVFPQIndex::load(path).expect("load should succeed");

        // Verify index properties match
        assert_eq!(loaded.len(), index.len());
        assert_eq!(loaded.n_partitions(), index.n_partitions());
        assert_eq!(loaded.nprobe(), index.nprobe());
        assert_eq!(loaded.dim(), index.dim());

        // Verify search results match
        let query = vectors[0].data.to_vec();
        let original_results = index.search(&query, 10);
        let loaded_results = loaded.search(&query, 10);

        assert_eq!(original_results.len(), loaded_results.len());
        // Results should find the same IDs (may have small distance differences)
        let original_ids: std::collections::HashSet<u64> =
            original_results.iter().map(|(id, _)| *id).collect();
        let loaded_ids: std::collections::HashSet<u64> =
            loaded_results.iter().map(|(id, _)| *id).collect();
        assert_eq!(original_ids, loaded_ids);

        // Cleanup
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_persistence_with_reranking() {
        use std::fs;
        use tempfile::NamedTempFile;

        // Build an index with reranking enabled
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let mut index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(4);
        index.enable_reranking(vectors.clone(), 10);

        // Create a temp file
        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Save and load
        index.save(path).expect("save should succeed");
        let loaded = IVFPQIndex::load(path).expect("load should succeed");

        // Verify reranking data was preserved
        let query = vectors[0].data.to_vec();
        let original_results = index.search(&query, 10);
        let loaded_results = loaded.search(&query, 10);

        // With reranking, exact vectors are used so results should match exactly
        assert_eq!(original_results.len(), loaded_results.len());
        for (orig, loaded) in original_results.iter().zip(loaded_results.iter()) {
            assert_eq!(orig.0, loaded.0); // Same IDs
            assert!((orig.1 - loaded.1).abs() < 1e-5); // Same distances
        }

        // Cleanup
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_persistence_corrupted_file() {
        use tempfile::NamedTempFile;
        use std::io::Write;

        let temp_file = NamedTempFile::new().unwrap();
        let path = temp_file.path();

        // Write invalid data
        let mut file = std::fs::File::create(path).unwrap();
        file.write_all(b"not a valid forge-db file").unwrap();

        // Load should fail
        let result = IVFPQIndex::load(path);
        assert!(result.is_err());
    }

    #[test]
    fn test_search_filtered() {
        use crate::metadata::{FilterCondition, MetadataStore, MetadataValue};

        // Build an index
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let mut index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(8);

        // Create metadata store
        let mut metadata_store = MetadataStore::new();
        for i in 0..500u64 {
            let category = if i % 3 == 0 {
                "electronics"
            } else if i % 3 == 1 {
                "clothing"
            } else {
                "books"
            };
            metadata_store.insert(i, "category", MetadataValue::String(category.into()));
            metadata_store.insert(i, "price", MetadataValue::Float((i as f64) * 2.0));
        }
        metadata_store.build_index("category");

        // Search without filter
        let query = vectors[0].data.to_vec();
        let all_results = index.search(&query, 10);
        assert_eq!(all_results.len(), 10);

        // Search with filter for electronics only
        let filter = FilterCondition::eq("category", "electronics");
        let filtered_results = index.search_filtered(&query, 10, &metadata_store, &filter);

        // All results should be from electronics category
        for (id, _) in &filtered_results {
            let meta = metadata_store.get(*id).unwrap();
            assert_eq!(
                meta.get("category"),
                Some(&MetadataValue::String("electronics".into()))
            );
        }

        // Results should be sorted by distance
        for i in 1..filtered_results.len() {
            assert!(filtered_results[i - 1].1 <= filtered_results[i].1);
        }
    }

    #[test]
    fn test_search_filtered_empty_result() {
        use crate::metadata::{FilterCondition, MetadataStore, MetadataValue};

        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let mut index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(8);

        // Create metadata store with no matching values
        let mut metadata_store = MetadataStore::new();
        for i in 0..500u64 {
            metadata_store.insert(i, "category", MetadataValue::String("electronics".into()));
        }
        metadata_store.build_index("category");

        // Search with filter for non-existent category
        let query = vectors[0].data.to_vec();
        let filter = FilterCondition::eq("category", "furniture");
        let results = index.search_filtered(&query, 10, &metadata_store, &filter);

        assert!(results.is_empty());
    }

    #[test]
    fn test_insert_batch() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let mut index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(8);

        let initial_len = index.len();

        // Insert new vectors
        let new_vectors: Vec<Vector> = (500..550).map(|i| Vector::random(i, 32)).collect();
        index.insert_batch(new_vectors.clone());

        assert_eq!(index.len(), initial_len + 50);

        // Search should find the new vectors
        let query = new_vectors[0].data.to_vec();
        let results = index.search(&query, 10);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_delete() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let mut index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(8);

        let initial_len = index.len();

        // Delete some vectors
        assert!(index.delete(0));
        assert!(index.delete(1));
        assert!(index.delete(2));

        assert_eq!(index.len(), initial_len - 3);
        assert_eq!(index.num_tombstones(), 3);

        // Deleted vectors should not appear in search results
        let query = vectors[0].data.to_vec();
        let results = index.search(&query, 10);
        let result_ids: std::collections::HashSet<u64> = results.iter().map(|(id, _)| *id).collect();
        assert!(!result_ids.contains(&0));
        assert!(!result_ids.contains(&1));
        assert!(!result_ids.contains(&2));
    }

    #[test]
    fn test_compact() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let mut index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(8);

        // Delete some vectors
        for i in 0..50 {
            index.delete(i as u64);
        }

        let len_before = index.len();
        let storage_before = index.len_with_tombstones();

        // Compact should remove tombstones
        index.compact();

        assert_eq!(index.len(), len_before);
        assert!(index.len_with_tombstones() < storage_before);
        assert_eq!(index.num_tombstones(), 0);
        assert_eq!(index.fragmentation_ratio(), 0.0);
    }

    #[test]
    fn test_delete_then_reinsert() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let mut index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(8);

        // Delete a vector
        index.delete(0);
        assert_eq!(index.num_tombstones(), 1);

        // Re-insert the same vector
        index.insert_batch(vec![vectors[0].clone()]);

        // Tombstone should be removed
        assert_eq!(index.num_tombstones(), 0);
    }

    #[test]
    fn test_statistics() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);

        let stats = index.statistics();

        assert_eq!(stats.num_vectors, 500);
        assert_eq!(stats.num_partitions, 8);
        assert_eq!(stats.dimension, 32);
        assert_eq!(stats.num_subvectors, 4);
        assert!(stats.compression_ratio > 1.0);
        assert!(stats.memory_bytes > 0);
        assert_eq!(stats.num_tombstones, 0);
        assert_eq!(stats.fragmentation_ratio, 0.0);
        assert!(!stats.reranking_enabled);
    }

    #[test]
    fn test_statistics_with_tombstones() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let mut index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);

        // Delete some vectors
        for i in 0..50 {
            index.delete(i);
        }

        let stats = index.statistics();

        assert_eq!(stats.num_vectors, 450);
        assert_eq!(stats.num_tombstones, 50);
        assert!(stats.fragmentation_ratio > 0.0);
    }

    #[test]
    fn test_search_with_stats() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(4);

        let query = vectors[0].data.to_vec();
        let (results, stats) = index.search_with_stats(&query, 10);

        // Results should be valid
        assert_eq!(results.len(), 10);

        // Stats should be populated
        assert_eq!(stats.partitions_probed, 4);
        assert!(stats.vectors_scanned > 0);
        assert!(stats.pq_distances_computed > 0);
        assert!(!stats.reranking_performed);
        assert!(stats.query_time_ms() >= 0.0);
    }

    #[test]
    fn test_search_with_stats_reranking() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let mut index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.enable_reranking(vectors.clone(), 10);
        index.set_nprobe(4);

        let query = vectors[0].data.to_vec();
        let (results, stats) = index.search_with_stats(&query, 10);

        assert_eq!(results.len(), 10);
        assert!(stats.reranking_performed);
        assert!(stats.candidates_reranked > 0);
    }

    #[test]
    fn test_validate() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let index = IVFPQIndex::build(vectors, 8, 4, DistanceMetric::Euclidean);

        // A freshly built index should validate
        assert!(index.validate().is_ok());
    }

    #[test]
    fn test_health_check_healthy() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let index = IVFPQIndex::build(vectors, 8, 4, DistanceMetric::Euclidean);

        let status = index.health_check();
        // Freshly built index should be healthy or have minor warnings
        assert!(!status.is_unhealthy());
    }

    #[test]
    fn test_health_check_fragmented() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let mut index = IVFPQIndex::build(vectors, 8, 4, DistanceMetric::Euclidean);

        // Delete more than 50% of vectors to trigger high fragmentation
        for i in 0..300 {
            index.delete(i);
        }

        let status = index.health_check();
        // Should be unhealthy due to high fragmentation
        assert!(status.is_unhealthy() || status.is_warning());
        assert!(status.issues().is_some());
    }

    #[test]
    fn test_health_check_empty() {
        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let mut index = IVFPQIndex::build(vectors, 8, 4, DistanceMetric::Euclidean);

        // Delete all vectors
        for i in 0..500 {
            index.delete(i);
        }

        let status = index.health_check();
        // Should have warnings about empty index
        assert!(!status.is_healthy());
    }

    #[test]
    fn test_search_with_timeout_success() {
        use std::time::Duration;

        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(4);

        let query = vectors[0].data.to_vec();
        // Use a generous timeout
        let result = index.search_with_timeout(&query, 10, Duration::from_secs(10));

        assert!(result.is_ok());
        let (results, elapsed) = result.unwrap();
        assert_eq!(results.len(), 10);
        assert!(elapsed < Duration::from_secs(10));
    }

    #[test]
    fn test_search_with_timeout_expired() {
        use std::time::Duration;

        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(8);

        let query = vectors[0].data.to_vec();
        // Use an extremely short timeout (0 nanoseconds)
        let result = index.search_with_timeout(&query, 10, Duration::from_nanos(0));

        // Should timeout
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::error::ForgeDbError::Timeout
        ));
    }

    #[test]
    fn test_check_resource_limits_pass() {
        use crate::metrics::ResourceLimits;

        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let index = IVFPQIndex::build(vectors, 8, 4, DistanceMetric::Euclidean);

        // Set generous limits
        let limits = ResourceLimits::none()
            .with_max_memory_bytes(100 * 1024 * 1024) // 100 MB
            .with_max_vectors(1000);

        assert!(index.check_resource_limits(&limits).is_ok());
    }

    #[test]
    fn test_check_resource_limits_memory_exceeded() {
        use crate::metrics::ResourceLimits;

        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let index = IVFPQIndex::build(vectors, 8, 4, DistanceMetric::Euclidean);

        // Set very small memory limit
        let limits = ResourceLimits::none().with_max_memory_bytes(100); // 100 bytes

        let result = index.check_resource_limits(&limits);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::error::ForgeDbError::MemoryLimitExceeded { .. }
        ));
    }

    #[test]
    fn test_check_resource_limits_vectors_exceeded() {
        use crate::metrics::ResourceLimits;

        let vectors: Vec<Vector> = (0..500).map(|i| Vector::random(i, 32)).collect();
        let index = IVFPQIndex::build(vectors, 8, 4, DistanceMetric::Euclidean);

        // Set very small vector limit
        let limits = ResourceLimits::none().with_max_vectors(100);

        let result = index.check_resource_limits(&limits);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            crate::error::ForgeDbError::VectorLimitExceeded { .. }
        ));
    }
}
