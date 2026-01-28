//! IVF-PQ Index: Inverted File with Product Quantization.
//!
//! Combines IVF (coarse quantization for partitioning) with PQ (fine
//! quantization for compression). This achieves both fast search through
//! reduced search space and low memory through vector compression.
//!
//! Search process:
//! 1. Find nprobe nearest IVF centroids to query
//! 2. Build PQ lookup table for the query
//! 3. Scan vectors in selected partitions using asymmetric PQ distance
//! 4. Return top-k results

use crate::distance::{euclidean_distance_squared, DistanceMetric};
use crate::kmeans::KMeans;
use crate::pq::ProductQuantizer;
use crate::vector::Vector;
use rayon::prelude::*;
use std::collections::HashMap;

// =============================================================================
// Software Prefetching for x86_64
// =============================================================================
// Prefetching hides memory latency by loading data into cache before it's needed.
// We prefetch 2-3 cache lines ahead to overlap memory access with computation.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0, _MM_HINT_T1};

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

/// Prefetch data into L2 cache (non-temporal - streaming access pattern).
/// Use for data that will only be read once.
#[inline(always)]
#[allow(dead_code)]
fn prefetch_read_nt<T>(ptr: *const T) {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        _mm_prefetch(ptr as *const i8, _MM_HINT_T1);
    }
    #[cfg(not(target_arch = "x86_64"))]
    let _ = ptr;
}

/// Data stored in each IVF partition.
struct PartitionData {
    /// Vector IDs in this partition
    ids: Vec<u64>,
    /// PQ codes for each vector (M bytes per vector)
    codes: Vec<Vec<u8>>,
}

/// IVF-PQ Index for scalable approximate nearest neighbor search.
///
/// Uses inverted file indexing for coarse partitioning and product
/// quantization for memory-efficient storage within partitions.
pub struct IVFPQIndex {
    /// IVF cluster centroids
    centroids: Vec<Vector>,
    /// Product quantizer for compression
    pq: ProductQuantizer,
    /// Vectors organized by partition
    partitions: Vec<PartitionData>,
    /// Number of partitions to probe during search
    nprobe: usize,
    /// Distance metric (used for IVF assignment)
    #[allow(dead_code)]
    metric: DistanceMetric,
    /// Original vectors for re-ranking (optional, improves recall)
    /// Stored as HashMap for O(1) lookup
    original_vectors: Option<std::collections::HashMap<u64, Vector>>,
    /// Re-ranking factor: fetch this many candidates, then re-rank
    rerank_factor: usize,
}

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

        println!(
            "Building IVF-PQ index: {} vectors, {} clusters, {} subvectors",
            vectors.len(),
            n_clusters,
            n_subvectors
        );

        // Step 1: Train IVF centroids on a sample (for speed)
        let train_sample_size = vectors.len().min(30_000);
        println!(
            "Step 1: Training IVF centroids (sample size: {})",
            train_sample_size
        );

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
        println!("Step 2: Assigning {} vectors to partitions", vectors.len());
        let assignments: Vec<usize> = vectors
            .par_iter()
            .map(|vector| {
                centroids
                    .iter()
                    .enumerate()
                    .map(|(idx, c)| (idx, euclidean_distance_squared(&vector.data, &c.data)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
                    .0
            })
            .collect();

        let mut partition_vectors: Vec<Vec<Vector>> = vec![Vec::new(); n_clusters];
        for (vector, &partition_id) in vectors.iter().zip(assignments.iter()) {
            partition_vectors[partition_id].push(vector.clone());
        }

        // Step 3: Train PQ on residuals (vector - centroid) for better recall
        println!("Step 3: Training PQ on residuals");
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
        println!("Step 4: Encoding residuals");
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
                PartitionData { ids, codes }
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

        Self {
            centroids,
            pq,
            partitions,
            nprobe: 1,
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
        use std::collections::HashMap;
        let map: HashMap<u64, Vector> = vectors.into_iter().map(|v| (v.id, v)).collect();
        self.original_vectors = Some(map);
        self.rerank_factor = rerank_factor.max(1);
    }

    /// Set the number of partitions to probe during search.
    ///
    /// Higher nprobe = better recall but slower search.
    /// Typically use 1-16 depending on desired recall/speed tradeoff.
    pub fn set_nprobe(&mut self, nprobe: usize) {
        self.nprobe = nprobe.min(self.centroids.len());
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
        let mut centroid_distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, euclidean_distance_squared(query, &c.data)))
            .collect();

        // Partial sort - only need top nprobe, not full sort
        let nth = self.nprobe.min(centroid_distances.len()).saturating_sub(1);
        centroid_distances.select_nth_unstable_by(nth, |a, b| {
            a.1.partial_cmp(&b.1).unwrap()
        });

        // Step 2-3: For each partition, compute residual and scan
        // Use a bounded heap to avoid storing all results
        use std::collections::BinaryHeap;
        use std::cmp::Ordering;

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

        for i in 0..self.nprobe.min(centroid_distances.len()) {
            let partition_id = centroid_distances[i].0;
            let partition = &self.partitions[partition_id];
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

            for i in 0..len {
                // Prefetch 3 vectors ahead to hide memory latency
                if i + 3 < len {
                    prefetch_read(codes_slice[i + 3].as_ptr());
                }

                let codes = &codes_slice[i];
                let id = ids_slice[i];
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
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Re-rank with original vectors if enabled
        if let Some(ref id_to_vec) = self.original_vectors {
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

            reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            reranked.truncate(k);
            return reranked;
        }

        results.truncate(k);
        results
    }

    /// Batch search optimized for cache efficiency.
    ///
    /// Instead of processing queries one-by-one, this groups queries by their
    /// nearest partitions and processes partition-by-partition. This keeps
    /// partition data hot in cache while scoring multiple queries.
    ///
    /// # Arguments
    /// * `queries` - Slice of query vectors (anything that can be dereferenced to &[f32])
    /// * `k` - Number of neighbors to return per query
    ///
    /// # Returns
    /// Vector of results for each query, in the same order as input.
    pub fn batch_search<Q: AsRef<[f32]>>(&self, queries: &[Q], k: usize) -> Vec<Vec<(u64, f32)>> {
        if queries.is_empty() {
            return Vec::new();
        }

        let fetch_k = if self.original_vectors.is_some() {
            k * self.rerank_factor
        } else {
            k
        };

        // Step 1: Find nearest partitions for all queries
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

                let nth = self.nprobe.min(centroid_distances.len()).saturating_sub(1);
                centroid_distances.select_nth_unstable_by(nth, |a, b| {
                    a.1.partial_cmp(&b.1).unwrap()
                });

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
            let partition = &self.partitions[partition_id];
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

            // Score all vectors in partition against all relevant queries
            let codes_slice = &partition.codes;
            let ids_slice = &partition.ids;
            let len = codes_slice.len();

            for i in 0..len {
                // Prefetch ahead
                if i + 3 < len {
                    prefetch_read(codes_slice[i + 3].as_ptr());
                }

                let codes = &codes_slice[i];
                let id = ids_slice[i];

                // Score against each query's lookup table
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

        // Step 5: Convert heaps to sorted results
        let mut results: Vec<Vec<(u64, f32)>> = heaps
            .into_iter()
            .map(|heap| {
                let mut r: Vec<(u64, f32)> = heap.into_iter().map(|h| (h.0, h.1)).collect();
                r.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                r
            })
            .collect();

        // Step 6: Re-rank with original vectors if enabled
        if let Some(ref id_to_vec) = self.original_vectors {
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

                reranked.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
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
        let table_bytes = self.pq.n_subvectors * 256 * 4;
        let chunk_size = (256 * 1024 / (self.nprobe.max(1) * table_bytes))
            .max(4)
            .min(64);

        // Process chunks in parallel
        let chunk_results: Vec<Vec<Vec<(u64, f32)>>> = queries
            .par_chunks(chunk_size)
            .map(|chunk| self.batch_search(chunk, k))
            .collect();

        // Flatten results
        chunk_results.into_iter().flatten().collect()
    }

    /// Return the total number of indexed vectors.
    pub fn len(&self) -> usize {
        self.partitions.iter().map(|p| p.ids.len()).sum()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the number of partitions.
    pub fn n_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// Get the current nprobe setting.
    pub fn get_nprobe(&self) -> usize {
        self.nprobe
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivf_pq_build() {
        let vectors: Vec<Vector> = (0..500)
            .map(|i| Vector::random(i, 32))
            .collect();

        let index = IVFPQIndex::build(vectors, 16, 4, DistanceMetric::Euclidean);

        assert_eq!(index.len(), 500);
        assert_eq!(index.n_partitions(), 16);
    }

    #[test]
    fn test_ivf_pq_search() {
        // Need 300+ vectors for PQ k-means (256 centroids per codebook)
        let vectors: Vec<Vector> = (0..500)
            .map(|i| Vector::random(i, 32))
            .collect();

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
        let vectors: Vec<Vector> = (0..500)
            .map(|i| Vector::random(i, 32))
            .collect();

        let mut index = IVFPQIndex::build(vectors, 10, 4, DistanceMetric::Euclidean);

        index.set_nprobe(5);
        assert_eq!(index.get_nprobe(), 5);

        // nprobe should be capped at number of partitions
        index.set_nprobe(100);
        assert_eq!(index.get_nprobe(), 10);
    }

    #[test]
    fn test_batch_search() {
        // Need 300+ vectors for PQ k-means (256 centroids per codebook)
        let vectors: Vec<Vector> = (0..500)
            .map(|i| Vector::random(i, 32))
            .collect();

        let mut index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(4);

        // Create query vectors as Vec<f32>
        let queries: Vec<Vec<f32>> = (0..5)
            .map(|i| vectors[i].data.to_vec())
            .collect();

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
        let vectors: Vec<Vector> = (0..500)
            .map(|i| Vector::random(i, 32))
            .collect();

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
        let vectors: Vec<Vector> = (0..500)
            .map(|i| Vector::random(i, 32))
            .collect();

        let mut index = IVFPQIndex::build(vectors.clone(), 8, 4, DistanceMetric::Euclidean);
        index.set_nprobe(8); // Search all partitions for determinism

        let queries: Vec<Vec<f32>> = (0..3)
            .map(|i| vectors[i * 10].data.to_vec())
            .collect();

        // Get batch results
        let batch_results = index.batch_search(&queries, 5);

        // Get individual results
        let individual_results: Vec<Vec<(u64, f32)>> = queries
            .iter()
            .map(|q| index.search(q, 5))
            .collect();

        // Results should match (same IDs, similar distances)
        assert_eq!(batch_results.len(), individual_results.len());
        for (batch, indiv) in batch_results.iter().zip(individual_results.iter()) {
            assert_eq!(batch.len(), indiv.len());
            // Check that we get the same IDs (order may vary slightly due to ties)
            let batch_ids: std::collections::HashSet<u64> = batch.iter().map(|(id, _)| *id).collect();
            let indiv_ids: std::collections::HashSet<u64> = indiv.iter().map(|(id, _)| *id).collect();
            assert_eq!(batch_ids, indiv_ids, "Batch and individual search should find same IDs");
        }
    }

    #[test]
    fn test_batch_search_empty() {
        // Need 300+ vectors for PQ k-means (256 centroids per codebook)
        let vectors: Vec<Vector> = (0..500)
            .map(|i| Vector::random(i, 32))
            .collect();

        let index = IVFPQIndex::build(vectors, 8, 4, DistanceMetric::Euclidean);

        let empty_queries: Vec<Vec<f32>> = vec![];
        let results = index.batch_search(&empty_queries, 10);

        assert!(results.is_empty());
    }
}
