//! Brute force index for exact nearest neighbor search.
//!
//! This implementation serves as the ground truth baseline for approximate
//! search algorithms. It computes distances to all vectors and returns the
//! k closest.

use crate::constants::cache::BRUTE_FORCE_CHUNK_SIZE;
use crate::distance::DistanceMetric;
use crate::error::{ForgeDbError, Result};
use crate::index::traits::{MutableVectorIndex, SearchResult, VectorIndex};
use crate::types::VectorId;
use crate::vector::Vector;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

/// A vector with its computed distance, used for heap operations.
#[derive(Clone)]
struct ScoredVector {
    id: u64,
    distance: f32,
}

impl PartialEq for ScoredVector {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for ScoredVector {}

impl PartialOrd for ScoredVector {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScoredVector {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .partial_cmp(&other.distance)
            .unwrap_or(Ordering::Equal)
    }
}

/// Brute force index that performs exact nearest neighbor search.
///
/// This index computes distances to all stored vectors for each query,
/// guaranteeing 100% recall at the cost of O(n) search time.
pub struct BruteForceIndex {
    vectors: Vec<Vector>,
    metric: DistanceMetric,
}

impl BruteForceIndex {
    /// Create a new empty brute force index with the given distance metric.
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            vectors: Vec::new(),
            metric,
        }
    }

    /// Add a vector to the index.
    pub fn add(&mut self, vector: Vector) {
        self.vectors.push(vector);
    }

    /// Return the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Return true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Return the dimensionality of vectors in this index.
    ///
    /// Returns 0 if the index is empty.
    pub fn dimension(&self) -> usize {
        self.vectors.first().map(|v| v.dim()).unwrap_or(0)
    }

    /// Search for the k nearest neighbors using basic linear scan.
    ///
    /// Returns a vector of (id, distance) pairs sorted by distance.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut heap: BinaryHeap<ScoredVector> = BinaryHeap::with_capacity(k);

        for vector in &self.vectors {
            let distance = self.metric.compute(query, &vector.data);

            if heap.len() < k {
                heap.push(ScoredVector {
                    id: vector.id,
                    distance,
                });
            } else if distance < heap.peek().unwrap().distance {
                heap.pop();
                heap.push(ScoredVector {
                    id: vector.id,
                    distance,
                });
            }
        }

        let mut results: Vec<(u64, f32)> =
            heap.into_iter().map(|sv| (sv.id, sv.distance)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Search with software prefetching for improved cache performance.
    ///
    /// Prefetches the next vector's data while processing the current one,
    /// hiding memory latency for sequential access patterns.
    pub fn search_prefetch(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut heap: BinaryHeap<ScoredVector> = BinaryHeap::with_capacity(k);

        for i in 0..self.vectors.len() {
            // Prefetch next vector's data
            #[cfg(target_arch = "x86_64")]
            if i + 1 < self.vectors.len() {
                // SAFETY: _mm_prefetch is always safe - it's a CPU hint that does not
                // dereference the pointer. Invalid addresses result in no-op, not crashes.
                unsafe {
                    _mm_prefetch(self.vectors[i + 1].data.as_ptr() as *const i8, _MM_HINT_T0);
                }
            }

            let vector = &self.vectors[i];
            let distance = self.metric.compute(query, &vector.data);

            if heap.len() < k {
                heap.push(ScoredVector {
                    id: vector.id,
                    distance,
                });
            } else if distance < heap.peek().unwrap().distance {
                heap.pop();
                heap.push(ScoredVector {
                    id: vector.id,
                    distance,
                });
            }
        }

        let mut results: Vec<(u64, f32)> =
            heap.into_iter().map(|sv| (sv.id, sv.distance)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Parallel search using Rayon for multi-core scaling.
    ///
    /// Divides the vector set into chunks, processes each chunk in parallel,
    /// then merges results. Provides near-linear scaling with CPU cores.
    pub fn search_parallel(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let final_heap = self
            .vectors
            .par_chunks(BRUTE_FORCE_CHUNK_SIZE)
            .map(|chunk| {
                let mut local_heap: BinaryHeap<ScoredVector> = BinaryHeap::with_capacity(k);

                for vector in chunk {
                    let distance = self.metric.compute(query, &vector.data);

                    if local_heap.len() < k {
                        local_heap.push(ScoredVector {
                            id: vector.id,
                            distance,
                        });
                    } else if distance < local_heap.peek().unwrap().distance {
                        local_heap.pop();
                        local_heap.push(ScoredVector {
                            id: vector.id,
                            distance,
                        });
                    }
                }

                local_heap
            })
            .reduce(
                || BinaryHeap::with_capacity(k),
                |mut a, b| {
                    for item in b {
                        if a.len() < k {
                            a.push(item);
                        } else if item.distance < a.peek().unwrap().distance {
                            a.pop();
                            a.push(item);
                        }
                    }
                    a
                },
            );

        let mut results: Vec<(u64, f32)> = final_heap
            .into_iter()
            .map(|sv| (sv.id, sv.distance))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Batch search for multiple queries in parallel.
    ///
    /// Processes all queries concurrently using Rayon, returning results
    /// for each query in the same order as the input.
    pub fn batch_search(&self, queries: &[Vector], k: usize) -> Vec<Vec<(u64, f32)>> {
        queries
            .par_iter()
            .map(|query| self.search_parallel(&query.data, k))
            .collect()
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

impl VectorIndex for BruteForceIndex {
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult> {
        self.search(query, k)
            .into_iter()
            .map(SearchResult::from)
            .collect()
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn dimension(&self) -> usize {
        self.dimension()
    }
}

impl MutableVectorIndex for BruteForceIndex {
    fn add(&mut self, id: VectorId, data: &[f32]) -> Result<()> {
        // Check dimension consistency
        if !self.vectors.is_empty() {
            let expected_dim = self.vectors[0].dim();
            if data.len() != expected_dim {
                return Err(ForgeDbError::dimension_mismatch(expected_dim, data.len()));
            }
        }

        self.vectors.push(Vector::new(id.0, data.to_vec()));
        Ok(())
    }

    fn remove(&mut self, id: VectorId) -> Result<bool> {
        let initial_len = self.vectors.len();
        self.vectors.retain(|v| v.id != id.0);
        Ok(self.vectors.len() < initial_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_search() {
        let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);

        // Add some vectors
        for i in 0..100 {
            index.add(Vector::random(i, 128));
        }

        let query = Vector::random(1000, 128);
        let results = index.search(&query.data, 10);

        assert_eq!(results.len(), 10);
        // Verify results are sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    }

    #[test]
    fn test_search_variants_consistency() {
        let mut index = BruteForceIndex::new(DistanceMetric::EuclideanSquared);

        for i in 0..1000 {
            index.add(Vector::random(i, 64));
        }

        let query = Vector::random(9999, 64);

        let basic = index.search(&query.data, 10);
        let prefetch = index.search_prefetch(&query.data, 10);
        let parallel = index.search_parallel(&query.data, 10);

        // All variants should return the same results
        assert_eq!(basic.len(), prefetch.len());
        assert_eq!(basic.len(), parallel.len());

        for i in 0..basic.len() {
            assert_eq!(basic[i].0, prefetch[i].0);
            assert_eq!(basic[i].0, parallel[i].0);
        }
    }
}
