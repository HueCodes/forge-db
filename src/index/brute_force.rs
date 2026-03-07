//! Brute force index for exact nearest neighbor search.
//!
//! This implementation serves as the ground truth baseline for approximate
//! search algorithms. It computes distances to all vectors and returns the
//! k closest.
//!
//! Uses contiguous `VectorStore` layout for optimal cache performance —
//! all vector data lives in a single flat buffer, eliminating pointer chasing.

use crate::constants::cache::BRUTE_FORCE_CHUNK_SIZE;
use crate::distance::DistanceMetric;
use crate::error::{ForgeDbError, Result};
use crate::index::traits::{MutableVectorIndex, SearchResult, VectorIndex};
use crate::types::VectorId;
use crate::vector::{Vector, VectorStore};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

/// Prefetch distance for brute force scan — how many vectors ahead to prefetch.
const PREFETCH_DISTANCE: usize = 4;

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
/// Uses contiguous `VectorStore` layout for optimal cache performance.
/// All vector data is stored in a single flat buffer, eliminating pointer
/// chasing and maximizing prefetcher efficiency.
pub struct BruteForceIndex {
    store: VectorStore,
    metric: DistanceMetric,
}

impl BruteForceIndex {
    /// Create a new empty brute force index with the given distance metric.
    pub fn new(metric: DistanceMetric) -> Self {
        Self {
            store: VectorStore::new(0),
            metric,
        }
    }

    /// Add a vector to the index.
    pub fn add(&mut self, vector: Vector) {
        if self.store.dim == 0 && !vector.data.is_empty() {
            self.store.dim = vector.dim();
        }
        self.store.push(vector.id, &vector.data);
    }

    /// Return the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.store.len()
    }

    /// Return true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Return the dimensionality of vectors in this index.
    ///
    /// Returns 0 if the index is empty.
    pub fn dimension(&self) -> usize {
        if self.store.is_empty() {
            0
        } else {
            self.store.dim
        }
    }

    /// Helper to run a linear scan over a range of indices and collect top-k into a heap.
    #[inline]
    fn scan_range(
        &self,
        query: &[f32],
        start: usize,
        end: usize,
        heap: &mut BinaryHeap<ScoredVector>,
        k: usize,
    ) {
        let dim = self.store.dim;
        let data_ptr = self.store.data.as_ptr();

        for i in start..end {
            // Prefetch ahead in the contiguous data buffer
            #[cfg(target_arch = "x86_64")]
            if i + PREFETCH_DISTANCE < end {
                // SAFETY: _mm_prefetch is a CPU hint — invalid addresses are no-ops.
                unsafe {
                    _mm_prefetch(
                        data_ptr.add((i + PREFETCH_DISTANCE) * dim) as *const i8,
                        _MM_HINT_T0,
                    );
                }
            }
            #[cfg(target_arch = "aarch64")]
            if i + PREFETCH_DISTANCE < end {
                unsafe {
                    let ptr = data_ptr.add((i + PREFETCH_DISTANCE) * dim);
                    // PRFM PLDL1KEEP — prefetch for load into L1 cache
                    std::arch::asm!("prfm pldl1keep, [{ptr}]", ptr = in(reg) ptr, options(nostack, preserves_flags));
                }
            }

            let data = self.store.get_data(i);
            let id = self.store.get_id(i);
            let distance = self.metric.compute(query, data);

            if heap.len() < k {
                heap.push(ScoredVector { id, distance });
            } else if distance < heap.peek().unwrap().distance {
                heap.pop();
                heap.push(ScoredVector { id, distance });
            }
        }
    }

    /// Drain a heap into sorted (id, distance) results.
    #[inline]
    fn heap_to_results(heap: BinaryHeap<ScoredVector>) -> Vec<(u64, f32)> {
        let mut results: Vec<(u64, f32)> =
            heap.into_iter().map(|sv| (sv.id, sv.distance)).collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results
    }

    /// Search for the k nearest neighbors using basic linear scan.
    ///
    /// Returns a vector of (id, distance) pairs sorted by distance.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let mut heap: BinaryHeap<ScoredVector> = BinaryHeap::with_capacity(k);
        self.scan_range(query, 0, self.store.len(), &mut heap, k);
        Self::heap_to_results(heap)
    }

    /// Search with software prefetching for improved cache performance.
    ///
    /// With VectorStore's contiguous layout, prefetching is built into the
    /// scan_range method, so this is equivalent to search().
    pub fn search_prefetch(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        self.search(query, k)
    }

    /// Parallel search using Rayon for multi-core scaling.
    ///
    /// Divides the vector set into chunks, processes each chunk in parallel,
    /// then merges results. Provides near-linear scaling with CPU cores.
    pub fn search_parallel(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let n = self.store.len();
        if n == 0 {
            return Vec::new();
        }

        let chunk_size = BRUTE_FORCE_CHUNK_SIZE;
        let n_chunks = n.div_ceil(chunk_size);

        let final_heap = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(n);
                let mut local_heap: BinaryHeap<ScoredVector> = BinaryHeap::with_capacity(k);
                self.scan_range(query, start, end, &mut local_heap, k);
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

        Self::heap_to_results(final_heap)
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
        if !self.store.is_empty() {
            let expected_dim = self.store.dim;
            if data.len() != expected_dim {
                return Err(ForgeDbError::dimension_mismatch(expected_dim, data.len()));
            }
        }

        self.add(Vector::new(id.0, data.to_vec()));
        Ok(())
    }

    fn remove(&mut self, id: VectorId) -> Result<bool> {
        if let Some(idx) = self.store.find_id(id.0) {
            self.store.swap_remove(idx);
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

impl crate::Persistable for BruteForceIndex {
    fn save(&self, path: impl AsRef<std::path::Path>) -> crate::error::Result<()> {
        let dim = self.dimension();
        let n = self.store.len();

        let metric_byte: u8 = match self.metric {
            DistanceMetric::Euclidean => 0,
            DistanceMetric::EuclideanSquared => 1,
            DistanceMetric::Cosine => 2,
            DistanceMetric::DotProduct => 3,
            DistanceMetric::Manhattan => 4,
        };

        // Payload: [metric:u8][n:u64][dim:u64][ids...][data...]
        // Use bulk write from contiguous buffers
        let id_bytes = n * 8;
        let data_bytes = n * dim * 4;
        let mut payload = Vec::with_capacity(1 + 8 + 8 + id_bytes + data_bytes);

        payload.push(metric_byte);
        payload.extend_from_slice(&(n as u64).to_le_bytes());
        payload.extend_from_slice(&(dim as u64).to_le_bytes());

        // Bulk write IDs
        for &id in &self.store.ids {
            payload.extend_from_slice(&id.to_le_bytes());
        }

        // Bulk write contiguous float data
        // SAFETY: f32 slice can be safely viewed as bytes
        let data_as_bytes = unsafe {
            std::slice::from_raw_parts(
                self.store.data.as_ptr() as *const u8,
                self.store.data.len() * 4,
            )
        };
        payload.extend_from_slice(data_as_bytes);

        crate::persistence::write_with_header(
            path,
            crate::persistence::IndexType::BruteForce,
            &payload,
        )
    }

    fn load(path: impl AsRef<std::path::Path>) -> crate::error::Result<Self> {
        let raw = std::fs::read(path)?;
        let payload =
            crate::persistence::verify_header(&raw, crate::persistence::IndexType::BruteForce)?;

        if payload.len() < 17 {
            return Err(crate::error::ForgeDbError::invalid_format(
                "brute-force payload too small",
            ));
        }

        let metric_byte = payload[0];
        let metric = match metric_byte {
            0 => DistanceMetric::Euclidean,
            1 => DistanceMetric::EuclideanSquared,
            2 => DistanceMetric::Cosine,
            3 => DistanceMetric::DotProduct,
            4 => DistanceMetric::Manhattan,
            _ => {
                return Err(crate::error::ForgeDbError::invalid_format(
                    "unknown metric byte",
                ))
            }
        };

        let n = u64::from_le_bytes(payload[1..9].try_into().unwrap()) as usize;
        let dim = u64::from_le_bytes(payload[9..17].try_into().unwrap()) as usize;

        let id_bytes = n * 8;
        let data_bytes = n * dim * 4;
        let expected_len = 17 + id_bytes + data_bytes;
        if payload.len() < expected_len {
            return Err(crate::error::ForgeDbError::invalid_format(
                "brute-force payload truncated",
            ));
        }

        let mut store = VectorStore::with_capacity(dim, n);

        // Bulk read IDs
        let mut offset = 17;
        let ids: Vec<u64> = (0..n)
            .map(|i| {
                let start = offset + i * 8;
                u64::from_le_bytes(payload[start..start + 8].try_into().unwrap())
            })
            .collect();
        offset += id_bytes;

        // Bulk read contiguous float data
        let data: Vec<f32> = (0..n * dim)
            .map(|i| {
                let start = offset + i * 4;
                f32::from_le_bytes(payload[start..start + 4].try_into().unwrap())
            })
            .collect();

        store.ids = ids;
        store.data = data;
        store.len = n;

        Ok(BruteForceIndex { store, metric })
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

    #[test]
    fn test_remove() {
        let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);
        for i in 0..10 {
            index.add(Vector::new(i, vec![i as f32; 4]));
        }
        assert_eq!(index.len(), 10);

        // Remove existing vector
        let removed = MutableVectorIndex::remove(&mut index, VectorId(5)).unwrap();
        assert!(removed);
        assert_eq!(index.len(), 9);

        // Remove non-existing vector
        let removed = MutableVectorIndex::remove(&mut index, VectorId(5)).unwrap();
        assert!(!removed);
        assert_eq!(index.len(), 9);

        // Verify the removed vector is not found in search
        let results = index.search(&[5.0; 4], 10);
        assert!(results.iter().all(|(id, _)| *id != 5));
    }

    #[test]
    fn test_persistence_roundtrip() {
        use crate::Persistable;

        let mut index = BruteForceIndex::new(DistanceMetric::Cosine);
        for i in 0..50u64 {
            index.add(Vector::random(i, 16));
        }

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bf.fdb");
        index.save(&path).unwrap();

        let loaded = BruteForceIndex::load(&path).unwrap();
        assert_eq!(loaded.len(), 50);
        assert_eq!(loaded.dimension(), 16);

        // Search results should be identical
        let query = Vector::random(999, 16);
        let a = index.search(&query.data, 5);
        let b = loaded.search(&query.data, 5);
        assert_eq!(a, b);
    }
}
