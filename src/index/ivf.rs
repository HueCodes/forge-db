//! Inverted File (IVF) Index for approximate nearest neighbor search.
//!
//! IVF partitions the vector space using k-means clustering, then searches
//! only the most relevant partitions at query time. This trades some recall
//! for dramatically faster search on large datasets.

use crate::distance::{euclidean_distance_squared, DistanceMetric};
use crate::kmeans::KMeans;
use crate::vector::Vector;
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

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
        // Max-heap ordering: BinaryHeap gives us largest first
        // We want to maintain k smallest, so peek() gives current worst
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for ScoredVector {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Inverted File Index for approximate nearest neighbor search.
///
/// Vectors are partitioned into clusters based on their nearest centroid.
/// At query time, only `nprobe` nearest partitions are searched, trading
/// recall for speed.
pub struct IVFIndex {
    /// Cluster centroids from k-means.
    centroids: Vec<Vector>,
    /// Vectors in each partition, indexed by cluster ID.
    partitions: Vec<Vec<Vector>>,
    /// Distance metric for search.
    metric: DistanceMetric,
    /// Number of partitions to search per query.
    nprobe: usize,
}

impl IVFIndex {
    /// Build an IVF index from vectors using k-means clustering.
    ///
    /// # Arguments
    /// * `vectors` - Vectors to index
    /// * `n_clusters` - Number of partitions (k-means clusters)
    /// * `metric` - Distance metric for search
    ///
    /// # Returns
    /// A new IVFIndex ready for search queries.
    pub fn build(vectors: Vec<Vector>, n_clusters: usize, metric: DistanceMetric) -> Self {
        println!("Building IVF index with {} clusters", n_clusters);

        // Train k-means clustering
        let mut kmeans = KMeans::new(n_clusters, 100);
        kmeans.fit(&vectors);

        println!("Assigning vectors to partitions");

        // Initialize empty partitions
        let mut partitions: Vec<Vec<Vector>> = (0..n_clusters).map(|_| Vec::new()).collect();

        // Assign each vector to its nearest centroid's partition
        for vector in vectors {
            let nearest = kmeans
                .centroids
                .iter()
                .enumerate()
                .map(|(idx, c)| (idx, euclidean_distance_squared(&vector.data, &c.data)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap()
                .0;

            partitions[nearest].push(vector);
        }

        // Print partition statistics
        let sizes: Vec<usize> = partitions.iter().map(|p| p.len()).collect();
        let max_size = sizes.iter().max().copied().unwrap_or(0);
        let min_size = sizes.iter().min().copied().unwrap_or(0);
        let avg_size = if !sizes.is_empty() {
            sizes.iter().sum::<usize>() / sizes.len()
        } else {
            0
        };

        println!(
            "Partition sizes - min: {}, max: {}, avg: {}",
            min_size, max_size, avg_size
        );

        Self {
            centroids: kmeans.centroids,
            partitions,
            metric,
            nprobe: 1,
        }
    }

    /// Set the number of partitions to probe during search.
    ///
    /// Higher nprobe increases recall but decreases speed.
    /// Clamped to the number of centroids.
    pub fn set_nprobe(&mut self, nprobe: usize) {
        self.nprobe = nprobe.min(self.centroids.len());
    }

    /// Get the current nprobe setting.
    pub fn nprobe(&self) -> usize {
        self.nprobe
    }

    /// Return the total number of vectors in the index.
    pub fn len(&self) -> usize {
        self.partitions.iter().map(|p| p.len()).sum()
    }

    /// Return true if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the number of partitions (clusters).
    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// Search for the k nearest neighbors.
    ///
    /// # Arguments
    /// * `query` - Query vector data
    /// * `k` - Number of neighbors to return
    ///
    /// # Returns
    /// Vector of (id, distance) pairs sorted by distance.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        // Step 1: Find nprobe nearest centroids
        let mut centroid_distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, euclidean_distance_squared(query, &c.data)))
            .collect();

        centroid_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let nearest_partitions: Vec<usize> = centroid_distances
            .iter()
            .take(self.nprobe)
            .map(|(idx, _)| *idx)
            .collect();

        // Step 2: Search within selected partitions
        let mut heap: BinaryHeap<ScoredVector> = BinaryHeap::with_capacity(k);

        for &partition_id in &nearest_partitions {
            for vector in &self.partitions[partition_id] {
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
        }

        // Step 3: Convert heap to sorted results
        let mut results: Vec<(u64, f32)> =
            heap.into_iter().map(|sv| (sv.id, sv.distance)).collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
    }

    /// Search within a single partition (for parallel search).
    fn search_partition(&self, query: &[f32], partition_id: usize, k: usize) -> Vec<ScoredVector> {
        let mut heap: BinaryHeap<ScoredVector> = BinaryHeap::with_capacity(k);

        for vector in &self.partitions[partition_id] {
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

        heap.into_vec()
    }

    /// Parallel search across multiple partitions.
    ///
    /// Uses Rayon to search partitions concurrently, then merges results.
    pub fn search_parallel(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        // Step 1: Find nprobe nearest centroids
        let mut centroid_distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, euclidean_distance_squared(query, &c.data)))
            .collect();

        centroid_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let nearest_partitions: Vec<usize> = centroid_distances
            .iter()
            .take(self.nprobe)
            .map(|(idx, _)| *idx)
            .collect();

        // Step 2: Search partitions in parallel
        let partial_results: Vec<Vec<ScoredVector>> = nearest_partitions
            .par_iter()
            .map(|&pid| self.search_partition(query, pid, k))
            .collect();

        // Step 3: Merge results
        let mut heap: BinaryHeap<ScoredVector> = BinaryHeap::with_capacity(k);

        for partition_results in partial_results {
            for sv in partition_results {
                if heap.len() < k {
                    heap.push(sv);
                } else if sv.distance < heap.peek().unwrap().distance {
                    heap.pop();
                    heap.push(sv);
                }
            }
        }

        let mut results: Vec<(u64, f32)> =
            heap.into_iter().map(|sv| (sv.id, sv.distance)).collect();

        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results
    }

    /// Batch search for multiple queries in parallel.
    ///
    /// Processes all queries concurrently using Rayon.
    pub fn batch_search(&self, queries: &[Vector], k: usize) -> Vec<Vec<(u64, f32)>> {
        queries
            .par_iter()
            .map(|query| self.search(&query.data, k))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ivf_basic() {
        let vectors: Vec<Vector> = (0..1000).map(|i| Vector::random(i, 32)).collect();

        let index = IVFIndex::build(vectors, 10, DistanceMetric::EuclideanSquared);

        assert_eq!(index.num_partitions(), 10);
        assert_eq!(index.len(), 1000);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_ivf_search() {
        let vectors: Vec<Vector> = (0..1000).map(|i| Vector::random(i, 64)).collect();

        let mut index = IVFIndex::build(vectors, 10, DistanceMetric::EuclideanSquared);
        index.set_nprobe(5);

        let query = Vector::random(9999, 64);
        let results = index.search(&query.data, 10);

        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    }

    #[test]
    fn test_ivf_search_parallel() {
        let vectors: Vec<Vector> = (0..1000).map(|i| Vector::random(i, 64)).collect();

        let mut index = IVFIndex::build(vectors, 10, DistanceMetric::EuclideanSquared);
        index.set_nprobe(10); // Search all partitions for comparison

        let query = Vector::random(9999, 64);

        let sequential = index.search(&query.data, 10);
        let parallel = index.search_parallel(&query.data, 10);

        // Both should return same results when searching all partitions
        assert_eq!(sequential.len(), parallel.len());
        for i in 0..sequential.len() {
            assert_eq!(sequential[i].0, parallel[i].0);
        }
    }

    #[test]
    fn test_ivf_batch_search() {
        let vectors: Vec<Vector> = (0..1000).map(|i| Vector::random(i, 64)).collect();

        let mut index = IVFIndex::build(vectors, 10, DistanceMetric::EuclideanSquared);
        index.set_nprobe(3);

        let queries: Vec<Vector> = (0..10).map(|i| Vector::random(10000 + i, 64)).collect();
        let results = index.batch_search(&queries, 5);

        assert_eq!(results.len(), 10);
        for r in &results {
            assert_eq!(r.len(), 5);
        }
    }

    #[test]
    fn test_nprobe_clamped() {
        let vectors: Vec<Vector> = (0..100).map(|i| Vector::random(i, 16)).collect();
        let mut index = IVFIndex::build(vectors, 5, DistanceMetric::Euclidean);

        index.set_nprobe(100); // More than number of partitions
        assert_eq!(index.nprobe(), 5); // Should be clamped
    }
}
