//! K-Means clustering for IVF index partitioning.
//!
//! Implements Lloyd's algorithm with k-means++ initialization for
//! high-quality centroid placement. Used to partition vectors into
//! clusters for inverted file indexing.

use crate::distance::euclidean_distance_squared;
use crate::vector::Vector;
use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;

/// K-Means clustering algorithm.
///
/// Uses k-means++ initialization for better convergence and
/// parallel assignment/update steps for performance.
pub struct KMeans {
    /// The computed cluster centroids.
    pub centroids: Vec<Vector>,
    /// Number of clusters.
    k: usize,
    /// Maximum iterations before stopping.
    max_iters: usize,
}

impl KMeans {
    /// Create a new K-Means instance.
    ///
    /// # Arguments
    /// * `k` - Number of clusters
    /// * `max_iters` - Maximum iterations for convergence
    pub fn new(k: usize, max_iters: usize) -> Self {
        Self {
            centroids: Vec::new(),
            k,
            max_iters,
        }
    }

    /// Fit the K-Means model to the given vectors.
    ///
    /// Initializes centroids using k-means++ and iteratively refines
    /// until convergence or max iterations reached.
    pub fn fit(&mut self, vectors: &[Vector]) {
        if vectors.is_empty() {
            return;
        }

        let dim = vectors[0].dim();

        // Initialize centroids: random for large k (k-means++ is O(k²n), too slow)
        self.centroids = if self.k > 64 {
            let mut rng = rand::thread_rng();
            let mut indices: Vec<usize> = (0..vectors.len()).collect();
            indices.shuffle(&mut rng);
            indices
                .into_iter()
                .take(self.k)
                .map(|i| vectors[i].clone())
                .collect()
        } else {
            self.kmeans_plus_plus_init(vectors, dim)
        };

        // Iteratively refine centroids
        for _iter in 0..self.max_iters {
            let assignments = self.assign_vectors(vectors);
            let new_centroids = self.update_centroids(vectors, &assignments, dim);
            let change = self.measure_change(&new_centroids);

            self.centroids = new_centroids;

            if change < 0.001 {
                break;
            }
        }
    }

    /// Initialize centroids using k-means++ algorithm.
    ///
    /// Selects initial centroids with probability proportional to
    /// squared distance from existing centroids, leading to better
    /// spread and faster convergence.
    fn kmeans_plus_plus_init(&self, vectors: &[Vector], _dim: usize) -> Vec<Vector> {
        let mut rng = rand::thread_rng();
        let mut centroids: Vec<Vector> = Vec::with_capacity(self.k);

        // Choose first centroid randomly
        if let Some(first) = vectors.choose(&mut rng) {
            centroids.push(first.clone());
        }

        // Choose remaining k-1 centroids
        for _ in 1..self.k {
            // Compute distance from each vector to nearest existing centroid
            let distances: Vec<f32> = vectors
                .par_iter()
                .map(|v| {
                    centroids
                        .iter()
                        .map(|c| euclidean_distance_squared(&v.data, &c.data))
                        .fold(f32::MAX, f32::min)
                })
                .collect();

            // Sum total distance for probability distribution
            let total: f32 = distances.iter().sum();

            if total == 0.0 {
                // All vectors are at centroid locations, pick randomly
                if let Some(v) = vectors.choose(&mut rng) {
                    centroids.push(v.clone());
                }
                continue;
            }

            // Select next centroid with probability proportional to distance squared
            let mut r = rng.gen_range(0.0..total);
            for (i, &d) in distances.iter().enumerate() {
                r -= d;
                if r <= 0.0 {
                    centroids.push(vectors[i].clone());
                    break;
                }
            }

            // Fallback if we didn't select (floating point edge case)
            if centroids.len() < self.k && centroids.len() == centroids.capacity() - 1 {
                if let Some(v) = vectors.choose(&mut rng) {
                    centroids.push(v.clone());
                }
            }
        }

        centroids
    }

    /// Assign each vector to its nearest centroid.
    ///
    /// Returns a vector of centroid indices, one per input vector.
    fn assign_vectors(&self, vectors: &[Vector]) -> Vec<usize> {
        vectors
            .par_iter()
            .map(|v| {
                self.centroids
                    .iter()
                    .enumerate()
                    .map(|(idx, c)| (idx, euclidean_distance_squared(&v.data, &c.data)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap()
                    .0
            })
            .collect()
    }

    /// Update centroids to be the mean of assigned vectors.
    ///
    /// For each cluster, computes the element-wise mean of all vectors
    /// assigned to that cluster. Empty clusters retain their old centroid.
    fn update_centroids(
        &self,
        vectors: &[Vector],
        assignments: &[usize],
        dim: usize,
    ) -> Vec<Vector> {
        (0..self.k)
            .into_par_iter()
            .map(|k| {
                // Collect vectors assigned to this cluster
                let cluster: Vec<&Vector> = vectors
                    .iter()
                    .zip(assignments.iter())
                    .filter(|(_, &a)| a == k)
                    .map(|(v, _)| v)
                    .collect();

                if cluster.is_empty() {
                    // Keep old centroid for empty cluster
                    return self.centroids[k].clone();
                }

                // Compute mean
                let mut mean = vec![0.0f32; dim];
                for v in &cluster {
                    for (i, &val) in v.data.iter().enumerate() {
                        mean[i] += val;
                    }
                }

                let count = cluster.len() as f32;
                for val in &mut mean {
                    *val /= count;
                }

                Vector::new(k as u64, mean)
            })
            .collect()
    }

    /// Measure the average change in centroid positions.
    ///
    /// Used to detect convergence - when change is small, the algorithm
    /// has stabilized.
    fn measure_change(&self, new_centroids: &[Vector]) -> f32 {
        let total: f32 = self
            .centroids
            .iter()
            .zip(new_centroids.iter())
            .map(|(old, new)| euclidean_distance_squared(&old.data, &new.data))
            .sum();

        total / self.k as f32
    }

    /// Find the k nearest centroids to a query point.
    ///
    /// Used by IVF index to determine which partitions to search.
    pub fn find_nearest_centroids(&self, query: &[f32], k: usize) -> Vec<usize> {
        let mut distances: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(idx, c)| (idx, euclidean_distance_squared(query, &c.data)))
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);
        distances.into_iter().map(|(idx, _)| idx).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kmeans_basic() {
        // Use random vectors to test k-means functionality
        let vectors: Vec<Vector> = (0..300).map(|i| Vector::random(i, 8)).collect();

        let mut kmeans = KMeans::new(5, 50);
        kmeans.fit(&vectors);

        // Basic checks
        assert_eq!(kmeans.centroids.len(), 5);

        // Each centroid should have the correct dimension
        for c in &kmeans.centroids {
            assert_eq!(c.dim(), 8);
        }

        // The inertia (sum of squared distances to nearest centroid) should be finite
        let mut total_inertia = 0.0f32;
        for v in &vectors {
            let min_dist = kmeans.centroids
                .iter()
                .map(|c| euclidean_distance_squared(&v.data, &c.data))
                .fold(f32::MAX, f32::min);
            total_inertia += min_dist;
        }
        assert!(total_inertia.is_finite());
        assert!(total_inertia > 0.0);
    }

    #[test]
    fn test_find_nearest_centroids() {
        let mut kmeans = KMeans::new(3, 1);
        kmeans.centroids = vec![
            Vector::new(0, vec![0.0, 0.0]),
            Vector::new(1, vec![10.0, 0.0]),
            Vector::new(2, vec![5.0, 10.0]),
        ];

        let query = [0.1, 0.1];
        let nearest = kmeans.find_nearest_centroids(&query, 2);

        assert_eq!(nearest.len(), 2);
        assert_eq!(nearest[0], 0); // (0,0) is closest
    }
}
