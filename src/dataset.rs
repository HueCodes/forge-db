//! Dataset utilities for generating, loading, and evaluating vector search.

use crate::vector::Vector;
use std::collections::HashSet;
use std::fs::File;
use std::io::{self, BufReader, Read};

/// A dataset containing vectors, queries, and ground truth for evaluation.
pub struct Dataset {
    pub vectors: Vec<Vector>,
    pub queries: Vec<Vector>,
    pub ground_truth: Vec<Vec<u64>>,
}

impl Dataset {
    /// Generate a random synthetic dataset.
    ///
    /// Creates `n_vectors` random vectors and `n_queries` random query vectors,
    /// all with the specified dimensionality.
    pub fn generate(n_vectors: usize, n_queries: usize, dim: usize) -> Self {
        let vectors: Vec<Vector> = (0..n_vectors)
            .map(|i| Vector::random(i as u64, dim))
            .collect();

        let queries: Vec<Vector> = (0..n_queries)
            .map(|i| Vector::random((n_vectors + i) as u64, dim))
            .collect();

        Self {
            vectors,
            queries,
            ground_truth: Vec::new(),
        }
    }

    /// Compute ground truth nearest neighbors using brute force search.
    ///
    /// For each query, finds the k nearest neighbors and stores their IDs.
    pub fn compute_ground_truth(&mut self, k: usize) {
        use crate::distance::DistanceMetric;
        use crate::index::brute_force::BruteForceIndex;

        let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);
        for vector in &self.vectors {
            index.add(vector.clone());
        }

        self.ground_truth = self
            .queries
            .iter()
            .map(|query| {
                index
                    .search(&query.data, k)
                    .into_iter()
                    .map(|(id, _)| id)
                    .collect()
            })
            .collect();
    }

    /// Load the SIFT1M dataset from disk.
    ///
    /// Expects files in the standard format:
    /// - `{base_path}/sift_base.fvecs` - 1M base vectors
    /// - `{base_path}/sift_query.fvecs` - 10K query vectors
    /// - `{base_path}/sift_groundtruth.ivecs` - ground truth neighbors
    pub fn load_sift1m(base_path: &str) -> io::Result<Self> {
        fn read_fvecs(path: &str) -> io::Result<Vec<Vec<f32>>> {
            let file = File::open(path)?;
            let mut reader = BufReader::new(file);
            let mut vectors = Vec::new();

            loop {
                // Read dimension (4 bytes, little-endian i32)
                let mut dim_buf = [0u8; 4];
                if reader.read_exact(&mut dim_buf).is_err() {
                    break;
                }
                let dim = i32::from_le_bytes(dim_buf) as usize;

                // Read vector data (dim * 4 bytes)
                let mut data_buf = vec![0u8; dim * 4];
                reader.read_exact(&mut data_buf)?;

                // Convert bytes to f32 array
                let data: Vec<f32> = data_buf
                    .chunks_exact(4)
                    .map(|chunk| {
                        let arr: [u8; 4] = chunk.try_into().unwrap();
                        f32::from_le_bytes(arr)
                    })
                    .collect();

                vectors.push(data);
            }

            Ok(vectors)
        }

        fn read_ivecs(path: &str) -> io::Result<Vec<Vec<u64>>> {
            let file = File::open(path)?;
            let mut reader = BufReader::new(file);
            let mut vectors = Vec::new();

            loop {
                // Read dimension (4 bytes, little-endian i32)
                let mut dim_buf = [0u8; 4];
                if reader.read_exact(&mut dim_buf).is_err() {
                    break;
                }
                let dim = i32::from_le_bytes(dim_buf) as usize;

                // Read vector data (dim * 4 bytes)
                let mut data_buf = vec![0u8; dim * 4];
                reader.read_exact(&mut data_buf)?;

                // Convert bytes to i32 then to u64
                let data: Vec<u64> = data_buf
                    .chunks_exact(4)
                    .map(|chunk| {
                        let arr: [u8; 4] = chunk.try_into().unwrap();
                        i32::from_le_bytes(arr) as u64
                    })
                    .collect();

                vectors.push(data);
            }

            Ok(vectors)
        }

        let base_vecs = read_fvecs(&format!("{}/sift_base.fvecs", base_path))?;
        let query_vecs = read_fvecs(&format!("{}/sift_query.fvecs", base_path))?;
        let ground_truth = read_ivecs(&format!("{}/sift_groundtruth.ivecs", base_path))?;

        let vectors: Vec<Vector> = base_vecs
            .into_iter()
            .enumerate()
            .map(|(i, data)| Vector::new(i as u64, data))
            .collect();

        let queries: Vec<Vector> = query_vecs
            .into_iter()
            .enumerate()
            .map(|(i, data)| Vector::new(i as u64, data))
            .collect();

        Ok(Self {
            vectors,
            queries,
            ground_truth,
        })
    }
}

/// Compute recall@k between predicted and ground truth results.
///
/// Recall is the fraction of true nearest neighbors that were found.
/// Returns a value between 0.0 and 1.0.
pub fn recall_at_k(predicted: &[u64], ground_truth: &[u64], k: usize) -> f32 {
    let pred_set: HashSet<u64> = predicted.iter().take(k).copied().collect();
    let truth_set: HashSet<u64> = ground_truth.iter().take(k).copied().collect();

    let intersection = pred_set.intersection(&truth_set).count();
    intersection as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_dataset() {
        let dataset = Dataset::generate(1000, 100, 128);
        assert_eq!(dataset.vectors.len(), 1000);
        assert_eq!(dataset.queries.len(), 100);
        assert_eq!(dataset.vectors[0].dim(), 128);
    }

    #[test]
    fn test_recall_perfect() {
        let predicted = vec![1, 2, 3, 4, 5];
        let ground_truth = vec![1, 2, 3, 4, 5];
        assert_eq!(recall_at_k(&predicted, &ground_truth, 5), 1.0);
    }

    #[test]
    fn test_recall_partial() {
        let predicted = vec![1, 2, 6, 7, 8];
        let ground_truth = vec![1, 2, 3, 4, 5];
        assert_eq!(recall_at_k(&predicted, &ground_truth, 5), 0.4);
    }

    #[test]
    fn test_recall_none() {
        let predicted = vec![6, 7, 8, 9, 10];
        let ground_truth = vec![1, 2, 3, 4, 5];
        assert_eq!(recall_at_k(&predicted, &ground_truth, 5), 0.0);
    }
}
