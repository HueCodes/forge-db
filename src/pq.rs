//! Product Quantization for vector compression.
//!
//! Product Quantization (PQ) compresses vectors by splitting them into
//! subvectors and quantizing each subspace independently. This achieves
//! 16-32x compression while maintaining reasonable recall.
//!
//! Key concepts:
//! - Split D-dimensional vectors into M subvectors of D/M dimensions
//! - Train K centroids (codebook) for each subspace using k-means
//! - Encode each vector as M bytes (one centroid index per subspace)
//! - Use asymmetric distance: query stays uncompressed, database is compressed

use crate::distance::euclidean_distance_squared;
use crate::kmeans::KMeans;
use crate::vector::Vector;
use rayon::prelude::*;

/// Product Quantizer for vector compression.
///
/// Splits vectors into subvectors and maintains a codebook (set of centroids)
/// for each subspace. Vectors are encoded as indices into these codebooks.
pub struct ProductQuantizer {
    /// M codebooks, each with 256 centroids (one per subspace)
    pub codebooks: Vec<Vec<Vector>>,
    /// Number of subvectors (M)
    pub n_subvectors: usize,
    /// Dimension of each subvector
    pub subvector_dim: usize,
    /// Original vector dimension
    pub dim: usize,
}

impl ProductQuantizer {
    /// Train a Product Quantizer on the given vectors.
    ///
    /// # Arguments
    /// * `vectors` - Training vectors
    /// * `n_subvectors` - Number of subvectors (M). Must divide vector dimension evenly.
    ///
    /// # Panics
    /// Panics if the vector dimension is not divisible by n_subvectors.
    pub fn train(vectors: &[Vector], n_subvectors: usize) -> Self {
        let dim = vectors[0].dim();
        assert!(
            dim % n_subvectors == 0,
            "Vector dimension {} must be divisible by n_subvectors {}",
            dim,
            n_subvectors
        );

        let subvector_dim = dim / n_subvectors;
        println!(
            "Training PQ with {} subvectors of dimension {}",
            n_subvectors, subvector_dim
        );

        // Train codebooks in parallel for each subspace
        let codebooks: Vec<Vec<Vector>> = (0..n_subvectors)
            .into_par_iter()
            .map(|m| {
                // Extract m-th subvector from all training vectors
                let subvectors: Vec<Vector> = vectors
                    .iter()
                    .map(|v| {
                        let start = m * subvector_dim;
                        let end = start + subvector_dim;
                        let sub_data = v.data[start..end].to_vec();
                        Vector::new(v.id, sub_data)
                    })
                    .collect();

                // Run k-means with k=256 centroids (25 iters is enough)
                let mut kmeans = KMeans::new(256, 25);
                kmeans.fit(&subvectors);
                kmeans.centroids
            })
            .collect();

        Self {
            codebooks,
            n_subvectors,
            subvector_dim,
            dim,
        }
    }

    /// Encode a vector into PQ codes.
    ///
    /// Returns M bytes, one per subvector, each being the index of the
    /// nearest centroid in that subspace.
    pub fn encode(&self, vector: &Vector) -> Vec<u8> {
        assert_eq!(
            vector.dim(),
            self.dim,
            "Vector dimension {} does not match PQ dimension {}",
            vector.dim(),
            self.dim
        );

        (0..self.n_subvectors)
            .map(|m| {
                let start = m * self.subvector_dim;
                let end = start + self.subvector_dim;
                let subvector = &vector.data[start..end];

                // Find nearest centroid in this codebook
                self.codebooks[m]
                    .iter()
                    .enumerate()
                    .map(|(idx, centroid)| (idx, euclidean_distance_squared(subvector, &centroid.data)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                    .unwrap()
                    .0 as u8
            })
            .collect()
    }

    /// Build a lookup table for asymmetric distance computation.
    ///
    /// Precomputes the squared distance from each query subvector to each
    /// centroid in each codebook. This allows O(M) distance computation
    /// per compressed vector instead of O(D).
    ///
    /// Returns a 2D table: `table[m][code]` = squared distance from query
    /// subvector m to centroid `code` in codebook m.
    pub fn build_lookup_table(&self, query: &[f32]) -> Vec<Vec<f32>> {
        assert_eq!(
            query.len(),
            self.dim,
            "Query dimension {} does not match PQ dimension {}",
            query.len(),
            self.dim
        );

        (0..self.n_subvectors)
            .map(|m| {
                let start = m * self.subvector_dim;
                let end = start + self.subvector_dim;
                let query_sub = &query[start..end];

                self.codebooks[m]
                    .iter()
                    .map(|centroid| euclidean_distance_squared(query_sub, &centroid.data))
                    .collect()
            })
            .collect()
    }

    /// Build a flat lookup table for better cache locality.
    /// Layout: [subvector0: 256 floats][subvector1: 256 floats]...
    #[inline]
    pub fn build_lookup_table_flat(&self, query: &[f32]) -> Vec<f32> {
        let mut table = Vec::with_capacity(self.n_subvectors * 256);

        for m in 0..self.n_subvectors {
            let start = m * self.subvector_dim;
            let end = start + self.subvector_dim;
            let query_sub = &query[start..end];

            for centroid in &self.codebooks[m] {
                table.push(euclidean_distance_squared(query_sub, &centroid.data));
            }
        }

        table
    }

    /// Fast distance using flat lookup table.
    #[inline(always)]
    pub fn asymmetric_distance_flat(&self, table: &[f32], codes: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for (m, &code) in codes.iter().enumerate() {
            sum += unsafe { *table.get_unchecked(m * 256 + code as usize) };
        }
        sum
    }

    /// Compute asymmetric distance using a precomputed lookup table.
    ///
    /// This is the fast path: instead of computing actual distances,
    /// we just look up precomputed values and sum them.
    #[inline]
    pub fn asymmetric_distance(&self, lookup_table: &[Vec<f32>], codes: &[u8]) -> f32 {
        let sum: f32 = codes
            .iter()
            .enumerate()
            .map(|(m, &code)| lookup_table[m][code as usize])
            .sum();
        sum.sqrt()
    }

    /// Fast asymmetric distance - optimized for latency.
    /// Returns squared distance (skip sqrt for comparison).
    #[inline(always)]
    pub fn asymmetric_distance_fast(&self, lookup_table: &[Vec<f32>], codes: &[u8]) -> f32 {
        // Manual unrolling for common case (avoids iterator overhead)
        let mut sum = 0.0f32;
        let n = codes.len();

        // Process 4 at a time
        let mut i = 0;
        while i + 4 <= n {
            sum += unsafe {
                *lookup_table.get_unchecked(i).get_unchecked(codes.get_unchecked(i).clone() as usize)
                    + *lookup_table.get_unchecked(i + 1).get_unchecked(codes.get_unchecked(i + 1).clone() as usize)
                    + *lookup_table.get_unchecked(i + 2).get_unchecked(codes.get_unchecked(i + 2).clone() as usize)
                    + *lookup_table.get_unchecked(i + 3).get_unchecked(codes.get_unchecked(i + 3).clone() as usize)
            };
            i += 4;
        }

        // Handle remainder
        while i < n {
            sum += unsafe {
                *lookup_table.get_unchecked(i).get_unchecked(codes.get_unchecked(i).clone() as usize)
            };
            i += 1;
        }

        sum // Return squared distance for faster comparison
    }

    /// Compute distance from query to compressed vector.
    ///
    /// This is a convenience method that builds the lookup table and
    /// computes the distance. For batch queries, use `build_lookup_table`
    /// and `asymmetric_distance` separately.
    pub fn compute_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let lookup_table = self.build_lookup_table(query);
        self.asymmetric_distance(&lookup_table, codes)
    }
}

/// Compressed vector storage using Product Quantization.
///
/// Stores a collection of vectors in compressed form for memory-efficient
/// approximate nearest neighbor search.
pub struct CompressedVectors {
    /// The trained Product Quantizer
    pub pq: ProductQuantizer,
    /// Compressed codes for each vector (M bytes per vector)
    pub codes: Vec<Vec<u8>>,
    /// Original vector IDs
    pub ids: Vec<u64>,
}

impl CompressedVectors {
    /// Create a new compressed vector store.
    ///
    /// Trains a Product Quantizer on the vectors and encodes them all.
    ///
    /// # Arguments
    /// * `vectors` - Vectors to compress
    /// * `n_subvectors` - Number of PQ subvectors (typically 8 for 128-dim vectors)
    pub fn new(vectors: Vec<Vector>, n_subvectors: usize) -> Self {
        // Train the PQ codebooks
        let pq = ProductQuantizer::train(&vectors, n_subvectors);

        // Encode all vectors in parallel
        println!("Encoding {} vectors", vectors.len());
        let codes: Vec<Vec<u8>> = vectors.par_iter().map(|v| pq.encode(v)).collect();

        // Extract IDs
        let ids: Vec<u64> = vectors.iter().map(|v| v.id).collect();

        // Calculate and print compression ratio
        let original_bytes = vectors.len() * vectors[0].dim() * 4; // f32 = 4 bytes
        let compressed_bytes = codes.len() * n_subvectors;
        let ratio = original_bytes as f32 / compressed_bytes as f32;

        println!(
            "Compression: {} MB -> {} MB ({:.1}x)",
            original_bytes / (1024 * 1024),
            compressed_bytes / (1024 * 1024),
            ratio
        );

        Self { pq, codes, ids }
    }

    /// Search for k nearest neighbors to the query.
    ///
    /// Uses asymmetric distance computation with a precomputed lookup table
    /// for efficient approximate search.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        // Build lookup table once for this query
        let lookup_table = self.pq.build_lookup_table(query);

        // Compute distances in parallel
        let mut results: Vec<(u64, f32)> = self
            .codes
            .par_iter()
            .zip(self.ids.par_iter())
            .map(|(codes, &id)| {
                let dist = self.pq.asymmetric_distance(&lookup_table, codes);
                (id, dist)
            })
            .collect();

        // Sort by distance and take top k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }

    /// Return the number of compressed vectors.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Check if the store is empty.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pq_encode_decode() {
        // Create simple test vectors
        let vectors: Vec<Vector> = (0..100)
            .map(|i| Vector::random(i, 16))
            .collect();

        let pq = ProductQuantizer::train(&vectors, 4);

        // Each vector should encode to 4 bytes
        let codes = pq.encode(&vectors[0]);
        assert_eq!(codes.len(), 4);

        // Codes should be valid u8 values (implicitly [0, 255])
        assert!(!codes.is_empty());
    }

    #[test]
    fn test_compressed_vectors_search() {
        // Create test vectors
        let vectors: Vec<Vector> = (0..100)
            .map(|i| Vector::random(i, 32))
            .collect();

        let compressed = CompressedVectors::new(vectors.clone(), 4);

        // Search should return k results
        let query = vectors[0].data.to_vec();
        let results = compressed.search(&query, 10);

        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    }

    #[test]
    fn test_lookup_table() {
        let vectors: Vec<Vector> = (0..50)
            .map(|i| Vector::random(i, 16))
            .collect();

        let pq = ProductQuantizer::train(&vectors, 4);

        let query = vectors[0].data.to_vec();
        let table = pq.build_lookup_table(&query);

        // Should have one table per subvector
        assert_eq!(table.len(), 4);

        // Each table should have 256 entries (one per centroid)
        for subtable in &table {
            assert_eq!(subtable.len(), 256);
        }
    }
}
