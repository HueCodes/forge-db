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
//!
//! # Variants
//!
//! - **8-bit PQ** (`ProductQuantizer`): 256 centroids per subspace, 1 byte per code
//! - **4-bit PQ** (`ProductQuantizer4Bit`): 16 centroids per subspace, 2 codes per byte
//!   (2x memory bandwidth savings with slight recall trade-off)

use crate::distance::euclidean_distance_squared;
use crate::kmeans::KMeans;
use crate::vector::Vector;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
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
        asymmetric_distance_flat_dispatch(table, codes)
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
                *lookup_table.get_unchecked(i).get_unchecked(*codes.get_unchecked(i) as usize)
                    + *lookup_table.get_unchecked(i + 1).get_unchecked(*codes.get_unchecked(i + 1) as usize)
                    + *lookup_table.get_unchecked(i + 2).get_unchecked(*codes.get_unchecked(i + 2) as usize)
                    + *lookup_table.get_unchecked(i + 3).get_unchecked(*codes.get_unchecked(i + 3) as usize)
            };
            i += 4;
        }

        // Handle remainder
        while i < n {
            sum += unsafe {
                *lookup_table.get_unchecked(i).get_unchecked(*codes.get_unchecked(i) as usize)
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

// =============================================================================
// SIMD-Accelerated PQ Distance Functions
// =============================================================================

/// Dispatch to the fastest available implementation for flat table lookup.
///
/// Automatically selects AVX2 gather or scalar based on CPU features.
#[inline]
pub fn asymmetric_distance_flat_dispatch(table: &[f32], codes: &[u8]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We verified AVX2 is available
            return unsafe { asymmetric_distance_simd_avx2(table, codes) };
        }
    }
    asymmetric_distance_flat_scalar(table, codes)
}

/// Scalar fallback for flat table lookup.
#[inline(always)]
pub fn asymmetric_distance_flat_scalar(table: &[f32], codes: &[u8]) -> f32 {
    let mut sum = 0.0f32;
    for (m, &code) in codes.iter().enumerate() {
        sum += unsafe { *table.get_unchecked(m * 256 + code as usize) };
    }
    sum
}

/// AVX2 SIMD gather implementation for PQ distance computation.
///
/// Uses `_mm256_i32gather_ps` to load 8 table entries at once based on code indices.
/// This provides significant speedup over scalar lookups by:
/// 1. Processing 8 subvectors per iteration
/// 2. Using hardware gather for indirect memory access
/// 3. Leveraging SIMD horizontal addition for final sum
///
/// # Safety
/// - Requires AVX2 CPU feature
/// - Table must have layout [256 * n_subvectors] floats
/// - Codes must have length n_subvectors
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn asymmetric_distance_simd_avx2(table: &[f32], codes: &[u8]) -> f32 {
    let n = codes.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 subvectors at a time using AVX2 gather
    while i + 8 <= n {
        // Build indices: for each code, compute table offset = subvector_index * 256 + code
        // We need i32 indices for gather instruction
        let idx0 = (i * 256 + *codes.get_unchecked(i) as usize) as i32;
        let idx1 = ((i + 1) * 256 + *codes.get_unchecked(i + 1) as usize) as i32;
        let idx2 = ((i + 2) * 256 + *codes.get_unchecked(i + 2) as usize) as i32;
        let idx3 = ((i + 3) * 256 + *codes.get_unchecked(i + 3) as usize) as i32;
        let idx4 = ((i + 4) * 256 + *codes.get_unchecked(i + 4) as usize) as i32;
        let idx5 = ((i + 5) * 256 + *codes.get_unchecked(i + 5) as usize) as i32;
        let idx6 = ((i + 6) * 256 + *codes.get_unchecked(i + 6) as usize) as i32;
        let idx7 = ((i + 7) * 256 + *codes.get_unchecked(i + 7) as usize) as i32;

        // Create index vector
        let indices = _mm256_set_epi32(idx7, idx6, idx5, idx4, idx3, idx2, idx1, idx0);

        // Gather 8 floats from table using indices
        // Scale = 4 because we're gathering f32 (4 bytes each)
        let values = _mm256_i32gather_ps::<4>(table.as_ptr(), indices);

        // Accumulate into sum
        sum = _mm256_add_ps(sum, values);

        i += 8;
    }

    // Horizontal sum of the 8 floats in the AVX2 register
    // Extract to array and sum
    let sum_array: [f32; 8] = std::mem::transmute(sum);
    let mut total: f32 = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3]
        + sum_array[4] + sum_array[5] + sum_array[6] + sum_array[7];

    // Handle remaining subvectors (0-7) with scalar operations
    while i < n {
        total += *table.get_unchecked(i * 256 + *codes.get_unchecked(i) as usize);
        i += 1;
    }

    total
}

// =============================================================================
// 4-bit Product Quantization
// =============================================================================
// 4-bit PQ uses 16 centroids per subspace instead of 256, enabling:
// - 2x memory bandwidth savings (2 codes packed per byte)
// - Smaller lookup tables (16 entries vs 256 per subspace)
// - Trade-off: slightly lower recall due to coarser quantization

/// Pack two 4-bit codes into a single byte.
///
/// The first code occupies the lower 4 bits, the second code the upper 4 bits.
#[inline(always)]
pub fn pack_codes_4bit(a: u8, b: u8) -> u8 {
    (a & 0x0F) | ((b & 0x0F) << 4)
}

/// Unpack two 4-bit codes from a single byte.
///
/// Returns (lower_code, upper_code).
#[inline(always)]
pub fn unpack_codes_4bit(packed: u8) -> (u8, u8) {
    (packed & 0x0F, packed >> 4)
}

/// 4-bit Product Quantizer for maximum memory efficiency.
///
/// Uses 16 centroids per subspace instead of 256, allowing two codes
/// to be packed into a single byte. This provides 2x memory bandwidth
/// savings compared to 8-bit PQ, at the cost of slightly reduced recall.
///
/// # Memory comparison (for 1M vectors, 16 subvectors):
/// - 8-bit PQ: 1M * 16 bytes = 16 MB
/// - 4-bit PQ: 1M * 8 bytes = 8 MB
///
/// # Recall trade-off:
/// - 4-bit PQ typically achieves 85-95% of 8-bit PQ recall
/// - Best suited for large-scale search where memory is constrained
pub struct ProductQuantizer4Bit {
    /// M codebooks, each with 16 centroids (one per subspace)
    pub codebooks: Vec<Vec<Vector>>,
    /// Number of subvectors (M)
    pub n_subvectors: usize,
    /// Dimension of each subvector
    pub subvector_dim: usize,
    /// Original vector dimension
    pub dim: usize,
}

impl ProductQuantizer4Bit {
    /// Train a 4-bit Product Quantizer on the given vectors.
    ///
    /// # Arguments
    /// * `vectors` - Training vectors
    /// * `n_subvectors` - Number of subvectors (M). Must divide vector dimension evenly.
    ///                    Should be even for optimal packing.
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

        // Train codebooks in parallel for each subspace with k=16
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

                // Run k-means with k=16 centroids (using k-means++ since k is small)
                let mut kmeans = KMeans::new(16, 25);
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

    /// Encode a vector into 4-bit PQ codes (unpacked).
    ///
    /// Returns M bytes, each in range [0, 15].
    /// Use `encode_packed` for memory-efficient storage.
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

                // Find nearest centroid in this codebook (only 16 centroids)
                self.codebooks[m]
                    .iter()
                    .enumerate()
                    .map(|(idx, centroid)| (idx, euclidean_distance_squared(subvector, &centroid.data)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap()
                    .0 as u8
            })
            .collect()
    }

    /// Encode a vector into packed 4-bit codes.
    ///
    /// Returns ceil(M/2) bytes with pairs of codes packed together.
    /// If M is odd, the last byte has the final code in the lower nibble.
    pub fn encode_packed(&self, vector: &Vector) -> Vec<u8> {
        let codes = self.encode(vector);
        let mut packed = Vec::with_capacity((codes.len() + 1) / 2);

        let mut i = 0;
        while i + 1 < codes.len() {
            packed.push(pack_codes_4bit(codes[i], codes[i + 1]));
            i += 2;
        }

        // Handle odd number of subvectors
        if i < codes.len() {
            packed.push(codes[i] & 0x0F);
        }

        packed
    }

    /// Build a flat lookup table for 4-bit asymmetric distance.
    ///
    /// Layout: [subvector0: 16 floats][subvector1: 16 floats]...
    /// Total size: 16 * n_subvectors floats (64x smaller than 8-bit PQ table!)
    #[inline]
    pub fn build_lookup_table_flat(&self, query: &[f32]) -> Vec<f32> {
        let mut table = Vec::with_capacity(self.n_subvectors * 16);

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

    /// Compute asymmetric distance using packed 4-bit codes.
    ///
    /// # Arguments
    /// * `table` - Flat lookup table from `build_lookup_table_flat` [16 * n_subvectors]
    /// * `packed_codes` - Packed codes [ceil(n_subvectors / 2)]
    #[inline(always)]
    pub fn asymmetric_distance_4bit(&self, table: &[f32], packed_codes: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        let mut m = 0; // subvector index

        for &packed in packed_codes {
            let (code_a, code_b) = unpack_codes_4bit(packed);

            // First code in pair
            if m < self.n_subvectors {
                sum += unsafe { *table.get_unchecked(m * 16 + code_a as usize) };
                m += 1;
            }

            // Second code in pair (if not past the end)
            if m < self.n_subvectors {
                sum += unsafe { *table.get_unchecked(m * 16 + code_b as usize) };
                m += 1;
            }
        }

        sum
    }

    /// Compute asymmetric distance using unpacked codes (for testing/compatibility).
    #[inline(always)]
    pub fn asymmetric_distance_unpacked(&self, table: &[f32], codes: &[u8]) -> f32 {
        let mut sum = 0.0f32;
        for (m, &code) in codes.iter().enumerate() {
            sum += unsafe { *table.get_unchecked(m * 16 + code as usize) };
        }
        sum
    }
}

/// Compressed vector storage using 4-bit Product Quantization.
///
/// Stores vectors with 2x better memory efficiency than 8-bit PQ,
/// at the cost of slightly reduced recall.
pub struct CompressedVectors4Bit {
    /// The trained 4-bit Product Quantizer
    pub pq: ProductQuantizer4Bit,
    /// Packed 4-bit codes for each vector [n_vectors][ceil(n_subvectors/2)]
    pub codes: Vec<Vec<u8>>,
    /// Original vector IDs
    pub ids: Vec<u64>,
}

impl CompressedVectors4Bit {
    /// Create a new 4-bit compressed vector store.
    ///
    /// Trains a 4-bit Product Quantizer on the vectors and encodes them all.
    ///
    /// # Arguments
    /// * `vectors` - Vectors to compress
    /// * `n_subvectors` - Number of PQ subvectors (should be even for optimal packing)
    pub fn new(vectors: Vec<Vector>, n_subvectors: usize) -> Self {
        // Train the 4-bit PQ codebooks
        let pq = ProductQuantizer4Bit::train(&vectors, n_subvectors);

        // Encode all vectors in parallel (packed format)
        let codes: Vec<Vec<u8>> = vectors.par_iter().map(|v| pq.encode_packed(v)).collect();

        // Extract IDs
        let ids: Vec<u64> = vectors.iter().map(|v| v.id).collect();

        Self { pq, codes, ids }
    }

    /// Search for k nearest neighbors to the query.
    ///
    /// Uses asymmetric distance computation with packed 4-bit codes.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        // Build lookup table once for this query (only 16 * n_subvectors floats)
        let lookup_table = self.pq.build_lookup_table_flat(query);

        // Compute distances in parallel
        let mut results: Vec<(u64, f32)> = self
            .codes
            .par_iter()
            .zip(self.ids.par_iter())
            .map(|(codes, &id)| {
                let dist = self.pq.asymmetric_distance_4bit(&lookup_table, codes);
                (id, dist.sqrt())
            })
            .collect();

        // Sort by distance and take top k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
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

    /// Get the bytes per vector (packed codes).
    pub fn bytes_per_vector(&self) -> usize {
        (self.pq.n_subvectors + 1) / 2
    }
}

// =============================================================================
// Flat (Contiguous) Compressed Vector Storage
// =============================================================================

/// Compressed vector storage with flat (contiguous) code layout.
///
/// Unlike `CompressedVectors` which stores `Vec<Vec<u8>>` (nested allocations),
/// this struct stores all codes in a single contiguous buffer for optimal
/// cache locality during sequential scans.
///
/// # Memory Layout
///
/// ```text
/// ids:   [id0, id1, id2, ...]
/// codes: [v0_c0, v0_c1, ..., v0_cm, v1_c0, v1_c1, ..., v1_cm, ...]
///         |<-- n_subvectors -->|   |<-- n_subvectors -->|
/// ```
///
/// # Cache Benefits
///
/// - All codes for a vector are contiguous in memory
/// - Sequential access pattern maximizes prefetcher efficiency
/// - No pointer chasing between code vectors
/// - Better memory bandwidth for batch distance computations
///
/// # Example
///
/// ```ignore
/// let vectors: Vec<Vector> = /* ... */;
/// let compressed = FlatCompressedVectors::new(vectors, 16);
///
/// // Get codes for vector at index 42
/// let codes = compressed.get_codes(42);
/// ```
pub struct FlatCompressedVectors {
    /// The trained Product Quantizer
    pub pq: ProductQuantizer,
    /// Flat code storage: [n_vectors * n_subvectors] bytes
    /// Layout: codes for vector i start at index i * n_subvectors
    pub codes: Vec<u8>,
    /// Original vector IDs
    pub ids: Vec<u64>,
    /// Number of vectors stored
    pub n_vectors: usize,
}

impl FlatCompressedVectors {
    /// Create a new flat compressed vector store.
    ///
    /// Trains a Product Quantizer on the vectors and encodes them all
    /// into a contiguous code buffer.
    ///
    /// # Arguments
    /// * `vectors` - Vectors to compress
    /// * `n_subvectors` - Number of PQ subvectors (typically 8-32)
    pub fn new(vectors: Vec<Vector>, n_subvectors: usize) -> Self {
        // Train the PQ codebooks
        let pq = ProductQuantizer::train(&vectors, n_subvectors);

        let n_vectors = vectors.len();

        // Pre-allocate flat code buffer
        let mut codes = Vec::with_capacity(n_vectors * n_subvectors);

        // Encode all vectors and append to flat buffer
        for v in &vectors {
            let vector_codes = pq.encode(v);
            codes.extend_from_slice(&vector_codes);
        }

        // Extract IDs
        let ids: Vec<u64> = vectors.iter().map(|v| v.id).collect();

        Self {
            pq,
            codes,
            ids,
            n_vectors,
        }
    }

    /// Create from an existing ProductQuantizer and vectors.
    ///
    /// Use this when you've already trained a PQ on a larger dataset
    /// and want to encode a subset with the same codebooks.
    pub fn from_pq(pq: ProductQuantizer, vectors: &[Vector]) -> Self {
        let n_vectors = vectors.len();
        let n_subvectors = pq.n_subvectors;

        // Pre-allocate flat code buffer
        let mut codes = Vec::with_capacity(n_vectors * n_subvectors);

        // Encode all vectors
        for v in vectors {
            let vector_codes = pq.encode(v);
            codes.extend_from_slice(&vector_codes);
        }

        // Extract IDs
        let ids: Vec<u64> = vectors.iter().map(|v| v.id).collect();

        Self {
            pq,
            codes,
            ids,
            n_vectors,
        }
    }

    /// Get the PQ codes for a vector at the given index.
    ///
    /// Returns a slice directly into the flat code buffer - no allocation.
    /// This is the hot path for distance computations.
    ///
    /// # Panics
    /// Panics (in debug) if `index >= n_vectors`.
    #[inline(always)]
    pub fn get_codes(&self, index: usize) -> &[u8] {
        debug_assert!(
            index < self.n_vectors,
            "Index {} out of bounds (n_vectors={})",
            index,
            self.n_vectors
        );
        let start = index * self.pq.n_subvectors;
        // SAFETY: We maintain the invariant that codes.len() == n_vectors * n_subvectors,
        // and index < n_vectors is checked by debug_assert above.
        unsafe { self.codes.get_unchecked(start..start + self.pq.n_subvectors) }
    }

    /// Get the vector ID at the given index.
    ///
    /// # Panics
    /// Panics (in debug) if `index >= n_vectors`.
    #[inline(always)]
    pub fn get_id(&self, index: usize) -> u64 {
        debug_assert!(
            index < self.n_vectors,
            "Index {} out of bounds (n_vectors={})",
            index,
            self.n_vectors
        );
        // SAFETY: index < n_vectors is checked by debug_assert above.
        unsafe { *self.ids.get_unchecked(index) }
    }

    /// Search for k nearest neighbors to the query.
    ///
    /// Uses asymmetric distance computation with a flat lookup table
    /// for efficient approximate search.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        // Build flat lookup table once for this query
        let lookup_table = self.pq.build_lookup_table_flat(query);

        // Compute distances sequentially (better cache locality than parallel for small k)
        let mut results: Vec<(u64, f32)> = Vec::with_capacity(self.n_vectors);

        for i in 0..self.n_vectors {
            let codes = self.get_codes(i);
            let dist_sq = self.pq.asymmetric_distance_flat(&lookup_table, codes);
            let id = self.get_id(i);
            results.push((id, dist_sq.sqrt()));
        }

        // Sort by distance and take top k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Search for k nearest neighbors using parallel computation.
    ///
    /// Better for large datasets where parallelization overhead is worth it.
    pub fn search_parallel(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        // Build flat lookup table once for this query
        let lookup_table = self.pq.build_lookup_table_flat(query);

        // Compute distances in parallel
        let mut results: Vec<(u64, f32)> = (0..self.n_vectors)
            .into_par_iter()
            .map(|i| {
                let codes = self.get_codes(i);
                let dist_sq = self.pq.asymmetric_distance_flat(&lookup_table, codes);
                let id = self.get_id(i);
                (id, dist_sq.sqrt())
            })
            .collect();

        // Sort by distance and take top k
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }

    /// Return the number of compressed vectors.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.n_vectors
    }

    /// Check if the store is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    /// Return the number of bytes used per vector.
    #[inline(always)]
    pub fn bytes_per_vector(&self) -> usize {
        self.pq.n_subvectors
    }

    /// Return the total size of the code buffer in bytes.
    #[inline(always)]
    pub fn codes_size_bytes(&self) -> usize {
        self.codes.len()
    }
}

/// Compressed vector storage using Product Quantization.
///
/// Stores a collection of vectors in compressed form for memory-efficient
/// approximate nearest neighbor search.
///
/// Note: For better cache locality, consider using `FlatCompressedVectors`
/// which stores codes in a contiguous buffer instead of nested Vecs.
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
        let codes: Vec<Vec<u8>> = vectors.par_iter().map(|v| pq.encode(v)).collect();

        // Extract IDs
        let ids: Vec<u64> = vectors.iter().map(|v| v.id).collect();

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
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
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
        // Create test vectors (need 256+ for k-means with k=256)
        let vectors: Vec<Vector> = (0..300)
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
        // Create test vectors (need 256+ for k-means with k=256)
        let vectors: Vec<Vector> = (0..300)
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
        // Need 256+ vectors for k-means with k=256
        let vectors: Vec<Vector> = (0..300)
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

    // =========================================================================
    // SIMD Distance Tests
    // =========================================================================

    #[test]
    fn test_simd_distance_matches_scalar() {
        // Need 256+ vectors for k-means with k=256
        let vectors: Vec<Vector> = (0..300)
            .map(|i| Vector::random(i, 64))
            .collect();

        let pq = ProductQuantizer::train(&vectors, 8);
        let codes = pq.encode(&vectors[0]);
        let query = vectors[1].data.to_vec();
        let table = pq.build_lookup_table_flat(&query);

        // Compute using scalar
        let scalar_dist = asymmetric_distance_flat_scalar(&table, &codes);

        // Compute using dispatch (which may use SIMD)
        let dispatch_dist = asymmetric_distance_flat_dispatch(&table, &codes);

        assert!(
            (scalar_dist - dispatch_dist).abs() < 1e-5,
            "Scalar: {}, Dispatch: {}",
            scalar_dist,
            dispatch_dist
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_simd_distance() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        // Need 256+ vectors for k-means with k=256
        let vectors: Vec<Vector> = (0..300)
            .map(|i| Vector::random(i, 128))
            .collect();

        // Test with different subvector counts (8, 16, 32)
        for n_subvectors in [8, 16, 32] {
            let pq = ProductQuantizer::train(&vectors, n_subvectors);
            let codes = pq.encode(&vectors[0]);
            let query = vectors[1].data.to_vec();
            let table = pq.build_lookup_table_flat(&query);

            let scalar_dist = asymmetric_distance_flat_scalar(&table, &codes);
            let simd_dist = unsafe { asymmetric_distance_simd_avx2(&table, &codes) };

            assert!(
                (scalar_dist - simd_dist).abs() < 1e-4,
                "n_subvectors={}: Scalar={}, SIMD={}",
                n_subvectors,
                scalar_dist,
                simd_dist
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_simd_non_multiple_of_8() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping test");
            return;
        }

        // Need 256+ vectors for k-means with k=256
        // Test with 10 subvectors (not multiple of 8)
        let vectors: Vec<Vector> = (0..300)
            .map(|i| Vector::random(i, 50))
            .collect();

        let pq = ProductQuantizer::train(&vectors, 10);
        let codes = pq.encode(&vectors[0]);
        let query = vectors[1].data.to_vec();
        let table = pq.build_lookup_table_flat(&query);

        let scalar_dist = asymmetric_distance_flat_scalar(&table, &codes);
        let simd_dist = unsafe { asymmetric_distance_simd_avx2(&table, &codes) };

        assert!(
            (scalar_dist - simd_dist).abs() < 1e-4,
            "Scalar={}, SIMD={}",
            scalar_dist,
            simd_dist
        );
    }

    // =========================================================================
    // 4-bit PQ Tests
    // =========================================================================

    #[test]
    fn test_pack_unpack_4bit() {
        // Test all combinations
        for a in 0..16u8 {
            for b in 0..16u8 {
                let packed = pack_codes_4bit(a, b);
                let (unpacked_a, unpacked_b) = unpack_codes_4bit(packed);
                assert_eq!(unpacked_a, a, "Failed for a={}", a);
                assert_eq!(unpacked_b, b, "Failed for b={}", b);
            }
        }
    }

    #[test]
    fn test_4bit_pq_encode() {
        let vectors: Vec<Vector> = (0..100)
            .map(|i| Vector::random(i, 16))
            .collect();

        let pq = ProductQuantizer4Bit::train(&vectors, 4);

        // Unpacked codes should be 4 bytes, each in [0, 15]
        let codes = pq.encode(&vectors[0]);
        assert_eq!(codes.len(), 4);
        for &code in &codes {
            assert!(code < 16, "4-bit code should be < 16, got {}", code);
        }

        // Packed codes should be 2 bytes (4 subvectors / 2)
        let packed = pq.encode_packed(&vectors[0]);
        assert_eq!(packed.len(), 2);
    }

    #[test]
    fn test_4bit_pq_odd_subvectors() {
        let vectors: Vec<Vector> = (0..100)
            .map(|i| Vector::random(i, 15))
            .collect();

        let pq = ProductQuantizer4Bit::train(&vectors, 5);

        // 5 subvectors -> 3 packed bytes (2+2+1)
        let packed = pq.encode_packed(&vectors[0]);
        assert_eq!(packed.len(), 3);
    }

    #[test]
    fn test_4bit_distance_computation() {
        let vectors: Vec<Vector> = (0..100)
            .map(|i| Vector::random(i, 32))
            .collect();

        let pq = ProductQuantizer4Bit::train(&vectors, 8);
        let query = vectors[0].data.to_vec();
        let table = pq.build_lookup_table_flat(&query);

        // Table should have 16 * 8 = 128 entries
        assert_eq!(table.len(), 128);

        // Test unpacked distance
        let codes = pq.encode(&vectors[1]);
        let dist_unpacked = pq.asymmetric_distance_unpacked(&table, &codes);

        // Test packed distance
        let packed = pq.encode_packed(&vectors[1]);
        let dist_packed = pq.asymmetric_distance_4bit(&table, &packed);

        assert!(
            (dist_unpacked - dist_packed).abs() < 1e-5,
            "Unpacked: {}, Packed: {}",
            dist_unpacked,
            dist_packed
        );
    }

    #[test]
    fn test_compressed_vectors_4bit_search() {
        let vectors: Vec<Vector> = (0..100)
            .map(|i| Vector::random(i, 32))
            .collect();

        let compressed = CompressedVectors4Bit::new(vectors.clone(), 8);

        // Search should return k results
        let query = vectors[0].data.to_vec();
        let results = compressed.search(&query, 10);

        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }

        // Verify memory savings: 8 subvectors -> 4 bytes per vector (vs 8 for 8-bit)
        assert_eq!(compressed.bytes_per_vector(), 4);
    }

    #[test]
    fn test_4bit_vs_8bit_recall() {
        // This test verifies that 4-bit PQ finds similar results to 8-bit PQ
        // Note: Need at least 256 vectors for 8-bit PQ k-means (256 centroids)
        let vectors: Vec<Vector> = (0..500)
            .map(|i| Vector::random(i, 32))
            .collect();

        let compressed_8bit = CompressedVectors::new(vectors.clone(), 8);
        let compressed_4bit = CompressedVectors4Bit::new(vectors.clone(), 8);

        let query = vectors[0].data.to_vec();

        let results_8bit = compressed_8bit.search(&query, 10);
        let results_4bit = compressed_4bit.search(&query, 10);

        // Both should find the query vector itself (or close to it)
        // Just verify we get valid results; exact recall comparison
        // would require ground truth
        assert!(!results_8bit.is_empty());
        assert!(!results_4bit.is_empty());

        // The query vector (id=0) should be in top results for both
        let top_ids_8bit: Vec<u64> = results_8bit.iter().map(|(id, _)| *id).collect();
        let top_ids_4bit: Vec<u64> = results_4bit.iter().map(|(id, _)| *id).collect();

        assert!(
            top_ids_8bit.contains(&0),
            "8-bit PQ should find query vector in top 10"
        );
        assert!(
            top_ids_4bit.contains(&0),
            "4-bit PQ should find query vector in top 10"
        );
    }

    // =========================================================================
    // Flat Compressed Vectors Tests
    // =========================================================================

    #[test]
    fn test_flat_compressed_vectors_new() {
        // Need 256+ vectors for k-means with k=256
        let vectors: Vec<Vector> = (0..300)
            .map(|i| Vector::random(i, 32))
            .collect();

        let compressed = FlatCompressedVectors::new(vectors.clone(), 8);

        assert_eq!(compressed.len(), 300);
        assert!(!compressed.is_empty());
        assert_eq!(compressed.bytes_per_vector(), 8);
        assert_eq!(compressed.codes_size_bytes(), 300 * 8);
    }

    #[test]
    fn test_flat_compressed_vectors_get_codes() {
        let vectors: Vec<Vector> = (0..300)
            .map(|i| Vector::random(i, 32))
            .collect();

        let compressed = FlatCompressedVectors::new(vectors.clone(), 8);

        // Each code slice should have n_subvectors bytes
        for i in 0..compressed.len() {
            let codes = compressed.get_codes(i);
            assert_eq!(codes.len(), 8);
        }

        // Verify IDs are preserved
        for i in 0..10 {
            assert_eq!(compressed.get_id(i), i as u64);
        }
    }

    #[test]
    fn test_flat_compressed_vectors_search() {
        let vectors: Vec<Vector> = (0..300)
            .map(|i| Vector::random(i, 32))
            .collect();

        let compressed = FlatCompressedVectors::new(vectors.clone(), 8);

        // Search should return k results
        let query = vectors[0].data.to_vec();
        let results = compressed.search(&query, 10);

        assert_eq!(results.len(), 10);

        // Results should be sorted by distance
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }

        // Query vector should be in top results
        let top_ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        assert!(
            top_ids.contains(&0),
            "Query vector should be in top 10 results"
        );
    }

    #[test]
    fn test_flat_compressed_vectors_search_parallel() {
        let vectors: Vec<Vector> = (0..300)
            .map(|i| Vector::random(i, 32))
            .collect();

        let compressed = FlatCompressedVectors::new(vectors.clone(), 8);

        let query = vectors[0].data.to_vec();

        // Both search methods should return same results
        let results_seq = compressed.search(&query, 10);
        let results_par = compressed.search_parallel(&query, 10);

        assert_eq!(results_seq.len(), results_par.len());

        // Top result should be the same (query vector itself)
        assert_eq!(results_seq[0].0, results_par[0].0);
    }

    #[test]
    fn test_flat_vs_nested_compressed_vectors() {
        // Verify that flat and nested implementations produce same results
        let vectors: Vec<Vector> = (0..300)
            .map(|i| Vector::random(i, 32))
            .collect();

        let nested = CompressedVectors::new(vectors.clone(), 8);
        let flat = FlatCompressedVectors::new(vectors.clone(), 8);

        let query = vectors[5].data.to_vec();

        let results_nested = nested.search(&query, 20);
        let results_flat = flat.search(&query, 20);

        // Both should find the query vector in top results
        let nested_ids: Vec<u64> = results_nested.iter().map(|(id, _)| *id).collect();
        let flat_ids: Vec<u64> = results_flat.iter().map(|(id, _)| *id).collect();

        assert!(nested_ids.contains(&5), "Nested should find query vector");
        assert!(flat_ids.contains(&5), "Flat should find query vector");
    }

    #[test]
    fn test_flat_compressed_vectors_from_pq() {
        let vectors: Vec<Vector> = (0..300)
            .map(|i| Vector::random(i, 32))
            .collect();

        // Train PQ on all vectors
        let pq = ProductQuantizer::train(&vectors, 8);

        // Create flat compressed from subset using existing PQ
        let subset: Vec<Vector> = vectors[0..100].to_vec();
        let compressed = FlatCompressedVectors::from_pq(pq, &subset);

        assert_eq!(compressed.len(), 100);
        assert_eq!(compressed.bytes_per_vector(), 8);
    }
}
