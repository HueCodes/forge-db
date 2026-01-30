use rand::Rng;
use std::sync::Arc;

/// A vector with an ID and floating-point data.
/// The data is stored in an Arc for cheap cloning.
#[derive(Clone, Debug)]
pub struct Vector {
    pub id: u64,
    pub data: Arc<[f32]>,
}

impl Vector {
    /// Create a new vector with the given ID and data.
    pub fn new(id: u64, data: Vec<f32>) -> Self {
        Self {
            id,
            data: data.into(),
        }
    }

    /// Create a random vector with values uniformly distributed in [-1.0, 1.0].
    pub fn random(id: u64, dim: usize) -> Self {
        let mut rng = rand::thread_rng();
        let data: Vec<f32> = (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Self::new(id, data)
    }

    /// Return the dimensionality of this vector.
    pub fn dim(&self) -> usize {
        self.data.len()
    }
}

/// A vector with 32-byte alignment for AVX2 operations.
/// Use this when you need guaranteed alignment for SIMD loads.
#[repr(align(32))]
pub struct AlignedVector {
    pub data: Vec<f32>,
}

impl AlignedVector {
    /// Create a new aligned vector from the given data.
    pub fn new(data: Vec<f32>) -> Self {
        Self { data }
    }

    /// Return the data as a slice.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

// =============================================================================
// Contiguous Vector Storage
// =============================================================================

/// Contiguous storage for many vectors - cache-friendly layout.
///
/// Uses Structure of Arrays (SoA) layout: IDs stored separately from data.
/// All vector data is stored in a single contiguous buffer for optimal
/// cache locality during sequential scans.
///
/// # Memory Layout
///
/// ```text
/// ids:  [id0, id1, id2, ...]
/// data: [v0_d0, v0_d1, ..., v0_dn, v1_d0, v1_d1, ..., v1_dn, ...]
///        |<---- dim floats ---->|  |<---- dim floats ---->|
/// ```
///
/// # Cache Benefits
///
/// - 64-byte alignment matches cache line size on most modern CPUs
/// - Sequential memory access pattern maximizes prefetcher efficiency
/// - No pointer chasing (unlike `Vec<Vector>` with `Arc<[f32]>`)
/// - Better memory bandwidth utilization for bulk operations
///
/// # Example
///
/// ```
/// use forge_db::VectorStore;
///
/// let mut store = VectorStore::with_capacity(128, 1000);
/// store.push(1, &vec![0.1; 128]);
/// store.push(2, &vec![0.2; 128]);
///
/// let (id, data) = store.get(0);
/// assert_eq!(id, 1);
/// assert_eq!(data.len(), 128);
/// ```
#[repr(align(64))] // Cache line aligned for optimal memory access
pub struct VectorStore {
    /// Vector IDs stored contiguously
    pub ids: Vec<u64>,
    /// Vector data stored contiguously: [vec0_dim0, vec0_dim1, ..., vec1_dim0, ...]
    pub data: Vec<f32>,
    /// Dimensionality of each vector
    pub dim: usize,
    /// Number of vectors stored
    pub len: usize,
}

impl VectorStore {
    /// Create a new empty VectorStore for vectors of the given dimension.
    #[inline]
    pub fn new(dim: usize) -> Self {
        Self {
            ids: Vec::new(),
            data: Vec::new(),
            dim,
            len: 0,
        }
    }

    /// Create a new VectorStore with pre-allocated capacity.
    ///
    /// Pre-allocation avoids reallocations during bulk insertion,
    /// which is important for maintaining contiguous memory layout.
    #[inline]
    pub fn with_capacity(dim: usize, capacity: usize) -> Self {
        Self {
            ids: Vec::with_capacity(capacity),
            data: Vec::with_capacity(capacity * dim),
            dim,
            len: 0,
        }
    }

    /// Add a vector to the store.
    ///
    /// # Panics
    /// Panics if `data.len() != self.dim`.
    #[inline]
    pub fn push(&mut self, id: u64, data: &[f32]) {
        debug_assert_eq!(
            data.len(),
            self.dim,
            "Vector dimension {} does not match store dimension {}",
            data.len(),
            self.dim
        );
        self.ids.push(id);
        self.data.extend_from_slice(data);
        self.len += 1;
    }

    /// Get a vector by index, returning (id, data_slice).
    ///
    /// # Panics
    /// Panics if `index >= self.len`.
    #[inline(always)]
    pub fn get(&self, index: usize) -> (u64, &[f32]) {
        debug_assert!(
            index < self.len,
            "Index {} out of bounds (len={})",
            index,
            self.len
        );
        let start = index * self.dim;
        // SAFETY: We maintain the invariant that data.len() == len * dim,
        // and index < len is checked by debug_assert above.
        unsafe {
            let id = *self.ids.get_unchecked(index);
            let data = self.data.get_unchecked(start..start + self.dim);
            (id, data)
        }
    }

    /// Get only the vector data by index (no ID).
    ///
    /// This is the hot path for distance computations - returns a slice
    /// directly into the contiguous buffer with minimal overhead.
    ///
    /// # Panics
    /// Panics if `index >= self.len`.
    #[inline(always)]
    pub fn get_data(&self, index: usize) -> &[f32] {
        debug_assert!(
            index < self.len,
            "Index {} out of bounds (len={})",
            index,
            self.len
        );
        let start = index * self.dim;
        // SAFETY: We maintain the invariant that data.len() == len * dim,
        // and index < len is checked by debug_assert above.
        unsafe { self.data.get_unchecked(start..start + self.dim) }
    }

    /// Get only the vector ID by index.
    ///
    /// # Panics
    /// Panics if `index >= self.len`.
    #[inline(always)]
    pub fn get_id(&self, index: usize) -> u64 {
        debug_assert!(
            index < self.len,
            "Index {} out of bounds (len={})",
            index,
            self.len
        );
        // SAFETY: index < len is checked by debug_assert above.
        unsafe { *self.ids.get_unchecked(index) }
    }

    /// Return the number of vectors in the store.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the store is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return the dimension of vectors in this store.
    #[inline(always)]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Iterate over all vectors as (id, data_slice) pairs.
    #[inline]
    pub fn iter(&self) -> VectorStoreIter<'_> {
        VectorStoreIter {
            store: self,
            index: 0,
        }
    }

    /// Reserve capacity for additional vectors.
    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.ids.reserve(additional);
        self.data.reserve(additional * self.dim);
    }

    /// Clear all vectors from the store, keeping allocated capacity.
    #[inline]
    pub fn clear(&mut self) {
        self.ids.clear();
        self.data.clear();
        self.len = 0;
    }

    /// Shrink the internal buffers to fit the current data.
    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.ids.shrink_to_fit();
        self.data.shrink_to_fit();
    }
}

/// Iterator over vectors in a VectorStore.
pub struct VectorStoreIter<'a> {
    store: &'a VectorStore,
    index: usize,
}

impl<'a> Iterator for VectorStoreIter<'a> {
    type Item = (u64, &'a [f32]);

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.store.len {
            let result = self.store.get(self.index);
            self.index += 1;
            Some(result)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.store.len - self.index;
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for VectorStoreIter<'_> {}

impl From<Vec<Vector>> for VectorStore {
    /// Convert a Vec<Vector> into contiguous VectorStore layout.
    ///
    /// This efficiently copies all vector data into a single contiguous buffer,
    /// eliminating the per-vector Arc allocations.
    fn from(vectors: Vec<Vector>) -> Self {
        if vectors.is_empty() {
            return Self::new(0);
        }

        let dim = vectors[0].dim();
        let mut store = Self::with_capacity(dim, vectors.len());

        for v in &vectors {
            debug_assert_eq!(v.dim(), dim, "All vectors must have the same dimension");
            store.push(v.id, &v.data);
        }

        store
    }
}

impl From<&[Vector]> for VectorStore {
    /// Convert a slice of Vectors into contiguous VectorStore layout.
    fn from(vectors: &[Vector]) -> Self {
        if vectors.is_empty() {
            return Self::new(0);
        }

        let dim = vectors[0].dim();
        let mut store = Self::with_capacity(dim, vectors.len());

        for v in vectors {
            debug_assert_eq!(v.dim(), dim, "All vectors must have the same dimension");
            store.push(v.id, &v.data);
        }

        store
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_store_new() {
        let store = VectorStore::new(128);
        assert_eq!(store.len(), 0);
        assert_eq!(store.dim(), 128);
        assert!(store.is_empty());
    }

    #[test]
    fn test_vector_store_push_and_get() {
        let mut store = VectorStore::new(4);
        let data1 = vec![1.0, 2.0, 3.0, 4.0];
        let data2 = vec![5.0, 6.0, 7.0, 8.0];

        store.push(100, &data1);
        store.push(200, &data2);

        assert_eq!(store.len(), 2);
        assert!(!store.is_empty());

        let (id1, d1) = store.get(0);
        assert_eq!(id1, 100);
        assert_eq!(d1, &data1[..]);

        let (id2, d2) = store.get(1);
        assert_eq!(id2, 200);
        assert_eq!(d2, &data2[..]);
    }

    #[test]
    fn test_vector_store_get_data_and_id() {
        let mut store = VectorStore::new(3);
        store.push(42, &[1.0, 2.0, 3.0]);

        assert_eq!(store.get_id(0), 42);
        assert_eq!(store.get_data(0), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_vector_store_iter() {
        let mut store = VectorStore::new(2);
        store.push(1, &[1.0, 2.0]);
        store.push(2, &[3.0, 4.0]);
        store.push(3, &[5.0, 6.0]);

        let collected: Vec<_> = store.iter().collect();
        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0], (1, &[1.0, 2.0][..]));
        assert_eq!(collected[1], (2, &[3.0, 4.0][..]));
        assert_eq!(collected[2], (3, &[5.0, 6.0][..]));
    }

    #[test]
    fn test_vector_store_from_vec_vector() {
        let vectors = vec![
            Vector::new(1, vec![1.0, 2.0, 3.0]),
            Vector::new(2, vec![4.0, 5.0, 6.0]),
            Vector::new(3, vec![7.0, 8.0, 9.0]),
        ];

        let store = VectorStore::from(vectors);

        assert_eq!(store.len(), 3);
        assert_eq!(store.dim(), 3);

        assert_eq!(store.get_id(0), 1);
        assert_eq!(store.get_data(0), &[1.0, 2.0, 3.0]);

        assert_eq!(store.get_id(2), 3);
        assert_eq!(store.get_data(2), &[7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_vector_store_with_capacity() {
        let store = VectorStore::with_capacity(128, 1000);
        assert_eq!(store.len(), 0);
        assert_eq!(store.dim(), 128);
        assert!(store.ids.capacity() >= 1000);
        assert!(store.data.capacity() >= 128 * 1000);
    }

    #[test]
    fn test_vector_store_clear() {
        let mut store = VectorStore::new(2);
        store.push(1, &[1.0, 2.0]);
        store.push(2, &[3.0, 4.0]);

        assert_eq!(store.len(), 2);
        store.clear();
        assert_eq!(store.len(), 0);
        assert!(store.is_empty());
    }

    #[test]
    fn test_vector_store_from_empty() {
        let vectors: Vec<Vector> = vec![];
        let store = VectorStore::from(vectors);
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_vector_store_alignment() {
        // Verify the struct is 64-byte aligned
        assert_eq!(std::mem::align_of::<VectorStore>(), 64);
    }
}
