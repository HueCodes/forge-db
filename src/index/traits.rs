//! Common traits for vector index implementations.
//!
//! These traits provide a unified interface for different index types,
//! enabling generic code that works with any index implementation.

use crate::error::Result;
use crate::types::VectorId;

/// A search result containing a vector ID and its distance to the query.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SearchResult {
    /// The ID of the matched vector.
    pub id: VectorId,
    /// The distance from the query vector.
    pub distance: f32,
}

impl SearchResult {
    /// Create a new SearchResult.
    #[inline]
    pub fn new(id: impl Into<VectorId>, distance: f32) -> Self {
        Self {
            id: id.into(),
            distance,
        }
    }

    /// Create a SearchResult from a raw (u64, f32) tuple.
    #[inline]
    pub fn from_tuple(tuple: (u64, f32)) -> Self {
        Self {
            id: VectorId(tuple.0),
            distance: tuple.1,
        }
    }

    /// Convert to a raw (u64, f32) tuple.
    #[inline]
    pub fn to_tuple(self) -> (u64, f32) {
        (self.id.0, self.distance)
    }
}

impl From<(u64, f32)> for SearchResult {
    fn from(tuple: (u64, f32)) -> Self {
        Self::from_tuple(tuple)
    }
}

impl From<SearchResult> for (u64, f32) {
    fn from(result: SearchResult) -> Self {
        result.to_tuple()
    }
}

/// Common interface for vector indices that support search operations.
///
/// All index types implement this trait, allowing generic code to work
/// with any index implementation.
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync` to support concurrent access.
/// Search operations should be safe to call from multiple threads.
pub trait VectorIndex: Send + Sync {
    /// Search for the k nearest neighbors to the query vector.
    ///
    /// Returns results sorted by distance (closest first).
    fn search(&self, query: &[f32], k: usize) -> Vec<SearchResult>;

    /// Return the number of vectors in the index.
    fn len(&self) -> usize;

    /// Return true if the index contains no vectors.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the dimensionality of vectors in this index.
    fn dimension(&self) -> usize;
}

/// Extension trait for indices that support dynamic insertion and deletion.
///
/// Not all indices support mutation after construction. For example,
/// IVF-PQ indices are typically built once from a batch of vectors.
pub trait MutableVectorIndex: VectorIndex {
    /// Add a vector to the index.
    ///
    /// # Errors
    ///
    /// Returns an error if the vector dimension doesn't match the index,
    /// or if the index has reached its capacity.
    fn add(&mut self, id: VectorId, data: &[f32]) -> Result<()>;

    /// Remove a vector from the index by ID.
    ///
    /// Returns `Ok(true)` if the vector was found and removed,
    /// `Ok(false)` if the vector was not found.
    fn remove(&mut self, id: VectorId) -> Result<bool>;
}

/// Extension trait for indices that require finalization before search.
///
/// Some indices (like HNSW) build an optimized search structure after
/// all vectors have been added. Call `finalize()` before searching.
pub trait FinalizableIndex: VectorIndex {
    /// Finalize the index for optimized search.
    ///
    /// This may build optimized data structures (like flattened graphs)
    /// that enable lock-free concurrent search.
    fn finalize(&mut self);

    /// Return true if the index has been finalized.
    fn is_finalized(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search_result() {
        let result = SearchResult::new(42u64, 0.5);
        assert_eq!(result.id, VectorId(42));
        assert_eq!(result.distance, 0.5);

        let tuple = result.to_tuple();
        assert_eq!(tuple, (42, 0.5));

        let from_tuple: SearchResult = (100, 1.5).into();
        assert_eq!(from_tuple.id.0, 100);
    }
}
