//! Core newtypes for type-safe vector database operations.
//!
//! These types provide compile-time guarantees that prevent mixing up
//! related but semantically different values (e.g., vector IDs vs indices).

use serde::{Deserialize, Serialize};
use std::fmt;

/// A unique identifier for a vector in the database.
///
/// Using a newtype prevents accidentally passing an index where an ID is expected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(transparent)]
pub struct VectorId(pub u64);

impl VectorId {
    /// Create a new VectorId.
    #[inline]
    pub const fn new(id: u64) -> Self {
        Self(id)
    }

    /// Get the raw u64 value.
    #[inline]
    pub const fn as_u64(self) -> u64 {
        self.0
    }
}

impl fmt::Display for VectorId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "VectorId({})", self.0)
    }
}

impl From<u64> for VectorId {
    #[inline]
    fn from(id: u64) -> Self {
        Self(id)
    }
}

impl From<VectorId> for u64 {
    #[inline]
    fn from(id: VectorId) -> Self {
        id.0
    }
}

/// The dimensionality of vectors in an index.
///
/// Ensures dimension values are used consistently throughout the codebase.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct Dimension(pub usize);

impl Dimension {
    /// Create a new Dimension.
    #[inline]
    pub const fn new(dim: usize) -> Self {
        Self(dim)
    }

    /// Get the raw usize value.
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0
    }

    /// Check if this dimension is divisible by a given number.
    #[inline]
    pub fn is_divisible_by(self, divisor: usize) -> bool {
        self.0.is_multiple_of(divisor)
    }
}

impl fmt::Display for Dimension {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<usize> for Dimension {
    #[inline]
    fn from(dim: usize) -> Self {
        Self(dim)
    }
}

impl From<Dimension> for usize {
    #[inline]
    fn from(dim: Dimension) -> Self {
        dim.0
    }
}

/// The number of subvectors for product quantization.
///
/// In PQ, each vector is split into `m` subvectors, each quantized independently.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NumSubvectors(pub usize);

impl NumSubvectors {
    /// Create a new NumSubvectors.
    #[inline]
    pub const fn new(m: usize) -> Self {
        Self(m)
    }

    /// Get the raw usize value.
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for NumSubvectors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<usize> for NumSubvectors {
    #[inline]
    fn from(m: usize) -> Self {
        Self(m)
    }
}

impl From<NumSubvectors> for usize {
    #[inline]
    fn from(m: NumSubvectors) -> Self {
        m.0
    }
}

/// The number of clusters for IVF partitioning.
///
/// In IVF indexes, vectors are assigned to one of `nlist` clusters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(transparent)]
pub struct NumClusters(pub usize);

impl NumClusters {
    /// Create a new NumClusters.
    #[inline]
    pub const fn new(nlist: usize) -> Self {
        Self(nlist)
    }

    /// Get the raw usize value.
    #[inline]
    pub const fn as_usize(self) -> usize {
        self.0
    }
}

impl fmt::Display for NumClusters {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<usize> for NumClusters {
    #[inline]
    fn from(nlist: usize) -> Self {
        Self(nlist)
    }
}

impl From<NumClusters> for usize {
    #[inline]
    fn from(nlist: NumClusters) -> Self {
        nlist.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_id() {
        let id = VectorId::new(42);
        assert_eq!(id.as_u64(), 42);
        assert_eq!(format!("{}", id), "VectorId(42)");

        let id2: VectorId = 100u64.into();
        assert_eq!(id2.as_u64(), 100);

        let raw: u64 = id.into();
        assert_eq!(raw, 42);
    }

    #[test]
    fn test_dimension() {
        let dim = Dimension::new(128);
        assert_eq!(dim.as_usize(), 128);
        assert!(dim.is_divisible_by(8));
        assert!(dim.is_divisible_by(16));
        assert!(!dim.is_divisible_by(3));
    }

    #[test]
    fn test_num_subvectors() {
        let m = NumSubvectors::new(8);
        assert_eq!(m.as_usize(), 8);
    }

    #[test]
    fn test_num_clusters() {
        let nlist = NumClusters::new(256);
        assert_eq!(nlist.as_usize(), 256);
    }

    #[test]
    fn test_ordering() {
        let id1 = VectorId::new(1);
        let id2 = VectorId::new(2);
        assert!(id1 < id2);
    }
}
