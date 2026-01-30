//! Error types for forge-db operations.
//!
//! This module provides comprehensive error handling for all forge-db operations,
//! including index building, searching, persistence, and metadata filtering.

use std::io;
use thiserror::Error;

/// Result type alias using [`ForgeDbError`].
pub type Result<T> = std::result::Result<T, ForgeDbError>;

/// Errors that can occur during forge-db operations.
#[derive(Error, Debug)]
pub enum ForgeDbError {
    /// Vector dimensions do not match the expected dimension.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch {
        /// Expected vector dimension.
        expected: usize,
        /// Actual vector dimension provided.
        actual: usize,
    },

    /// Operation requires a non-empty vector set but received empty input.
    #[error("empty vector set: operation requires at least one vector")]
    EmptyVectorSet,

    /// Invalid parameter value provided.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// Attempted operation on an index that has not been built.
    #[error("index not built: call build() before searching")]
    IndexNotBuilt,

    /// Insufficient vectors for the requested operation.
    #[error("insufficient vectors: required {required}, got {actual}")]
    InsufficientVectors {
        /// Minimum number of vectors required.
        required: usize,
        /// Actual number of vectors provided.
        actual: usize,
    },

    /// I/O error during file operations.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// Vector with the specified ID was not found.
    #[error("vector not found: ID {0}")]
    VectorNotFound(u64),

    /// Error during serialization or deserialization.
    #[error("serialization error: {0}")]
    SerializationError(String),

    /// Checksum verification failed during file loading.
    #[error("checksum mismatch: file may be corrupted")]
    ChecksumMismatch,

    /// Operation exceeded the configured timeout.
    #[error("operation timed out")]
    Timeout,

    /// Index file has an invalid or unrecognized format.
    #[error("invalid file format: {0}")]
    InvalidFormat(String),

    /// Metadata field type mismatch during filtering.
    #[error("metadata type mismatch for field '{field}': expected {expected}, got {actual}")]
    MetadataTypeMismatch {
        /// Field name.
        field: String,
        /// Expected type name.
        expected: String,
        /// Actual type name.
        actual: String,
    },

    /// Metadata field not found.
    #[error("metadata field not found: {0}")]
    MetadataFieldNotFound(String),

    /// Index is corrupted or in an invalid state.
    #[error("index corrupted: {0}")]
    IndexCorrupted(String),

    /// Operation not supported for this index type.
    #[error("operation not supported: {0}")]
    NotSupported(String),

    /// Memory limit exceeded.
    #[error("memory limit exceeded: requested {requested} bytes, limit is {limit} bytes")]
    MemoryLimitExceeded {
        /// Bytes requested.
        requested: usize,
        /// Maximum bytes allowed.
        limit: usize,
    },

    /// Maximum vector count exceeded.
    #[error("vector limit exceeded: attempted to add vector {attempted}, limit is {limit}")]
    VectorLimitExceeded {
        /// Attempted vector count.
        attempted: usize,
        /// Maximum vector count allowed.
        limit: usize,
    },
}

impl ForgeDbError {
    /// Creates a new `DimensionMismatch` error.
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Creates a new `InsufficientVectors` error.
    pub fn insufficient_vectors(required: usize, actual: usize) -> Self {
        Self::InsufficientVectors { required, actual }
    }

    /// Creates a new `InvalidParameter` error.
    pub fn invalid_parameter(msg: impl Into<String>) -> Self {
        Self::InvalidParameter(msg.into())
    }

    /// Creates a new `SerializationError`.
    pub fn serialization_error(msg: impl Into<String>) -> Self {
        Self::SerializationError(msg.into())
    }

    /// Creates a new `InvalidFormat` error.
    pub fn invalid_format(msg: impl Into<String>) -> Self {
        Self::InvalidFormat(msg.into())
    }

    /// Creates a new `IndexCorrupted` error.
    pub fn index_corrupted(msg: impl Into<String>) -> Self {
        Self::IndexCorrupted(msg.into())
    }

    /// Creates a new `NotSupported` error.
    pub fn not_supported(msg: impl Into<String>) -> Self {
        Self::NotSupported(msg.into())
    }

    /// Creates a new `MemoryLimitExceeded` error.
    pub fn memory_limit_exceeded(requested: usize, limit: usize) -> Self {
        Self::MemoryLimitExceeded { requested, limit }
    }

    /// Creates a new `VectorLimitExceeded` error.
    pub fn vector_limit_exceeded(attempted: usize, limit: usize) -> Self {
        Self::VectorLimitExceeded { attempted, limit }
    }
}

impl From<bincode::Error> for ForgeDbError {
    fn from(err: bincode::Error) -> Self {
        Self::SerializationError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = ForgeDbError::dimension_mismatch(128, 256);
        assert_eq!(err.to_string(), "dimension mismatch: expected 128, got 256");

        let err = ForgeDbError::insufficient_vectors(100, 50);
        assert_eq!(
            err.to_string(),
            "insufficient vectors: required 100, got 50"
        );

        let err = ForgeDbError::EmptyVectorSet;
        assert_eq!(
            err.to_string(),
            "empty vector set: operation requires at least one vector"
        );

        let err = ForgeDbError::Timeout;
        assert_eq!(err.to_string(), "operation timed out");
    }

    #[test]
    fn test_error_from_io() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err: ForgeDbError = io_err.into();
        assert!(matches!(err, ForgeDbError::Io(_)));
    }
}
