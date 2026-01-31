//! Metadata filtering for vector search.
//!
//! This module provides support for attaching metadata to vectors and filtering
//! search results based on metadata conditions. It uses bitmap indices for
//! efficient filtering of common fields.
//!
//! # Example
//!
//! ```ignore
//! use forge_db::metadata::{MetadataValue, FilterCondition, MetadataStore};
//!
//! // Create a metadata store
//! let mut store = MetadataStore::new();
//!
//! // Add metadata for vectors
//! store.insert(1, "category", MetadataValue::String("electronics".into()));
//! store.insert(1, "price", MetadataValue::Float(299.99));
//!
//! // Build indices for efficient filtering
//! store.build_index("category");
//!
//! // Create a filter
//! let filter = FilterCondition::Equals {
//!     field: "category".into(),
//!     value: MetadataValue::String("electronics".into()),
//! };
//!
//! // Get matching IDs
//! let matching = store.filter(&filter);
//! ```

mod filter;
mod store;
mod value;

pub use filter::FilterCondition;
pub use store::MetadataStore;
pub use value::MetadataValue;

use crate::vector::Vector;

/// A vector with associated metadata.
#[derive(Clone, Debug)]
pub struct VectorWithMetadata {
    /// The vector data.
    pub vector: Vector,
    /// Metadata key-value pairs.
    pub metadata: std::collections::HashMap<String, MetadataValue>,
}

impl VectorWithMetadata {
    /// Create a new vector with metadata.
    pub fn new(vector: Vector) -> Self {
        Self {
            vector,
            metadata: std::collections::HashMap::new(),
        }
    }

    /// Add a metadata field.
    pub fn with_metadata(mut self, key: impl Into<String>, value: MetadataValue) -> Self {
        self.metadata.insert(key.into(), value);
        self
    }

    /// Get the vector ID.
    pub fn id(&self) -> u64 {
        self.vector.id
    }
}

impl From<Vector> for VectorWithMetadata {
    fn from(vector: Vector) -> Self {
        Self::new(vector)
    }
}
