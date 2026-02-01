//! forge-db: A high-performance vector database in Rust.
//!
//! This crate provides SIMD-optimized distance functions and data structures
//! for building efficient similarity search systems. It is designed for
//! applications like semantic search, recommendation systems, and image
//! similarity.
//!
//! # Features
//!
//! - **SIMD Distance Functions**: AVX2/AVX-512/NEON optimized distance computation
//! - **Multiple Index Types**: IVF-PQ (memory-efficient), HNSW (fast graph-based)
//! - **Product Quantization**: 8-bit and 4-bit compression for large-scale indexes
//! - **Metadata Filtering**: Filter search results by vector attributes
//! - **Persistence**: Save and load indexes with checksum verification
//! - **Thread Safety**: Concurrent reads with atomic parameter updates
//! - **Parallel Search**: Multi-core scaling with Rayon
//!
//! # Quick Start
//!
//! ```
//! use forge_db::{Vector, IVFPQIndex, DistanceMetric};
//!
//! // Create vectors (in practice, load your embeddings)
//! let vectors: Vec<Vector> = (0..1000)
//!     .map(|i| Vector::random(i, 128))
//!     .collect();
//!
//! // Build the IVF-PQ index
//! let mut index = IVFPQIndex::build(vectors, 32, 8, DistanceMetric::Euclidean);
//! index.set_nprobe(4);
//!
//! // Search
//! let query = Vector::random(9999, 128);
//! let results = index.search(&query.data, 10);
//!
//! for (id, distance) in results {
//!     println!("Vector {} at distance {}", id, distance);
//! }
//! ```
//!
//! # Choosing an Index
//!
//! | Index         | Best For                          | Memory    | Recall  |
//! |---------------|-----------------------------------|-----------|---------|
//! | `IVFPQIndex`  | Large scale (1M+ vectors)         | Low       | Good    |
//! | `HNSWIndex`   | Low latency, high recall          | High      | Best    |
//! | `IVFIndex`    | Medium scale without compression  | Medium    | Good    |
//! | `BruteForce`  | Small datasets, ground truth      | High      | 100%    |
//!
//! # Distance Metrics
//!
//! - `Euclidean`: L2 distance, most common for embeddings
//! - `EuclideanSquared`: L2 squared (faster when only ranking matters)
//! - `Cosine`: For normalized vectors (text embeddings)
//! - `DotProduct`: Maximum inner product search (negated for min-heap)
//! - `Manhattan`: L1 distance (city block)
//!
//! # Modules
//!
//! - [`index`]: Vector index implementations (IVF-PQ, HNSW, IVF, BruteForce)
//! - [`distance`]: SIMD-optimized distance functions
//! - [`pq`]: Product Quantization for vector compression
//! - [`metadata`]: Filtered search with metadata attributes
//! - [`persistence`]: Save/load indexes to disk
//! - [`metrics`]: Index statistics and health monitoring

pub mod constants;
pub mod dataset;
pub mod distance;
pub mod error;
pub mod index;
pub mod kmeans;
pub mod metadata;
pub mod metrics;
pub mod persistence;
pub mod pq;
pub mod types;
pub mod vector;

// Re-export commonly used types at crate root
pub use dataset::{recall_at_k, Dataset};
pub use distance::DistanceMetric;
pub use error::{ForgeDbError, Result};
pub use index::{BruteForceIndex, HNSWIndex, IVFIndex, IVFPQIndex, IVFPQIndexBuilder};
pub use kmeans::KMeans;
pub use metrics::{HealthStatus, IndexStatistics, ResourceLimits, SearchStatistics};
pub use pq::{CompressedVectors, FlatCompressedVectors, ProductQuantizer};
pub use metadata::{FilterCondition, MetadataStore, MetadataValue, VectorWithMetadata};
pub use persistence::Persistable;
pub use types::{Dimension, NumClusters, NumSubvectors, VectorId};
pub use vector::{AlignedVector, Vector, VectorStore};
