//! vector-db: A high-performance vector database in Rust.
//!
//! This crate provides SIMD-optimized distance functions and data structures
//! for building efficient similarity search systems.
//!
//! # Features
//!
//! - **SIMD Distance Functions**: AVX2/FMA optimized Euclidean and dot product
//! - **Automatic CPU Detection**: Falls back to scalar on unsupported hardware
//! - **Multiple Distance Metrics**: Euclidean, Cosine, Dot Product
//! - **Brute Force Index**: Exact nearest neighbor search (ground truth baseline)
//! - **Parallel Search**: Multi-core scaling with Rayon
//!
//! # Example
//!
//! ```
//! use forge_db::{Vector, DistanceMetric};
//!
//! let v1 = Vector::random(1, 128);
//! let v2 = Vector::random(2, 128);
//!
//! let distance = DistanceMetric::Euclidean.compute(&v1.data, &v2.data);
//! println!("Distance: {}", distance);
//! ```

pub mod dataset;
pub mod distance;
pub mod index;
pub mod kmeans;
pub mod pq;
pub mod vector;

// Re-export commonly used types at crate root
pub use dataset::{recall_at_k, Dataset};
pub use distance::DistanceMetric;
pub use index::{BruteForceIndex, HNSWIndex, IVFIndex, IVFPQIndex};
pub use kmeans::KMeans;
pub use pq::{CompressedVectors, ProductQuantizer};
pub use vector::{AlignedVector, Vector};
