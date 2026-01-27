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
