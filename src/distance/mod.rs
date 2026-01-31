//! Distance computation module providing both scalar and SIMD implementations.
//!
//! The public API automatically selects the fastest available implementation
//! based on CPU feature detection at runtime.

pub mod scalar;
pub mod simd;

// Re-export the auto-dispatching functions as the primary API
pub use simd::{dot_product, euclidean_distance, euclidean_distance_squared, manhattan_distance};

/// Supported distance metrics for similarity search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    /// Euclidean (L2) distance: sqrt(sum((a[i] - b[i])^2))
    Euclidean,
    /// Squared Euclidean distance: sum((a[i] - b[i])^2)
    /// Faster than Euclidean when only relative ordering matters.
    EuclideanSquared,
    /// Cosine distance: 1 - cosine_similarity(a, b)
    /// Range [0, 2] where 0 means identical direction.
    Cosine,
    /// Negative dot product: -dot(a, b)
    /// Negated for min-heap compatibility (larger dot = smaller distance).
    DotProduct,
    /// Manhattan (L1) distance: sum(|a[i] - b[i]|)
    /// Also known as taxicab distance or city block distance.
    Manhattan,
}

impl DistanceMetric {
    /// Compute the distance between two vectors using this metric.
    ///
    /// # Panics
    /// Panics if the vectors have different dimensions.
    #[inline]
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        match self {
            DistanceMetric::Euclidean => euclidean_distance(a, b),
            DistanceMetric::EuclideanSquared => euclidean_distance_squared(a, b),
            DistanceMetric::Cosine => scalar::cosine_distance(a, b),
            DistanceMetric::DotProduct => -dot_product(a, b), // Negative for min-heap
            DistanceMetric::Manhattan => manhattan_distance(a, b),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distance_metric_euclidean() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = DistanceMetric::Euclidean.compute(&a, &b);
        assert!((dist - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_distance_metric_dot_product_negated() {
        let a = vec![1.0, 0.0];
        let b = vec![1.0, 0.0];
        // dot product is 1.0, so distance should be -1.0
        let dist = DistanceMetric::DotProduct.compute(&a, &b);
        assert!((dist - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn test_distance_metric_manhattan() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        let dist = DistanceMetric::Manhattan.compute(&a, &b);
        assert!((dist - 7.0).abs() < 1e-5);
    }
}
