//! Scalar (non-SIMD) distance function implementations.
//! These serve as baselines for comparison and fallbacks on non-x86_64 platforms.

/// Compute the Euclidean (L2) distance between two vectors.
///
/// Returns sqrt(sum((a[i] - b[i])^2))
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let sum: f32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();

    sum.sqrt()
}

/// Compute the squared Euclidean distance between two vectors.
///
/// Returns sum((a[i] - b[i])^2)
///
/// This is faster than `euclidean_distance` when you only need relative
/// distances (e.g., finding nearest neighbors) since it skips the sqrt.
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum()
}

/// Compute the dot product of two vectors.
///
/// Returns sum(a[i] * b[i])
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the cosine similarity between two vectors.
///
/// Returns dot(a, b) / (||a|| * ||b||)
///
/// Range: [-1.0, 1.0] where 1.0 means identical direction.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    let denom = (norm_a * norm_b).sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}

/// Compute the cosine distance between two vectors.
///
/// Returns 1.0 - cosine_similarity(a, b)
///
/// Range: [0.0, 2.0] where 0.0 means identical direction.
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    1.0 - cosine_similarity(a, b)
}

/// Compute the Manhattan (L1) distance between two vectors.
///
/// Returns sum(|a[i] - b[i]|)
///
/// Also known as taxicab distance or city block distance.
#[inline]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((euclidean_distance(&a, &a)) < 1e-6);
    }

    #[test]
    fn test_euclidean_distance_simple() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((euclidean_distance(&a, &b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_dot_product_simple() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot_product(&a, &b) - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance_simple() {
        let a = vec![0.0, 0.0];
        let b = vec![3.0, 4.0];
        assert!((manhattan_distance(&a, &b) - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_manhattan_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!(manhattan_distance(&a, &a) < 1e-6);
    }

    #[test]
    fn test_manhattan_distance_negative() {
        let a = vec![1.0, -2.0, 3.0];
        let b = vec![-1.0, 2.0, -3.0];
        // |1-(-1)| + |-2-2| + |3-(-3)| = 2 + 4 + 6 = 12
        assert!((manhattan_distance(&a, &b) - 12.0).abs() < 1e-6);
    }
}
