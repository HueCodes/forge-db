//! SIMD-optimized distance function implementations using AVX2 and FMA intrinsics.
//! These provide 5-10x speedup over scalar implementations for typical vector dimensions.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use super::scalar;

/// Compute Euclidean distance using AVX2 and FMA intrinsics.
///
/// # Safety
/// - Requires AVX2 and FMA CPU features to be available.
/// - The caller must ensure the CPU supports these features before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let len = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 floats at a time with AVX2
    while i + 8 <= len {
        // Load 8 floats from each vector (unaligned load)
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));

        // Compute difference: diff = a - b
        let diff = _mm256_sub_ps(va, vb);

        // Accumulate: sum = diff * diff + sum (fused multiply-add)
        sum = _mm256_fmadd_ps(diff, diff, sum);

        i += 8;
    }

    // Horizontal sum of the 8 floats in the AVX2 register
    // Transmute to array for summing
    let sum_array: [f32; 8] = std::mem::transmute(sum);
    let mut total: f32 = sum_array.iter().sum();

    // Handle remaining elements with scalar operations
    while i < len {
        let diff = a[i] - b[i];
        total += diff * diff;
        i += 1;
    }

    total.sqrt()
}

/// Compute squared Euclidean distance using AVX2 and FMA intrinsics.
///
/// # Safety
/// - Requires AVX2 and FMA CPU features to be available.
/// - The caller must ensure the CPU supports these features before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn euclidean_distance_squared_avx2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let len = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 floats at a time with AVX2
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(va, vb);
        sum = _mm256_fmadd_ps(diff, diff, sum);
        i += 8;
    }

    // Horizontal sum
    let sum_array: [f32; 8] = std::mem::transmute(sum);
    let mut total: f32 = sum_array.iter().sum();

    // Handle remaining elements
    while i < len {
        let diff = a[i] - b[i];
        total += diff * diff;
        i += 1;
    }

    total
}

/// Compute dot product using AVX2 and FMA intrinsics.
///
/// # Safety
/// - Requires AVX2 and FMA CPU features to be available.
/// - The caller must ensure the CPU supports these features before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let len = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 floats at a time with AVX2
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));

        // Accumulate: sum = a * b + sum (fused multiply-add)
        sum = _mm256_fmadd_ps(va, vb, sum);

        i += 8;
    }

    // Horizontal sum
    let sum_array: [f32; 8] = std::mem::transmute(sum);
    let mut total: f32 = sum_array.iter().sum();

    // Handle remaining elements
    while i < len {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

/// Compute Euclidean distance with automatic CPU feature detection.
///
/// Uses AVX2/FMA if available, falls back to scalar implementation otherwise.
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We just verified that AVX2 and FMA are supported
            return unsafe { euclidean_distance_avx2(a, b) };
        }
    }
    scalar::euclidean_distance(a, b)
}

/// Compute squared Euclidean distance with automatic CPU feature detection.
///
/// Uses AVX2/FMA if available, falls back to scalar implementation otherwise.
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We just verified that AVX2 and FMA are supported
            return unsafe { euclidean_distance_squared_avx2(a, b) };
        }
    }
    scalar::euclidean_distance_squared(a, b)
}

/// Compute dot product with automatic CPU feature detection.
///
/// Uses AVX2/FMA if available, falls back to scalar implementation otherwise.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We just verified that AVX2 and FMA are supported
            return unsafe { dot_product_avx2(a, b) };
        }
    }
    scalar::dot_product(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_avx2_simple() {
        let a = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = euclidean_distance(&a, &b);
        assert!((result - 5.0).abs() < 1e-5, "Expected 5.0, got {}", result);
    }

    #[test]
    fn test_dot_product_avx2_simple() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let result = dot_product(&a, &b);
        assert!((result - 36.0).abs() < 1e-5, "Expected 36.0, got {}", result);
    }

    #[test]
    fn test_non_multiple_of_8() {
        // Test with dimension not divisible by 8
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let scalar_result = scalar::euclidean_distance(&a, &b);
        let simd_result = euclidean_distance(&a, &b);

        assert!(
            (scalar_result - simd_result).abs() < 1e-5,
            "Scalar: {}, SIMD: {}",
            scalar_result,
            simd_result
        );
    }
}
