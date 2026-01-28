//! SIMD-optimized distance function implementations.
//!
//! Supports multiple SIMD instruction sets with automatic runtime detection:
//! - **AVX-512** (x86_64): Processes 16 floats at a time, 30-50% faster than AVX2 on supported CPUs
//!   (Requires nightly Rust and the `avx512` cargo feature)
//! - **AVX2+FMA** (x86_64): Processes 8 floats at a time, 5-10x faster than scalar
//! - **NEON** (aarch64): Processes 4 floats at a time, 3-5x faster than scalar
//! - **Scalar**: Fallback for all platforms
//!
//! The public API functions automatically select the fastest available implementation.
//!
//! # Feature Flags
//!
//! - `avx512`: Enable AVX-512 support. Requires nightly Rust due to unstable intrinsics.
//!
//! # Building with AVX-512
//!
//! ```bash
//! # On nightly Rust with the avx512 feature:
//! cargo +nightly build --features avx512
//! ```

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

use super::scalar;

// =============================================================================
// AVX-512 Implementations (x86_64)
// =============================================================================
// These provide ~30-50% speedup over AVX2 on supported CPUs (Skylake-X, Ice Lake, etc.)
// by processing 16 floats per iteration instead of 8.
//
// Note: AVX-512 intrinsics require nightly Rust due to `stdarch_x86_avx512` being unstable.
// The implementations are provided behind the `avx512` cargo feature flag.

/// Compute Euclidean distance using AVX-512 intrinsics.
///
/// Processes 16 floats per iteration, providing ~30-50% speedup over AVX2
/// on supported CPUs (Skylake-X, Ice Lake, Zen 4, etc.).
///
/// # Safety
/// - Requires AVX-512F CPU feature to be available.
/// - The caller must ensure the CPU supports this feature before calling.
///
/// # Feature Flag
/// This function requires the `avx512` cargo feature and nightly Rust.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn euclidean_distance_avx512(a: &[f32], b: &[f32]) -> f32 {
    euclidean_distance_squared_avx512(a, b).sqrt()
}

/// Compute squared Euclidean distance using AVX-512 intrinsics.
///
/// Processes 16 floats per iteration using 512-bit registers.
///
/// # Safety
/// - Requires AVX-512F CPU feature to be available.
/// - The caller must ensure the CPU supports this feature before calling.
///
/// # Feature Flag
/// This function requires the `avx512` cargo feature and nightly Rust.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn euclidean_distance_squared_avx512(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let len = a.len();
    let mut i = 0;

    // Accumulator for sum of squared differences (512-bit = 16 floats)
    let mut sum = _mm512_setzero_ps();

    // Main loop: process 16 floats at a time with AVX-512
    while i + 16 <= len {
        // Load 16 floats from each vector (unaligned load)
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));

        // Compute difference: diff = a - b
        let diff = _mm512_sub_ps(va, vb);

        // Accumulate: sum += diff * diff (fused multiply-add)
        sum = _mm512_fmadd_ps(diff, diff, sum);

        i += 16;
    }

    // Horizontal sum using AVX-512's built-in reduce operation
    // This is much more efficient than the AVX2 approach of extracting to array
    let mut total = _mm512_reduce_add_ps(sum);

    // Handle tail elements (0-15 remaining) with scalar operations
    // For small tails, scalar is faster than masked AVX-512 operations
    while i < len {
        let diff = a[i] - b[i];
        total += diff * diff;
        i += 1;
    }

    total
}

/// Compute dot product using AVX-512 intrinsics.
///
/// Processes 16 floats per iteration, providing ~30-50% speedup over AVX2.
///
/// # Safety
/// - Requires AVX-512F CPU feature to be available.
/// - The caller must ensure the CPU supports this feature before calling.
///
/// # Feature Flag
/// This function requires the `avx512` cargo feature and nightly Rust.
#[cfg(all(target_arch = "x86_64", feature = "avx512"))]
#[target_feature(enable = "avx512f")]
#[inline]
pub unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let len = a.len();
    let mut i = 0;

    // Accumulator for dot product (512-bit = 16 floats)
    let mut sum = _mm512_setzero_ps();

    // Main loop: process 16 floats at a time
    while i + 16 <= len {
        // Load 16 floats from each vector
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));

        // Accumulate: sum += a * b (fused multiply-add)
        sum = _mm512_fmadd_ps(va, vb, sum);

        i += 16;
    }

    // Horizontal sum using AVX-512's efficient reduce operation
    let mut total = _mm512_reduce_add_ps(sum);

    // Handle tail elements with scalar operations
    while i < len {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

// =============================================================================
// AVX2+FMA Implementations (x86_64)
// =============================================================================

/// Compute Euclidean distance using AVX2 and FMA intrinsics.
///
/// Processes 8 floats per iteration, providing 5-10x speedup over scalar.
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

// =============================================================================
// ARM NEON Implementations (aarch64)
// =============================================================================
// NEON is always available on aarch64, providing 3-5x speedup over scalar.

/// Compute squared Euclidean distance using ARM NEON intrinsics.
///
/// Processes 4 floats per iteration using 128-bit NEON registers.
/// NEON is always available on aarch64, so no runtime detection needed.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn euclidean_distance_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let len = a.len();
    let mut i = 0;

    // Accumulator for sum of squared differences (128-bit = 4 floats)
    let mut sum = unsafe { vdupq_n_f32(0.0) };

    // Main loop: process 4 floats at a time
    while i + 4 <= len {
        unsafe {
            // Load 4 floats from each vector
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));

            // Compute difference: diff = a - b
            let diff = vsubq_f32(va, vb);

            // Accumulate: sum += diff * diff (fused multiply-add)
            sum = vfmaq_f32(sum, diff, diff);
        }
        i += 4;
    }

    // Horizontal sum of the 4 floats
    let mut total = unsafe { vaddvq_f32(sum) };

    // Handle tail elements
    while i < len {
        let diff = a[i] - b[i];
        total += diff * diff;
        i += 1;
    }

    total
}

/// Compute Euclidean distance using ARM NEON intrinsics.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn euclidean_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    euclidean_distance_squared_neon(a, b).sqrt()
}

/// Compute dot product using ARM NEON intrinsics.
///
/// Processes 4 floats per iteration using 128-bit NEON registers.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let len = a.len();
    let mut i = 0;

    // Accumulator for dot product
    let mut sum = unsafe { vdupq_n_f32(0.0) };

    // Main loop: process 4 floats at a time
    while i + 4 <= len {
        unsafe {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));

            // Accumulate: sum += a * b
            sum = vfmaq_f32(sum, va, vb);
        }
        i += 4;
    }

    // Horizontal sum
    let mut total = unsafe { vaddvq_f32(sum) };

    // Handle tail elements
    while i < len {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

// =============================================================================
// Auto-dispatching Public API
// =============================================================================
// These functions automatically select the fastest available SIMD implementation
// at runtime based on CPU feature detection.

/// Compute Euclidean distance with automatic CPU feature detection.
///
/// Dispatch order (fastest first):
/// 1. AVX-512F (x86_64, requires `avx512` cargo feature) - 16 floats/iteration
/// 2. AVX2+FMA (x86_64) - 8 floats/iteration
/// 3. NEON (aarch64) - 4 floats/iteration
/// 4. Scalar fallback
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: We just verified that AVX-512F is supported
                return unsafe { euclidean_distance_avx512(a, b) };
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We just verified that AVX2 and FMA are supported
            return unsafe { euclidean_distance_avx2(a, b) };
        }
        return scalar::euclidean_distance(a, b);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return euclidean_distance_neon(a, b);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::euclidean_distance(a, b)
}

/// Compute squared Euclidean distance with automatic CPU feature detection.
///
/// Dispatch order (fastest first):
/// 1. AVX-512F (x86_64, requires `avx512` cargo feature) - 16 floats/iteration
/// 2. AVX2+FMA (x86_64) - 8 floats/iteration
/// 3. NEON (aarch64) - 4 floats/iteration
/// 4. Scalar fallback
#[inline]
pub fn euclidean_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: We just verified that AVX-512F is supported
                return unsafe { euclidean_distance_squared_avx512(a, b) };
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We just verified that AVX2 and FMA are supported
            return unsafe { euclidean_distance_squared_avx2(a, b) };
        }
        return scalar::euclidean_distance_squared(a, b);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return euclidean_distance_squared_neon(a, b);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::euclidean_distance_squared(a, b)
}

/// Compute dot product with automatic CPU feature detection.
///
/// Dispatch order (fastest first):
/// 1. AVX-512F (x86_64, requires `avx512` cargo feature) - 16 floats/iteration
/// 2. AVX2+FMA (x86_64) - 8 floats/iteration
/// 3. NEON (aarch64) - 4 floats/iteration
/// 4. Scalar fallback
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: We just verified that AVX-512F is supported
                return unsafe { dot_product_avx512(a, b) };
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: We just verified that AVX2 and FMA are supported
            return unsafe { dot_product_avx2(a, b) };
        }
        return scalar::dot_product(a, b);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        return dot_product_neon(a, b);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::dot_product(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_euclidean_simple() {
        let a = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = euclidean_distance(&a, &b);
        assert!((result - 5.0).abs() < 1e-5, "Expected 5.0, got {}", result);
    }

    #[test]
    fn test_dot_product_simple() {
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

    #[test]
    fn test_non_multiple_of_16() {
        // Test with dimension divisible by 8 but not 16 (AVX-512 tail case)
        let a: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let b: Vec<f32> = (1..=24).map(|x| (x * 2) as f32).collect();

        let scalar_result = scalar::euclidean_distance(&a, &b);
        let simd_result = euclidean_distance(&a, &b);

        assert!(
            (scalar_result - simd_result).abs() < 1e-4,
            "Scalar: {}, SIMD: {}",
            scalar_result,
            simd_result
        );
    }

    #[test]
    fn test_large_vectors() {
        // Test with typical embedding dimensions (128, 256, 512, 768)
        for dim in [128, 256, 512, 768] {
            let a: Vec<f32> = (0..dim).map(|x| (x as f32) * 0.01).collect();
            let b: Vec<f32> = (0..dim).map(|x| (x as f32) * 0.02).collect();

            let scalar_dist = scalar::euclidean_distance(&a, &b);
            let simd_dist = euclidean_distance(&a, &b);

            assert!(
                (scalar_dist - simd_dist).abs() < 1e-3,
                "Dimension {}: Scalar: {}, SIMD: {}",
                dim,
                scalar_dist,
                simd_dist
            );

            let scalar_dot = scalar::dot_product(&a, &b);
            let simd_dot = dot_product(&a, &b);

            assert!(
                (scalar_dot - simd_dot).abs() / scalar_dot.abs().max(1.0) < 1e-5,
                "Dimension {}: Scalar dot: {}, SIMD dot: {}",
                dim,
                scalar_dot,
                simd_dot
            );
        }
    }

    #[test]
    fn test_identical_vectors() {
        let a: Vec<f32> = (0..64).map(|x| x as f32).collect();
        let result = euclidean_distance(&a, &a);
        assert!(
            result.abs() < 1e-6,
            "Distance to self should be 0, got {}",
            result
        );
    }

    #[test]
    fn test_squared_distance() {
        let a = vec![0.0; 16];
        let b: Vec<f32> = (0..16).map(|x| x as f32).collect();

        let scalar_result = scalar::euclidean_distance_squared(&a, &b);
        let simd_result = euclidean_distance_squared(&a, &b);

        assert!(
            (scalar_result - simd_result).abs() < 1e-4,
            "Scalar: {}, SIMD: {}",
            scalar_result,
            simd_result
        );
    }

    // Direct AVX-512 tests (only compiled with avx512 feature on nightly)
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn test_avx512_if_available() {
        if !is_x86_feature_detected!("avx512f") {
            println!("AVX-512 not available on this CPU, skipping direct test");
            return;
        }

        let a: Vec<f32> = (0..64).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..64).map(|x| (x * 2) as f32).collect();

        let scalar_result = scalar::euclidean_distance(&a, &b);
        let avx512_result = unsafe { euclidean_distance_avx512(&a, &b) };

        assert!(
            (scalar_result - avx512_result).abs() < 1e-4,
            "Scalar: {}, AVX-512: {}",
            scalar_result,
            avx512_result
        );

        let scalar_dot = scalar::dot_product(&a, &b);
        let avx512_dot = unsafe { dot_product_avx512(&a, &b) };

        assert!(
            (scalar_dot - avx512_dot).abs() / scalar_dot.abs().max(1.0) < 1e-5,
            "Scalar dot: {}, AVX-512 dot: {}",
            scalar_dot,
            avx512_dot
        );
    }

    // Direct AVX2 tests
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_directly() {
        if !is_x86_feature_detected!("avx2") || !is_x86_feature_detected!("fma") {
            println!("AVX2+FMA not available, skipping direct test");
            return;
        }

        let a: Vec<f32> = (0..64).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..64).map(|x| (x * 2) as f32).collect();

        let scalar_result = scalar::euclidean_distance(&a, &b);
        let avx2_result = unsafe { euclidean_distance_avx2(&a, &b) };

        assert!(
            (scalar_result - avx2_result).abs() < 1e-4,
            "Scalar: {}, AVX2: {}",
            scalar_result,
            avx2_result
        );
    }
}
