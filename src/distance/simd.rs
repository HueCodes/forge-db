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
//!
//! # Safety Notes
//!
//! The SIMD functions in this module use `unsafe` blocks for:
//!
//! 1. **SIMD intrinsic calls**: All SIMD intrinsics (AVX2, AVX-512, NEON) are inherently
//!    unsafe in Rust's std::arch. However, they are memory-safe when:
//!    - The CPU supports the required feature (enforced by `#[target_feature]` or runtime checks)
//!    - Pointer arithmetic stays within the input slice bounds (ensured by loop bounds)
//!
//! 2. **Horizontal reductions**: Converting SIMD registers to scalar values uses
//!    `transmute` or extraction intrinsics, which are safe given correct register types.
//!
//! All unsafe blocks in this module follow these invariants and are guarded by
//! appropriate bounds checking.

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

// Used in tests and x86_64 scalar fallbacks
#[allow(unused_imports)]
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

/// Compute Manhattan (L1) distance using AVX-512 intrinsics.
///
/// Processes 16 floats per iteration using native `_mm512_abs_ps`.
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
pub unsafe fn manhattan_distance_avx512(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let len = a.len();
    let mut i = 0;

    // Accumulator for sum of absolute differences (512-bit = 16 floats)
    let mut sum = _mm512_setzero_ps();

    // Main loop: process 16 floats at a time with AVX-512
    while i + 16 <= len {
        // Load 16 floats from each vector (unaligned load)
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));

        // Compute difference: diff = a - b
        let diff = _mm512_sub_ps(va, vb);

        // Compute absolute value using native AVX-512 abs instruction
        let abs_diff = _mm512_abs_ps(diff);

        // Accumulate: sum += |diff|
        sum = _mm512_add_ps(sum, abs_diff);

        i += 16;
    }

    // Horizontal sum using AVX-512's built-in reduce operation
    let mut total = _mm512_reduce_add_ps(sum);

    // Handle tail elements with scalar operations
    while i < len {
        total += (a[i] - b[i]).abs();
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

/// Compute cosine distance using AVX-512 intrinsics.
///
/// Computes dot product and norms in a single pass:
/// `cosine_distance = 1 - (a · b) / (||a|| * ||b||)`
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
pub unsafe fn cosine_distance_avx512(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let len = a.len();
    let mut i = 0;

    let mut dot_sum = _mm512_setzero_ps();
    let mut norm_a_sum = _mm512_setzero_ps();
    let mut norm_b_sum = _mm512_setzero_ps();

    while i + 16 <= len {
        let va = _mm512_loadu_ps(a.as_ptr().add(i));
        let vb = _mm512_loadu_ps(b.as_ptr().add(i));

        dot_sum = _mm512_fmadd_ps(va, vb, dot_sum);
        norm_a_sum = _mm512_fmadd_ps(va, va, norm_a_sum);
        norm_b_sum = _mm512_fmadd_ps(vb, vb, norm_b_sum);

        i += 16;
    }

    let mut dot = _mm512_reduce_add_ps(dot_sum);
    let mut norm_a = _mm512_reduce_add_ps(norm_a_sum);
    let mut norm_b = _mm512_reduce_add_ps(norm_b_sum);

    while i < len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }

    let norm_product = (norm_a * norm_b).sqrt();
    if norm_product == 0.0 {
        0.0
    } else {
        1.0 - (dot / norm_product)
    }
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

/// Compute Manhattan (L1) distance using AVX2 intrinsics.
///
/// Uses `_mm256_and_ps` with a sign mask to compute absolute value.
///
/// # Safety
/// - Requires AVX2 CPU feature to be available.
/// - The caller must ensure the CPU supports this feature before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
pub unsafe fn manhattan_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let len = a.len();
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Sign mask for absolute value: clear the sign bit (0x7FFFFFFF for each f32)
    let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFF_FFFF));

    // Process 8 floats at a time with AVX2
    while i + 8 <= len {
        // Load 8 floats from each vector (unaligned load)
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));

        // Compute difference: diff = a - b
        let diff = _mm256_sub_ps(va, vb);

        // Compute absolute value by clearing sign bit: |diff| = diff & 0x7FFFFFFF
        let abs_diff = _mm256_and_ps(diff, sign_mask);

        // Accumulate: sum += |diff|
        sum = _mm256_add_ps(sum, abs_diff);

        i += 8;
    }

    // Horizontal sum of the 8 floats in the AVX2 register
    let sum_array: [f32; 8] = std::mem::transmute(sum);
    let mut total: f32 = sum_array.iter().sum();

    // Handle remaining elements with scalar operations
    while i < len {
        total += (a[i] - b[i]).abs();
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

/// Compute cosine distance using AVX2 and FMA intrinsics.
///
/// Computes dot product and norms in a single pass for efficiency:
/// `cosine_distance = 1 - (a · b) / (||a|| * ||b||)`
///
/// # Safety
/// - Requires AVX2 and FMA CPU features to be available.
/// - The caller must ensure the CPU supports these features before calling.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
#[inline]
pub unsafe fn cosine_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let len = a.len();
    let mut dot_sum = _mm256_setzero_ps();
    let mut norm_a_sum = _mm256_setzero_ps();
    let mut norm_b_sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 floats at a time, computing all three sums in parallel
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));

        // dot_sum += a * b
        dot_sum = _mm256_fmadd_ps(va, vb, dot_sum);
        // norm_a_sum += a * a
        norm_a_sum = _mm256_fmadd_ps(va, va, norm_a_sum);
        // norm_b_sum += b * b
        norm_b_sum = _mm256_fmadd_ps(vb, vb, norm_b_sum);

        i += 8;
    }

    // Horizontal sums
    let dot_array: [f32; 8] = std::mem::transmute(dot_sum);
    let norm_a_array: [f32; 8] = std::mem::transmute(norm_a_sum);
    let norm_b_array: [f32; 8] = std::mem::transmute(norm_b_sum);

    let mut dot: f32 = dot_array.iter().sum();
    let mut norm_a: f32 = norm_a_array.iter().sum();
    let mut norm_b: f32 = norm_b_array.iter().sum();

    // Handle remaining elements
    while i < len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }

    // Compute cosine distance: 1 - similarity
    let norm_product = (norm_a * norm_b).sqrt();
    if norm_product == 0.0 {
        0.0
    } else {
        1.0 - (dot / norm_product)
    }
}

// =============================================================================
// ARM NEON Implementations (aarch64)
// =============================================================================
// NEON is always available on aarch64, providing 3-5x speedup over scalar.

/// Compute squared Euclidean distance using ARM NEON intrinsics.
///
/// Processes 8 floats per iteration using 2 independent accumulators
/// for better instruction-level parallelism on modern ARM cores.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn euclidean_distance_squared_neon(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let len = a.len();
    let mut i = 0;

    let mut sum1 = unsafe { vdupq_n_f32(0.0) };
    let mut sum2 = unsafe { vdupq_n_f32(0.0) };

    // Main loop: process 8 floats at a time with 2 accumulators
    while i + 8 <= len {
        unsafe {
            let va1 = vld1q_f32(a.as_ptr().add(i));
            let vb1 = vld1q_f32(b.as_ptr().add(i));
            let diff1 = vsubq_f32(va1, vb1);
            sum1 = vfmaq_f32(sum1, diff1, diff1);

            let va2 = vld1q_f32(a.as_ptr().add(i + 4));
            let vb2 = vld1q_f32(b.as_ptr().add(i + 4));
            let diff2 = vsubq_f32(va2, vb2);
            sum2 = vfmaq_f32(sum2, diff2, diff2);
        }
        i += 8;
    }

    // Handle 4-float tail
    if i + 4 <= len {
        unsafe {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            let diff = vsubq_f32(va, vb);
            sum1 = vfmaq_f32(sum1, diff, diff);
        }
        i += 4;
    }

    // Combine accumulators and horizontal sum
    let mut total = unsafe { vaddvq_f32(vaddq_f32(sum1, sum2)) };

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

/// Compute Manhattan (L1) distance using ARM NEON intrinsics.
///
/// Processes 8 floats per iteration using 2 independent accumulators
/// for better instruction-level parallelism on modern ARM cores.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn manhattan_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let len = a.len();
    let mut i = 0;

    let mut sum1 = unsafe { vdupq_n_f32(0.0) };
    let mut sum2 = unsafe { vdupq_n_f32(0.0) };

    while i + 8 <= len {
        unsafe {
            let diff1 = vsubq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            sum1 = vaddq_f32(sum1, vabsq_f32(diff1));

            let diff2 = vsubq_f32(vld1q_f32(a.as_ptr().add(i + 4)), vld1q_f32(b.as_ptr().add(i + 4)));
            sum2 = vaddq_f32(sum2, vabsq_f32(diff2));
        }
        i += 8;
    }

    if i + 4 <= len {
        unsafe {
            let diff = vsubq_f32(vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            sum1 = vaddq_f32(sum1, vabsq_f32(diff));
        }
        i += 4;
    }

    let mut total = unsafe { vaddvq_f32(vaddq_f32(sum1, sum2)) };

    while i < len {
        total += (a[i] - b[i]).abs();
        i += 1;
    }

    total
}

/// Compute dot product using ARM NEON intrinsics.
///
/// Processes 8 floats per iteration using 2 independent accumulators
/// for better instruction-level parallelism on modern ARM cores.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let len = a.len();
    let mut i = 0;

    let mut sum1 = unsafe { vdupq_n_f32(0.0) };
    let mut sum2 = unsafe { vdupq_n_f32(0.0) };

    while i + 8 <= len {
        unsafe {
            sum1 = vfmaq_f32(sum1, vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
            sum2 = vfmaq_f32(sum2, vld1q_f32(a.as_ptr().add(i + 4)), vld1q_f32(b.as_ptr().add(i + 4)));
        }
        i += 8;
    }

    if i + 4 <= len {
        unsafe {
            sum1 = vfmaq_f32(sum1, vld1q_f32(a.as_ptr().add(i)), vld1q_f32(b.as_ptr().add(i)));
        }
        i += 4;
    }

    let mut total = unsafe { vaddvq_f32(vaddq_f32(sum1, sum2)) };

    while i < len {
        total += a[i] * b[i];
        i += 1;
    }

    total
}

/// Compute cosine distance using ARM NEON intrinsics.
///
/// Computes dot product and norms in a single pass:
/// `cosine_distance = 1 - (a · b) / (||a|| * ||b||)`
///
/// Processes 8 floats per iteration using 2 independent accumulators
/// for better instruction-level parallelism on modern ARM cores.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
pub fn cosine_distance_neon(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector dimensions must match");

    let len = a.len();
    let mut i = 0;

    let mut dot1 = unsafe { vdupq_n_f32(0.0) };
    let mut dot2 = unsafe { vdupq_n_f32(0.0) };
    let mut na1 = unsafe { vdupq_n_f32(0.0) };
    let mut na2 = unsafe { vdupq_n_f32(0.0) };
    let mut nb1 = unsafe { vdupq_n_f32(0.0) };
    let mut nb2 = unsafe { vdupq_n_f32(0.0) };

    while i + 8 <= len {
        unsafe {
            let va1 = vld1q_f32(a.as_ptr().add(i));
            let vb1 = vld1q_f32(b.as_ptr().add(i));
            dot1 = vfmaq_f32(dot1, va1, vb1);
            na1 = vfmaq_f32(na1, va1, va1);
            nb1 = vfmaq_f32(nb1, vb1, vb1);

            let va2 = vld1q_f32(a.as_ptr().add(i + 4));
            let vb2 = vld1q_f32(b.as_ptr().add(i + 4));
            dot2 = vfmaq_f32(dot2, va2, vb2);
            na2 = vfmaq_f32(na2, va2, va2);
            nb2 = vfmaq_f32(nb2, vb2, vb2);
        }
        i += 8;
    }

    if i + 4 <= len {
        unsafe {
            let va = vld1q_f32(a.as_ptr().add(i));
            let vb = vld1q_f32(b.as_ptr().add(i));
            dot1 = vfmaq_f32(dot1, va, vb);
            na1 = vfmaq_f32(na1, va, va);
            nb1 = vfmaq_f32(nb1, vb, vb);
        }
        i += 4;
    }

    let mut dot = unsafe { vaddvq_f32(vaddq_f32(dot1, dot2)) };
    let mut norm_a = unsafe { vaddvq_f32(vaddq_f32(na1, na2)) };
    let mut norm_b = unsafe { vaddvq_f32(vaddq_f32(nb1, nb2)) };

    while i < len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
        i += 1;
    }

    let norm_product = (norm_a * norm_b).sqrt();
    if norm_product == 0.0 {
        0.0
    } else {
        1.0 - (dot / norm_product)
    }
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
        scalar::euclidean_distance(a, b)
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        euclidean_distance_neon(a, b)
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
        scalar::euclidean_distance_squared(a, b)
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        euclidean_distance_squared_neon(a, b)
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
        scalar::dot_product(a, b)
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        dot_product_neon(a, b)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::dot_product(a, b)
}

/// Compute Manhattan (L1) distance with automatic CPU feature detection.
///
/// Manhattan distance is the sum of absolute differences: sum(|a[i] - b[i]|).
/// Also known as taxicab distance or city block distance.
///
/// Dispatch order (fastest first):
/// 1. AVX-512F (x86_64, requires `avx512` cargo feature) - 16 floats/iteration
/// 2. AVX2 (x86_64) - 8 floats/iteration
/// 3. NEON (aarch64) - 4 floats/iteration
/// 4. Scalar fallback
#[inline]
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if is_x86_feature_detected!("avx512f") {
                // SAFETY: We just verified that AVX-512F is supported
                return unsafe { manhattan_distance_avx512(a, b) };
            }
        }
        if is_x86_feature_detected!("avx2") {
            // SAFETY: We just verified that AVX2 is supported
            return unsafe { manhattan_distance_avx2(a, b) };
        }
        scalar::manhattan_distance(a, b)
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON is always available on aarch64
        manhattan_distance_neon(a, b)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::manhattan_distance(a, b)
}

/// Compute cosine distance with automatic CPU feature detection.
///
/// Cosine distance = 1 - cosine_similarity(a, b).
/// Range [0, 2] where 0 means identical direction.
///
/// Dispatch order (fastest first):
/// 1. AVX-512F (x86_64, requires `avx512` cargo feature) - 16 floats/iteration
/// 2. AVX2+FMA (x86_64) - 8 floats/iteration
/// 3. NEON (aarch64) - 4 floats/iteration
/// 4. Scalar fallback
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        #[cfg(feature = "avx512")]
        {
            if is_x86_feature_detected!("avx512f") {
                return unsafe { cosine_distance_avx512(a, b) };
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { cosine_distance_avx2(a, b) };
        }
        scalar::cosine_distance(a, b)
    }

    #[cfg(target_arch = "aarch64")]
    {
        cosine_distance_neon(a, b)
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::cosine_distance(a, b)
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
        assert!(
            (result - 36.0).abs() < 1e-5,
            "Expected 36.0, got {}",
            result
        );
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

    #[test]
    fn test_manhattan_simple() {
        let a = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let result = manhattan_distance(&a, &b);
        assert!((result - 7.0).abs() < 1e-5, "Expected 7.0, got {}", result);
    }

    #[test]
    fn test_manhattan_identical() {
        let a: Vec<f32> = (0..64).map(|x| x as f32).collect();
        let result = manhattan_distance(&a, &a);
        assert!(
            result.abs() < 1e-6,
            "Manhattan distance to self should be 0, got {}",
            result
        );
    }

    #[test]
    fn test_manhattan_matches_scalar() {
        // Test with various dimensions including non-multiples of 8
        for dim in [3, 7, 8, 15, 16, 31, 32, 64, 100, 128, 256] {
            let a: Vec<f32> = (0..dim).map(|x| (x as f32) * 0.1 - 5.0).collect();
            let b: Vec<f32> = (0..dim).map(|x| (x as f32) * -0.2 + 3.0).collect();

            let scalar_result = scalar::manhattan_distance(&a, &b);
            let simd_result = manhattan_distance(&a, &b);

            // Use relative tolerance for larger values
            let tol = scalar_result.abs() * 1e-5 + 1e-4;
            assert!(
                (scalar_result - simd_result).abs() < tol,
                "Dimension {}: Scalar: {}, SIMD: {}",
                dim,
                scalar_result,
                simd_result
            );
        }
    }

    #[test]
    fn test_manhattan_with_negatives() {
        let a: Vec<f32> = (0..16)
            .map(|x| if x % 2 == 0 { x as f32 } else { -(x as f32) })
            .collect();
        let b: Vec<f32> = (0..16)
            .map(|x| if x % 2 == 1 { x as f32 } else { -(x as f32) })
            .collect();

        let scalar_result = scalar::manhattan_distance(&a, &b);
        let simd_result = manhattan_distance(&a, &b);

        assert!(
            (scalar_result - simd_result).abs() < 1e-4,
            "With negatives - Scalar: {}, SIMD: {}",
            scalar_result,
            simd_result
        );
    }

    // Direct AVX2 Manhattan test
    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_manhattan_avx2_directly() {
        if !is_x86_feature_detected!("avx2") {
            println!("AVX2 not available, skipping direct test");
            return;
        }

        let a: Vec<f32> = (0..64).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..64).map(|x| (x * 2) as f32).collect();

        let scalar_result = scalar::manhattan_distance(&a, &b);
        let avx2_result = unsafe { manhattan_distance_avx2(&a, &b) };

        assert!(
            (scalar_result - avx2_result).abs() < 1e-4,
            "Manhattan - Scalar: {}, AVX2: {}",
            scalar_result,
            avx2_result
        );
    }

    #[test]
    fn test_cosine_distance_matches_scalar() {
        for dim in [3, 7, 8, 15, 16, 31, 32, 64, 100, 128, 256, 768] {
            let a: Vec<f32> = (0..dim).map(|x| (x as f32) * 0.1 + 0.1).collect();
            let b: Vec<f32> = (0..dim).map(|x| (x as f32) * -0.2 + 3.0).collect();

            let scalar_result = scalar::cosine_distance(&a, &b);
            let simd_result = cosine_distance(&a, &b);

            let tol = scalar_result.abs() * 1e-5 + 1e-5;
            assert!(
                (scalar_result - simd_result).abs() < tol,
                "Dimension {}: Scalar: {}, SIMD: {}",
                dim,
                scalar_result,
                simd_result
            );
        }
    }

    #[test]
    fn test_cosine_distance_identical() {
        let a: Vec<f32> = (0..64).map(|x| (x as f32) + 1.0).collect();
        let result = cosine_distance(&a, &a);
        assert!(
            result.abs() < 1e-5,
            "Cosine distance to self should be 0, got {}",
            result
        );
    }

    #[test]
    fn test_cosine_distance_orthogonal() {
        let mut a = vec![0.0; 8];
        let mut b = vec![0.0; 8];
        a[0] = 1.0;
        b[1] = 1.0;
        let result = cosine_distance(&a, &b);
        assert!(
            (result - 1.0).abs() < 1e-5,
            "Cosine distance of orthogonal vectors should be 1.0, got {}",
            result
        );
    }

    #[test]
    fn test_cosine_distance_zero_vector() {
        let a = vec![0.0; 8];
        let b = vec![1.0; 8];
        let result = cosine_distance(&a, &b);
        assert!(
            result.abs() < 1e-5,
            "Cosine distance with zero vector should be 0.0, got {}",
            result
        );
    }

    // Direct AVX-512 Manhattan test
    #[cfg(all(target_arch = "x86_64", feature = "avx512"))]
    #[test]
    fn test_manhattan_avx512_if_available() {
        if !is_x86_feature_detected!("avx512f") {
            println!("AVX-512 not available on this CPU, skipping direct test");
            return;
        }

        let a: Vec<f32> = (0..64).map(|x| x as f32).collect();
        let b: Vec<f32> = (0..64).map(|x| (x * 2) as f32).collect();

        let scalar_result = scalar::manhattan_distance(&a, &b);
        let avx512_result = unsafe { manhattan_distance_avx512(&a, &b) };

        assert!(
            (scalar_result - avx512_result).abs() < 1e-4,
            "Manhattan - Scalar: {}, AVX-512: {}",
            scalar_result,
            avx512_result
        );
    }
}
