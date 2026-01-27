//! Correctness tests verifying SIMD implementations match scalar baselines.
//!
//! Run with: cargo test

use forge_db::distance::{scalar, simd};
use forge_db::Vector;

#[test]
fn test_simd_matches_scalar() {
    // Test various dimensions including edge cases
    let dims = vec![1, 7, 8, 15, 16, 64, 128, 256];

    for dim in dims {
        let v1 = Vector::random(1, dim);
        let v2 = Vector::random(2, dim);

        let scalar_result = scalar::euclidean_distance(&v1.data, &v2.data);
        let simd_result = simd::euclidean_distance(&v1.data, &v2.data);
        let diff = (scalar_result - simd_result).abs();

        assert!(
            diff < 1e-5,
            "Euclidean mismatch at dim {}: scalar={}, simd={}, diff={}",
            dim,
            scalar_result,
            simd_result,
            diff
        );
    }
}

#[test]
fn test_simd_squared_matches_scalar() {
    let dims = vec![1, 7, 8, 15, 16, 64, 128, 256];

    for dim in dims {
        let v1 = Vector::random(1, dim);
        let v2 = Vector::random(2, dim);

        let scalar_result = scalar::euclidean_distance_squared(&v1.data, &v2.data);
        let simd_result = simd::euclidean_distance_squared(&v1.data, &v2.data);
        let diff = (scalar_result - simd_result).abs();

        assert!(
            diff < 1e-5,
            "Squared Euclidean mismatch at dim {}: scalar={}, simd={}, diff={}",
            dim,
            scalar_result,
            simd_result,
            diff
        );
    }
}

#[test]
fn test_dot_product_simd_matches_scalar() {
    let dims = vec![1, 7, 8, 15, 16, 64, 128, 256];

    for dim in dims {
        let v1 = Vector::random(1, dim);
        let v2 = Vector::random(2, dim);

        let scalar_result = scalar::dot_product(&v1.data, &v2.data);
        let simd_result = simd::dot_product(&v1.data, &v2.data);
        let diff = (scalar_result - simd_result).abs();

        assert!(
            diff < 1e-5,
            "Dot product mismatch at dim {}: scalar={}, simd={}, diff={}",
            dim,
            scalar_result,
            simd_result,
            diff
        );
    }
}

#[test]
fn test_euclidean_properties() {
    let v1 = Vector::random(1, 128);
    let v2 = Vector::random(2, 128);

    // Property 1: Distance to self is zero
    let self_distance = simd::euclidean_distance(&v1.data, &v1.data);
    assert!(
        self_distance < 1e-6,
        "Distance to self should be 0, got {}",
        self_distance
    );

    // Property 2: Symmetry - d(a,b) = d(b,a)
    let d1 = simd::euclidean_distance(&v1.data, &v2.data);
    let d2 = simd::euclidean_distance(&v2.data, &v1.data);
    let diff = (d1 - d2).abs();
    assert!(
        diff < 1e-6,
        "Distance should be symmetric: d(v1,v2)={}, d(v2,v1)={}, diff={}",
        d1,
        d2,
        diff
    );
}

#[test]
fn test_dot_product_properties() {
    let v1 = Vector::random(1, 128);
    let v2 = Vector::random(2, 128);

    // Property 1: Commutativity - dot(a,b) = dot(b,a)
    let d1 = simd::dot_product(&v1.data, &v2.data);
    let d2 = simd::dot_product(&v2.data, &v1.data);
    let diff = (d1 - d2).abs();
    assert!(
        diff < 1e-6,
        "Dot product should be commutative: dot(v1,v2)={}, dot(v2,v1)={}, diff={}",
        d1,
        d2,
        diff
    );

    // Property 2: Self dot product is non-negative (squared norm)
    let self_dot = simd::dot_product(&v1.data, &v1.data);
    assert!(
        self_dot >= 0.0,
        "Self dot product should be non-negative, got {}",
        self_dot
    );
}

#[test]
fn test_known_values() {
    // Test with known values to verify correctness
    let a = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let b = vec![3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Euclidean distance should be 5 (3-4-5 triangle)
    let dist = simd::euclidean_distance(&a, &b);
    assert!(
        (dist - 5.0).abs() < 1e-5,
        "Expected distance 5.0, got {}",
        dist
    );

    // Squared distance should be 25
    let dist_sq = simd::euclidean_distance_squared(&a, &b);
    assert!(
        (dist_sq - 25.0).abs() < 1e-5,
        "Expected squared distance 25.0, got {}",
        dist_sq
    );
}

#[test]
fn test_unit_vectors() {
    // Unit vectors along axes
    let e1 = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let e2 = vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

    // Dot product of orthogonal unit vectors should be 0
    let dot = simd::dot_product(&e1, &e2);
    assert!(dot.abs() < 1e-6, "Orthogonal vectors should have dot product 0, got {}", dot);

    // Euclidean distance between orthogonal unit vectors should be sqrt(2)
    let dist = simd::euclidean_distance(&e1, &e2);
    let expected = std::f32::consts::SQRT_2;
    assert!(
        (dist - expected).abs() < 1e-5,
        "Expected distance sqrt(2)={}, got {}",
        expected,
        dist
    );
}
