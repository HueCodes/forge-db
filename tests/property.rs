//! Property-based tests using proptest.
//!
//! Covers serialization roundtrips, distance invariants, and index correctness.

use proptest::prelude::*;
use tempfile::TempDir;

use forge_db::distance::{scalar, simd};
use forge_db::index::brute_force::BruteForceIndex;
use forge_db::index::hnsw::HNSWIndex;
use forge_db::persistence::Persistable;
use forge_db::{DistanceMetric, Vector};

/// Generate a random f32 vector of given dimension.
fn arb_vector(dim: usize) -> impl Strategy<Value = Vec<f32>> {
    proptest::collection::vec(-100.0f32..100.0f32, dim)
}

/// Generate a pair of vectors with the same dimension.
fn arb_vector_pair(dim: usize) -> impl Strategy<Value = (Vec<f32>, Vec<f32>)> {
    (arb_vector(dim), arb_vector(dim))
}

// ─────────────────────────────────────────────────────────────────────────────
// Distance function properties
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    #[test]
    fn euclidean_non_negative(dim in 1usize..256, seed in any::<u64>()) {
        let a = Vector::random(seed, dim);
        let b = Vector::random(seed + 1, dim);
        let d = simd::euclidean_distance(&a.data, &b.data);
        prop_assert!(d >= 0.0, "distance must be non-negative, got {d}");
    }

    #[test]
    fn euclidean_self_is_zero((v,) in (arb_vector(128),)) {
        let d = simd::euclidean_distance(&v, &v);
        prop_assert!(d.abs() < 1e-5, "self-distance must be ~0, got {d}");
    }

    #[test]
    fn euclidean_symmetric((a, b) in arb_vector_pair(128)) {
        let d1 = simd::euclidean_distance(&a, &b);
        let d2 = simd::euclidean_distance(&b, &a);
        prop_assert!((d1 - d2).abs() < 1e-5, "d(a,b)={d1} != d(b,a)={d2}");
    }

    #[test]
    fn simd_matches_scalar_euclidean(dim in 1usize..512, seed in any::<u64>()) {
        let a = Vector::random(seed, dim);
        let b = Vector::random(seed + 1, dim);
        let s = scalar::euclidean_distance_squared(&a.data, &b.data);
        let d = simd::euclidean_distance_squared(&a.data, &b.data);
        let tol = s.abs() * 1e-5 + 1e-5;
        prop_assert!((s - d).abs() < tol, "scalar={s} vs simd={d} at dim={dim}");
    }

    #[test]
    fn simd_matches_scalar_dot(dim in 1usize..512, seed in any::<u64>()) {
        let a = Vector::random(seed, dim);
        let b = Vector::random(seed + 1, dim);
        let s = scalar::dot_product(&a.data, &b.data);
        let d = simd::dot_product(&a.data, &b.data);
        let tol = s.abs() * 1e-4 + 1e-4;
        prop_assert!((s - d).abs() < tol, "scalar={s} vs simd={d} at dim={dim}");
    }

    #[test]
    fn simd_matches_scalar_cosine(dim in 1usize..512, seed in any::<u64>()) {
        let a = Vector::random(seed, dim);
        let b = Vector::random(seed + 1, dim);
        let s = scalar::cosine_distance(&a.data, &b.data);
        let d = simd::cosine_distance(&a.data, &b.data);
        let tol = 1e-4;
        prop_assert!((s - d).abs() < tol, "scalar={s} vs simd={d} at dim={dim}");
    }

    #[test]
    fn simd_matches_scalar_manhattan(dim in 1usize..512, seed in any::<u64>()) {
        let a = Vector::random(seed, dim);
        let b = Vector::random(seed + 1, dim);
        let s = scalar::manhattan_distance(&a.data, &b.data);
        let d = simd::manhattan_distance(&a.data, &b.data);
        let tol = s.abs() * 1e-5 + 1e-5;
        prop_assert!((s - d).abs() < tol, "scalar={s} vs simd={d} at dim={dim}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Serialization roundtrips
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(20))]

    #[test]
    fn brute_force_roundtrip(
        n in 1usize..200,
        dim in 2usize..64,
        metric_idx in 0u8..5,
    ) {
        let metric = match metric_idx {
            0 => DistanceMetric::Euclidean,
            1 => DistanceMetric::EuclideanSquared,
            2 => DistanceMetric::Cosine,
            3 => DistanceMetric::DotProduct,
            _ => DistanceMetric::Manhattan,
        };

        let mut index = BruteForceIndex::new(metric);
        for i in 0..n {
            index.add(Vector::random(i as u64, dim));
        }

        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.fdb");
        index.save(&path).unwrap();
        let loaded = BruteForceIndex::load(&path).unwrap();

        // Verify same number of vectors
        prop_assert_eq!(index.len(), loaded.len());

        // Verify search results match
        let query = Vector::random(9999, dim);
        let k = n.min(10);
        let original = index.search(&query.data, k);
        let restored = loaded.search(&query.data, k);
        prop_assert_eq!(original.len(), restored.len());

        for (o, r) in original.iter().zip(restored.iter()) {
            prop_assert_eq!(o.0, r.0, "id mismatch");
            prop_assert!((o.1 - r.1).abs() < 1e-5, "distance mismatch");
        }
    }

    #[test]
    fn wal_roundtrip(
        n_ops in 1usize..50,
        dim in 2usize..32,
    ) {
        use forge_db::wal::{WriteAheadLog, WalOperation};

        let dir = TempDir::new().unwrap();

        // Write operations
        {
            let mut wal = WriteAheadLog::open(dir.path()).unwrap();
            for i in 0..n_ops {
                let v = Vector::random(i as u64, dim);
                let _ = wal.append(&WalOperation::Insert {
                    id: v.id,
                    vector: v.data.to_vec(),
                    metadata: None,
                });
            }
        }

        // Replay and verify
        let wal = WriteAheadLog::open(dir.path()).unwrap();
        let entries = wal.replay_all().unwrap();
        let expected = n_ops;
        prop_assert_eq!(entries.len(), expected);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Index correctness
// ─────────────────────────────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(10))]

    #[test]
    fn brute_force_finds_nearest(
        n in 10usize..200,
        dim in 2usize..64,
    ) {
        let mut index = BruteForceIndex::new(DistanceMetric::EuclideanSquared);
        let vectors: Vec<Vector> = (0..n).map(|i| Vector::random(i as u64, dim)).collect();
        for v in &vectors {
            index.add(v.clone());
        }

        let query = Vector::random(9999, dim);
        let results = index.search(&query.data, 1);

        // Verify the top result is actually the nearest
        let mut best_dist = f32::MAX;
        let mut best_id = 0u64;
        for v in &vectors {
            let d = scalar::euclidean_distance_squared(&query.data, &v.data);
            if d < best_dist {
                best_dist = d;
                best_id = v.id;
            }
        }

        prop_assert_eq!(results[0].0, best_id, "brute force didn't find true nearest");
    }

    #[test]
    fn hnsw_high_recall(
        n in 100usize..500,
        dim in 8usize..64,
    ) {
        let vectors: Vec<Vector> = (0..n).map(|i| Vector::random(i as u64, dim)).collect();
        let hnsw = HNSWIndex::build(vectors.clone(), 16, 200, DistanceMetric::EuclideanSquared);

        let mut bf = BruteForceIndex::new(DistanceMetric::EuclideanSquared);
        for v in &vectors {
            bf.add(v.clone());
        }

        let k = 10.min(n);
        let query = Vector::random(9999, dim);
        let hnsw_results = hnsw.search(&query.data, k);
        let bf_results = bf.search(&query.data, k);

        // HNSW should find at least 80% of exact top-k
        let bf_ids: std::collections::HashSet<u64> = bf_results.iter().map(|r| r.0).collect();
        let recall = hnsw_results.iter().filter(|r| bf_ids.contains(&r.0)).count() as f32 / k as f32;
        prop_assert!(recall >= 0.8, "HNSW recall too low: {recall:.2} at n={n}, dim={dim}");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Large dimension correctness
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn test_large_dimension_correctness() {
    for dim in [1024, 2048, 4096] {
        let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);
        let vectors: Vec<Vector> = (0..100).map(|i| Vector::random(i, dim)).collect();
        for v in &vectors {
            index.add(v.clone());
        }

        let query = Vector::random(9999, dim);
        let results = index.search(&query.data, 5);
        assert_eq!(results.len(), 5, "dim={dim}: expected 5 results");

        // Verify distances are non-negative and sorted
        for w in results.windows(2) {
            assert!(w[0].1 <= w[1].1, "dim={dim}: results not sorted");
        }

        // Verify SIMD matches scalar at large dims
        let s = scalar::euclidean_distance(&query.data, &vectors[0].data);
        let d = simd::euclidean_distance(&query.data, &vectors[0].data);
        assert!(
            (s - d).abs() < s * 1e-4 + 1e-4,
            "dim={dim}: scalar={s} vs simd={d}"
        );
    }
}
