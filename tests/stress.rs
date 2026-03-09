//! Stress tests for concurrent insert + search + delete operations.

use std::sync::Arc;
use std::thread;

use forge_db::index::brute_force::BruteForceIndex;
use forge_db::index::hnsw::HNSWIndex;
use forge_db::{DistanceMetric, Vector};

#[test]
fn stress_brute_force_concurrent_insert_search() {
    let index = Arc::new(std::sync::RwLock::new(BruteForceIndex::new(
        DistanceMetric::EuclideanSquared,
    )));

    let dim = 128;
    let n_writers = 4;
    let n_readers = 4;
    let ops_per_thread = 500;

    // Pre-populate with some vectors
    {
        let mut idx = index.write().unwrap();
        for i in 0..100 {
            idx.add(Vector::random(i, dim));
        }
    }

    let mut handles = Vec::new();

    // Writer threads
    for t in 0..n_writers {
        let idx = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            for i in 0..ops_per_thread {
                let id = (t * ops_per_thread + i + 1000) as u64;
                let v = Vector::random(id, dim);
                idx.write().unwrap().add(v);
            }
        }));
    }

    // Reader threads
    for t in 0..n_readers {
        let idx = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            for i in 0..ops_per_thread {
                let query = Vector::random((t * 10000 + i) as u64, dim);
                let guard = idx.read().unwrap();
                let results = guard.search(&query.data, 10);
                // Results may be fewer than 10 if index is still being populated
                assert!(!results.is_empty() || guard.is_empty());
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }

    // Verify final state
    let idx = index.read().unwrap();
    let expected = 100 + n_writers * ops_per_thread;
    assert_eq!(
        idx.len(),
        expected,
        "expected {expected} vectors, got {}",
        idx.len()
    );
}

/// Test concurrent inserts and searches don't interfere.
#[test]
fn stress_brute_force_interleaved_insert_search() {
    let index = Arc::new(std::sync::RwLock::new(BruteForceIndex::new(
        DistanceMetric::EuclideanSquared,
    )));

    let dim = 64;
    let n = 500;

    // Populate with initial vectors
    {
        let mut idx = index.write().unwrap();
        for i in 0..n {
            idx.add(Vector::random(i as u64, dim));
        }
    }

    let mut handles = Vec::new();

    // Interleaved: each thread alternates insert and search
    for t in 0..8 {
        let idx = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            for i in 0..100 {
                if i % 2 == 0 {
                    // Insert
                    let id = (n as u64) + (t * 1000 + i) as u64;
                    let v = Vector::random(id, dim);
                    idx.write().unwrap().add(v);
                } else {
                    // Search
                    let query = Vector::random((t * 10000 + i) as u64, dim);
                    let guard = idx.read().unwrap();
                    let results = guard.search(&query.data, 5);
                    assert!(!results.is_empty());
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}

#[test]
fn stress_hnsw_concurrent_search() {
    let dim = 128;
    let n = 2000;
    let vectors: Vec<Vector> = (0..n).map(|i| Vector::random(i as u64, dim)).collect();
    let index = Arc::new(HNSWIndex::build(
        vectors,
        16,
        200,
        DistanceMetric::EuclideanSquared,
    ));

    let mut handles = Vec::new();
    let n_threads = 8;
    let queries_per_thread = 200;

    for t in 0..n_threads {
        let idx = Arc::clone(&index);
        handles.push(thread::spawn(move || {
            for i in 0..queries_per_thread {
                let query = Vector::random((t * 10000 + i) as u64, dim);
                let results = idx.search(&query.data, 10);
                assert_eq!(results.len(), 10, "expected 10 results");

                // Verify results are sorted by distance
                for w in results.windows(2) {
                    assert!(
                        w[0].1 <= w[1].1,
                        "results not sorted: {} > {}",
                        w[0].1,
                        w[1].1
                    );
                }
            }
        }));
    }

    for h in handles {
        h.join().unwrap();
    }
}
