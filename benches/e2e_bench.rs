//! End-to-end benchmarks for insertion + search across index types and dimensions.
//!
//! Run with: cargo bench --bench e2e_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use forge_db::{BruteForceIndex, Dataset, DistanceMetric, Vector};
use forge_db::index::hnsw::HNSWIndex;
use std::sync::Arc;
use std::time::Duration;

fn benchmark_brute_force_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("bf_insert");
    group.measurement_time(Duration::from_secs(10));

    for &(n, dim) in &[
        (10_000, 128),
        (10_000, 768),
        (10_000, 1536),
        (100_000, 128),
    ] {
        let vectors: Vec<Vector> = (0..n).map(|i| Vector::random(i, dim)).collect();

        group.throughput(Throughput::Elements(n));
        group.bench_with_input(
            BenchmarkId::new(format!("n{}_d{}", n, dim), n),
            &vectors,
            |b, vecs| {
                b.iter(|| {
                    let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);
                    for v in vecs {
                        index.add(v.clone());
                    }
                    black_box(&index);
                })
            },
        );
    }
    group.finish();
}

fn benchmark_hnsw_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_insert");
    group.measurement_time(Duration::from_secs(15));
    group.sample_size(10);

    for &(n, dim) in &[(10_000, 128), (10_000, 768), (10_000, 1536)] {
        let vectors: Vec<Vector> = (0..n).map(|i| Vector::random(i, dim)).collect();

        group.throughput(Throughput::Elements(n));
        group.bench_with_input(
            BenchmarkId::new(format!("n{}_d{}", n, dim), n),
            &vectors,
            |b, vecs| {
                b.iter(|| {
                    let index =
                        HNSWIndex::build(vecs.clone(), 16, 200, DistanceMetric::Euclidean);
                    black_box(&index);
                })
            },
        );
    }
    group.finish();
}

fn benchmark_brute_force_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("bf_search");

    for &(n, dim) in &[
        (1_000, 128),
        (10_000, 128),
        (10_000, 768),
        (10_000, 1536),
        (100_000, 128),
    ] {
        let dataset = Dataset::generate(n as usize, 100, dim);
        let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);
        for v in &dataset.vectors {
            index.add(v.clone());
        }
        let query = &dataset.queries[0];

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new(format!("n{}_d{}", n, dim), n),
            &query,
            |b, q| b.iter(|| index.search(black_box(&q.data), black_box(10))),
        );
    }
    group.finish();
}

fn benchmark_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");

    for &(n, dim) in &[
        (1_000, 128),
        (10_000, 128),
        (10_000, 768),
        (10_000, 1536),
        (100_000, 128),
    ] {
        let dataset = Dataset::generate(n as usize, 100, dim);
        let index =
            HNSWIndex::build(dataset.vectors.clone(), 16, 200, DistanceMetric::Euclidean);
        let query = &dataset.queries[0];

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::new(format!("n{}_d{}", n, dim), n),
            &query,
            |b, q| b.iter(|| index.search(black_box(&q.data), black_box(10))),
        );
    }
    group.finish();
}

fn benchmark_distance_metrics_bf(c: &mut Criterion) {
    let mut group = c.benchmark_group("metric_comparison");
    let dataset = Dataset::generate(10_000, 100, 128);
    let query = &dataset.queries[0];

    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::EuclideanSquared,
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
        DistanceMetric::Manhattan,
    ] {
        let mut index = BruteForceIndex::new(metric);
        for v in &dataset.vectors {
            index.add(v.clone());
        }

        group.bench_with_input(
            BenchmarkId::new("bf_10k", format!("{:?}", metric)),
            &index,
            |b, idx| b.iter(|| idx.search(black_box(&query.data), black_box(10))),
        );
    }
    group.finish();
}

fn benchmark_concurrent_read_write(c: &mut Criterion) {
    use std::sync::RwLock;
    use std::thread;

    let mut group = c.benchmark_group("concurrent");
    group.measurement_time(Duration::from_secs(10));
    group.sample_size(10);

    // Setup: pre-populate an index with 10k vectors
    let dim = 128;
    let n = 10_000;
    let dataset = Dataset::generate(n, 100, dim);
    let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);
    for v in &dataset.vectors {
        index.add(v.clone());
    }
    let index = Arc::new(RwLock::new(index));
    let queries: Arc<Vec<Vector>> = Arc::new(dataset.queries);

    // Benchmark: concurrent reads while inserting
    group.bench_function("mixed_read_write_bf_10k", |b| {
        b.iter(|| {
            let index = Arc::clone(&index);
            let queries = Arc::clone(&queries);

            // Spawn reader threads
            let mut handles = Vec::new();
            for t in 0..4 {
                let idx = Arc::clone(&index);
                let qs = Arc::clone(&queries);
                handles.push(thread::spawn(move || {
                    for i in 0..25 {
                        let q = &qs[(t * 25 + i) % qs.len()];
                        let guard = idx.read().unwrap();
                        black_box(guard.search(&q.data, 10));
                    }
                }));
            }

            // Writer thread: insert 100 new vectors
            {
                let idx = Arc::clone(&index);
                handles.push(thread::spawn(move || {
                    for i in 0..100 {
                        let v = Vector::random(n as u64 + i, dim);
                        idx.write().unwrap().add(v);
                    }
                }));
            }

            for h in handles {
                h.join().unwrap();
            }
        })
    });

    group.finish();
}

fn benchmark_batch_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insert");
    group.measurement_time(Duration::from_secs(10));

    for &(n, dim) in &[(1_000, 128), (10_000, 128), (10_000, 768)] {
        let vectors: Vec<Vector> = (0..n).map(|i| Vector::random(i as u64, dim)).collect();

        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(
            BenchmarkId::new(format!("n{}_d{}", n, dim), n),
            &vectors,
            |b, vecs| {
                b.iter(|| {
                    let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);
                    // Batch: add all at once
                    for v in vecs {
                        index.add(v.clone());
                    }
                    black_box(&index);
                })
            },
        );
    }
    group.finish();
}

criterion_group!(
    benches,
    benchmark_brute_force_insert,
    benchmark_hnsw_insert,
    benchmark_brute_force_search,
    benchmark_hnsw_search,
    benchmark_distance_metrics_bf,
    benchmark_concurrent_read_write,
    benchmark_batch_insert,
);
criterion_main!(benches);
