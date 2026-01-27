//! Benchmarks for brute force search variants.
//!
//! Run with: cargo bench --bench search_bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use forge_db::{BruteForceIndex, Dataset, DistanceMetric};

/// Benchmark different search implementations across dataset sizes.
fn benchmark_search_variants(c: &mut Criterion) {
    let sizes = vec![1_000, 10_000, 100_000];

    for size in sizes {
        let dataset = Dataset::generate(size, 100, 128);

        let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);
        for vector in &dataset.vectors {
            index.add(vector.clone());
        }

        let query = &dataset.queries[0];

        let mut group = c.benchmark_group(format!("search_{}", size));
        group.throughput(Throughput::Elements(1));

        group.bench_function("basic", |b| {
            b.iter(|| index.search(black_box(&query.data), black_box(10)))
        });

        group.bench_function("prefetch", |b| {
            b.iter(|| index.search_prefetch(black_box(&query.data), black_box(10)))
        });

        group.bench_function("parallel", |b| {
            b.iter(|| index.search_parallel(black_box(&query.data), black_box(10)))
        });

        group.finish();
    }
}

/// Benchmark batch search performance with many queries.
fn benchmark_batch_search(c: &mut Criterion) {
    let dataset = Dataset::generate(10_000, 1_000, 128);

    let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);
    for vector in &dataset.vectors {
        index.add(vector.clone());
    }

    let mut group = c.benchmark_group("batch_search");
    group.throughput(Throughput::Elements(dataset.queries.len() as u64));

    group.bench_function("batch_1000_queries", |b| {
        b.iter(|| index.batch_search(black_box(&dataset.queries), black_box(10)))
    });

    group.finish();
}

/// Benchmark to measure queries per second on a realistic workload.
fn benchmark_qps(c: &mut Criterion) {
    let dataset = Dataset::generate(100_000, 10_000, 128);

    let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);
    for vector in &dataset.vectors {
        index.add(vector.clone());
    }

    let queries = &dataset.queries;

    c.bench_function("qps_100k_vectors", |b| {
        let mut query_idx = 0usize;
        b.iter(|| {
            let query = &queries[query_idx % queries.len()];
            query_idx += 1;
            index.search_parallel(black_box(&query.data), black_box(10))
        })
    });
}

/// Benchmark different k values to understand scaling.
fn benchmark_k_values(c: &mut Criterion) {
    let dataset = Dataset::generate(10_000, 100, 128);

    let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);
    for vector in &dataset.vectors {
        index.add(vector.clone());
    }

    let query = &dataset.queries[0];

    let mut group = c.benchmark_group("k_values");
    group.throughput(Throughput::Elements(1));

    for k in [1, 10, 50, 100, 500] {
        group.bench_with_input(BenchmarkId::new("search", k), &k, |b, &k| {
            b.iter(|| index.search(black_box(&query.data), black_box(k)))
        });
    }

    group.finish();
}

/// Benchmark different distance metrics.
fn benchmark_distance_metrics(c: &mut Criterion) {
    let dataset = Dataset::generate(10_000, 100, 128);
    let query = &dataset.queries[0];

    let mut group = c.benchmark_group("distance_metrics");
    group.throughput(Throughput::Elements(1));

    for metric in [
        DistanceMetric::Euclidean,
        DistanceMetric::EuclideanSquared,
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
    ] {
        let mut index = BruteForceIndex::new(metric);
        for vector in &dataset.vectors {
            index.add(vector.clone());
        }

        group.bench_with_input(BenchmarkId::new("metric", format!("{:?}", metric)), &index, |b, idx| {
            b.iter(|| idx.search(black_box(&query.data), black_box(10)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_search_variants,
    benchmark_batch_search,
    benchmark_qps,
    benchmark_k_values,
    benchmark_distance_metrics,
);

criterion_main!(benches);
