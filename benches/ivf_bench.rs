//! Benchmarks for IVF index build and search performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use forge_db::dataset::Dataset;
use forge_db::distance::DistanceMetric;
use forge_db::index::ivf::IVFIndex;

/// Benchmark IVF index construction with different cluster counts.
fn benchmark_ivf_build(c: &mut Criterion) {
    let sizes = vec![10_000, 100_000];

    for size in sizes {
        let dataset = Dataset::generate(size, 100, 128);

        let mut group = c.benchmark_group(format!("ivf_build_{}", size));
        group.sample_size(10); // Reduce samples for slow builds

        for n_clusters in [100, 256, 1024] {
            // Skip if too many clusters for dataset size
            if n_clusters > size / 10 {
                continue;
            }

            group.bench_with_input(
                BenchmarkId::from_parameter(n_clusters),
                &n_clusters,
                |b, &n_clusters| {
                    b.iter(|| {
                        black_box(IVFIndex::build(
                            dataset.vectors.clone(),
                            n_clusters,
                            DistanceMetric::Euclidean,
                        ))
                    })
                },
            );
        }

        group.finish();
    }
}

/// Benchmark IVF search with different nprobe values.
fn benchmark_ivf_search(c: &mut Criterion) {
    let dataset = Dataset::generate(100_000, 1000, 128);

    for n_clusters in [256, 1024] {
        let mut index = IVFIndex::build(
            dataset.vectors.clone(),
            n_clusters,
            DistanceMetric::Euclidean,
        );

        let mut group = c.benchmark_group(format!("ivf_search_{}_clusters", n_clusters));

        for nprobe in [1, 4, 16, 64] {
            if nprobe > n_clusters {
                continue;
            }

            index.set_nprobe(nprobe);

            let mut query_idx = 0usize;

            group.bench_with_input(BenchmarkId::from_parameter(nprobe), &nprobe, |b, _| {
                b.iter(|| {
                    let query = &dataset.queries[query_idx % dataset.queries.len()];
                    query_idx += 1;
                    black_box(index.search(&query.data, 10))
                })
            });
        }

        group.finish();
    }
}

/// Benchmark batch search throughput.
fn benchmark_ivf_batch_search(c: &mut Criterion) {
    let dataset = Dataset::generate(100_000, 100, 128);
    let mut index = IVFIndex::build(dataset.vectors.clone(), 256, DistanceMetric::Euclidean);

    let mut group = c.benchmark_group("ivf_batch_search");
    group.sample_size(20);

    for nprobe in [1, 4, 16] {
        index.set_nprobe(nprobe);

        group.bench_with_input(
            BenchmarkId::from_parameter(format!("nprobe_{}_100q", nprobe)),
            &nprobe,
            |b, _| b.iter(|| black_box(index.batch_search(&dataset.queries, 10))),
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    benchmark_ivf_build,
    benchmark_ivf_search,
    benchmark_ivf_batch_search
);
criterion_main!(benches);
