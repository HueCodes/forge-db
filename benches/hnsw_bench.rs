//! HNSW index benchmarks.
//!
//! Measures build time and search throughput for various configurations.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use forge_db::dataset::Dataset;
use forge_db::distance::DistanceMetric;
use forge_db::index::hnsw::HNSWIndex;

fn benchmark_hnsw_build(c: &mut Criterion) {
    let sizes = vec![1000, 10000];

    for size in sizes {
        let dataset = Dataset::generate(size, 100, 128);

        let mut group = c.benchmark_group(format!("hnsw_build_{}", size));

        for m in vec![16, 32] {
            group.bench_with_input(BenchmarkId::from_parameter(m), &m, |b, &m| {
                b.iter(|| {
                    let mut index = HNSWIndex::new(m, 200, DistanceMetric::Euclidean);
                    for vector in &dataset.vectors {
                        index.add(vector.clone());
                    }
                    index.finalize();
                    black_box(index)
                });
            });
        }

        group.finish();
    }
}

fn benchmark_hnsw_search(c: &mut Criterion) {
    let dataset = Dataset::generate(10000, 1000, 128);

    for m in vec![16, 32] {
        let mut index = HNSWIndex::new(m, 200, DistanceMetric::Euclidean);

        println!("Building HNSW index with M={}", m);
        for vector in &dataset.vectors {
            index.add(vector.clone());
        }
        index.finalize();

        let mut group = c.benchmark_group(format!("hnsw_search_m{}", m));

        for ef in vec![50, 100, 200, 400] {
            index.set_ef_search(ef);

            group.bench_with_input(BenchmarkId::from_parameter(ef), &ef, |b, _| {
                let mut query_idx = 0;
                b.iter(|| {
                    let query = &dataset.queries[query_idx % dataset.queries.len()];
                    query_idx += 1;
                    black_box(index.search(&query.data, 10))
                });
            });
        }

        group.finish();
    }
}

criterion_group!(benches, benchmark_hnsw_build, benchmark_hnsw_search);
criterion_main!(benches);
