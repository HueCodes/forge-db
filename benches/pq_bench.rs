//! Benchmarks for Product Quantization and IVF-PQ index.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use forge_db::dataset::Dataset;
use forge_db::distance::DistanceMetric;
use forge_db::index::ivf_pq::IVFPQIndex;

fn benchmark_sift1m(c: &mut Criterion) {
    // Load SIFT1M dataset
    println!("Loading SIFT1M dataset...");
    let dataset = match Dataset::load_sift1m("data/sift1m") {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to load SIFT1M: {}. Using synthetic data.", e);
            Dataset::generate(100_000, 1000, 128)
        }
    };

    let n_vectors = dataset.vectors.len();
    println!("Loaded {} vectors", n_vectors);

    // Build IVF-PQ index: 1024 clusters, 32 subvectors
    println!("Building IVF-PQ index...");
    let mut index = IVFPQIndex::build(
        dataset.vectors.clone(),
        1024,
        32,
        DistanceMetric::Euclidean,
    );

    let queries = &dataset.queries;

    // Benchmark without re-ranking
    let mut group = c.benchmark_group("ivf_pq");
    group.sample_size(500);

    for nprobe in [1, 2, 4, 8, 16, 32] {
        index.set_nprobe(nprobe);
        let mut query_idx = 0;

        group.bench_with_input(
            BenchmarkId::new("nprobe", nprobe),
            &nprobe,
            |b, _| {
                b.iter(|| {
                    let query = &queries[query_idx % queries.len()];
                    query_idx += 1;
                    black_box(index.search(&query.data, 10))
                })
            },
        );
    }
    group.finish();

    // Benchmark with re-ranking
    println!("Enabling re-ranking...");
    index.enable_reranking(dataset.vectors.clone(), 4);

    let mut group = c.benchmark_group("ivf_pq_rerank");
    group.sample_size(500);

    for nprobe in [1, 2, 4, 8, 16, 32] {
        index.set_nprobe(nprobe);
        let mut query_idx = 0;

        group.bench_with_input(
            BenchmarkId::new("nprobe", nprobe),
            &nprobe,
            |b, _| {
                b.iter(|| {
                    let query = &queries[query_idx % queries.len()];
                    query_idx += 1;
                    black_box(index.search(&query.data, 10))
                })
            },
        );
    }
    group.finish();
}

criterion_group!(benches, benchmark_sift1m);
criterion_main!(benches);
