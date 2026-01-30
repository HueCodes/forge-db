//! Benchmarks for distance function implementations.
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use forge_db::distance::{scalar, simd};
use forge_db::Vector;

fn benchmark_distances(c: &mut Criterion) {
    let dimensions = vec![64, 128, 256, 512, 768, 1024];

    for dim in &dimensions {
        let v1 = Vector::random(1, *dim);
        let v2 = Vector::random(2, *dim);

        let mut group = c.benchmark_group(format!("euclidean_{}", dim));

        group.bench_function("scalar", |b| {
            b.iter(|| scalar::euclidean_distance(black_box(&v1.data), black_box(&v2.data)))
        });

        group.bench_function("simd", |b| {
            b.iter(|| simd::euclidean_distance(black_box(&v1.data), black_box(&v2.data)))
        });

        group.finish();
    }

    // Throughput benchmark: 10K vectors at 128 dimensions
    let dim = 128;
    let vectors: Vec<Vector> = (0..10_000).map(|i| Vector::random(i, dim)).collect();
    let query = Vector::random(10_001, dim);

    c.bench_function("distance_throughput_10k_128d", |b| {
        b.iter(|| {
            let sum: f32 = vectors
                .iter()
                .map(|v| simd::euclidean_distance(black_box(&query.data), black_box(&v.data)))
                .sum();
            black_box(sum)
        })
    });

    // Additional benchmarks for squared distance and dot product
    let v1 = Vector::random(1, 128);
    let v2 = Vector::random(2, 128);

    let mut group = c.benchmark_group("squared_euclidean_128");
    group.bench_function("scalar", |b| {
        b.iter(|| scalar::euclidean_distance_squared(black_box(&v1.data), black_box(&v2.data)))
    });
    group.bench_function("simd", |b| {
        b.iter(|| simd::euclidean_distance_squared(black_box(&v1.data), black_box(&v2.data)))
    });
    group.finish();

    let mut group = c.benchmark_group("dot_product_128");
    group.bench_function("scalar", |b| {
        b.iter(|| scalar::dot_product(black_box(&v1.data), black_box(&v2.data)))
    });
    group.bench_function("simd", |b| {
        b.iter(|| simd::dot_product(black_box(&v1.data), black_box(&v2.data)))
    });
    group.finish();
}

criterion_group!(benches, benchmark_distances);
criterion_main!(benches);
