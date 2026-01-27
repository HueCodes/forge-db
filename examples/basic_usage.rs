//! Basic usage example demonstrating the vector-db distance functions.
//!
//! Run with: cargo run --example basic_usage

use forge_db::{DistanceMetric, Vector};

fn main() {
    println!("=== Vector Database Distance Functions Demo ===\n");

    // Create some random vectors
    let dim = 128;
    let v1 = Vector::random(1, dim);
    let v2 = Vector::random(2, dim);

    println!("Created two random {}-dimensional vectors\n", dim);

    // Demonstrate different distance metrics
    let metrics = [
        DistanceMetric::Euclidean,
        DistanceMetric::EuclideanSquared,
        DistanceMetric::Cosine,
        DistanceMetric::DotProduct,
    ];

    for metric in &metrics {
        let distance = metric.compute(&v1.data, &v2.data);
        println!("{:?} distance: {:.6}", metric, distance);
    }

    println!("\n=== Self-Distance Test ===\n");

    // Distance to self should be 0 (or -norm^2 for dot product)
    for metric in &metrics {
        let distance = metric.compute(&v1.data, &v1.data);
        println!("{:?} distance to self: {:.6}", metric, distance);
    }

    println!("\n=== Performance Demo ===\n");

    // Quick throughput estimate
    let num_vectors = 10_000;
    let vectors: Vec<Vector> = (0..num_vectors).map(|i| Vector::random(i, dim)).collect();
    let query = Vector::random(num_vectors as u64, dim);

    let start = std::time::Instant::now();
    let _sum: f32 = vectors
        .iter()
        .map(|v| DistanceMetric::Euclidean.compute(&query.data, &v.data))
        .sum();
    let elapsed = start.elapsed();

    let throughput = num_vectors as f64 / elapsed.as_secs_f64();
    println!(
        "Computed {} distances in {:.2?}",
        num_vectors, elapsed
    );
    println!("Throughput: {:.0} distances/second", throughput);

    // Check if SIMD is being used
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            println!("\nSIMD: Using AVX2 + FMA acceleration");
        } else {
            println!("\nSIMD: Falling back to scalar (AVX2/FMA not available)");
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("\nSIMD: Using scalar implementation (non-x86_64 platform)");
    }
}
