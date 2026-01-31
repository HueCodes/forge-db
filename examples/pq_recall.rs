//! Recall measurement for Product Quantization and IVF-PQ.
//!
//! This example demonstrates the recall/performance tradeoffs of PQ and IVF-PQ
//! compared to exact brute-force search.

use forge_db::dataset::{recall_at_k, Dataset};
use forge_db::distance::DistanceMetric;
use forge_db::index::ivf_pq::IVFPQIndex;
use forge_db::pq::CompressedVectors;
use std::time::Instant;

fn main() {
    println!("=== Product Quantization Recall Measurement ===\n");

    // Generate dataset
    println!("Generating dataset: 10K vectors, 100 queries, 128 dimensions");
    let mut dataset = Dataset::generate(10_000, 100, 128);

    // Compute ground truth
    println!("Computing ground truth...\n");
    dataset.compute_ground_truth(100);

    // =====================
    // Pure PQ
    // =====================
    println!("=== Pure PQ (brute-force on compressed vectors) ===");

    let compressed = CompressedVectors::new(dataset.vectors.clone(), 8);

    let mut total_recall = 0.0;
    let start = Instant::now();

    for (i, query) in dataset.queries.iter().enumerate() {
        let results = compressed.search(&query.data, 10);
        let predicted: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        let recall = recall_at_k(&predicted, &dataset.ground_truth[i], 10);
        total_recall += recall;
    }

    let duration = start.elapsed();
    let avg_recall = total_recall / dataset.queries.len() as f32;
    let qps = dataset.queries.len() as f64 / duration.as_secs_f64();

    println!("Recall@10: {:.2}%", avg_recall * 100.0);
    println!("QPS: {:.0}", qps);
    println!(
        "Latency: {:.2} ms/query\n",
        duration.as_millis() as f64 / dataset.queries.len() as f64
    );

    // =====================
    // IVF-PQ
    // =====================
    println!("=== IVF-PQ (partitioned compressed search) ===");

    let mut index = IVFPQIndex::build(dataset.vectors.clone(), 256, 8, DistanceMetric::Euclidean);

    println!("\n{:>8} | {:>12} | {:>10}", "nprobe", "Recall@10", "QPS");
    println!("{}", "-".repeat(38));

    for nprobe in [1, 4, 16, 64] {
        index.set_nprobe(nprobe);

        let mut total_recall = 0.0;
        let start = Instant::now();

        for (i, query) in dataset.queries.iter().enumerate() {
            let results = index.search(&query.data, 10);
            let predicted: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
            let recall = recall_at_k(&predicted, &dataset.ground_truth[i], 10);
            total_recall += recall;
        }

        let duration = start.elapsed();
        let avg_recall = total_recall / dataset.queries.len() as f32;
        let qps = dataset.queries.len() as f64 / duration.as_secs_f64();

        println!(
            "{:>8} | {:>11.2}% | {:>10.0}",
            nprobe,
            avg_recall * 100.0,
            qps
        );
    }

    println!("\n=== Memory Comparison ===");
    let dim = 128;
    let n_vectors = dataset.vectors.len();
    let n_subvectors = 8;

    let uncompressed_bytes = n_vectors * dim * 4; // f32 = 4 bytes
    let compressed_bytes = n_vectors * n_subvectors; // 1 byte per subvector

    println!("Uncompressed: {} MB", uncompressed_bytes / (1024 * 1024));
    println!("Compressed:   {} MB", compressed_bytes / (1024 * 1024));
    println!(
        "Compression:  {:.1}x",
        uncompressed_bytes as f64 / compressed_bytes as f64
    );
}
