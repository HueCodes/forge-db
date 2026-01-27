//! IVF Index recall and performance evaluation.
//!
//! Measures recall@k and queries per second across different configurations.

use std::time::Instant;
use forge_db::dataset::{recall_at_k, Dataset};
use forge_db::distance::DistanceMetric;
use forge_db::index::ivf::IVFIndex;

fn main() {
    println!("=== IVF Index Recall Evaluation ===\n");

    // Generate dataset
    println!("Generating dataset...");
    let mut dataset = Dataset::generate(10_000, 100, 128);

    println!("Computing ground truth...");
    dataset.compute_ground_truth(100);

    println!("Testing different configurations\n");

    // Test different cluster counts
    for n_clusters in [100, 256, 512] {
        println!("==============================");
        println!("=== {} clusters ===", n_clusters);
        println!("==============================");

        let mut index = IVFIndex::build(
            dataset.vectors.clone(),
            n_clusters,
            DistanceMetric::Euclidean,
        );

        println!();
        println!(
            "{:>8} | {:>12} | {:>12}",
            "nprobe", "Recall@10", "QPS"
        );
        println!("{}", "-".repeat(40));

        for nprobe in [1, 4, 16, 32, 64] {
            if nprobe > n_clusters {
                continue;
            }

            index.set_nprobe(nprobe);

            let mut total_recall = 0.0f32;
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
                "{:>8} | {:>11.2}% | {:>12.0}",
                nprobe,
                avg_recall * 100.0,
                qps
            );
        }

        println!();
    }

    // Large scale test
    println!("==============================");
    println!("=== Large Scale Test (100K vectors) ===");
    println!("==============================\n");

    println!("Generating large dataset...");
    let mut large_dataset = Dataset::generate(100_000, 100, 128);

    println!("Computing ground truth (this may take a while)...");
    large_dataset.compute_ground_truth(100);

    let mut index = IVFIndex::build(
        large_dataset.vectors.clone(),
        1024,
        DistanceMetric::Euclidean,
    );

    println!();
    println!(
        "{:>8} | {:>12} | {:>12}",
        "nprobe", "Recall@10", "QPS"
    );
    println!("{}", "-".repeat(40));

    for nprobe in [1, 4, 16, 64, 128] {
        index.set_nprobe(nprobe);

        let mut total_recall = 0.0f32;
        let start = Instant::now();

        for (i, query) in large_dataset.queries.iter().enumerate() {
            let results = index.search(&query.data, 10);
            let predicted: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
            let recall = recall_at_k(&predicted, &large_dataset.ground_truth[i], 10);
            total_recall += recall;
        }

        let duration = start.elapsed();
        let avg_recall = total_recall / large_dataset.queries.len() as f32;
        let qps = large_dataset.queries.len() as f64 / duration.as_secs_f64();

        println!(
            "{:>8} | {:>11.2}% | {:>12.0}",
            nprobe,
            avg_recall * 100.0,
            qps
        );
    }

    println!("\n=== Evaluation Complete ===");
}
