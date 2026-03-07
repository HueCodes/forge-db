//! HNSW recall measurement example.
//!
//! Measures recall@1, recall@10, recall@100 and asserts minimum quality.
//!
//! Run with: cargo run --example hnsw_recall --release

use forge_db::dataset::{recall_at_k, Dataset};
use forge_db::distance::DistanceMetric;
use forge_db::index::hnsw::HNSWIndex;
use std::time::Instant;

fn main() {
    println!("Generating dataset...");
    let mut dataset = Dataset::generate(10000, 100, 128);

    println!("Computing ground truth...");
    dataset.compute_ground_truth(100);

    println!("Testing different configurations\n");

    for m in [16, 32] {
        println!("=== M={} ===", m);

        println!("Building index...");
        let build_start = Instant::now();
        let index = HNSWIndex::build(dataset.vectors.clone(), m, 200, DistanceMetric::Euclidean);
        let build_time = build_start.elapsed();
        let inserts_per_sec = dataset.vectors.len() as f64 / build_time.as_secs_f64();
        println!(
            "Built index with {} vectors in {:.2}s ({:.0} inserts/sec)",
            index.len(),
            build_time.as_secs_f64(),
            inserts_per_sec
        );

        for ef in [50, 100, 200, 400] {
            index.set_ef_search(ef);

            let start = Instant::now();

            let mut recall_1_sum = 0.0f32;
            let mut recall_10_sum = 0.0f32;
            let mut recall_100_sum = 0.0f32;

            for (i, query) in dataset.queries.iter().enumerate() {
                let results = index.search(&query.data, 100);
                let predicted: Vec<u64> = results.into_iter().map(|(id, _)| id).collect();

                recall_1_sum += recall_at_k(&predicted[..1.min(predicted.len())], &dataset.ground_truth[i], 1);
                recall_10_sum += recall_at_k(&predicted[..10.min(predicted.len())], &dataset.ground_truth[i], 10);
                recall_100_sum += recall_at_k(&predicted, &dataset.ground_truth[i], 100);
            }

            let n_queries = dataset.queries.len() as f32;
            let duration = start.elapsed();
            let qps = dataset.queries.len() as f64 / duration.as_secs_f64();

            let r1 = recall_1_sum / n_queries;
            let r10 = recall_10_sum / n_queries;
            let r100 = recall_100_sum / n_queries;

            println!(
                "ef={:3} | Recall@1: {:5.2}% | Recall@10: {:5.2}% | Recall@100: {:5.2}% | QPS: {:8.0}",
                ef,
                r1 * 100.0,
                r10 * 100.0,
                r100 * 100.0,
                qps
            );

            // Assert minimum recall quality with high ef
            if ef >= 200 {
                assert!(
                    r10 >= 0.90,
                    "Recall@10 ({:.2}%) should be >= 90% with ef={}, m={}",
                    r10 * 100.0,
                    ef,
                    m
                );
            }
        }

        println!();
    }

    println!("All recall assertions passed!");
}
