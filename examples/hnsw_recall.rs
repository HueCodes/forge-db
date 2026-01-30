//! HNSW recall measurement example.
//!
//! Measures recall@k at various ef_search settings to demonstrate
//! the speed/recall tradeoff.

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

    for m in vec![16, 32] {
        println!("=== M={} ===", m);

        println!("Building index...");
        let build_start = Instant::now();
        let mut index = HNSWIndex::new(m, 200, DistanceMetric::Euclidean);
        for vector in &dataset.vectors {
            index.add(vector.clone());
        }
        index.finalize();
        let build_time = build_start.elapsed();
        let inserts_per_sec = dataset.vectors.len() as f64 / build_time.as_secs_f64();
        println!(
            "Built index with {} vectors in {:.2}s ({:.0} inserts/sec)",
            index.len(),
            build_time.as_secs_f64(),
            inserts_per_sec
        );

        for ef in vec![50, 100, 200, 400] {
            index.set_ef_search(ef);

            let mut total_recall = 0.0;
            let start = Instant::now();

            for (i, query) in dataset.queries.iter().enumerate() {
                let results = index.search(&query.data, 10);
                let predicted: Vec<u64> = results.into_iter().map(|(id, _)| id).collect();
                let recall = recall_at_k(&predicted, &dataset.ground_truth[i], 10);
                total_recall += recall;
            }

            let duration = start.elapsed();
            let avg_recall = total_recall / dataset.queries.len() as f32;
            let qps = dataset.queries.len() as f64 / duration.as_secs_f64();

            println!(
                "ef={:3} | Recall@10: {:5.2}% | QPS: {:8.0}",
                ef,
                avg_recall * 100.0,
                qps
            );
        }

        println!();
    }
}
