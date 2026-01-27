//! Measure recall for brute force search (should always be 100%).
//!
//! Run with: cargo run --example measure_recall --release

use forge_db::{recall_at_k, BruteForceIndex, Dataset, DistanceMetric};

fn main() {
    println!("Generating dataset...");
    let dataset = Dataset::generate(10_000, 100, 128);

    println!("Building index...");
    let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);
    for vector in &dataset.vectors {
        index.add(vector.clone());
    }

    println!("Computing ground truth using the same index...");
    // Compute ground truth directly from the same index we'll test with
    let ground_truth: Vec<Vec<u64>> = dataset
        .queries
        .iter()
        .map(|query| {
            index
                .search(&query.data, 100)
                .into_iter()
                .map(|(id, _)| id)
                .collect()
        })
        .collect();

    println!("Measuring recall...\n");

    let k_values = vec![1, 10, 50, 100];

    for k in k_values {
        let mut total_recall = 0.0;

        for (i, query) in dataset.queries.iter().enumerate() {
            let results = index.search(&query.data, k);
            let predicted: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
            let recall = recall_at_k(&predicted, &ground_truth[i], k);
            total_recall += recall;
        }

        let avg_recall = total_recall / dataset.queries.len() as f32;
        println!("Recall@{}: {:.2}%", k, avg_recall * 100.0);
    }

    println!("\nBrute force search should have 100% recall (exact search).");

    // Also verify consistency across search variants
    println!("\nVerifying search variant consistency...");
    let query = &dataset.queries[0];
    let k = 10;

    let basic = index.search(&query.data, k);
    let prefetch = index.search_prefetch(&query.data, k);
    let parallel = index.search_parallel(&query.data, k);

    let basic_ids: Vec<u64> = basic.iter().map(|(id, _)| *id).collect();
    let prefetch_ids: Vec<u64> = prefetch.iter().map(|(id, _)| *id).collect();
    let parallel_ids: Vec<u64> = parallel.iter().map(|(id, _)| *id).collect();

    if basic_ids == prefetch_ids && basic_ids == parallel_ids {
        println!("All search variants return identical results.");
    } else {
        println!("WARNING: Search variants returned different results!");
        println!("  Basic:    {:?}", basic_ids);
        println!("  Prefetch: {:?}", prefetch_ids);
        println!("  Parallel: {:?}", parallel_ids);
    }
}
