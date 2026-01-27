//! SIFT1M Benchmark for Product Quantization.
//!
//! Tests PQ and IVF-PQ on the industry-standard SIFT1M dataset
//! (1 million 128-dimensional SIFT descriptors).
//!
//! Download dataset from: http://corpus-texmex.irisa.fr/
//! Extract to: data/sift1m/

use forge_db::dataset::{recall_at_k, Dataset};
use forge_db::distance::DistanceMetric;
use forge_db::index::ivf_pq::IVFPQIndex;
use forge_db::pq::CompressedVectors;
use std::time::Instant;

fn main() {
    println!("SIFT1M Benchmark: 1M vectors, 128 dims, 10K queries\n");

    // Load SIFT1M dataset
    println!("Loading SIFT1M dataset...");
    let start = Instant::now();
    let dataset = match Dataset::load_sift1m("data/sift1m") {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Error loading SIFT1M: {}", e);
            eprintln!("\nPlease download the dataset:");
            eprintln!("  curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz");
            eprintln!("  tar -xzf sift.tar.gz");
            eprintln!("  mv sift data/sift1m");
            return;
        }
    };
    println!(
        "Loaded {} vectors, {} queries in {:.2}s\n",
        dataset.vectors.len(),
        dataset.queries.len(),
        start.elapsed().as_secs_f64()
    );

    let k = 10;
    let n_queries = 1000; // Use subset for faster iteration

    // =====================================================================
    // Memory stats
    // =====================================================================
    let dim = 128;
    let n_vectors = dataset.vectors.len();
    let n_subvectors = 32; // 32 subvectors = 16x compression, higher recall
    let uncompressed_mb = (n_vectors * dim * 4) as f64 / (1024.0 * 1024.0);
    let compressed_mb = (n_vectors * n_subvectors) as f64 / (1024.0 * 1024.0);

    println!("═══════════════════════════════════════════════════════════════");
    println!("                      MEMORY COMPARISON");
    println!("═══════════════════════════════════════════════════════════════");
    println!("  Uncompressed: {:>8.1} MB  (128 × f32 per vector)", uncompressed_mb);
    println!("  Compressed:   {:>8.1} MB  ({} bytes per vector)", compressed_mb, n_subvectors);
    println!("  Compression:  {:>8.1}x", uncompressed_mb / compressed_mb);
    println!();

    // =====================================================================
    // Pure PQ on subset (100K) for quick test
    // =====================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("              PURE PQ (brute-force on 100K subset)");
    println!("═══════════════════════════════════════════════════════════════");

    let subset: Vec<_> = dataset.vectors.iter().take(100_000).cloned().collect();
    println!("Building PQ index on 100K vectors...");
    let start = Instant::now();
    let compressed = CompressedVectors::new(subset, n_subvectors);
    println!("Build time: {:.2}s\n", start.elapsed().as_secs_f64());

    let mut total_recall = 0.0;
    let start = Instant::now();
    for i in 0..n_queries {
        let query = &dataset.queries[i];
        let results = compressed.search(&query.data, k);
        let predicted: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
        let recall = recall_at_k(&predicted, &dataset.ground_truth[i], k);
        total_recall += recall;
    }
    let duration = start.elapsed();
    let avg_recall = total_recall / n_queries as f32;
    let qps = n_queries as f64 / duration.as_secs_f64();

    println!("  Recall@{}: {:>6.2}%", k, avg_recall * 100.0);
    println!("  QPS:       {:>6.0}", qps);
    println!("  Latency:   {:>6.2} ms/query", duration.as_millis() as f64 / n_queries as f64);
    println!();

    // =====================================================================
    // IVF-PQ on full 1M dataset
    // =====================================================================
    println!("═══════════════════════════════════════════════════════════════");
    println!("                   IVF-PQ (full 1M vectors)");
    println!("═══════════════════════════════════════════════════════════════");

    let n_clusters = 1024; // Optimal balance: fast centroid search + small partitions
    println!("Building IVF-PQ index ({} clusters, {} subvectors)...", n_clusters, n_subvectors);
    let start = Instant::now();
    let mut index = IVFPQIndex::build(
        dataset.vectors.clone(),
        n_clusters,
        n_subvectors,
        DistanceMetric::Euclidean,
    );
    println!("Build time: {:.2}s\n", start.elapsed().as_secs_f64());

    println!("{:>8} │ {:>10} │ {:>12} │ {:>10}", "nprobe", "Recall@10", "QPS", "Latency");
    println!("─────────┼────────────┼──────────────┼───────────");

    for nprobe in [1, 2, 4, 8, 16, 32, 64, 128] {
        index.set_nprobe(nprobe);

        let mut total_recall = 0.0;
        let start = Instant::now();

        for i in 0..n_queries {
            let query = &dataset.queries[i];
            let results = index.search(&query.data, k);
            let predicted: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
            let recall = recall_at_k(&predicted, &dataset.ground_truth[i], k);
            total_recall += recall;
        }

        let duration = start.elapsed();
        let avg_recall = total_recall / n_queries as f32;
        let qps = n_queries as f64 / duration.as_secs_f64();
        let latency_ms = duration.as_millis() as f64 / n_queries as f64;

        println!(
            "{:>8} │ {:>9.2}% │ {:>12.0} │ {:>8.2} ms",
            nprobe,
            avg_recall * 100.0,
            qps,
            latency_ms
        );
    }

    // =====================================================================
    // IVF-PQ with Re-ranking (higher recall)
    // =====================================================================
    println!("\n═══════════════════════════════════════════════════════════════");
    println!("           IVF-PQ + Re-ranking (4x candidates)");
    println!("═══════════════════════════════════════════════════════════════");

    index.enable_reranking(dataset.vectors.clone(), 4);

    println!("{:>8} │ {:>10} │ {:>12} │ {:>10}", "nprobe", "Recall@10", "QPS", "Latency");
    println!("─────────┼────────────┼──────────────┼───────────");

    for nprobe in [1, 2, 4, 8, 16, 32] {
        index.set_nprobe(nprobe);

        let mut total_recall = 0.0;
        let start = Instant::now();

        for i in 0..n_queries {
            let query = &dataset.queries[i];
            let results = index.search(&query.data, k);
            let predicted: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
            let recall = recall_at_k(&predicted, &dataset.ground_truth[i], k);
            total_recall += recall;
        }

        let duration = start.elapsed();
        let avg_recall = total_recall / n_queries as f32;
        let qps = n_queries as f64 / duration.as_secs_f64();
        let latency_ms = duration.as_millis() as f64 / n_queries as f64;

        println!(
            "{:>8} │ {:>9.2}% │ {:>12.0} │ {:>8.2} ms",
            nprobe,
            avg_recall * 100.0,
            qps,
            latency_ms
        );
    }

    println!("\n[SUMMARY]");
    println!("  Dataset:     SIFT1M (1M vectors, 128 dims)");
    println!("  Compression: {:.0}x ({:.0}MB to {:.0}MB)", uncompressed_mb / compressed_mb, uncompressed_mb, compressed_mb);
    println!("  Pure Rust implementation");
    println!();
}
