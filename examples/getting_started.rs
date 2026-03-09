//! Getting Started with forge-db
//!
//! This example shows how to:
//! - Build an HNSW index and search it
//! - Build an IVF-PQ index with the builder API
//! - Save and load indexes from disk
//!
//! Run with: cargo run --example getting_started

use forge_db::{DistanceMetric, HNSWIndex, IVFPQIndexBuilder, Vector};

fn main() {
    // ─────────────────────────────────────────────
    // 1. HNSW Index
    // ─────────────────────────────────────────────
    println!("=== HNSW Index ===\n");

    // Create an HNSW index (m=16, ef_construction=200, Euclidean metric).
    let mut hnsw = HNSWIndex::new(16, 200, DistanceMetric::Euclidean);

    // Add 1 000 random 128-dimensional vectors.
    let dim = 128usize;
    let n = 1_000usize;

    println!("Inserting {} vectors of dimension {} ...", n, dim);
    for i in 0..n {
        let v = Vector::random(i as u64, dim);
        hnsw.add(v);
    }

    // Finalize builds the lock-free flat graph used during search.
    hnsw.finalize();

    println!("Index size: {} vectors", hnsw.len());

    // Search for the 10 nearest neighbours of a query vector.
    let query = Vector::random(9_999, dim);
    let results = hnsw.search(&query.data, 10);

    println!("Top-10 nearest neighbours (HNSW):");
    for (id, dist) in &results {
        println!("  id={id}  distance={dist:.4}");
    }

    // ─────────────────────────────────────────────
    // 2. IVF-PQ Index via Builder
    // ─────────────────────────────────────────────
    println!("\n=== IVF-PQ Index (builder API) ===\n");

    // Gather 2 000 training vectors.
    let train_vecs: Vec<Vector> = (0..2_000u64)
        .map(|i| Vector::random(i, dim))
        .collect();

    println!("Building IVF-PQ index with {} vectors (auto-tune=true) ...", train_vecs.len());

    let ivf_pq = IVFPQIndexBuilder::new()
        .vectors(train_vecs)
        .auto_tune(true)
        .build()
        .expect("IVF-PQ build failed");

    ivf_pq.set_nprobe(8);
    println!("Index size: {} vectors", ivf_pq.len());

    let results = ivf_pq.search(&query.data, 10);
    println!("Top-10 nearest neighbours (IVF-PQ):");
    for (id, dist) in &results {
        println!("  id={id}  distance={dist:.4}");
    }

    // ─────────────────────────────────────────────
    // 3. Save and load (IVF-PQ is persistable)
    // ─────────────────────────────────────────────
    println!("\n=== Save / Load ===\n");

    use forge_db::Persistable;
    let tmp = std::env::temp_dir().join("forge_getting_started.fdb");

    ivf_pq.save(&tmp).expect("save failed");
    println!("Saved index to {}", tmp.display());

    let loaded = forge_db::IVFPQIndex::load(&tmp).expect("load failed");
    println!("Loaded index — size: {} vectors", loaded.len());

    // Quick sanity check: results should be identical.
    let loaded_results = loaded.search(&query.data, 10);
    assert_eq!(results, loaded_results, "search results differ after load");
    println!("Save/load round-trip: OK");

    // Clean up temp file.
    let _ = std::fs::remove_file(&tmp);

    println!("\nDone.");
}
