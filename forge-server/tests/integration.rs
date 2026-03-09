//! Integration tests for forge-server state and collection management.

use std::sync::Arc;

use forge_db::{
    config::ForgeConfig,
    distance::DistanceMetric,
    vector::Vector,
    MetadataValue,
};

use forge_server::collections::{Collection, CollectionMeta};
use forge_server::state::AppState;

fn test_meta(name: &str) -> CollectionMeta {
    CollectionMeta {
        name: name.to_string(),
        dimension: 4,
        index_type: "brute_force".to_string(),
        distance_metric: "euclidean".to_string(),
        n_clusters: 8,
        n_subvectors: 2,
        nprobe: 2,
        m: 16,
        ef_construction: 200,
        ef_search: 50,
        enable_reranking: false,
        rerank_factor: 4,
    }
}

fn sample_vectors(n: usize, dim: usize) -> Vec<Vector> {
    (0..n)
        .map(|i| Vector::new(i as u64, vec![i as f32; dim]))
        .collect()
}

#[tokio::test]
async fn test_create_and_list_collections() {
    let dir = tempfile::tempdir().unwrap();
    let config = ForgeConfig { data_dir: dir.path().to_path_buf(), ..ForgeConfig::default() };
    let state = Arc::new(AppState::new(config).await.unwrap());

    assert!(state.list_collections().is_empty());

    let meta = test_meta("test_coll");
    let coll = Collection::build_brute_force(meta, DistanceMetric::Euclidean);
    state.create_collection("test_coll", coll).unwrap();

    let names = state.list_collections();
    assert_eq!(names.len(), 1);
    assert_eq!(names[0], "test_coll");
}

#[tokio::test]
async fn test_duplicate_collection_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let config = ForgeConfig { data_dir: dir.path().to_path_buf(), ..ForgeConfig::default() };
    let state = Arc::new(AppState::new(config).await.unwrap());

    let meta = test_meta("dupe");
    let coll1 = Collection::build_brute_force(meta.clone(), DistanceMetric::Euclidean);
    let coll2 = Collection::build_brute_force(meta, DistanceMetric::Euclidean);

    state.create_collection("dupe", coll1).unwrap();
    let result = state.create_collection("dupe", coll2);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_drop_collection() {
    let dir = tempfile::tempdir().unwrap();
    let config = ForgeConfig { data_dir: dir.path().to_path_buf(), ..ForgeConfig::default() };
    let state = Arc::new(AppState::new(config).await.unwrap());

    let meta = test_meta("to_drop");
    let coll = Collection::build_brute_force(meta, DistanceMetric::Euclidean);
    state.create_collection("to_drop", coll).unwrap();

    assert!(state.drop_collection("to_drop"));
    assert!(!state.drop_collection("to_drop")); // second drop returns false
    assert!(state.list_collections().is_empty());
}

#[tokio::test]
async fn test_upsert_and_search() {
    let dir = tempfile::tempdir().unwrap();
    let config = ForgeConfig { data_dir: dir.path().to_path_buf(), ..ForgeConfig::default() };
    let state = Arc::new(AppState::new(config).await.unwrap());

    let meta = test_meta("vectors");
    let coll = Collection::build_brute_force(meta, DistanceMetric::Euclidean);
    state.create_collection("vectors", coll).unwrap();

    // Insert 10 vectors of dim=4
    let vecs = sample_vectors(10, 4);
    {
        let coll = state.get_collection("vectors").unwrap();
        coll.write().upsert_batch(vecs);
    }

    assert_eq!(state.total_vectors(), 10);

    // Search for the vector closest to [5.0, 5.0, 5.0, 5.0]
    let query = vec![5.0f32; 4];
    let coll = state.get_collection("vectors").unwrap();
    let (results, _) = coll.read().search(&query, 3);

    assert_eq!(results.len(), 3);
    // The closest should be id=5 (exact match)
    assert_eq!(results[0].0, 5);
    assert!((results[0].1).abs() < 1e-4);
}

#[tokio::test]
async fn test_stats_tracking() {
    let dir = tempfile::tempdir().unwrap();
    let config = ForgeConfig { data_dir: dir.path().to_path_buf(), ..ForgeConfig::default() };
    let state = Arc::new(AppState::new(config).await.unwrap());

    let meta = test_meta("stats_test");
    let coll = Collection::build_brute_force(meta, DistanceMetric::Euclidean);
    state.create_collection("stats_test", coll).unwrap();

    let vecs = sample_vectors(5, 4);
    {
        let coll = state.get_collection("stats_test").unwrap();
        coll.write().upsert_batch(vecs);
    }

    assert_eq!(state.total_vectors(), 5);
    // Memory estimate: 5 vectors × 4 dims × 4 bytes = 80 bytes
    assert!(state.total_memory_bytes() >= 80);
}

#[tokio::test]
async fn test_brute_force_persistence_roundtrip() {
    let dir = tempfile::tempdir().unwrap();
    let config = ForgeConfig { data_dir: dir.path().to_path_buf(), ..ForgeConfig::default() };
    let state = Arc::new(AppState::new(config).await.unwrap());

    // Create and populate a collection
    let meta = test_meta("persist_test");
    let coll = Collection::build_brute_force(meta, DistanceMetric::Euclidean);
    state.create_collection("persist_test", coll).unwrap();

    let vecs = sample_vectors(20, 4);
    {
        let c = state.get_collection("persist_test").unwrap();
        c.write().upsert_batch(vecs);
    }

    // Save to disk
    {
        let c = state.get_collection("persist_test").unwrap();
        c.read().save_to_disk(dir.path()).unwrap();
    }

    // Reload from disk
    let path = dir.path().join("persist_test.fdb");
    assert!(path.exists(), "saved file should exist");

    let loaded = Collection::load_from_disk("persist_test", &path).unwrap();
    assert_eq!(loaded.len(), 20);
    assert_eq!(loaded.meta.dimension, 4);
}

#[tokio::test]
async fn test_metadata_storage_and_retrieval() {
    let dir = tempfile::tempdir().unwrap();
    let config = ForgeConfig { data_dir: dir.path().to_path_buf(), ..ForgeConfig::default() };
    let state = Arc::new(AppState::new(config).await.unwrap());

    let meta = test_meta("meta_coll");
    let coll = Collection::build_brute_force(meta, DistanceMetric::Euclidean);
    state.create_collection("meta_coll", coll).unwrap();

    let vecs = sample_vectors(3, 4);
    let metadata_jsons = vec![
        Some(r#"{"category": "sports", "score": 9}"#.to_string()),
        Some(r#"{"category": "music", "score": 5}"#.to_string()),
        None,
    ];

    {
        let c = state.get_collection("meta_coll").unwrap();
        c.write().upsert_batch_with_metadata(vecs, metadata_jsons);
    }

    let c = state.get_collection("meta_coll").unwrap();
    let guard = c.read();

    // Check id=0 has "category" = "sports"
    let cat = guard.metadata.get_field(0, "category");
    assert_eq!(cat, Some(&MetadataValue::String("sports".to_string())));

    // Check id=1 has "score" = 5
    let score = guard.metadata.get_field(1, "score");
    assert_eq!(score, Some(&MetadataValue::Integer(5)));

    // id=2 has no metadata
    assert!(guard.metadata.get(2).is_none());
}

#[tokio::test]
async fn test_hnsw_upsert_and_search() {
    let dir = tempfile::tempdir().unwrap();
    let config = ForgeConfig { data_dir: dir.path().to_path_buf(), ..ForgeConfig::default() };
    let state = Arc::new(AppState::new(config).await.unwrap());

    let mut meta = test_meta("hnsw_coll");
    meta.index_type = "hnsw".to_string();
    meta.dimension = 8;
    let coll = Collection::build_hnsw("hnsw_coll".to_string(), meta, DistanceMetric::Euclidean);
    state.create_collection("hnsw_coll", coll).unwrap();

    // Insert 50 random-ish vectors
    let vecs: Vec<Vector> = (0u64..50)
        .map(|i| Vector::new(i, (0..8).map(|d| (i * 8 + d) as f32 / 10.0).collect()))
        .collect();

    {
        let c = state.get_collection("hnsw_coll").unwrap();
        c.write().upsert_batch(vecs);
    }

    assert_eq!(state.total_vectors(), 50);

    // Search — HNSW should be finalized and searchable
    let query = vec![2.5f32; 8];
    let c = state.get_collection("hnsw_coll").unwrap();
    let (results, _) = c.read().search(&query, 5);
    assert!(!results.is_empty());
    assert!(results.len() <= 5);
    // Results should be sorted by distance
    for w in results.windows(2) {
        assert!(w[0].1 <= w[1].1);
    }
}

#[tokio::test]
async fn test_ivfpq_build() {
    let dir = tempfile::tempdir().unwrap();
    let config = ForgeConfig { data_dir: dir.path().to_path_buf(), ..ForgeConfig::default() };
    let state = Arc::new(AppState::new(config).await.unwrap());

    let meta = test_meta("ivfpq_coll");
    let coll = Collection::build_brute_force(meta, DistanceMetric::Euclidean);
    state.create_collection("ivfpq_coll", coll).unwrap();

    // Build an IVF-PQ index from 512 vectors of dim=4
    let vecs: Vec<Vector> = (0u64..512)
        .map(|i| Vector::new(i, vec![(i % 16) as f32, (i % 8) as f32, (i % 4) as f32, (i % 2) as f32]))
        .collect();

    {
        let c = state.get_collection("ivfpq_coll").unwrap();
        let count = c.write().build_ivfpq(vecs, 0, 0, true).unwrap();
        assert_eq!(count, 512);
    }

    // Should now be an IVF-PQ index
    let c = state.get_collection("ivfpq_coll").unwrap();
    assert_eq!(c.read().len(), 512);

    // Search should return results
    let query = vec![3.0f32, 2.0, 1.0, 0.0];
    let (results, _) = c.read().search(&query, 5);
    assert!(!results.is_empty());
}
