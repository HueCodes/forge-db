//! Shared application state for the forge-server.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use anyhow::Context;
use parking_lot::{Mutex, RwLock};
use tracing::{info, warn};

use forge_db::{
    config::ForgeConfig,
    WriteAheadLog,
};

use crate::collections::Collection;

/// Global server statistics.
pub struct ServerStats {
    pub total_searches: AtomicU64,
    pub total_upserts: AtomicU64,
    pub total_deletes: AtomicU64,
    pub search_latency_sum_us: AtomicU64,
    pub start_time: SystemTime,
}

impl Default for ServerStats {
    fn default() -> Self {
        Self {
            total_searches: AtomicU64::new(0),
            total_upserts: AtomicU64::new(0),
            total_deletes: AtomicU64::new(0),
            search_latency_sum_us: AtomicU64::new(0),
            start_time: SystemTime::now(),
        }
    }
}

impl ServerStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn uptime_seconds(&self) -> u64 {
        self.start_time
            .elapsed()
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    pub fn avg_search_latency_ms(&self) -> f32 {
        let searches = self.total_searches.load(Ordering::Relaxed);
        if searches == 0 {
            return 0.0;
        }
        let sum_us = self.search_latency_sum_us.load(Ordering::Relaxed);
        (sum_us as f64 / searches as f64 / 1000.0) as f32
    }
}

/// Shared application state, accessible from all request handlers.
pub struct AppState {
    /// All active collections, keyed by name.
    pub collections: RwLock<HashMap<String, Arc<RwLock<Collection>>>>,
    /// Server configuration.
    #[allow(dead_code)]
    pub config: ForgeConfig,
    /// Global server statistics.
    pub stats: Arc<ServerStats>,
    /// Write-ahead log for the server.
    pub wal: Arc<Mutex<WriteAheadLog>>,
}

impl AppState {
    /// Create a new AppState, loading any persisted collections from disk.
    pub async fn new(config: ForgeConfig) -> anyhow::Result<Self> {
        // Open (or create) the WAL
        let wal_dir = config.wal_dir();
        std::fs::create_dir_all(&wal_dir)
            .with_context(|| format!("creating WAL dir {}", wal_dir.display()))?;

        let wal = WriteAheadLog::open(&wal_dir)
            .with_context(|| format!("opening WAL at {}", wal_dir.display()))?;

        let mut collections: HashMap<String, Arc<RwLock<Collection>>> = HashMap::new();

        // Load any persisted collections
        let data_dir = config.data_dir.clone();
        if data_dir.exists() {
            if let Ok(entries) = std::fs::read_dir(&data_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|e| e.to_str()) == Some("fdb") {
                        if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                            match Collection::load_from_disk(name, &path) {
                                Ok(collection) => {
                                    info!(name, path = %path.display(), "loaded collection from disk");
                                    collections.insert(
                                        name.to_string(),
                                        Arc::new(RwLock::new(collection)),
                                    );
                                }
                                Err(e) => {
                                    warn!(name, error = %e, "failed to load collection, skipping");
                                }
                            }
                        }
                    }
                }
            }
        }

        // Replay WAL entries (best-effort — a richer WAL would include collection name)
        let pending = wal.replay_all().with_context(|| "replaying WAL")?;
        if !pending.is_empty() {
            info!(count = pending.len(), "replaying WAL entries (informational)");
        }

        Ok(Self {
            collections: RwLock::new(collections),
            config,
            stats: Arc::new(ServerStats::new()),
            wal: Arc::new(Mutex::new(wal)),
        })
    }

    /// Get a collection by name.
    pub fn get_collection(&self, name: &str) -> Option<Arc<RwLock<Collection>>> {
        self.collections.read().get(name).cloned()
    }

    /// Create a new collection.
    pub fn create_collection(
        &self,
        name: impl Into<String>,
        collection: Collection,
    ) -> Result<(), String> {
        let name = name.into();
        let mut collections = self.collections.write();
        if collections.contains_key(&name) {
            return Err(format!("collection '{name}' already exists"));
        }
        collections.insert(name, Arc::new(RwLock::new(collection)));
        Ok(())
    }

    /// Drop a collection by name.
    pub fn drop_collection(&self, name: &str) -> bool {
        let mut collections = self.collections.write();
        collections.remove(name).is_some()
    }

    /// List all collection names.
    pub fn list_collections(&self) -> Vec<String> {
        self.collections.read().keys().cloned().collect()
    }

    /// Total number of vectors across all collections.
    pub fn total_vectors(&self) -> u64 {
        self.collections
            .read()
            .values()
            .map(|c| c.read().len() as u64)
            .sum()
    }

    /// Total memory usage estimate.
    pub fn total_memory_bytes(&self) -> u64 {
        self.collections
            .read()
            .values()
            .map(|c| c.read().memory_bytes() as u64)
            .sum()
    }
}
