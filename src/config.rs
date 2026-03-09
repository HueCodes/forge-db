//! Configuration system for forge-db.
//!
//! Supports loading configuration from TOML files, environment variables,
//! or programmatic construction.
//!
//! # Example
//!
//! ```
//! use forge_db::config::ForgeConfig;
//!
//! // Load from TOML file
//! // let config = ForgeConfig::from_file("forge.toml").unwrap();
//!
//! // Or build programmatically
//! let config = ForgeConfig::default();
//! ```

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::error::{ForgeDbError, Result};

/// Top-level configuration for forge-db.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ForgeConfig {
    /// Directory where index files are persisted.
    pub data_dir: PathBuf,

    /// Directory for write-ahead log files.
    /// Defaults to `{data_dir}/wal` if not set.
    pub wal_dir: Option<PathBuf>,

    /// Global memory limit in bytes across all collections.
    /// Defaults to 8 GiB.
    pub max_memory_bytes: usize,

    /// Default configuration applied to newly created indexes.
    pub default_index: IndexConfig,

    /// Server configuration (used by forge-server).
    pub server: ServerConfig,

    /// Logging configuration.
    pub log: LogConfig,
}

impl Default for ForgeConfig {
    fn default() -> Self {
        Self {
            data_dir: PathBuf::from("./forge_data"),
            wal_dir: None,
            max_memory_bytes: 8 * 1024 * 1024 * 1024, // 8 GiB
            default_index: IndexConfig::default(),
            server: ServerConfig::default(),
            log: LogConfig::default(),
        }
    }
}

impl ForgeConfig {
    /// Load configuration from a TOML file.
    ///
    /// # Errors
    /// Returns an error if the file cannot be read or contains invalid TOML.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            ForgeDbError::Io(std::io::Error::new(
                e.kind(),
                format!("failed to read config file: {}", e),
            ))
        })?;
        Self::from_toml_str(&content)
    }

    /// Parse configuration from a TOML string.
    ///
    /// # Errors
    /// Returns an error if the string is not valid TOML or has type errors.
    pub fn from_toml_str(toml: &str) -> Result<Self> {
        toml::from_str(toml).map_err(|e| ForgeDbError::invalid_format(format!("invalid config TOML: {e}")))
    }

    /// Serialize configuration to a TOML string.
    pub fn to_toml_string(&self) -> String {
        toml::to_string_pretty(self).unwrap_or_default()
    }

    /// Save configuration to a TOML file.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let content = self.to_toml_string();
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Resolve the WAL directory, defaulting to `{data_dir}/wal`.
    pub fn wal_dir(&self) -> PathBuf {
        self.wal_dir
            .clone()
            .unwrap_or_else(|| self.data_dir.join("wal"))
    }

    /// Validate the configuration, returning errors for invalid settings.
    pub fn validate(&self) -> Result<()> {
        if self.max_memory_bytes == 0 {
            return Err(ForgeDbError::invalid_parameter("max_memory_bytes must be > 0"));
        }
        self.default_index.validate()?;
        self.server.validate()?;
        Ok(())
    }
}

/// Configuration for a vector index.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexConfig {
    /// Index type to use when automatically selecting.
    pub index_type: IndexType,

    /// IVF-specific configuration.
    pub ivf: IvfConfig,

    /// HNSW-specific configuration.
    pub hnsw: HnswConfig,

    /// Product Quantization configuration.
    pub pq: PqConfig,

    /// Maximum number of vectors this index can hold.
    pub max_vectors: usize,

    /// Maximum memory in bytes for this index (0 = use global limit).
    pub max_memory_bytes: usize,

    /// Whether to enable re-ranking for higher recall.
    pub enable_reranking: bool,

    /// Re-ranking factor: how many extra candidates to fetch.
    pub rerank_factor: usize,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            index_type: IndexType::IvfPq,
            ivf: IvfConfig::default(),
            hnsw: HnswConfig::default(),
            pq: PqConfig::default(),
            max_vectors: 100_000_000,
            max_memory_bytes: 0,
            enable_reranking: false,
            rerank_factor: 4,
        }
    }
}

impl IndexConfig {
    fn validate(&self) -> Result<()> {
        if self.max_vectors == 0 {
            return Err(ForgeDbError::invalid_parameter("max_vectors must be > 0"));
        }
        if self.rerank_factor == 0 {
            return Err(ForgeDbError::invalid_parameter("rerank_factor must be > 0"));
        }
        self.ivf.validate()?;
        self.hnsw.validate()?;
        self.pq.validate()?;
        Ok(())
    }
}

/// Which index type to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum IndexType {
    /// IVF with Product Quantization — best for large scale with memory constraints.
    #[default]
    IvfPq,
    /// HNSW graph index — best for lowest latency with high recall.
    Hnsw,
    /// IVF without compression — medium scale, full precision.
    Ivf,
    /// Brute force — small datasets or ground truth generation.
    BruteForce,
}

/// IVF (Inverted File) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IvfConfig {
    /// Number of IVF clusters (partitions).
    /// Rule of thumb: sqrt(n_vectors) is a good starting point.
    pub n_clusters: usize,

    /// Number of partitions to probe during search.
    /// Higher values increase recall at the cost of latency.
    pub nprobe: usize,

    /// Maximum k-means iterations for centroid training.
    pub max_iterations: usize,
}

impl Default for IvfConfig {
    fn default() -> Self {
        Self {
            n_clusters: 1024,
            nprobe: 16,
            max_iterations: 100,
        }
    }
}

impl IvfConfig {
    fn validate(&self) -> Result<()> {
        if self.n_clusters == 0 {
            return Err(ForgeDbError::invalid_parameter("n_clusters must be > 0"));
        }
        if self.nprobe == 0 {
            return Err(ForgeDbError::invalid_parameter("nprobe must be > 0"));
        }
        if self.nprobe > self.n_clusters {
            return Err(ForgeDbError::invalid_parameter(
                "nprobe must be <= n_clusters",
            ));
        }
        Ok(())
    }
}

/// HNSW (Hierarchical Navigable Small World) configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct HnswConfig {
    /// Maximum connections per node per layer (M parameter).
    pub m: usize,

    /// Beam width during index construction.
    /// Higher values produce better graphs but slower builds.
    pub ef_construction: usize,

    /// Beam width during search (ef parameter).
    /// Higher values increase recall at the cost of latency.
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

impl HnswConfig {
    fn validate(&self) -> Result<()> {
        if self.m == 0 {
            return Err(ForgeDbError::invalid_parameter("hnsw.m must be > 0"));
        }
        if self.ef_construction < self.m {
            return Err(ForgeDbError::invalid_parameter(
                "ef_construction must be >= m",
            ));
        }
        Ok(())
    }
}

/// Product Quantization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct PqConfig {
    /// Number of subvectors for PQ compression.
    /// Higher values improve recall but increase memory.
    pub n_subvectors: usize,

    /// Bits per code (8 = 256 centroids, 4 = 16 centroids).
    pub bits_per_code: u8,
}

impl Default for PqConfig {
    fn default() -> Self {
        Self {
            n_subvectors: 32,
            bits_per_code: 8,
        }
    }
}

impl PqConfig {
    fn validate(&self) -> Result<()> {
        if self.n_subvectors == 0 {
            return Err(ForgeDbError::invalid_parameter("pq.n_subvectors must be > 0"));
        }
        if self.bits_per_code != 4 && self.bits_per_code != 8 {
            return Err(ForgeDbError::invalid_parameter(
                "pq.bits_per_code must be 4 or 8",
            ));
        }
        Ok(())
    }
}

/// Server configuration (used by `forge-server`).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    /// gRPC server bind address.
    pub grpc_addr: String,

    /// HTTP/REST server bind address.
    pub http_addr: String,

    /// Maximum number of concurrent requests.
    pub max_concurrency: usize,

    /// Request timeout in milliseconds.
    pub request_timeout_ms: u64,

    /// TLS configuration (optional).
    pub tls: Option<TlsConfig>,

    /// API key authentication (optional).
    pub auth: Option<AuthConfig>,

    /// Rate limit in requests per second. 0 = disabled.
    pub requests_per_second: u64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            grpc_addr: "0.0.0.0:50051".to_string(),
            http_addr: "0.0.0.0:8080".to_string(),
            max_concurrency: 256,
            request_timeout_ms: 30_000,
            tls: None,
            auth: None,
            requests_per_second: 0,
        }
    }
}

impl ServerConfig {
    /// Get the request timeout as a Duration.
    pub fn request_timeout(&self) -> Duration {
        Duration::from_millis(self.request_timeout_ms)
    }

    fn validate(&self) -> Result<()> {
        if self.max_concurrency == 0 {
            return Err(ForgeDbError::invalid_parameter("max_concurrency must be > 0"));
        }
        Ok(())
    }
}

/// TLS configuration for server transport encryption.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Path to the TLS certificate file (PEM format).
    pub cert_path: PathBuf,

    /// Path to the TLS private key file (PEM format).
    pub key_path: PathBuf,

    /// Optional path to CA certificate for client verification.
    pub ca_cert_path: Option<PathBuf>,
}

/// API key authentication configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuthConfig {
    /// API keys allowed to access the server.
    /// Keys should be stored hashed in production.
    pub api_keys: Vec<String>,

    /// Whether to require authentication on all endpoints.
    pub required: bool,
}

/// Logging configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct LogConfig {
    /// Log level filter string (e.g., "info", "debug", "forge_db=trace,warn").
    pub level: String,

    /// Log format: "json" for production, "pretty" for development.
    pub format: LogFormat,

    /// Whether to include span traces in log output.
    pub include_spans: bool,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: "info".to_string(),
            format: LogFormat::Pretty,
            include_spans: false,
        }
    }
}

/// Log output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum LogFormat {
    /// Human-readable format (development).
    #[default]
    Pretty,
    /// JSON format (production, structured logging).
    Json,
    /// Compact single-line format.
    Compact,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_is_valid() {
        let config = ForgeConfig::default();
        config.validate().expect("default config should be valid");
    }

    #[test]
    fn test_config_toml_roundtrip() {
        let config = ForgeConfig::default();
        let toml_str = config.to_toml_string();
        let parsed = ForgeConfig::from_toml_str(&toml_str).expect("should parse back");
        assert_eq!(config.data_dir, parsed.data_dir);
        assert_eq!(config.max_memory_bytes, parsed.max_memory_bytes);
        assert_eq!(config.server.grpc_addr, parsed.server.grpc_addr);
    }

    #[test]
    fn test_config_from_toml_partial() {
        let toml = r#"
data_dir = "/var/lib/forge"
max_memory_bytes = 4294967296

[server]
grpc_addr = "127.0.0.1:50051"
http_addr = "127.0.0.1:8080"

[default_index]
index_type = "hnsw"
enable_reranking = true
"#;
        let config = ForgeConfig::from_toml_str(toml).expect("should parse");
        assert_eq!(config.data_dir, PathBuf::from("/var/lib/forge"));
        assert_eq!(config.max_memory_bytes, 4_294_967_296);
        assert_eq!(config.server.grpc_addr, "127.0.0.1:50051");
        assert_eq!(config.default_index.index_type, IndexType::Hnsw);
        assert!(config.default_index.enable_reranking);
    }

    #[test]
    fn test_config_validation_errors() {
        let config = ForgeConfig { max_memory_bytes: 0, ..ForgeConfig::default() };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_ivf_config_validation() {
        let mut ivf = IvfConfig { nprobe: 2000, n_clusters: 1024, ..IvfConfig::default() };
        assert!(ivf.validate().is_err(), "nprobe > n_clusters should error");

        ivf.nprobe = 16;
        assert!(ivf.validate().is_ok());
    }

    #[test]
    fn test_wal_dir_defaults_to_data_dir() {
        let config = ForgeConfig::default();
        let wal_dir = config.wal_dir();
        assert!(wal_dir.starts_with(&config.data_dir));
    }

    #[test]
    fn test_request_timeout() {
        let server = ServerConfig::default();
        assert_eq!(server.request_timeout(), Duration::from_millis(30_000));
    }
}
