# forge-db

[![Crates.io](https://img.shields.io/crates/v/forge-db)](https://crates.io/crates/forge-db)
[![docs.rs](https://img.shields.io/docsrs/forge-db)](https://docs.rs/forge-db)
[![CI](https://github.com/HueCodes/forge-db/actions/workflows/ci.yml/badge.svg)](https://github.com/HueCodes/forge-db/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](LICENSE)

A high-performance vector database written in pure Rust with SIMD-accelerated
distance computation. forge-db provides multiple index types (IVF-PQ, HNSW, IVF,
BruteForce), runtime SIMD dispatch (AVX-512, AVX2, NEON, scalar fallback),
product quantization for 16-32x memory compression, and persistence with
checksum verification. It is designed for semantic search, recommendation
systems, and embedding-based retrieval at scale.

## Features

- **Multiple index types** -- IVF-PQ (compressed, large-scale), HNSW (low-latency graph), IVF (uncompressed partitions), BruteForce (exact search)
- **SIMD acceleration** -- Runtime detection: AVX-512 / AVX2+FMA / NEON / scalar
- **Product quantization** -- 8-bit codes with 16-32x memory compression
- **Five distance metrics** -- Euclidean, Euclidean squared, cosine, dot product, Manhattan
- **Metadata filtering** -- Attach key-value metadata to vectors and filter during search
- **Persistence** -- Save/load indexes to disk with CRC32 checksum verification
- **Write-ahead log** -- Durable insert/delete operations with replay and checkpointing
- **Thread safety** -- Concurrent reads with atomic parameter updates; parallel search via Rayon
- **Observability** -- Prometheus metrics endpoint, structured JSON logging, health checks
- **gRPC + REST** -- Dual-protocol server with TLS support
- **Python bindings** -- NumPy-native interface via PyO3

## Quick Start

Add the dependency:

```toml
[dependencies]
forge-db = "0.1"
```

Build an HNSW index, insert vectors, and search:

```rust
use forge_db::{DistanceMetric, HNSWIndex, Vector};

// Create an HNSW index (m=16, ef_construction=200, Euclidean distance).
let mut hnsw = HNSWIndex::new(16, 200, DistanceMetric::Euclidean);

// Insert vectors.
for i in 0..1_000u64 {
    hnsw.add(Vector::random(i, 128));
}
hnsw.finalize();

// Search for the 10 nearest neighbours.
let query = Vector::random(9_999, 128);
let results = hnsw.search(&query.data, 10);

for (id, dist) in &results {
    println!("id={id}  distance={dist:.4}");
}
```

Build an IVF-PQ index with the builder API for large-scale datasets:

```rust
use forge_db::{IVFPQIndexBuilder, Persistable, Vector};

let vectors: Vec<Vector> = (0..10_000u64)
    .map(|i| Vector::random(i, 128))
    .collect();

let index = IVFPQIndexBuilder::new()
    .vectors(vectors)
    .auto_tune(true)
    .build()
    .expect("IVF-PQ build failed");

index.set_nprobe(8);

let query = Vector::random(99_999, 128);
let results = index.search(&query.data, 10);

// Save to disk and reload.
index.save("my_index.fdb").expect("save failed");
let loaded = forge_db::IVFPQIndex::load("my_index.fdb").expect("load failed");
```

### Choosing an Index

| Index | Best for | Memory | Recall |
|-------|----------|--------|--------|
| `IVFPQIndex` | Large scale (1M+ vectors) | Low | Good |
| `HNSWIndex` | Low latency, high recall | High | Best |
| `IVFIndex` | Medium scale, no compression | Medium | Good |
| `BruteForceIndex` | Small datasets, ground truth | High | 100% |

## Server

### Docker

```bash
docker compose up -d
```

This starts three services:

- **forge-server** -- gRPC on port 50051, REST on port 8080, metrics on port 9090
- **Prometheus** -- scrapes metrics, available on port 9091
- **Grafana** -- dashboards on port 3000

### REST API

Create a collection, insert vectors, and search:

```bash
# Create a collection
curl -X POST http://localhost:8080/v1/collections \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "embeddings",
    "dimension": 128,
    "index_type": "hnsw",
    "distance_metric": "euclidean"
  }'

# Upsert vectors
curl -X POST http://localhost:8080/v1/collections/embeddings/vectors \
  -H 'Content-Type: application/json' \
  -d '{"vectors": [{"id": 1, "data": [0.1, 0.2, 0.3]}]}'

# Search
curl -X POST http://localhost:8080/v1/collections/embeddings/search \
  -H 'Content-Type: application/json' \
  -d '{"query": [0.15, 0.25, 0.35], "top_k": 10}'

# Health check
curl http://localhost:8080/health
```

Full endpoint list:

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |
| GET | `/v1/collections` | List collections |
| POST | `/v1/collections` | Create collection |
| GET | `/v1/collections/{name}` | Collection info |
| DELETE | `/v1/collections/{name}` | Drop collection |
| POST | `/v1/collections/{name}/vectors` | Upsert vectors |
| DELETE | `/v1/collections/{name}/vectors` | Delete vectors by ID |
| POST | `/v1/collections/{name}/search` | Search |
| POST | `/v1/collections/{name}/batch-search` | Batch search |
| POST | `/v1/collections/{name}/checkpoint` | Checkpoint WAL |
| POST | `/v1/collections/{name}/compact` | Compact collection |
| GET | `/v1/stats` | Global statistics |

## Python

Build and install from source (requires a Rust toolchain and [maturin](https://github.com/PyO3/maturin)):

```bash
pip install maturin
cd forge-py
maturin develop --release
```

Usage:

```python
import forge_py
import numpy as np

# Build an IVF-PQ index
vectors = np.random.rand(10_000, 128).astype(np.float32)
ids = list(range(10_000))

index = forge_py.IvfPqIndex(
    vectors=vectors,
    ids=ids,
    n_clusters=32,
    n_subvectors=8,
    metric="euclidean",
)

# Search
query = np.random.rand(128).astype(np.float32).tolist()
results = index.search(query, k=10)
for vid, distance in results:
    print(f"  id={vid} distance={distance:.4f}")

# Save / load
index.save("my_index.fdb")
loaded = forge_py.IvfPqIndex.load("my_index.fdb")
```

## Architecture

forge-db is a Cargo workspace with three crates:

- **`forge-db`** (library) -- Core vector database engine. Contains index
  implementations (IVF-PQ, HNSW, IVF, BruteForce), SIMD-optimized distance
  functions, product quantization, k-means clustering, metadata filtering,
  persistence, and the write-ahead log.

- **`forge-server`** (binary) -- Production server exposing forge-db over gRPC
  (tonic) and REST (axum). Includes collection management, authentication, TLS,
  concurrency limiting, request timeouts, Prometheus metrics, and structured
  logging.

- **`forge-py`** (cdylib) -- Python bindings via PyO3 with NumPy integration.
  Provides `IvfPqIndex`, `HnswIndex`, and `BruteForceIndex` classes that wrap
  the Rust implementations.

## Benchmarks

Run the full benchmark suite:

```bash
cargo bench
```

Individual benchmarks for specific components:

```bash
cargo bench --bench distance_bench   # SIMD distance functions
cargo bench --bench search_bench     # End-to-end search
cargo bench --bench hnsw_bench       # HNSW index operations
cargo bench --bench ivf_bench        # IVF partitioning
cargo bench --bench pq_bench         # Product quantization
cargo bench --bench e2e_bench        # Full pipeline
```

Representative results on SIFT1M (1 million 128-d vectors):

| Configuration | Latency | QPS | Recall@10 | Memory |
|---------------|---------|-----|-----------|--------|
| IVF-PQ nprobe=1 | 75 us | 13,333 | 33% | 31 MB |
| IVF-PQ nprobe=16 | 520 us | 1,923 | 69% | 31 MB |
| IVF-PQ nprobe=16 + rerank | 560 us | 1,786 | 91% | 31 MB |
| IVF-PQ nprobe=32 + rerank | 1.0 ms | 1,000 | 96% | 31 MB |
| HNSW ef=50 | <100 us | -- | 95% | 512 MB |

Uncompressed memory for 1M vectors: 488 MB. IVF-PQ compressed: 31 MB (16x reduction).

SIMD distance throughput improvement (128-d vectors): 3.2x faster vs scalar.

Build with profile-guided optimization for best throughput:

```bash
# Instrumented build
cargo build --profile release-instrumented
# Run a representative workload to generate profile data
# PGO-optimized build
RUSTFLAGS="-C profile-use=default.profdata" cargo build --profile release-pgo
```

## Configuration

The server reads a TOML configuration file. The default Docker configuration
lives at [`docker/forge.toml`](docker/forge.toml). Key options:

```toml
data_dir = "/var/lib/forge-db/data"
wal_dir  = "/var/lib/forge-db/wal"
max_memory_bytes = 8589934592  # 8 GiB

[server]
grpc_addr = "0.0.0.0:50051"
http_addr = "0.0.0.0:8080"
max_concurrency = 256
request_timeout_ms = 30000

[default_index]
index_type = "ivf_pq"       # ivf_pq | hnsw | ivf | brute_force

[default_index.hnsw]
m = 16
ef_construction = 200
ef_search = 50

[default_index.pq]
n_subvectors = 32
bits_per_code = 8

[log]
level = "info"              # trace | debug | info | warn | error
format = "json"             # json | pretty
```

Override the log level at runtime with the `RUST_LOG` environment variable:

```bash
RUST_LOG=info,forge_db=debug cargo run -p forge-server
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, coding standards,
and the pull request process.

## License

MIT -- see [LICENSE](LICENSE) for details.
