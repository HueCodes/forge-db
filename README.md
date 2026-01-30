# forge-db

## Features

- IVF-PQ indexing with residual quantization
- SIMD-accelerated distance functions (AVX2/AVX-512 with runtime detection)
- 16x memory compression via Product Quantization (8-bit) or 32x with 4-bit PQ
- Optional re-ranking for higher recall
- Batch query optimization with partition-centric processing
- Lock-free HNSW search after index finalization
- Profile-guided optimization (PGO) build support
- Pure Rust, no external dependencies for core algorithm

## Benchmarks

SIFT1M dataset (1M vectors, 128 dimensions):

| Mode | Latency | QPS | Recall@10 | Memory |
|------|---------|-----|-----------|--------|
| nprobe=1 | 75 µs | 13,333 | 33% | 31 MB |
| nprobe=16 | 520 µs | 1,923 | 69% | 31 MB |
| nprobe=16 + rerank | 560 µs | 1,786 | 91% | 31 MB |
| nprobe=32 + rerank | 1.0 ms | 1,000 | 96% | 31 MB |

Uncompressed memory: 488 MB. Compressed: 31 MB (16x reduction).

SIMD distance improvements (128 dimensions): **3.2x faster** vs scalar.

## Usage

```rust
use forge_db::{Vector, IVFPQIndex, DistanceMetric};

// Build index
let vectors: Vec<Vector> = load_vectors();
let mut index = IVFPQIndex::build(
    vectors.clone(),
    1024,  // clusters
    32,    // subvectors (compression)
    DistanceMetric::Euclidean,
);

// Optional: enable re-ranking for higher recall
index.enable_reranking(vectors, 4);

// Search
index.set_nprobe(16);
let results = index.search(&query, 10);
```

## Build

```
cargo build --release
cargo test
cargo bench --bench pq_bench
```

For maximum performance, use profile-guided optimization:

```
./pgo-build.sh
```

## Run SIFT1M benchmark

```
# Download dataset
curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xzf sift.tar.gz
mkdir -p data && mv sift data/sift1m

# Run benchmark
cargo run --release --example sift1m_benchmark
```

## License

MIT
