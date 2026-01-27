# forge-db

High-performance vector database in Rust.

## Features

- IVF-PQ indexing with residual quantization
- SIMD-accelerated distance functions (AVX2/FMA)
- 16x memory compression via Product Quantization
- Optional re-ranking for higher recall
- Pure Rust, no external dependencies for core algorithm

## Benchmarks

SIFT1M dataset (1M vectors, 128 dimensions):

| Mode | Latency | QPS | Recall@10 | Memory |
|------|---------|-----|-----------|--------|
| nprobe=1 | 106 us | 9,452 | 33% | 31 MB |
| nprobe=16 | 548 us | 1,826 | 69% | 31 MB |
| nprobe=16 + rerank | 588 us | 1,700 | 91% | 31 MB |
| nprobe=32 + rerank | 1.05 ms | 951 | 96% | 31 MB |

Uncompressed memory: 488 MB. Compressed: 31 MB (16x reduction).

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
