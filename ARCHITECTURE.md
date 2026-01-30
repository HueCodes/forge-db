# forge-db Architecture Guide

A comprehensive technical guide to the forge-db vector database library, covering architecture, algorithms, design decisions, and interview preparation material.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture Layers](#2-architecture-layers)
3. [Design Patterns & Decisions](#3-design-patterns--decisions)
4. [Algorithm Deep-Dives](#4-algorithm-deep-dives)
5. [Memory Optimization Techniques](#5-memory-optimization-techniques)
6. [SIMD Implementation Details](#6-simd-implementation-details)
7. [Interview Questions & Answers](#7-interview-questions--answers)
8. [Key Concepts Glossary](#8-key-concepts-glossary)
9. [File Reference](#9-file-reference)

---

## 1. Project Overview

### What is forge-db?

forge-db is a high-performance vector database library written in pure Rust, designed for approximate nearest neighbor (ANN) search. It enables similarity search over large collections of high-dimensional vectors, commonly used for:

- Semantic search and RAG (Retrieval-Augmented Generation)
- Image/audio similarity matching
- Recommendation systems
- Anomaly detection

### Key Features

| Feature | Description |
|---------|-------------|
| **SIMD Acceleration** | AVX2/AVX-512/NEON with runtime CPU detection |
| **Multiple Index Types** | BruteForce, IVF, IVF-PQ, HNSW |
| **Product Quantization** | 16-32x memory compression (8-bit and 4-bit PQ) |
| **Lock-Free Search** | HNSW search is completely lock-free after finalization |
| **Batch Optimization** | Partition-centric processing for cache efficiency |
| **Parallel Processing** | Rayon-based multi-core scaling |

### Performance Numbers (SIFT1M Dataset)

| Configuration | Latency | QPS | Recall@10 | Memory |
|--------------|---------|-----|-----------|--------|
| nprobe=1 | 75 µs | 13,333 | 33% | 31 MB |
| nprobe=16 | 520 µs | 1,923 | 69% | 31 MB |
| nprobe=16 + rerank | 560 µs | 1,786 | 91% | 31 MB |
| nprobe=32 + rerank | 1.0 ms | 1,000 | 96% | 31 MB |

**Compression**: 488 MB (uncompressed) → 31 MB (compressed) = **16x reduction**

**SIMD speedup**: 3.2x faster than scalar (128 dimensions)

---

## 2. Architecture Layers

The library is organized into distinct layers, each with clear responsibilities:

```
┌─────────────────────────────────────────────────────────────┐
│                      Application Layer                       │
│              (IVFPQIndex, HNSWIndex, BruteForceIndex)       │
├─────────────────────────────────────────────────────────────┤
│                      Index Layer                             │
│         (index/ivf.rs, index/ivf_pq.rs, index/hnsw.rs)      │
├─────────────────────────────────────────────────────────────┤
│                   Compression Layer                          │
│              (pq.rs - ProductQuantizer, kmeans.rs)          │
├─────────────────────────────────────────────────────────────┤
│                     Distance Layer                           │
│        (distance/simd.rs, distance/scalar.rs, DistanceMetric)│
├─────────────────────────────────────────────────────────────┤
│                      Vector Layer                            │
│           (vector.rs - Vector, AlignedVector, VectorStore)  │
└─────────────────────────────────────────────────────────────┘
```

### 2.1 Vector Layer (`src/vector.rs`)

The foundation of the library, providing vector storage abstractions.

#### Core Types

**`Vector`** - Basic vector with ID and data:
```rust
pub struct Vector {
    pub id: u64,
    pub data: Arc<[f32]>,  // Cheap cloning via Arc
}
```

**`AlignedVector`** - 32-byte aligned for AVX2 operations:
```rust
#[repr(align(32))]
pub struct AlignedVector {
    pub data: Vec<f32>,
}
```

**`VectorStore`** - Contiguous storage with Structure of Arrays (SoA) layout:
```rust
#[repr(align(64))]  // Cache-line aligned
pub struct VectorStore {
    pub ids: Vec<u64>,    // IDs stored separately
    pub data: Vec<f32>,   // All vector data contiguous
    pub dim: usize,
    pub len: usize,
}
```

#### Design Rationale

- **Arc<[f32]>**: Enables cheap cloning of vectors without copying data
- **64-byte alignment**: Matches cache line size on modern CPUs
- **SoA layout**: Better cache utilization during sequential scans

### 2.2 Distance Layer (`src/distance/`)

Provides distance metrics with automatic SIMD dispatch.

#### Supported Metrics

| Metric | Formula | Use Case |
|--------|---------|----------|
| `Euclidean` | √Σ(aᵢ - bᵢ)² | General similarity |
| `EuclideanSquared` | Σ(aᵢ - bᵢ)² | When only ordering matters |
| `Cosine` | 1 - cos(a, b) | Text embeddings |
| `DotProduct` | -Σ(aᵢ × bᵢ) | Normalized vectors |

#### SIMD Dispatch Hierarchy

```
┌─────────────────────────────────────────────┐
│           euclidean_distance()              │
├─────────────────────────────────────────────┤
│ 1. AVX-512F? → euclidean_distance_avx512()  │  16 floats/iter
│ 2. AVX2+FMA? → euclidean_distance_avx2()    │   8 floats/iter
│ 3. NEON?     → euclidean_distance_neon()    │   4 floats/iter
│ 4. Fallback  → scalar::euclidean_distance() │   1 float/iter
└─────────────────────────────────────────────┘
```

### 2.3 Index Layer (`src/index/`)

Four index implementations with different speed/recall/memory trade-offs:

| Index | Search Time | Memory | Recall | Use Case |
|-------|-------------|--------|--------|----------|
| BruteForce | O(n) | O(n×d) | 100% | Ground truth, small datasets |
| IVF | O(n/k × nprobe) | O(n×d) | 60-95% | Medium datasets |
| IVF-PQ | O(n/k × nprobe) | O(n×M) | 50-96% | Large datasets, memory constrained |
| HNSW | O(log n) | O(n×d + n×M) | 95-99% | Low latency requirements |

### 2.4 Compression Layer (`src/pq.rs`, `src/kmeans.rs`)

Product Quantization provides massive memory savings with configurable precision.

| PQ Type | Centroids | Codes/byte | Compression | Typical Recall |
|---------|-----------|------------|-------------|----------------|
| 8-bit PQ | 256 | 1 | 16x | 85-95% |
| 4-bit PQ | 16 | 2 | 32x | 75-90% |

---

## 3. Design Patterns & Decisions

### 3.1 Runtime SIMD Dispatch

**Problem**: Different CPUs support different SIMD instruction sets.

**Solution**: Check CPU features at runtime using `is_x86_feature_detected!()`:

```rust
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { euclidean_distance_avx512(a, b) };
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { euclidean_distance_avx2(a, b) };
        }
    }
    scalar::euclidean_distance(a, b)
}
```

**Benefits**:
- Single binary works on all CPUs
- No compile-time feature flags needed for deployment
- Graceful degradation on older hardware

### 3.2 Cache-Line Alignment (64 bytes)

**Problem**: Random memory access patterns cause cache misses.

**Solution**: Align data structures to cache line boundaries:

```rust
#[repr(align(64))]
pub struct VectorStore { ... }
```

**Benefits**:
- Prevents false sharing in parallel code
- Maximizes prefetcher efficiency
- Ensures aligned SIMD loads where possible

### 3.3 Lock-Free Search (HNSW)

**Problem**: Locks cause contention in multi-threaded search.

**Solution**: Two-phase design with mutable construction and immutable search:

```rust
// Construction phase (uses RwLock)
graph: Vec<RwLock<NodeConnections>>

// Finalization converts to immutable structure
flat_graph: Option<FlatGraph>

pub fn finalize(&mut self) {
    self.flat_graph = Some(FlatGraph::build_from(&self.graph, self.max_layer));
    self.finalized = true;
}
```

**After finalization**:
- `search()` performs zero locking
- Multiple threads can search concurrently
- Only `add()` requires re-finalization

### 3.4 Structure of Arrays (SoA) Layout

**Problem**: Array of Structures (AoS) layout causes poor cache utilization.

**Traditional AoS**:
```
[{id0, data0[0..dim]}, {id1, data1[0..dim]}, ...]
```

**SoA Layout** (VectorStore):
```
ids:  [id0, id1, id2, ...]
data: [v0_d0, v0_d1, ..., v0_dn, v1_d0, v1_d1, ...]
```

**Benefits**:
- Sequential access patterns
- Better memory bandwidth utilization
- No pointer chasing

### 3.5 Heap-Based Top-K Selection

**Problem**: Sorting all N results to get top K is O(N log N).

**Solution**: Maintain a max-heap of size K:

```rust
let mut heap: BinaryHeap<ScoredVector> = BinaryHeap::with_capacity(k);

for vector in &vectors {
    let distance = compute_distance(query, vector);

    if heap.len() < k {
        heap.push(ScoredVector { id, distance });
    } else if distance < heap.peek().unwrap().distance {
        heap.pop();
        heap.push(ScoredVector { id, distance });
    }
}
```

**Complexity**: O(N log K) instead of O(N log N)

### 3.6 Residual Quantization

**Problem**: Direct PQ encoding loses information about cluster assignment.

**Solution**: Encode the residual (vector - centroid) instead of the vector:

```rust
// Instead of: encode(vector)
// We do:      encode(vector - centroid)

let residual: Vec<f32> = v.data.iter()
    .zip(centroid.data.iter())
    .map(|(a, b)| a - b)
    .collect();
pq.encode(&residual)
```

**Benefits**:
- Residuals have smaller variance
- PQ codebooks fit the data better
- 10-20% recall improvement

---

## 4. Algorithm Deep-Dives

### 4.1 Brute Force Index

The simplest index, computing distances to all vectors.

**Algorithm**:
1. For each vector in the index
2. Compute distance to query
3. Maintain top-K using max-heap

**Time Complexity**: O(N × D)
**Space Complexity**: O(N × D)

**Optimizations**:
- Software prefetching (`_mm_prefetch`)
- Parallel chunk processing with Rayon
- SIMD distance computation

```rust
// Prefetch next vector while processing current
if i + 1 < self.vectors.len() {
    _mm_prefetch(self.vectors[i + 1].data.as_ptr() as *const i8, _MM_HINT_T0);
}
```

### 4.2 IVF Index (Inverted File)

Partitions the vector space using k-means, then searches only relevant partitions.

**Build Phase**:
1. Run k-means to find `n_clusters` centroids
2. Assign each vector to its nearest centroid's partition

**Search Phase**:
1. Find `nprobe` nearest centroids to query
2. Search only vectors in those partitions
3. Return top-K results

**Trade-off**: `nprobe` controls recall vs speed:
- Higher nprobe → Higher recall, slower search
- Lower nprobe → Lower recall, faster search

**Time Complexity**: O((N/k) × nprobe × D) for k clusters
**Space Complexity**: O(N × D + k × D)

### 4.3 IVF-PQ Index

Combines IVF partitioning with Product Quantization compression.

**Build Phase**:
1. Train IVF centroids on sample vectors
2. Assign vectors to partitions
3. Train PQ on residuals (vector - centroid)
4. Encode residuals in each partition

**Search Phase**:
1. Find `nprobe` nearest IVF centroids
2. For each partition:
   - Compute query residual: query - centroid
   - Build PQ lookup table for residual
   - Scan partition using ADC (Asymmetric Distance Computation)
3. Optionally re-rank with original vectors

**Asymmetric Distance Computation (ADC)**:
```
Query (uncompressed): [q₀, q₁, ..., qₘ₋₁]  (M subvectors)
Database (compressed): [c₀, c₁, ..., cₘ₋₁]  (M code indices)

Distance ≈ Σ lookup_table[m][cₘ]  (M lookups instead of D operations)
```

**Time Complexity**: O((N/k) × nprobe × M) where M << D
**Space Complexity**: O(N × M + k × D + 256 × M × D/M)

### 4.4 HNSW Index (Hierarchical Navigable Small World)

A multi-layer graph structure enabling O(log N) search.

**Structure**:
```
Layer 3:  o ─────────── o  (sparse, long-range links)
          |             |
Layer 2:  o ─── o ───── o ─── o
          |     |       |     |
Layer 1:  o─o─o─o─o─o─o─o─o─o─o─o
          | | | | | | | | | | | |
Layer 0:  o─o─o─o─o─o─o─o─o─o─o─o─o─o─o─o  (all nodes, dense)
```

**Layer Assignment**:
- Node assigned to layers 0..L where L ~ -ln(random) × mL
- Higher layers are exponentially sparser

**Build Phase** (for each new vector):
1. Generate random layer L
2. Search from top layer down to L+1 (greedy, ef=1)
3. For layers L down to 0:
   - Beam search with ef=ef_construction
   - Select M best neighbors
   - Create bidirectional links
   - Prune if neighbor count exceeds M_max

**Search Phase**:
1. Start at entry point (highest layer node)
2. Greedy descent through layers L..1 (follow best neighbor)
3. Beam search on layer 0 with ef=ef_search
4. Return top-K from beam

**Parameters**:
- `M`: Maximum connections per node (typically 16-64)
- `ef_construction`: Beam width during build (100-400)
- `ef_search`: Beam width during search (controls recall)

**Time Complexity**: O(log N × M × D) average case
**Space Complexity**: O(N × D + N × M × layers)

### 4.5 Product Quantization

Compresses vectors by splitting into subvectors and encoding each independently.

**Concept**:
```
Original vector (D dimensions):
[x₀, x₁, x₂, x₃, ..., x_{D-1}]

Split into M subvectors (each D/M dimensions):
[x₀..x_{D/M-1}] [x_{D/M}..x_{2D/M-1}] ... [x_{D-D/M}..x_{D-1}]
     ↓                  ↓                        ↓
  code₀=42          code₁=17                  codeₘ₋₁=231

Compressed: [42, 17, ..., 231]  (M bytes vs 4D bytes)
```

**Training**:
1. For each subspace m ∈ [0, M):
   - Extract m-th subvector from all training vectors
   - Run k-means with K=256 centroids
   - Store centroids as codebook[m]

**Encoding**:
1. For each subspace:
   - Find nearest centroid index
   - Store as 1-byte code

**Distance Computation (ADC)**:
1. Build lookup table: `table[m][k] = ||query_m - codebook[m][k]||²`
2. For each compressed vector with codes [c₀, c₁, ..., cₘ₋₁]:
   - `distance = Σ table[m][cₘ]`

**Compression Ratio**:
- Original: D × 4 bytes (f32)
- Compressed: M bytes (8-bit codes)
- Ratio: 4D/M (e.g., D=128, M=8 → 64x compression)

---

## 5. Memory Optimization Techniques

### 5.1 VectorStore Contiguous Layout

**Before** (Vec<Vector> with Arc):
```
Vec<Vector>: [ptr₀, ptr₁, ptr₂, ...]
                ↓      ↓      ↓
             Arc₀   Arc₁   Arc₂
                ↓      ↓      ↓
            [data]  [data]  [data]  (scattered in heap)
```

**After** (VectorStore):
```
ids:  [id₀, id₁, id₂, ...]  (contiguous)
data: [v0_d0, v0_d1, ..., v1_d0, v1_d1, ...]  (contiguous)
```

**Benefits**:
- No Arc reference counting overhead
- No heap fragmentation
- Perfect for prefetching

### 5.2 PQ Compression Ratios

| Configuration | Memory per Vector | Compression |
|--------------|-------------------|-------------|
| Original (128-dim f32) | 512 bytes | 1x |
| 8-bit PQ, M=32 | 32 bytes | 16x |
| 8-bit PQ, M=16 | 16 bytes | 32x |
| 4-bit PQ, M=32 | 16 bytes | 32x |
| 4-bit PQ, M=16 | 8 bytes | 64x |

### 5.3 u32 Node IDs in HNSW

**Optimization**: Use `u32` instead of `u64` for neighbor IDs:

```rust
type NodeId = u32;  // Supports up to 4 billion vectors
```

**Memory Savings**:
- Each edge: 4 bytes instead of 8 bytes
- For M=16 neighbors × N nodes × 2 directions: 50% edge memory savings

### 5.4 SmallVec for Neighbor Lists

**Problem**: Most nodes have few neighbors (≤32), but Vec allocates on heap.

**Solution**: Use SmallVec with inline storage:

```rust
use smallvec::SmallVec;
layers: Vec<SmallVec<[NodeId; 32]>>  // Up to 32 neighbors stored inline
```

**Benefits**:
- No heap allocation for typical cases
- Reduces allocator pressure during construction
- Better cache locality

### 5.5 Flat Code Storage

**Before** (nested Vecs):
```rust
codes: Vec<Vec<u8>>  // N allocations, pointer chasing
```

**After** (FlatCompressedVectors):
```rust
codes: Vec<u8>  // Single contiguous allocation
// Access: codes[i * n_subvectors .. (i+1) * n_subvectors]
```

**Benefits**:
- Single allocation vs N allocations
- Sequential access pattern
- Cache-friendly during scans

---

## 6. SIMD Implementation Details

### 6.1 AVX2 Intrinsics Used

| Intrinsic | Purpose |
|-----------|---------|
| `_mm256_loadu_ps` | Load 8 unaligned floats |
| `_mm256_sub_ps` | Subtract 8 float pairs |
| `_mm256_fmadd_ps` | Fused multiply-add (a*b+c) |
| `_mm256_setzero_ps` | Initialize to zeros |
| `_mm256_i32gather_ps` | Gather 8 floats by indices |

### 6.2 AVX2 Euclidean Distance

```rust
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len();
    let mut sum = _mm256_setzero_ps();  // [0,0,0,0,0,0,0,0]
    let mut i = 0;

    // Process 8 floats at a time
    while i + 8 <= len {
        let va = _mm256_loadu_ps(a.as_ptr().add(i));  // Load a[i..i+8]
        let vb = _mm256_loadu_ps(b.as_ptr().add(i));  // Load b[i..i+8]
        let diff = _mm256_sub_ps(va, vb);             // diff = a - b
        sum = _mm256_fmadd_ps(diff, diff, sum);       // sum += diff²
        i += 8;
    }

    // Horizontal sum
    let sum_array: [f32; 8] = transmute(sum);
    let mut total: f32 = sum_array.iter().sum();

    // Handle remaining 0-7 elements
    while i < len {
        let diff = a[i] - b[i];
        total += diff * diff;
        i += 1;
    }

    total.sqrt()
}
```

### 6.3 AVX2 Gather for PQ Distance

```rust
#[target_feature(enable = "avx2")]
pub unsafe fn asymmetric_distance_simd_avx2(table: &[f32], codes: &[u8]) -> f32 {
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= n {
        // Build indices: table offset = subvector_index * 256 + code
        let indices = _mm256_set_epi32(
            (i+7) * 256 + codes[i+7] as i32,
            (i+6) * 256 + codes[i+6] as i32,
            // ...
            (i+0) * 256 + codes[i+0] as i32,
        );

        // Gather 8 floats from table using indices
        let values = _mm256_i32gather_ps::<4>(table.as_ptr(), indices);
        sum = _mm256_add_ps(sum, values);
        i += 8;
    }
    // ... horizontal sum and remainder handling
}
```

### 6.4 Feature Detection Pattern

```rust
#[inline]
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // Compile-time: only include AVX-512 code if feature enabled
        #[cfg(feature = "avx512")]
        {
            // Runtime: check if CPU actually supports it
            if is_x86_feature_detected!("avx512f") {
                return unsafe { euclidean_distance_avx512(a, b) };
            }
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { euclidean_distance_avx2(a, b) };
        }
        return scalar::euclidean_distance(a, b);
    }

    #[cfg(target_arch = "aarch64")]
    {
        // NEON always available on aarch64
        return euclidean_distance_neon(a, b);
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    scalar::euclidean_distance(a, b)
}
```

### 6.5 Handling Non-Aligned Dimensions

When vector dimension isn't a multiple of 8 (AVX2) or 16 (AVX-512):

```rust
// Main SIMD loop
while i + 8 <= len {
    // ... SIMD operations ...
    i += 8;
}

// Scalar tail for remaining 0-7 elements
while i < len {
    let diff = a[i] - b[i];
    total += diff * diff;
    i += 1;
}
```

### 6.6 Prefetching Strategy

```rust
// Prefetch 3 vectors ahead to hide memory latency
for i in 0..len {
    if i + 3 < len {
        prefetch_read(partition[i + 3].data.as_ptr());
    }
    // Process partition[i]
}
```

**Why 3 ahead?**
- Memory latency: ~100-300 cycles
- Processing time per vector: ~50-100 cycles
- 3 × 100 = 300 cycles ≈ memory latency

---

## 7. Interview Questions & Answers

### Fundamentals

**Q1: What is approximate nearest neighbor (ANN) search?**

ANN search finds vectors that are *close* to a query vector, trading exact accuracy for dramatically faster search. Instead of O(N) comparisons, ANN achieves O(log N) or O(N/k) using clever data structures.

Key insight: For most applications (search, recommendations), finding the exact closest vector isn't necessary—finding one that's "close enough" is sufficient.

---

**Q2: What is recall@K and why does it matter?**

Recall@K measures what fraction of the true K nearest neighbors were found by an approximate search.

```
Recall@K = |Predicted ∩ GroundTruth| / K
```

Example: If ground truth top-10 is {1,2,3,4,5,6,7,8,9,10} and we return {1,2,3,4,5,11,12,13,14,15}, recall@10 = 5/10 = 50%.

It matters because:
- 90%+ recall is often indistinguishable from exact search for users
- Higher recall requires more computation (trade-off)
- Different applications have different recall requirements

---

**Q3: Explain the recall vs speed trade-off in vector search.**

More thorough search → higher recall, lower QPS (queries per second).

| Parameter | Higher Value Effect |
|-----------|---------------------|
| IVF nprobe | More partitions searched → higher recall, slower |
| HNSW ef_search | Wider beam → higher recall, slower |
| PQ M (subvectors) | Less compression → higher recall, more memory |
| Re-ranking | Exact distances for candidates → higher recall, slower |

---

**Q4: What are the main vector index types and when to use each?**

| Index | When to Use |
|-------|-------------|
| **Brute Force** | <10K vectors, need 100% recall, ground truth generation |
| **IVF** | 10K-1M vectors, acceptable memory, need tunable recall |
| **IVF-PQ** | >1M vectors, memory constrained, 80-95% recall acceptable |
| **HNSW** | Low latency critical, higher memory acceptable, 95%+ recall needed |

---

### Product Quantization

**Q5: How does Product Quantization achieve compression?**

PQ splits a D-dimensional vector into M subvectors and quantizes each independently to one of 256 centroids. Instead of storing D floats (4D bytes), we store M centroid indices (M bytes).

Example: D=128, M=8 → 512 bytes → 8 bytes = 64x compression.

The magic: We can compute approximate distances using only the compressed codes via lookup tables.

---

**Q6: What is Asymmetric Distance Computation (ADC)?**

In ADC, the query remains uncompressed while the database vectors are compressed.

```
Distance(query, compressed_vector) ≈ Σ lookup_table[m][code_m]
```

The lookup table is precomputed once per query:
```
lookup_table[m][k] = ||query_subvector_m - centroid[m][k]||²
```

This converts O(D) distance computation to O(M) table lookups where M << D.

---

**Q7: Why use residual quantization instead of direct quantization?**

Residual quantization encodes (vector - centroid) instead of the vector itself.

Benefits:
1. **Smaller variance**: Residuals cluster near zero
2. **Better codebook fit**: PQ centroids can represent smaller differences more precisely
3. **10-20% recall improvement** for the same memory

---

**Q8: What's the difference between 8-bit and 4-bit PQ?**

| Aspect | 8-bit PQ | 4-bit PQ |
|--------|----------|----------|
| Centroids per subspace | 256 | 16 |
| Codes per byte | 1 | 2 |
| Memory per vector | M bytes | M/2 bytes |
| Lookup table size | 256 × M floats | 16 × M floats |
| Recall | Higher | ~85-95% of 8-bit |

4-bit PQ is better when memory is extremely constrained and slight recall loss is acceptable.

---

### HNSW

**Q9: How does HNSW achieve O(log N) search?**

HNSW uses a multi-layer graph where higher layers are exponentially sparser:
- Layer 0: All N nodes
- Layer 1: ~N/e nodes
- Layer 2: ~N/e² nodes
- Layer L: ~N/e^L nodes

Search:
1. Start at top layer (few nodes, long-range connections)
2. Greedy descent finds good entry point for next layer
3. At layer 0, beam search finds final results

Total traversal: O(log N) layers × O(M) neighbors per layer × O(D) distance.

---

**Q10: Explain HNSW's layer assignment strategy.**

Each node is assigned to layers 0 through L where:
```
L = floor(-ln(uniform_random) × mL)
```

This creates exponential decay:
- Most nodes only in layer 0 (~63%)
- Fewer nodes in layer 1 (~23%)
- Even fewer in layer 2 (~9%)
- etc.

The result: higher layers act as "express lanes" for long-range navigation.

---

**Q11: What is the purpose of ef_construction vs ef_search?**

| Parameter | Phase | Effect |
|-----------|-------|--------|
| ef_construction | Build | Beam width when adding nodes; higher = better graph, slower build |
| ef_search | Query | Beam width when searching; higher = better recall, slower search |

Typical values:
- ef_construction: 100-400 (set once, pay cost during build)
- ef_search: 10-200 (tunable per-query for recall/speed trade-off)

---

**Q12: How does forge-db achieve lock-free HNSW search?**

Two-phase design:

1. **Construction phase**: Uses `RwLock` per node for safe concurrent modification
2. **Finalization**: Copies graph to immutable `FlatGraph` structure
3. **Search phase**: Reads only from `FlatGraph` with no locking

```rust
// After finalize(), this is completely lock-free
fn search_lock_free(&self, query: &[f32], k: usize, flat_graph: &FlatGraph) -> Vec<(u64, f32)>
```

---

### IVF

**Q13: How does IVF reduce search space?**

IVF partitions vectors into clusters using k-means:
1. Train k centroids on the data
2. Assign each vector to its nearest centroid's partition
3. At search time, only scan the `nprobe` nearest partitions

If k=100 and nprobe=10, we only search 10% of the data.

---

**Q14: What is nprobe and how does it affect search?**

nprobe is the number of IVF partitions to search per query.

| nprobe | Vectors Searched | Recall | Speed |
|--------|------------------|--------|-------|
| 1 | N/k | Low | Fast |
| k/2 | N/2 | Medium | Medium |
| k | N (all) | ~100% | Slow (like brute force) |

Rule of thumb: Start with nprobe=sqrt(k) and tune based on recall requirements.

---

**Q15: Why combine IVF with PQ (IVF-PQ)?**

IVF alone reduces search space but doesn't reduce memory.
PQ alone reduces memory but requires scanning all compressed vectors.

Together:
- IVF reduces what we search
- PQ reduces memory and makes scanning faster (smaller codes, ADC)

Result: Large-scale search with low memory and acceptable recall.

---

### SIMD & Performance

**Q16: How does SIMD improve vector search performance?**

SIMD (Single Instruction, Multiple Data) processes multiple values in one CPU instruction.

| Instruction Set | Floats per Instruction | Speedup |
|----------------|------------------------|---------|
| Scalar | 1 | 1x |
| NEON (ARM) | 4 | ~3-4x |
| AVX2 (x86) | 8 | ~5-8x |
| AVX-512 (x86) | 16 | ~10-15x |

For distance computation (mostly multiply and add), SIMD is nearly ideal.

---

**Q17: What is fused multiply-add (FMA) and why is it important?**

FMA computes `a * b + c` in a single instruction instead of two.

Benefits:
1. **Throughput**: 2 operations in 1 instruction
2. **Latency**: ~5 cycles instead of ~10 cycles
3. **Precision**: Single rounding instead of two
4. **Power**: One instruction dispatch instead of two

Essential for distance computation: `sum += diff * diff`

---

**Q18: How does runtime SIMD dispatch work?**

```rust
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    // Runtime check (cached by CPU after first call)
    if is_x86_feature_detected!("avx2") {
        return unsafe { euclidean_distance_avx2(a, b) };
    }
    scalar::euclidean_distance(a, b)
}
```

The `is_x86_feature_detected!` macro:
- Uses CPUID instruction
- Result is cached by std library
- Near-zero overhead after first call

---

**Q19: Why use prefetching in vector search?**

Memory latency (100-300 cycles) dominates when data isn't in cache.

Prefetching loads data before it's needed:
```rust
if i + 3 < len {
    _mm_prefetch(vectors[i + 3].data.as_ptr(), _MM_HINT_T0);
}
// Process vectors[i] while vectors[i+3] is being loaded
```

This overlaps memory access with computation, hiding latency.

---

### Memory & Cache

**Q20: Why is cache-line alignment important?**

Modern CPUs load memory in 64-byte cache lines. Misalignment causes:
1. **Two loads** instead of one when data spans lines
2. **False sharing** in parallel code when threads modify adjacent data
3. **Wasted bandwidth** when only part of a line is used

forge-db uses `#[repr(align(64))]` for hot data structures.

---

**Q21: Explain the memory access patterns in vector search.**

| Operation | Access Pattern | Optimization |
|-----------|----------------|--------------|
| Brute force scan | Sequential | Prefetching |
| IVF scan | Sequential per partition | Prefetching |
| HNSW traversal | Random (graph walk) | Compact neighbor lists |
| PQ lookup | Random (table access) | Small tables, gather SIMD |

Sequential patterns benefit from prefetching; random patterns need smaller working sets.

---

**Q22: What are the memory costs of different indexes?**

For N vectors of dimension D:

| Index | Memory |
|-------|--------|
| Brute Force | N × D × 4 bytes |
| IVF | N × D × 4 + k × D × 4 bytes |
| IVF-PQ | N × M + k × D × 4 + 256 × D × 4 bytes |
| HNSW | N × D × 4 + N × M × 4 × avg_layers bytes |

---

### Design Decisions

**Q23: Why use u32 instead of u64 for HNSW node IDs?**

Most indexes have <4 billion vectors. Using u32:
- Saves 4 bytes per edge
- For M=32 neighbors, N=1M nodes: saves 128MB
- Still supports 4+ billion vectors

---

**Q24: Why use SmallVec for HNSW neighbor lists?**

Most nodes have ≤32 neighbors (M_max). SmallVec stores these inline without heap allocation:

```rust
SmallVec<[NodeId; 32]>  // 32 × 4 = 128 bytes inline
```

Benefits:
- No allocator calls for typical cases
- Better cache locality
- Still supports arbitrarily large lists (falls back to heap)

---

**Q25: Why is VectorStore better than Vec<Vector>?**

Vec<Vector> with Arc<[f32]>:
- Each vector has separate heap allocation
- Arc adds reference counting overhead
- Pointer chasing during iteration

VectorStore:
- Single contiguous allocation for all data
- No reference counting
- Sequential memory access

For 1M vectors: potentially 1M fewer allocations.

---

### System Design

**Q26: How would you design a distributed vector search system?**

Key considerations:
1. **Sharding**: Partition vectors across nodes (by ID hash or by cluster)
2. **Replication**: Each shard replicated for availability
3. **Query routing**: Broadcast to all shards, merge results
4. **Index distribution**: Each node has local index (IVF-PQ or HNSW)

Challenges:
- Maintaining global top-K requires coordination
- Index updates need consistency
- Balancing shards as data grows

---

**Q27: How would you handle real-time updates in a vector index?**

Options:
1. **Append-only with periodic rebuild**: Simple, good for batch updates
2. **Delta index**: New vectors in small index, periodically merge
3. **HNSW dynamic insertion**: Supports add() but needs re-finalize for optimal search
4. **LSM-style**: Write to memory, compact to disk periodically

forge-db uses approach 1/3: HNSW supports add(), but `finalize()` creates optimal search structure.

---

**Q28: What trade-offs would you make for billion-scale search?**

| Aspect | Trade-off |
|--------|-----------|
| Memory | Use 4-bit PQ or disk-based index |
| Latency | Accept higher latency, batch queries |
| Recall | Lower nprobe, accept 80-90% recall |
| Build time | Sample-based training, incremental builds |
| Hardware | GPU acceleration, multiple machines |

---

### Practical

**Q29: How would you tune an IVF-PQ index for 95% recall?**

Steps:
1. Start with reasonable defaults: k=sqrt(N), M=D/4, nprobe=k/10
2. Measure recall on held-out queries
3. Increase nprobe until recall reaches 95%
4. If memory allows, enable re-ranking with rerank_factor=4
5. Profile to ensure latency is acceptable

---

**Q30: When would you choose HNSW over IVF-PQ?**

Choose HNSW when:
- Latency is critical (<1ms)
- Memory is available (~2-3x vector data)
- Need 95%+ recall consistently
- Updates are infrequent

Choose IVF-PQ when:
- Memory is constrained
- Can tolerate higher latency
- 80-95% recall is acceptable
- Dataset is very large (>10M vectors)

---

**Q31: How does batch search optimization work in forge-db?**

Partition-centric processing:
1. Find nearest partitions for ALL queries upfront
2. Group queries by which partitions they need
3. Process partition-by-partition (data stays in cache)
4. Score each vector against all relevant queries at once

```rust
// Instead of: for each query, search partitions
// Do: for each partition, score against all queries needing it
for (partition_id, query_indices) in partition_queries {
    let partition = &self.partitions[partition_id];
    for vector in partition {
        for &qi in &query_indices {
            // vector data is hot in cache for all queries
            score(queries[qi], vector);
        }
    }
}
```

---

**Q32: What are common pitfalls in vector search implementations?**

1. **Sorting all results** instead of using heap for top-K
2. **Allocating per-query** instead of reusing buffers
3. **Not prefetching** sequential data
4. **Using f64** when f32 is sufficient (2x memory, slower SIMD)
5. **Ignoring cache effects** (random access patterns kill performance)
6. **Over-parallelizing** (thread overhead > computation for small tasks)

---

## 8. Key Concepts Glossary

### Algorithms

| Term | Definition |
|------|------------|
| **ANN** | Approximate Nearest Neighbor - finding close vectors without guaranteeing the absolute closest |
| **Brute Force** | Exhaustive search comparing query to all vectors |
| **IVF** | Inverted File - partitions vectors into clusters for faster search |
| **HNSW** | Hierarchical Navigable Small World - multi-layer graph for O(log N) search |
| **PQ** | Product Quantization - compression by quantizing subvectors independently |
| **k-means** | Clustering algorithm to find k centroids minimizing within-cluster variance |
| **ADC** | Asymmetric Distance Computation - query uncompressed, database compressed |

### Metrics

| Term | Definition |
|------|------------|
| **Recall@K** | Fraction of true K nearest neighbors found |
| **QPS** | Queries Per Second - throughput measure |
| **Latency** | Time to answer a single query (p50, p99) |
| **Compression Ratio** | Original size / Compressed size |

### SIMD

| Term | Definition |
|------|------------|
| **SIMD** | Single Instruction, Multiple Data - parallel processing of multiple values |
| **AVX2** | Advanced Vector Extensions 2 - 256-bit SIMD (8 floats) |
| **AVX-512** | 512-bit SIMD (16 floats), available on newer Intel/AMD CPUs |
| **NEON** | ARM's 128-bit SIMD (4 floats) |
| **FMA** | Fused Multiply-Add - computes a*b+c in one instruction |

### Data Structures

| Term | Definition |
|------|------------|
| **Centroid** | Cluster center point in k-means |
| **Codebook** | Set of centroids for a PQ subspace |
| **Inverted Index** | Maps cluster ID to list of vectors in that cluster |
| **Lookup Table** | Precomputed distances for fast PQ search |
| **Max-Heap** | Priority queue where largest element is at top |

### Performance

| Term | Definition |
|------|------------|
| **Cache Line** | Unit of memory transfer (typically 64 bytes) |
| **Prefetching** | Loading data into cache before it's needed |
| **False Sharing** | Cache invalidation when threads modify adjacent data |
| **Memory Bandwidth** | Rate of data transfer between CPU and RAM |
| **Branch Prediction** | CPU's guess about which branch a conditional will take |

### Search Parameters

| Term | Definition |
|------|------------|
| **nprobe** | Number of IVF partitions to search |
| **ef_search** | HNSW beam width during search |
| **ef_construction** | HNSW beam width during index construction |
| **M** | Maximum connections per HNSW node |
| **rerank_factor** | How many extra candidates to fetch for re-ranking |

---

## 9. File Reference

### Source Files

| File | Lines | Purpose |
|------|-------|---------|
| `src/lib.rs` | 39 | Crate root, re-exports public API |
| `src/vector.rs` | 420 | Vector, AlignedVector, VectorStore types |
| `src/dataset.rs` | 200 | Dataset loading, SIFT1M support, recall computation |
| `src/kmeans.rs` | 289 | K-means clustering with k-means++ initialization |
| `src/pq.rs` | 1323 | Product Quantization (8-bit, 4-bit), compressed storage |
| `src/distance/mod.rs` | 64 | Distance metric enum, dispatch |
| `src/distance/scalar.rs` | 128 | Scalar distance implementations |
| `src/distance/simd.rs` | 656 | SIMD distance (AVX2, AVX-512, NEON) |
| `src/index/mod.rs` | 11 | Index module exports |
| `src/index/brute_force.rs` | 267 | Exact search baseline with prefetching |
| `src/index/ivf.rs` | 542 | Inverted File index with batch optimization |
| `src/index/ivf_pq.rs` | 723 | IVF-PQ with residual quantization |
| `src/index/hnsw.rs` | 885 | HNSW graph index with lock-free search |

**Total**: ~5,500 lines of code

### Key Functions by File

**`pq.rs`**:
- `ProductQuantizer::train()` - Train PQ codebooks
- `ProductQuantizer::encode()` - Compress vector to codes
- `build_lookup_table_flat()` - Build ADC lookup table
- `asymmetric_distance_simd_avx2()` - SIMD PQ distance

**`ivf_pq.rs`**:
- `IVFPQIndex::build()` - Construct index (train IVF, PQ, encode)
- `IVFPQIndex::search()` - Single query search
- `IVFPQIndex::batch_search()` - Partition-centric batch search

**`hnsw.rs`**:
- `HNSWIndex::add()` - Insert vector into graph
- `HNSWIndex::finalize()` - Build lock-free search structure
- `HNSWIndex::search_lock_free()` - Lock-free beam search

**`distance/simd.rs`**:
- `euclidean_distance()` - Auto-dispatching distance
- `euclidean_distance_avx2()` - AVX2 implementation
- `euclidean_distance_neon()` - ARM NEON implementation

### Test Coverage

All modules include comprehensive tests:
- Unit tests for core algorithms
- Consistency tests (SIMD vs scalar)
- Edge cases (empty inputs, non-aligned dimensions)
- Integration tests (build + search)

Run all tests:
```bash
cargo test
```

---

## Summary

forge-db demonstrates production-quality vector search implementation with:

1. **Multiple index types** for different use cases
2. **Sophisticated compression** via Product Quantization
3. **SIMD acceleration** with runtime dispatch
4. **Memory optimization** through careful data layout
5. **Lock-free concurrency** for high-throughput search

The codebase balances readability with performance, using Rust's type system for safety while achieving competitive benchmark results.
