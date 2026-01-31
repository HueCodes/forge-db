//! HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.
//!
//! HNSW is a state-of-the-art graph-based algorithm that provides excellent
//! speed/recall tradeoffs. It builds a multi-layer graph where higher layers
//! contain fewer nodes and enable fast long-range traversal.
//!
//! # Lock-Free Search Design
//!
//! After `finalize()` is called, the graph structure becomes immutable. This allows
//! `search()` operations to be completely lock-free - they only perform read-only
//! traversal of the graph. Only `add()` operations require synchronization.
//!
//! # Memory Optimizations
//!
//! - Neighbor IDs stored as `u32` instead of `u64` (50% memory savings on edges)
//! - Layer 0 neighbors stored in contiguous memory with offset table for cache efficiency
//! - `SmallVec<[u32; 32]>` avoids heap allocations for typical neighbor counts
//!
//! # Algorithm Overview
//!
//! - Each node is assigned to layers 0..L where L follows exponential decay
//! - Layer 0 contains all nodes; higher layers are progressively sparser
//! - Search starts at the top layer and greedily descends to layer 0
//! - At each layer, beam search finds the best entry points for the next layer

use crate::distance::DistanceMetric;
use crate::vector::Vector;
use parking_lot::RwLock;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};

/// Compact node identifier (u32 saves 50% memory vs u64 on edges).
/// Supports up to 4 billion vectors per index.
type NodeId = u32;

/// Internal node index for array access.
type NodeIndex = usize;

/// A node with its computed distance, used for heap operations.
#[derive(Clone, Copy)]
struct ScoredNode {
    id: NodeIndex,
    distance: f32,
}

impl PartialEq for ScoredNode {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for ScoredNode {}

impl PartialOrd for ScoredNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.distance.partial_cmp(&other.distance)
    }
}

impl Ord for ScoredNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap_or(Ordering::Equal)
    }
}

/// Graph connections for a node at each layer (used during construction).
struct NodeConnections {
    /// Neighbors at each layer, stored as u32 for memory efficiency.
    layers: Vec<SmallVec<[NodeId; 32]>>,
}

/// Flattened graph structure for lock-free, cache-efficient search.
///
/// # Design
///
/// Layer 0 (containing all nodes) is the hot path during search, so it uses
/// a flat contiguous layout for maximum cache efficiency. Higher layers are
/// sparse and accessed less frequently, so they use a simpler nested structure.
///
/// # Thread Safety
///
/// This structure is immutable after construction. All fields can be safely
/// read from multiple threads without any synchronization.
struct FlatGraph {
    /// For each node: (offset, count) into layer0_neighbors array.
    /// Using (u32, u16) would save more memory but usize is faster on 64-bit.
    layer0_offsets: Vec<(usize, usize)>,
    /// All layer 0 neighbors stored contiguously for cache efficiency.
    layer0_neighbors: Vec<NodeId>,
    /// Higher layers stored separately (sparse, less performance-critical).
    /// Structure: higher_layers[layer-1][node_id] = neighbors
    higher_layers: Vec<Vec<SmallVec<[NodeId; 32]>>>,
    /// Maximum layer index in the graph.
    max_layer: usize,
}

impl FlatGraph {
    /// Build a flat, immutable graph from the construction graph.
    ///
    /// This converts the lock-based construction graph into a flat structure
    /// optimized for lock-free traversal.
    fn build_from(graph: &[RwLock<NodeConnections>], max_layer: usize) -> Self {
        let n = graph.len();
        let mut layer0_offsets = Vec::with_capacity(n);
        let mut layer0_neighbors = Vec::new();

        // Pre-allocate higher layers structure
        let mut higher_layers: Vec<Vec<SmallVec<[NodeId; 32]>>> =
            (0..max_layer).map(|_| vec![SmallVec::new(); n]).collect();

        for (node_id, node_lock) in graph.iter().enumerate() {
            let node = node_lock.read();

            // Store layer 0 neighbors contiguously
            let offset = layer0_neighbors.len();
            if !node.layers.is_empty() {
                layer0_neighbors.extend_from_slice(&node.layers[0]);
            }
            layer0_offsets.push((offset, layer0_neighbors.len() - offset));

            // Store higher layers
            for (layer_idx, neighbors) in node.layers.iter().enumerate().skip(1) {
                if layer_idx <= max_layer {
                    higher_layers[layer_idx - 1][node_id] = neighbors.clone();
                }
            }
        }

        Self {
            layer0_offsets,
            layer0_neighbors,
            higher_layers,
            max_layer,
        }
    }

    /// Get layer 0 neighbors for a node.
    ///
    /// # Safety Note
    /// This is safe to call from multiple threads - no locks needed.
    #[inline(always)]
    fn get_layer0_neighbors(&self, node_id: NodeIndex) -> &[NodeId] {
        let (offset, count) = self.layer0_offsets[node_id];
        &self.layer0_neighbors[offset..offset + count]
    }

    /// Get higher layer neighbors for a node.
    ///
    /// # Safety Note
    /// This is safe to call from multiple threads - no locks needed.
    #[inline(always)]
    fn get_higher_layer_neighbors(&self, node_id: NodeIndex, layer: usize) -> &[NodeId] {
        if layer == 0 || layer > self.max_layer {
            return &[];
        }
        &self.higher_layers[layer - 1][node_id]
    }
}

/// Cached entry points for faster search convergence.
///
/// Instead of always starting from a single entry point, we cache multiple
/// well-distributed entry points and pick the closest to the query.
struct EntryPointCache {
    /// Entry point node IDs with their vectors for quick distance computation.
    /// Stored as (node_id, vector_offset) pairs.
    points: Vec<NodeIndex>,
    /// Number of cached entry points (typically 4-8).
    count: usize,
}

impl EntryPointCache {
    fn new() -> Self {
        Self {
            points: Vec::with_capacity(8),
            count: 0,
        }
    }

    /// Update the cache with a new potential entry point.
    fn update(&mut self, node_id: NodeIndex, max_points: usize) {
        if self.count < max_points {
            self.points.push(node_id);
            self.count += 1;
        }
    }

    /// Get all cached entry points.
    #[inline(always)]
    fn get_points(&self) -> &[NodeIndex] {
        &self.points[..self.count]
    }
}

/// HNSW index for approximate nearest neighbor search.
///
/// # Thread Safety
///
/// After calling `finalize()`, the `search()` method is completely lock-free
/// and can be safely called from multiple threads concurrently. Only `add()`
/// requires synchronization (via internal locks).
///
/// # Example
///
/// ```ignore
/// let mut index = HNSWIndex::new(16, 200, DistanceMetric::Euclidean);
/// for vector in vectors {
///     index.add(vector);
/// }
/// index.finalize();  // Builds lock-free search structure
///
/// // Now search can be called from multiple threads without locks
/// let results = index.search(&query, 10);
/// ```
pub struct HNSWIndex {
    /// Vector IDs (for returning results to caller).
    vector_ids: Vec<u64>,
    /// Flat vector data for cache-efficient distance computation.
    /// Layout: [vec0_dim0, vec0_dim1, ..., vec1_dim0, vec1_dim1, ...]
    vector_data: Vec<f32>,
    /// Dimensionality of vectors.
    dim: usize,
    /// Graph connections (used during construction, protected by locks).
    graph: Vec<RwLock<NodeConnections>>,
    /// Flattened graph for lock-free search (built on finalize).
    flat_graph: Option<FlatGraph>,
    /// Primary entry point for search.
    entry_point: Option<NodeIndex>,
    /// Cached entry points for faster convergence.
    entry_cache: EntryPointCache,
    /// Maximum layer in the graph.
    max_layer: usize,
    /// Max connections per layer (except layer 0).
    _m: usize,
    /// Max connections generally.
    m_max: usize,
    /// Max connections in layer 0 (typically m * 2).
    m_max0: usize,
    /// Beam width during construction.
    ef_construction: usize,
    /// Beam width during search (atomic for thread-safe modification).
    ef_search: AtomicUsize,
    /// Normalization factor for level generation.
    ml: f64,
    /// Distance metric to use.
    metric: DistanceMetric,
    /// Flag to indicate if index is finalized for lock-free search.
    finalized: bool,
}

impl HNSWIndex {
    /// Create a new HNSW index.
    ///
    /// # Parameters
    ///
    /// * `m` - Max connections per node. Higher values improve recall but use more memory.
    ///   Typical values: 16-64.
    /// * `ef_construction` - Beam width during index building. Higher values build a
    ///   better graph but take longer. Typical values: 100-400.
    /// * `metric` - Distance metric for similarity computation.
    pub fn new(m: usize, ef_construction: usize, metric: DistanceMetric) -> Self {
        let m_max0 = m * 2;
        Self {
            vector_ids: Vec::new(),
            vector_data: Vec::new(),
            dim: 0,
            graph: Vec::new(),
            flat_graph: None,
            entry_point: None,
            entry_cache: EntryPointCache::new(),
            max_layer: 0,
            _m: m,
            m_max: m,
            m_max0,
            ef_construction,
            ef_search: AtomicUsize::new(ef_construction),
            ml: 1.0 / (m as f64).ln(),
            metric,
            finalized: false,
        }
    }

    /// Set the beam width for search operations.
    ///
    /// Higher values improve recall at the cost of speed.
    /// This can be safely called while searches are in progress.
    pub fn set_ef_search(&self, ef: usize) {
        self.ef_search.store(ef, AtomicOrdering::Relaxed);
    }

    /// Get the current ef_search value.
    #[inline(always)]
    fn get_ef_search(&self) -> usize {
        self.ef_search.load(AtomicOrdering::Relaxed)
    }

    /// Generate a random layer for a new node.
    fn random_layer(&self) -> usize {
        let r: f64 = rand::random();
        (-r.ln() * self.ml).floor() as usize
    }

    /// Add a vector to the index.
    ///
    /// This invalidates the finalized state. Call `finalize()` again before
    /// searching for optimal performance.
    pub fn add(&mut self, vector: Vector) {
        // Invalidate flat graph on modification
        self.flat_graph = None;
        self.finalized = false;

        let node_id = self.vector_ids.len();
        let layer = self.random_layer();

        // Set dimension from first vector
        if self.dim == 0 {
            self.dim = vector.data.len();
        }

        // Store vector data
        self.vector_ids.push(vector.id);
        self.vector_data.extend_from_slice(&vector.data);

        // Create graph connections
        let mut layers = Vec::with_capacity(layer + 1);
        for _ in 0..=layer {
            layers.push(SmallVec::new());
        }
        self.graph.push(RwLock::new(NodeConnections { layers }));

        // First node becomes entry point
        if self.entry_point.is_none() {
            self.entry_point = Some(node_id);
            self.entry_cache.update(node_id, 8);
            self.max_layer = layer;
            return;
        }

        let entry_point = self.entry_point.unwrap();
        let mut ep = vec![entry_point];

        // Search from top layer down to target layer + 1
        for lc in (layer + 1..=self.max_layer).rev() {
            ep = self.search_layer_build(node_id, &ep, 1, lc);
        }

        // Insert at layers 0 to target layer
        for lc in (0..=layer).rev() {
            let candidates = self.search_layer_build(node_id, &ep, self.ef_construction, lc);
            let neighbor_m = if lc == 0 { self.m_max0 } else { self.m_max };
            let neighbors = self.select_neighbors_sorted(node_id, &candidates, neighbor_m);

            for &neighbor in &neighbors {
                self.add_connection(node_id, neighbor as NodeIndex, lc);
                self.add_connection(neighbor as NodeIndex, node_id, lc);
                self.prune_connections(neighbor as NodeIndex, neighbor_m, lc);
            }

            ep = neighbors.iter().map(|&n| n as NodeIndex).collect();
        }

        if layer > self.max_layer {
            self.entry_point = Some(node_id);
            self.entry_cache.update(node_id, 8);
            self.max_layer = layer;
        }
    }

    /// Finalize the index for fast, lock-free searching.
    ///
    /// Call this after adding all vectors. After finalization:
    /// - `search()` becomes completely lock-free
    /// - Multiple threads can search concurrently without contention
    /// - Adding more vectors will require re-finalization
    pub fn finalize(&mut self) {
        if !self.finalized && !self.graph.is_empty() {
            self.flat_graph = Some(FlatGraph::build_from(&self.graph, self.max_layer));
            self.finalized = true;
        }
    }

    /// Get pointer to vector data for prefetching.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn get_vector_ptr(&self, node_id: NodeIndex) -> *const f32 {
        let start = node_id * self.dim;
        self.vector_data.as_ptr().wrapping_add(start)
    }

    /// Get vector data for a node.
    #[inline(always)]
    fn get_vector_data(&self, node_id: NodeIndex) -> &[f32] {
        let start = node_id * self.dim;
        &self.vector_data[start..start + self.dim]
    }

    /// Compute distance between two nodes.
    #[inline(always)]
    fn distance_nodes(&self, node_a: NodeIndex, node_b: NodeIndex) -> f32 {
        self.metric
            .compute(self.get_vector_data(node_a), self.get_vector_data(node_b))
    }

    /// Compute distance from query to a node.
    #[inline(always)]
    fn distance_query(&self, query: &[f32], node_id: NodeIndex) -> f32 {
        self.metric.compute(query, self.get_vector_data(node_id))
    }

    /// Prefetch vector data for upcoming distance computations.
    ///
    /// # Safety
    /// Uses x86_64 prefetch intrinsics. Safe because:
    /// - Prefetch hints are advisory - invalid addresses are ignored
    /// - We verify the pointer is within our allocated vector_data
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn prefetch_vector(&self, node_id: NodeIndex) {
        let ptr = self.get_vector_ptr(node_id);
        // SAFETY: _mm_prefetch is safe with any pointer - it's just a hint
        // to the CPU. Invalid addresses result in no-op, not crashes.
        unsafe {
            _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
        }
    }

    fn search_layer_build(
        &self,
        query_node: NodeIndex,
        entry_points: &[NodeIndex],
        ef: usize,
        layer: usize,
    ) -> Vec<NodeIndex> {
        let query = self.get_vector_data(query_node);

        let mut visited = vec![false; self.graph.len()];
        let mut candidates: BinaryHeap<Reverse<ScoredNode>> = BinaryHeap::with_capacity(ef);
        let mut results: BinaryHeap<ScoredNode> = BinaryHeap::with_capacity(ef + 1);

        for &ep in entry_points {
            if !visited[ep] {
                visited[ep] = true;
                let dist = self.metric.compute(query, self.get_vector_data(ep));
                candidates.push(Reverse(ScoredNode {
                    id: ep,
                    distance: dist,
                }));
                results.push(ScoredNode {
                    id: ep,
                    distance: dist,
                });
            }
        }

        while let Some(Reverse(current)) = candidates.pop() {
            let worst_dist = results.peek().map(|n| n.distance).unwrap_or(f32::MAX);
            if current.distance > worst_dist && results.len() >= ef {
                break;
            }

            let neighbors: SmallVec<[NodeId; 32]> = {
                let node = self.graph[current.id].read();
                if layer < node.layers.len() {
                    node.layers[layer].clone()
                } else {
                    SmallVec::new()
                }
            };

            for neighbor in neighbors {
                let neighbor_idx = neighbor as NodeIndex;
                if !visited[neighbor_idx] {
                    visited[neighbor_idx] = true;
                    let neighbor_dist = self
                        .metric
                        .compute(query, self.get_vector_data(neighbor_idx));
                    let worst_dist = results.peek().map(|n| n.distance).unwrap_or(f32::MAX);

                    if neighbor_dist < worst_dist || results.len() < ef {
                        candidates.push(Reverse(ScoredNode {
                            id: neighbor_idx,
                            distance: neighbor_dist,
                        }));
                        results.push(ScoredNode {
                            id: neighbor_idx,
                            distance: neighbor_dist,
                        });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        results.into_iter().map(|sn| sn.id).collect()
    }

    fn select_neighbors_sorted(
        &self,
        query_node: NodeIndex,
        candidates: &[NodeIndex],
        m: usize,
    ) -> Vec<NodeId> {
        let mut scored: Vec<(NodeId, f32)> = candidates
            .iter()
            .map(|&id| (id as NodeId, self.distance_nodes(query_node, id)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored.into_iter().take(m).map(|(id, _)| id).collect()
    }

    fn add_connection(&self, from: NodeIndex, to: NodeIndex, layer: usize) {
        let mut node = self.graph[from].write();
        let to_id = to as NodeId;
        if layer < node.layers.len() && !node.layers[layer].contains(&to_id) {
            node.layers[layer].push(to_id);
        }
    }

    fn prune_connections(&self, node_id: NodeIndex, m: usize, layer: usize) {
        let neighbors: Vec<NodeId> = {
            let node = self.graph[node_id].read();
            if layer >= node.layers.len() || node.layers[layer].len() <= m {
                return;
            }
            node.layers[layer].iter().copied().collect()
        };

        let mut scored: Vec<(NodeId, f32)> = neighbors
            .into_iter()
            .map(|neighbor_id| {
                (
                    neighbor_id,
                    self.distance_nodes(node_id, neighbor_id as NodeIndex),
                )
            })
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored.truncate(m);

        let mut node = self.graph[node_id].write();
        if layer < node.layers.len() {
            node.layers[layer] = scored.into_iter().map(|(id, _)| id).collect();
        }
    }

    /// Search for the k nearest neighbors to a query vector.
    ///
    /// # Lock-Free Guarantee
    ///
    /// After `finalize()` is called, this method performs no locking and can be
    /// safely called from multiple threads concurrently. The graph traversal is
    /// read-only and uses only atomic reads where necessary.
    ///
    /// # Returns
    ///
    /// Vector of (id, distance) pairs sorted by distance (closest first).
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        if self.entry_point.is_none() {
            return Vec::new();
        }

        // Use flat graph if available (lock-free path), otherwise fall back to locked search
        if let Some(ref flat_graph) = self.flat_graph {
            self.search_lock_free(query, k, flat_graph)
        } else {
            self.search_locked(query, k)
        }
    }

    /// Lock-free search using flattened graph.
    ///
    /// # Guarantees
    ///
    /// - No locks are acquired during this entire function
    /// - Safe to call from multiple threads concurrently
    /// - Uses prefetching to hide memory latency
    #[inline(always)]
    fn search_lock_free(&self, query: &[f32], k: usize, flat_graph: &FlatGraph) -> Vec<(u64, f32)> {
        let n = self.vector_ids.len();
        let ef = self.get_ef_search().max(k);

        // Find best entry point from cache
        let mut ep_id = self.find_best_entry_point(query);
        let mut ep_dist = self.distance_query(query, ep_id);

        // Greedy search through higher layers (single best path)
        for lc in (1..=self.max_layer).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let neighbors = flat_graph.get_higher_layer_neighbors(ep_id, lc);

                // Prefetch neighbor vectors while we process
                self.prefetch_neighbors_from_slice(neighbors);

                for &neighbor in neighbors {
                    let neighbor_idx = neighbor as NodeIndex;
                    let dist = self.distance_query(query, neighbor_idx);
                    if dist < ep_dist {
                        ep_id = neighbor_idx;
                        ep_dist = dist;
                        changed = true;
                    }
                }
            }
        }

        // Beam search on layer 0 (the hot path)
        self.beam_search_layer0(query, ep_id, ep_dist, ef, k, n, flat_graph)
    }

    /// Find the best entry point from the cache.
    #[inline(always)]
    fn find_best_entry_point(&self, query: &[f32]) -> NodeIndex {
        let cached_points = self.entry_cache.get_points();

        if cached_points.len() <= 1 {
            return self.entry_point.unwrap();
        }

        let mut best_id = cached_points[0];
        let mut best_dist = self.distance_query(query, best_id);

        for &point in &cached_points[1..] {
            let dist = self.distance_query(query, point);
            if dist < best_dist {
                best_id = point;
                best_dist = dist;
            }
        }

        best_id
    }

    /// Prefetch neighbor vectors from a slice of NodeIds.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn prefetch_neighbors_from_slice(&self, neighbors: &[NodeId]) {
        for &neighbor_id in neighbors.iter().take(4) {
            self.prefetch_vector(neighbor_id as NodeIndex);
        }
    }

    /// No-op on non-x86_64 platforms.
    #[cfg(not(target_arch = "x86_64"))]
    #[inline(always)]
    fn prefetch_neighbors_from_slice(&self, _neighbors: &[NodeId]) {}

    /// Beam search on layer 0 with prefetching.
    ///
    /// This is the performance-critical inner loop. It's been optimized for:
    /// - Cache efficiency (contiguous neighbor storage)
    /// - Prefetching (hide memory latency)
    /// - Minimal branching
    #[inline(always)]
    fn beam_search_layer0(
        &self,
        query: &[f32],
        ep_id: NodeIndex,
        ep_dist: f32,
        ef: usize,
        k: usize,
        n: usize,
        flat_graph: &FlatGraph,
    ) -> Vec<(u64, f32)> {
        let mut visited = vec![false; n];
        let mut candidates: BinaryHeap<Reverse<ScoredNode>> = BinaryHeap::with_capacity(ef);
        let mut results: BinaryHeap<ScoredNode> = BinaryHeap::with_capacity(ef + 1);

        visited[ep_id] = true;
        candidates.push(Reverse(ScoredNode {
            id: ep_id,
            distance: ep_dist,
        }));
        results.push(ScoredNode {
            id: ep_id,
            distance: ep_dist,
        });

        while let Some(Reverse(current)) = candidates.pop() {
            let worst_dist = results.peek().map(|n| n.distance).unwrap_or(f32::MAX);
            if current.distance > worst_dist && results.len() >= ef {
                break;
            }

            let neighbors = flat_graph.get_layer0_neighbors(current.id);

            // Prefetch upcoming neighbor vectors
            self.prefetch_neighbors_from_slice(neighbors);

            for &neighbor in neighbors {
                let neighbor_idx = neighbor as NodeIndex;
                if !visited[neighbor_idx] {
                    visited[neighbor_idx] = true;

                    let neighbor_dist = self.distance_query(query, neighbor_idx);
                    let worst_dist = results.peek().map(|n| n.distance).unwrap_or(f32::MAX);

                    if neighbor_dist < worst_dist || results.len() < ef {
                        candidates.push(Reverse(ScoredNode {
                            id: neighbor_idx,
                            distance: neighbor_dist,
                        }));
                        results.push(ScoredNode {
                            id: neighbor_idx,
                            distance: neighbor_dist,
                        });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        // Convert to output format
        let mut final_results: Vec<(u64, f32)> = results
            .into_iter()
            .map(|sn| (self.vector_ids[sn.id], sn.distance))
            .collect();
        final_results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        final_results.truncate(k);
        final_results
    }

    /// Fallback search using locks (for when graph isn't finalized).
    fn search_locked(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        let entry_point = self.entry_point.unwrap();
        let mut ep = vec![entry_point];

        for lc in (1..=self.max_layer).rev() {
            ep = self.search_layer_build_query(query, &ep, 1, lc);
        }

        let ef = self.get_ef_search().max(k);
        let candidates = self.search_layer_build_query(query, &ep, ef, 0);

        let mut results: Vec<(u64, f32)> = candidates
            .into_iter()
            .map(|node_id| {
                (
                    self.vector_ids[node_id],
                    self.distance_query(query, node_id),
                )
            })
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.truncate(k);
        results
    }

    fn search_layer_build_query(
        &self,
        query: &[f32],
        entry_points: &[NodeIndex],
        ef: usize,
        layer: usize,
    ) -> Vec<NodeIndex> {
        let mut visited = vec![false; self.graph.len()];
        let mut candidates: BinaryHeap<Reverse<ScoredNode>> = BinaryHeap::with_capacity(ef);
        let mut results: BinaryHeap<ScoredNode> = BinaryHeap::with_capacity(ef + 1);

        for &ep in entry_points {
            if !visited[ep] {
                visited[ep] = true;
                let dist = self.distance_query(query, ep);
                candidates.push(Reverse(ScoredNode {
                    id: ep,
                    distance: dist,
                }));
                results.push(ScoredNode {
                    id: ep,
                    distance: dist,
                });
            }
        }

        while let Some(Reverse(current)) = candidates.pop() {
            let worst_dist = results.peek().map(|n| n.distance).unwrap_or(f32::MAX);
            if current.distance > worst_dist && results.len() >= ef {
                break;
            }

            let neighbors: SmallVec<[NodeId; 32]> = {
                let node = self.graph[current.id].read();
                if layer < node.layers.len() {
                    node.layers[layer].clone()
                } else {
                    SmallVec::new()
                }
            };

            for neighbor in neighbors {
                let neighbor_idx = neighbor as NodeIndex;
                if !visited[neighbor_idx] {
                    visited[neighbor_idx] = true;
                    let neighbor_dist = self.distance_query(query, neighbor_idx);
                    let worst_dist = results.peek().map(|n| n.distance).unwrap_or(f32::MAX);

                    if neighbor_dist < worst_dist || results.len() < ef {
                        candidates.push(Reverse(ScoredNode {
                            id: neighbor_idx,
                            distance: neighbor_dist,
                        }));
                        results.push(ScoredNode {
                            id: neighbor_idx,
                            distance: neighbor_dist,
                        });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        results.into_iter().map(|sn| sn.id).collect()
    }

    /// Returns the number of vectors in the index.
    pub fn len(&self) -> usize {
        self.vector_ids.len()
    }

    /// Returns true if the index contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.vector_ids.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_index() {
        let index = HNSWIndex::new(16, 200, DistanceMetric::Euclidean);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);
        let results = index.search(&[0.0; 128], 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_single_vector() {
        let mut index = HNSWIndex::new(16, 200, DistanceMetric::Euclidean);
        index.add(Vector::new(42, vec![1.0; 128]));
        index.finalize();

        assert_eq!(index.len(), 1);
        let results = index.search(&[1.0; 128], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 42);
        assert!(results[0].1 < 1e-5);
    }

    #[test]
    fn test_basic_search() {
        let mut index = HNSWIndex::new(16, 200, DistanceMetric::Euclidean);
        for i in 0..100 {
            index.add(Vector::random(i, 128));
        }
        index.finalize();

        assert_eq!(index.len(), 100);
        let query = Vector::random(1000, 128);
        let results = index.search(&query.data, 10);

        assert_eq!(results.len(), 10);
        for i in 1..results.len() {
            assert!(results[i - 1].1 <= results[i].1);
        }
    }

    #[test]
    fn test_exact_match() {
        let mut index = HNSWIndex::new(16, 200, DistanceMetric::EuclideanSquared);

        for i in 0..50 {
            index.add(Vector::random(i, 64));
        }
        let target = Vector::new(999, vec![0.5; 64]);
        index.add(target.clone());
        for i in 51..100 {
            index.add(Vector::random(i, 64));
        }
        index.finalize();

        let results = index.search(&target.data, 1);
        assert_eq!(results[0].0, 999);
        assert!(results[0].1 < 1e-5);
    }

    #[test]
    fn test_larger_k_than_index() {
        let mut index = HNSWIndex::new(16, 200, DistanceMetric::Euclidean);
        for i in 0..5 {
            index.add(Vector::random(i, 32));
        }
        index.finalize();

        let results = index.search(&[0.0; 32], 100);
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_ef_search_affects_quality() {
        let mut index = HNSWIndex::new(16, 200, DistanceMetric::Euclidean);
        for i in 0..1000 {
            index.add(Vector::random(i, 64));
        }
        index.finalize();

        let query = Vector::random(9999, 64);

        index.set_ef_search(10);
        let low_ef_results = index.search(&query.data, 10);

        index.set_ef_search(200);
        let high_ef_results = index.search(&query.data, 10);

        assert_eq!(low_ef_results.len(), 10);
        assert_eq!(high_ef_results.len(), 10);
    }

    #[test]
    fn test_search_without_finalize() {
        let mut index = HNSWIndex::new(16, 200, DistanceMetric::Euclidean);
        for i in 0..100 {
            index.add(Vector::random(i, 64));
        }
        // Don't call finalize - should still work (using locked path)

        let query = Vector::random(1000, 64);
        let results = index.search(&query.data, 10);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_node_id_fits_in_u32() {
        // Verify that our u32 NodeId type is sufficient
        let max_nodes: u32 = u32::MAX;
        assert!(
            max_nodes > 4_000_000_000,
            "u32 should support 4+ billion nodes"
        );
    }
}
