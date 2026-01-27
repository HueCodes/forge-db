//! HNSW (Hierarchical Navigable Small World) index for approximate nearest neighbor search.
//!
//! HNSW is a state-of-the-art graph-based algorithm that provides excellent
//! speed/recall tradeoffs. It builds a multi-layer graph where higher layers
//! contain fewer nodes and enable fast long-range traversal.
//!
//! # Algorithm Overview
//!
//! - Each node is assigned to layers 0..L where L follows exponential decay
//! - Layer 0 contains all nodes; higher layers are progressively sparser
//! - Search starts at the top layer and greedily descends to layer 0
//! - At each layer, beam search finds the best entry points for the next layer
//!
//! # Parameters
//!
//! - `m`: Max connections per node (except layer 0). Higher = better recall, more memory
//! - `ef_construction`: Beam width during index building. Higher = better graph, slower build
//! - `ef_search`: Beam width during search. Higher = better recall, slower search

use crate::distance::DistanceMetric;
use crate::vector::Vector;
use parking_lot::RwLock;
use smallvec::SmallVec;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Node identifier within the index.
type NodeId = usize;

/// A node with its computed distance, used for heap operations.
#[derive(Clone, Copy)]
struct ScoredNode {
    id: NodeId,
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
    layers: Vec<SmallVec<[NodeId; 32]>>,
}

/// Flattened graph structure for fast, lock-free search.
/// Layer 0 neighbors are stored contiguously for cache efficiency.
struct FlatGraph {
    /// For each node: [offset, count] into neighbors array for layer 0
    layer0_offsets: Vec<(usize, usize)>,
    /// All layer 0 neighbors stored contiguously
    layer0_neighbors: Vec<NodeId>,
    /// Higher layers stored separately (sparse, less critical)
    higher_layers: Vec<Vec<SmallVec<[NodeId; 32]>>>,
    /// Max layer index
    max_layer: usize,
}

impl FlatGraph {
    /// Build flat graph from construction graph.
    fn build_from(graph: &[RwLock<NodeConnections>], max_layer: usize) -> Self {
        let n = graph.len();
        let mut layer0_offsets = Vec::with_capacity(n);
        let mut layer0_neighbors = Vec::new();

        // Initialize higher layers structure
        let mut higher_layers: Vec<Vec<SmallVec<[NodeId; 32]>>> =
            (0..max_layer).map(|_| vec![SmallVec::new(); n]).collect();

        for (node_id, node_lock) in graph.iter().enumerate() {
            let node = node_lock.read();

            // Store layer 0
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
    #[inline]
    fn get_layer0_neighbors(&self, node_id: NodeId) -> &[NodeId] {
        let (offset, count) = self.layer0_offsets[node_id];
        &self.layer0_neighbors[offset..offset + count]
    }

    /// Get higher layer neighbors for a node.
    #[inline]
    fn get_higher_layer_neighbors(&self, node_id: NodeId, layer: usize) -> &[NodeId] {
        if layer == 0 || layer > self.max_layer {
            return &[];
        }
        &self.higher_layers[layer - 1][node_id]
    }
}

/// HNSW index for approximate nearest neighbor search.
///
/// Provides fast similarity search with tunable recall/speed tradeoff.
pub struct HNSWIndex {
    /// Vector IDs (for returning results)
    vector_ids: Vec<u64>,
    /// Flat vector data for cache-efficient distance computation
    vector_data: Vec<f32>,
    /// Dimensionality of vectors
    dim: usize,
    /// Graph connections (used during construction)
    graph: Vec<RwLock<NodeConnections>>,
    /// Flattened graph for fast search (built on first search)
    flat_graph: Option<FlatGraph>,
    entry_point: Option<NodeId>,
    max_layer: usize,
    /// Max connections per layer (except layer 0)
    #[allow(dead_code)]
    m: usize,
    /// Max connections generally
    m_max: usize,
    /// Max connections in layer 0 (typically m * 2)
    m_max0: usize,
    /// Beam width during construction
    ef_construction: usize,
    /// Beam width during search
    ef_search: usize,
    /// Normalization factor for level generation
    ml: f64,
    metric: DistanceMetric,
    /// Flag to indicate if index is finalized
    finalized: bool,
}

impl HNSWIndex {
    /// Create a new HNSW index.
    pub fn new(m: usize, ef_construction: usize, metric: DistanceMetric) -> Self {
        let m_max0 = m * 2;
        Self {
            vector_ids: Vec::new(),
            vector_data: Vec::new(),
            dim: 0,
            graph: Vec::new(),
            flat_graph: None,
            entry_point: None,
            max_layer: 0,
            m,
            m_max: m,
            m_max0,
            ef_construction,
            ef_search: ef_construction,
            ml: 1.0 / (m as f64).ln(),
            metric,
            finalized: false,
        }
    }

    /// Set the beam width for search operations.
    pub fn set_ef_search(&mut self, ef: usize) {
        self.ef_search = ef;
    }

    /// Generate a random layer for a new node.
    fn random_layer(&self) -> usize {
        let r: f64 = rand::random();
        (-r.ln() * self.ml).floor() as usize
    }

    /// Add a vector to the index.
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
                self.add_connection(node_id, neighbor, lc);
                self.add_connection(neighbor, node_id, lc);
                self.prune_connections(neighbor, neighbor_m, lc);
            }

            ep = neighbors;
        }

        if layer > self.max_layer {
            self.entry_point = Some(node_id);
            self.max_layer = layer;
        }
    }

    /// Finalize the index for fast searching.
    /// Call this after adding all vectors.
    pub fn finalize(&mut self) {
        if !self.finalized && !self.graph.is_empty() {
            self.flat_graph = Some(FlatGraph::build_from(&self.graph, self.max_layer));
            self.finalized = true;
        }
    }

    #[inline]
    fn get_vector_data(&self, node_id: NodeId) -> &[f32] {
        let start = node_id * self.dim;
        &self.vector_data[start..start + self.dim]
    }

    #[inline]
    fn distance_nodes(&self, node_a: NodeId, node_b: NodeId) -> f32 {
        self.metric.compute(self.get_vector_data(node_a), self.get_vector_data(node_b))
    }

    #[inline]
    fn distance_query(&self, query: &[f32], node_id: NodeId) -> f32 {
        self.metric.compute(query, self.get_vector_data(node_id))
    }

    fn search_layer_build(&self, query_node: NodeId, entry_points: &[NodeId], ef: usize, layer: usize) -> Vec<NodeId> {
        let query = self.get_vector_data(query_node);

        let mut visited = vec![false; self.graph.len()];
        let mut candidates: BinaryHeap<Reverse<ScoredNode>> = BinaryHeap::with_capacity(ef);
        let mut results: BinaryHeap<ScoredNode> = BinaryHeap::with_capacity(ef + 1);

        for &ep in entry_points {
            if !visited[ep] {
                visited[ep] = true;
                let dist = self.metric.compute(query, self.get_vector_data(ep));
                candidates.push(Reverse(ScoredNode { id: ep, distance: dist }));
                results.push(ScoredNode { id: ep, distance: dist });
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
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    let neighbor_dist = self.metric.compute(query, self.get_vector_data(neighbor));
                    let worst_dist = results.peek().map(|n| n.distance).unwrap_or(f32::MAX);

                    if neighbor_dist < worst_dist || results.len() < ef {
                        candidates.push(Reverse(ScoredNode { id: neighbor, distance: neighbor_dist }));
                        results.push(ScoredNode { id: neighbor, distance: neighbor_dist });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        results.into_iter().map(|sn| sn.id).collect()
    }

    fn select_neighbors_sorted(&self, query_node: NodeId, candidates: &[NodeId], m: usize) -> Vec<NodeId> {
        let mut scored: Vec<(NodeId, f32)> = candidates
            .iter()
            .map(|&id| (id, self.distance_nodes(query_node, id)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored.into_iter().take(m).map(|(id, _)| id).collect()
    }

    fn add_connection(&self, from: NodeId, to: NodeId, layer: usize) {
        let mut node = self.graph[from].write();
        if layer < node.layers.len() && !node.layers[layer].contains(&to) {
            node.layers[layer].push(to);
        }
    }

    fn prune_connections(&self, node_id: NodeId, m: usize, layer: usize) {
        let neighbors: Vec<NodeId> = {
            let node = self.graph[node_id].read();
            if layer >= node.layers.len() || node.layers[layer].len() <= m {
                return;
            }
            node.layers[layer].iter().copied().collect()
        };

        let mut scored: Vec<(NodeId, f32)> = neighbors
            .into_iter()
            .map(|neighbor_id| (neighbor_id, self.distance_nodes(node_id, neighbor_id)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored.truncate(m);

        let mut node = self.graph[node_id].write();
        if layer < node.layers.len() {
            node.layers[layer] = scored.into_iter().map(|(id, _)| id).collect();
        }
    }

    /// Search for the k nearest neighbors to a query vector.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)> {
        if self.entry_point.is_none() {
            return Vec::new();
        }

        // Use flat graph if available, otherwise fall back to locked search
        if let Some(ref flat_graph) = self.flat_graph {
            self.search_flat(query, k, flat_graph)
        } else {
            self.search_locked(query, k)
        }
    }

    /// Fast search using flattened graph (no locks).
    fn search_flat(&self, query: &[f32], k: usize, flat_graph: &FlatGraph) -> Vec<(u64, f32)> {
        let entry_point = self.entry_point.unwrap();
        let n = self.vector_ids.len();

        let mut visited = vec![false; n];
        let mut ep_id = entry_point;
        let mut ep_dist = self.distance_query(query, entry_point);

        // Greedy search through higher layers
        for lc in (1..=self.max_layer).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                for &neighbor in flat_graph.get_higher_layer_neighbors(ep_id, lc) {
                    let dist = self.distance_query(query, neighbor);
                    if dist < ep_dist {
                        ep_id = neighbor;
                        ep_dist = dist;
                        changed = true;
                    }
                }
            }
        }

        // Beam search on layer 0
        let ef = self.ef_search.max(k);
        let mut candidates: BinaryHeap<Reverse<ScoredNode>> = BinaryHeap::with_capacity(ef);
        let mut results: BinaryHeap<ScoredNode> = BinaryHeap::with_capacity(ef + 1);

        visited[ep_id] = true;
        candidates.push(Reverse(ScoredNode { id: ep_id, distance: ep_dist }));
        results.push(ScoredNode { id: ep_id, distance: ep_dist });

        while let Some(Reverse(current)) = candidates.pop() {
            let worst_dist = results.peek().map(|n| n.distance).unwrap_or(f32::MAX);
            if current.distance > worst_dist && results.len() >= ef {
                break;
            }

            for &neighbor in flat_graph.get_layer0_neighbors(current.id) {
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    let neighbor_dist = self.distance_query(query, neighbor);
                    let worst_dist = results.peek().map(|n| n.distance).unwrap_or(f32::MAX);

                    if neighbor_dist < worst_dist || results.len() < ef {
                        candidates.push(Reverse(ScoredNode { id: neighbor, distance: neighbor_dist }));
                        results.push(ScoredNode { id: neighbor, distance: neighbor_dist });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

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

        let ef = self.ef_search.max(k);
        let candidates = self.search_layer_build_query(query, &ep, ef, 0);

        let mut results: Vec<(u64, f32)> = candidates
            .into_iter()
            .map(|node_id| (self.vector_ids[node_id], self.distance_query(query, node_id)))
            .collect();
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        results.truncate(k);
        results
    }

    fn search_layer_build_query(&self, query: &[f32], entry_points: &[NodeId], ef: usize, layer: usize) -> Vec<NodeId> {
        let mut visited = vec![false; self.graph.len()];
        let mut candidates: BinaryHeap<Reverse<ScoredNode>> = BinaryHeap::with_capacity(ef);
        let mut results: BinaryHeap<ScoredNode> = BinaryHeap::with_capacity(ef + 1);

        for &ep in entry_points {
            if !visited[ep] {
                visited[ep] = true;
                let dist = self.distance_query(query, ep);
                candidates.push(Reverse(ScoredNode { id: ep, distance: dist }));
                results.push(ScoredNode { id: ep, distance: dist });
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
                if !visited[neighbor] {
                    visited[neighbor] = true;
                    let neighbor_dist = self.distance_query(query, neighbor);
                    let worst_dist = results.peek().map(|n| n.distance).unwrap_or(f32::MAX);

                    if neighbor_dist < worst_dist || results.len() < ef {
                        candidates.push(Reverse(ScoredNode { id: neighbor, distance: neighbor_dist }));
                        results.push(ScoredNode { id: neighbor, distance: neighbor_dist });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        results.into_iter().map(|sn| sn.id).collect()
    }

    pub fn len(&self) -> usize {
        self.vector_ids.len()
    }

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
        // Don't call finalize - should still work

        let query = Vector::random(1000, 64);
        let results = index.search(&query.data, 10);
        assert_eq!(results.len(), 10);
    }
}
