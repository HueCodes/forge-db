#![allow(clippy::useless_conversion)] // PyO3 macros generate Into::<PyErr>::into on PyErr values
//! Python bindings for forge-db using PyO3.
//!
//! # Usage
//!
//! ```python
//! import forge_py
//!
//! # Build an IVF-PQ index
//! index = forge_py.IvfPqIndex(
//!     vectors=[[0.1, 0.2, 0.3], ...],  # list of lists or numpy array
//!     ids=[1, 2, 3, ...],
//!     n_clusters=32,
//!     n_subvectors=8,
//!     metric="euclidean",
//! )
//!
//! # Search
//! results = index.search([0.15, 0.25, 0.35], k=10)
//! for id, distance in results:
//!     print(f"  id={id} distance={distance:.4f}")
//!
//! # Save / load
//! index.save("my_index.fdb")
//! loaded = forge_py.IvfPqIndex.load("my_index.fdb")
//! ```

use forge_db::{
    distance::DistanceMetric,

    vector::Vector,
    BruteForceIndex, HNSWIndex, IVFPQIndex, Persistable,
};
use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Helper: parse a distance metric string
// ─────────────────────────────────────────────────────────────────────────────

fn parse_metric(s: &str) -> PyResult<DistanceMetric> {
    match s.to_lowercase().as_str() {
        "euclidean" | "l2" => Ok(DistanceMetric::Euclidean),
        "euclidean_squared" | "l2_squared" => Ok(DistanceMetric::EuclideanSquared),
        "cosine" => Ok(DistanceMetric::Cosine),
        "dot_product" | "inner_product" | "ip" => Ok(DistanceMetric::DotProduct),
        other => Err(PyValueError::new_err(format!(
            "unknown distance metric '{}'. Valid options: euclidean, cosine, dot_product",
            other
        ))),
    }
}

/// Convert a Python list-of-lists or numpy 2D array to forge_db Vectors.
fn to_vectors(
    ids: Vec<u64>,
    array: PyReadonlyArray2<f32>,
) -> PyResult<Vec<Vector>> {
    let arr = array.as_array();
    if arr.nrows() != ids.len() {
        return Err(PyValueError::new_err(format!(
            "ids length {} != vectors rows {}",
            ids.len(),
            arr.nrows()
        )));
    }

    Ok(ids
        .into_iter()
        .enumerate()
        .map(|(i, id)| {
            let row: Vec<f32> = arr.row(i).to_vec();
            Vector::new(id, row)
        })
        .collect())
}

// ─────────────────────────────────────────────────────────────────────────────
// IvfPqIndex
// ─────────────────────────────────────────────────────────────────────────────

/// IVF-PQ vector index — memory-efficient, good for 1M+ vectors.
#[pyclass(name = "IvfPqIndex")]
pub struct PyIvfPqIndex {
    inner: IVFPQIndex,
}

#[pymethods]
impl PyIvfPqIndex {
    /// Build an IVF-PQ index from a numpy array of vectors.
    ///
    /// Args:
    ///     vectors: numpy float32 array of shape (n, dim)
    ///     ids: list of integer IDs (length n)
    ///     n_clusters: number of IVF partitions (default 1024)
    ///     n_subvectors: PQ subvectors, must divide dim (default 32)
    ///     metric: distance metric ("euclidean", "cosine", "dot_product")
    #[new]
    #[pyo3(signature = (vectors, ids, n_clusters=1024, n_subvectors=32, metric="euclidean"))]
    pub fn new(
        vectors: PyReadonlyArray2<f32>,
        ids: Vec<u64>,
        n_clusters: usize,
        n_subvectors: usize,
        metric: &str,
    ) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        let vecs = to_vectors(ids, vectors)?;

        let inner = IVFPQIndex::build(vecs, n_clusters, n_subvectors, metric);
        Ok(Self { inner })
    }

    /// Search for the k nearest neighbors.
    ///
    /// Args:
    ///     query: float32 array or list of shape (dim,)
    ///     k: number of results
    ///     nprobe: number of partitions to search (optional override)
    ///
    /// Returns:
    ///     list of (id, distance) tuples, sorted by distance
    #[pyo3(signature = (query, k, nprobe=None))]
    pub fn search(&self, query: Vec<f32>, k: usize, nprobe: Option<usize>) -> Vec<(u64, f32)> {
        if let Some(np) = nprobe {
            self.inner.set_nprobe(np);
        }
        self.inner.search(&query, k)
    }

    /// Set the number of partitions to probe during search.
    pub fn set_nprobe(&self, nprobe: usize) {
        self.inner.set_nprobe(nprobe);
    }

    /// Enable re-ranking for higher recall (requires storing original vectors).
    ///
    /// Args:
    ///     vectors: numpy float32 array of original vectors (same as used to build index)
    ///     ids: list of IDs
    ///     rerank_factor: multiplier for candidates to consider
    pub fn enable_reranking(
        &mut self,
        vectors: PyReadonlyArray2<f32>,
        ids: Vec<u64>,
        rerank_factor: usize,
    ) -> PyResult<()> {
        let vecs = to_vectors(ids, vectors)?;
        self.inner.enable_reranking(vecs, rerank_factor);
        Ok(())
    }

    /// Delete a vector by ID (marks as tombstone).
    pub fn delete(&mut self, id: u64) -> bool {
        self.inner.delete(id)
    }

    /// Compact the index, removing tombstoned vectors.
    pub fn compact(&mut self) {
        self.inner.compact();
    }

    /// Save the index to a file.
    pub fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .save(path)
            .map_err(|e| PyRuntimeError::new_err(format!("save failed: {e}")))
    }

    /// Load an index from a file.
    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let inner = IVFPQIndex::load(path)
            .map_err(|e| PyRuntimeError::new_err(format!("load failed: {e}")))?;
        Ok(Self { inner })
    }

    /// Return the number of vectors in the index (excluding tombstones).
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Return index statistics as a dict.
    pub fn stats(&self, py: Python<'_>) -> PyResult<PyObject> {
        let s = self.inner.statistics();
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("num_vectors", s.num_vectors)?;
        dict.set_item("num_partitions", s.num_partitions)?;
        dict.set_item("dimension", s.dimension)?;
        dict.set_item("memory_mb", s.memory_bytes as f32 / (1024.0 * 1024.0))?;
        dict.set_item("compression_ratio", s.compression_ratio)?;
        dict.set_item("num_tombstones", s.num_tombstones)?;
        dict.set_item("nprobe", s.nprobe)?;
        Ok(dict.into())
    }

    pub fn __repr__(&self) -> String {
        format!(
            "IvfPqIndex(vectors={}, dim={}, partitions={})",
            self.inner.len(),
            self.inner.dim(),
            self.inner.n_partitions(),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HnswIndex
// ─────────────────────────────────────────────────────────────────────────────

/// HNSW graph index — fastest search, highest recall, more memory.
#[pyclass(name = "HnswIndex")]
pub struct PyHnswIndex {
    inner: HNSWIndex,
}

#[pymethods]
impl PyHnswIndex {
    /// Create a new empty HNSW index.
    ///
    /// Args:
    ///     dimension: vector dimensionality
    ///     m: max connections per node per layer (default 16)
    ///     ef_construction: beam width during build (default 200)
    ///     metric: distance metric
    #[new]
    #[pyo3(signature = (_dimension, m=16, ef_construction=200, metric="euclidean"))]
    pub fn new(_dimension: usize, m: usize, ef_construction: usize, metric: &str) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        let inner = HNSWIndex::new(m, ef_construction, metric);
        Ok(Self { inner })
    }

    /// Add vectors to the index.
    ///
    /// Args:
    ///     vectors: numpy float32 array of shape (n, dim)
    ///     ids: list of integer IDs
    pub fn add(&mut self, vectors: PyReadonlyArray2<f32>, ids: Vec<u64>) -> PyResult<()> {
        let vecs = to_vectors(ids, vectors)?;
        for v in vecs {
            self.inner.add(v);
        }
        Ok(())
    }

    /// Finalize the index for optimized lock-free search.
    /// Call this after adding all vectors and before searching.
    pub fn finalize(&mut self) {
        self.inner.finalize();
    }

    /// Search for the k nearest neighbors.
    pub fn search(&self, query: Vec<f32>, k: usize) -> Vec<(u64, f32)> {
        self.inner.search(&query, k)
    }

    /// Number of vectors in the index.
    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    pub fn __repr__(&self) -> String {
        format!(
            "HnswIndex(vectors={}, dim={})",
            self.inner.len(),
            self.inner.dimension(),
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BruteForceIndex
// ─────────────────────────────────────────────────────────────────────────────

/// Brute-force exact search — 100% recall, O(N) per query.
#[pyclass(name = "BruteForceIndex")]
pub struct PyBruteForceIndex {
    inner: BruteForceIndex,
}

#[pymethods]
impl PyBruteForceIndex {
    #[new]
    #[pyo3(signature = (metric="euclidean"))]
    pub fn new(metric: &str) -> PyResult<Self> {
        let metric = parse_metric(metric)?;
        Ok(Self {
            inner: BruteForceIndex::new(metric),
        })
    }

    pub fn add(&mut self, vectors: PyReadonlyArray2<f32>, ids: Vec<u64>) -> PyResult<()> {
        let vecs = to_vectors(ids, vectors)?;
        for v in vecs {
            self.inner.add(v);
        }
        Ok(())
    }

    pub fn search(&self, query: Vec<f32>, k: usize) -> Vec<(u64, f32)> {
        self.inner.search(&query, k)
    }

    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    pub fn __repr__(&self) -> String {
        format!("BruteForceIndex(vectors={})", self.inner.len())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module registration
// ─────────────────────────────────────────────────────────────────────────────

/// Python module for forge-db bindings.
#[pymodule]
fn forge_py(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyIvfPqIndex>()?;
    m.add_class::<PyHnswIndex>()?;
    m.add_class::<PyBruteForceIndex>()?;

    // Module metadata
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("__author__", "HueCodes")?;

    Ok(())
}
