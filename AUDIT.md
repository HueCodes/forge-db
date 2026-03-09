# forge-db Security & Correctness Audit

**Date:** 2026-03-08
**Scope:** Full codebase — `forge-db` (lib), `forge-server`, `forge-py`

---

## Critical

### C1. No vector dimension validation at API boundary
**Location:** `forge-server/src/grpc.rs:111-114`, `forge-server/src/http.rs:394-397`
**Impact:** Inserting vectors of wrong dimension can cause panics or undefined behavior in index structures. Query vectors are also unvalidated.
**Fix:** Enforce dimension consistency per collection on insert and search.

### C2. WAL write failures silently suppressed
**Location:** `forge-server/src/grpc.rs:97,164,445`, `forge-server/src/http.rs:384,451`
**Impact:** `let _ = wal.append(...)` means durability guarantees are silently violated. Users believe data is persisted when it may not be.
**Fix:** Propagate WAL errors to callers or at minimum log them.

---

## High

### H1. HNSW entry_point unwraps
**Location:** `src/index/hnsw.rs:403, 676, 786`
**Impact:** `self.entry_point.unwrap()` panics if called before any vectors are added. While guarded by `is_none()` checks that return early, the pattern is fragile — any code path reaching these lines without vectors causes a panic.
**Fix:** Return `Result` with `ForgeDbError::IndexNotBuilt`.

### H2. PQ unchecked lookup table access
**Location:** `src/pq.rs:222-235, 404, 681, 687, 700`
**Impact:** `get_unchecked(codes[i] as usize)` assumes codes are valid (0-255 for 8-bit PQ). If codes are corrupted (e.g., from deserialization), this is out-of-bounds UB.
**Fix:** Add `debug_assert!` bounds checks, or validate codes on deserialization.

### H3. Lock poisoning panics in IVF-PQ
**Location:** `src/index/ivf_pq.rs` (~20+ occurrences)
**Impact:** All `RwLock::read().unwrap()` / `write().unwrap()` calls panic if another thread panicked while holding the lock, cascading one failure into total system failure.
**Fix:** Use `.read().map_err(...)` or accept the risk with a code comment.

---

## Medium

### M1. Signal handler setup panics
**Location:** `forge-server/src/main.rs:218, 224`
**Impact:** `signal::ctrl_c().await.expect(...)` panics if signal handler setup fails, preventing any graceful shutdown path.
**Fix:** Log error and exit cleanly.

### M2. Health monitor task never stops
**Location:** `forge-server/src/main.rs:93-101`
**Impact:** Infinite loop with no shutdown signal check. Task leaks on server shutdown.
**Fix:** Select on shutdown signal inside the health loop.

### M3. WAL lock held during batch cloning
**Location:** `forge-server/src/grpc.rs:94-107`, `forge-server/src/http.rs:382-390`
**Impact:** WAL mutex held while cloning potentially large vector batches, blocking all other WAL writers.
**Fix:** Build the operations list first, then lock-and-append.

### M4. Invalid metadata JSON silently ignored
**Location:** `forge-server/src/collections.rs:180`
**Impact:** Malformed metadata JSON is silently dropped — no error, no log. Users don't know their metadata wasn't stored.
**Fix:** Return error or log warning.

### M5. No validation on top_k, batch size, collection names
**Location:** `forge-server/src/grpc.rs`, `forge-server/src/http.rs`
**Impact:** k=0, k=MAX, empty names, oversized batches all pass through to the library.
**Fix:** Validate at API boundary with clear error messages.

### M6. K-means unwraps on empty input
**Location:** `src/kmeans.rs:156, 266`
**Impact:** `min_by().unwrap()` panics if centroids vector is empty.
**Fix:** Check for empty centroids and return error.

### M7. forge-py stats() unwraps
**Location:** `forge-py/src/lib.rs:188-194`
**Impact:** 6 `dict.set_item().unwrap()` calls that could panic and crash the Python interpreter.
**Fix:** Use `?` operator since the method can return `PyResult`.

---

## Low

### L1. Unused proto fields (filter_json, nprobe in gRPC SearchRequest)
**Location:** `forge-server/proto/forge.proto`, `forge-server/src/grpc.rs`
**Impact:** Defined in proto but not wired up in gRPC handler (HTTP handler does use metadata filtering).

### L2. max_distance sentinel value (0.0)
**Location:** `forge-server/src/grpc.rs:201`
**Impact:** Uses 0.0 as "no limit" — NaN or Inf inputs cause undefined filter behavior.

### L3. Grafana default password in docker-compose.yml
**Location:** `docker-compose.yml`
**Impact:** `admin:admin` — acceptable for local dev, not for production deployment.

### L4. BruteForce/dataset file parsing unwraps
**Location:** `src/index/brute_force.rs:347-375`, `src/dataset.rs:89,121`
**Impact:** Corrupted on-disk data causes panic instead of error. Low risk since these paths are less commonly hit.

---

## Positive Findings

- **SIMD safety:** All unsafe SIMD blocks are properly guarded by runtime CPU detection (`is_x86_feature_detected!`) or compile-time `#[cfg]`. NEON is always available on aarch64. No UB found.
- **No RwLock held across .await:** All async handlers correctly use `spawn_blocking` with locks acquired inside the closure.
- **WAL has CRC32 checksums:** Already implemented with proper verification on replay.
- **Auth is secure:** SHA-256 hashing + constant-time comparison for API keys.
- **No hardcoded secrets:** Zero `.env` files, no embedded credentials.
- **Docker image hardened:** Non-root user, minimal base, health checks.
- **CI covers security:** `cargo-audit`, `clippy -D warnings`, MSRV check, multi-platform testing.
- **Zero TODO/FIXME/HACK comments** in the codebase.
- **No unsafe in forge-server or forge-py.**
- **VectorStore unsafe accesses** are properly bounded by debug_assert invariants.
