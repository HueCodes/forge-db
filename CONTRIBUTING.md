# Contributing to forge-db

Thank you for your interest in contributing to forge-db. This document explains
how to build the project, run tests, operate the server locally, and how to
submit changes.

---

## Prerequisites

- Rust stable toolchain (1.75+). Install via [rustup](https://rustup.rs/).
- `protoc` (protobuf compiler) for re-generating gRPC stubs:
  - macOS: `brew install protobuf`
  - Ubuntu/Debian: `apt install protobuf-compiler`
- Python 3.9+ and [maturin](https://maturin.rs) if working on `forge-py`.

---

## Building

```bash
# Build everything (all workspace crates)
cargo build

# Build in release mode
cargo build --release

# Build only the server
cargo build -p forge-server

# Build only the Python bindings
cd forge-py && maturin develop
```

---

## Running Tests

```bash
# Run all tests in the workspace
cargo test

# Run tests for a specific crate
cargo test -p forge-db
cargo test -p forge-server

# Run a specific test by name
cargo test -p forge-db metadata_store

# Run tests with output shown
cargo test -- --nocapture
```

The test suite should always pass on the `main` branch. If your changes break
any tests, fix them before opening a PR.

---

## Running the Server Locally

```bash
# Start the server with the default (in-memory) configuration
cargo run -p forge-server

# Start with a custom config file
cargo run -p forge-server -- --config forge_server.toml

# Start with environment overrides
FORGE_SERVER_HOST=0.0.0.0 FORGE_SERVER_PORT=8080 cargo run -p forge-server
```

The server exposes:
- **gRPC** on `[::1]:50051` (default)
- **HTTP/REST** on `127.0.0.1:3000` (default)
- **Health**: `GET /health`
- **Metrics**: `GET /metrics`

---

## Examples

```bash
cargo run --example getting_started
cargo run --example hnsw_recall
cargo run --example ivf_recall
```

---

## Code Style

- Follow the [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/).
- Run `cargo fmt` before every commit.
- Run `cargo clippy -- -D warnings` and fix all lints.
- Keep functions small and focused on a single responsibility.
- Document public APIs with `///` doc comments including at least one example
  where the usage is non-obvious.
- Avoid `unwrap()` and `expect()` in library code; use `Result` / `Option`
  propagation instead.
- Prefer `anyhow::Result` for server-side / binary error handling and
  `thiserror` for library error types.

---

## Project Layout

```
forge-db/           # Root library crate
  src/
    index/          # HNSW, IVF-PQ, BruteForce index implementations
    distance/       # SIMD-optimized distance functions
    metadata/       # Filtered search via MetadataStore
    persistence/    # Save/load with checksums
    wal/            # Write-Ahead Log
    pq/             # Product Quantisation
    ...
  examples/         # Runnable examples
  benches/          # Criterion benchmarks

forge-server/       # gRPC + REST server (tonic + axum)
  src/
    grpc.rs         # tonic service implementation
    http.rs         # axum router and handlers
    collections.rs  # Collection abstraction
    state.rs        # Shared server state
    auth.rs         # API key authentication
  proto/
    forge.proto     # Protobuf service definition

forge-py/           # Python bindings via PyO3 + maturin
```

---

## Pull Request Process

1. Fork the repository and create a branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```
2. Make your changes and add tests where appropriate.
3. Ensure everything passes:
   ```bash
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo test
   ```
4. Update documentation if you changed a public API.
5. Open a PR against `main` with a clear description of what was changed and
   why. Reference any relevant issues.
6. A maintainer will review your PR and may request changes. Once approved it
   will be merged using a squash merge.

---

## Reporting Bugs

Open an issue on GitHub with:
- The version of forge-db and your Rust toolchain (`rustc --version`).
- A minimal reproducible example.
- The expected and actual behaviour.

---

## License

By contributing you agree that your contributions will be licensed under the
MIT license, the same as the project.
