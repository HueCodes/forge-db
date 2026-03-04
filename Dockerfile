# ─────────────────────────────────────────────────────────────────────────────
# Stage 1: Builder
# ─────────────────────────────────────────────────────────────────────────────
FROM rust:1.82-slim-bookworm AS builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    protobuf-compiler \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy workspace manifests first for better layer caching
COPY Cargo.toml Cargo.lock ./
COPY forge-server/Cargo.toml forge-server/
COPY forge-server/build.rs forge-server/
COPY forge-server/proto forge-server/proto/
COPY forge-py/Cargo.toml forge-py/

# Create dummy source files to cache dependency compilation
RUN mkdir -p src && echo "// dummy" > src/lib.rs
RUN mkdir -p forge-server/src && echo "fn main() {}" > forge-server/src/main.rs && echo "" > forge-server/src/lib.rs
RUN mkdir -p forge-py/src && echo "" > forge-py/src/lib.rs

# Build dependencies only (cached layer)
RUN cargo build --release --bin forge-server 2>&1 || true

# Remove the dummy sources
RUN rm -rf src forge-server/src forge-py/src

# Copy actual source code
COPY src ./src
COPY forge-server ./forge-server
COPY forge-py ./forge-py

# Build the release binary
RUN cargo build --release --bin forge-server

# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: Runtime
# ─────────────────────────────────────────────────────────────────────────────
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd --gid 1001 forge && \
    useradd --uid 1001 --gid forge --no-create-home --shell /usr/sbin/nologin forge

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/forge-server /usr/local/bin/forge-server

# Default config (can be overridden by volume mount)
COPY --chown=forge:forge docker/forge.toml /etc/forge-db/forge.toml

# Create data and WAL directories
RUN mkdir -p /var/lib/forge-db/data /var/lib/forge-db/wal && \
    chown -R forge:forge /var/lib/forge-db

USER forge

# Expose ports
EXPOSE 50051  # gRPC
EXPOSE 8080   # HTTP/REST
EXPOSE 9090   # Prometheus metrics

VOLUME ["/var/lib/forge-db"]

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD wget -qO- http://localhost:8080/health || exit 1

ENTRYPOINT ["/usr/local/bin/forge-server"]
CMD ["--config", "/etc/forge-db/forge.toml"]
