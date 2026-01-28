#!/bin/bash
# Profile-Guided Optimization build script for forge-db
# Usage: ./pgo-build.sh
#
# This script:
# 1. Builds an instrumented binary
# 2. Runs benchmarks to generate profile data
# 3. Rebuilds with profile data for optimized binary
#
# Requirements:
# - Rust toolchain with LLVM (rustup default)
# - llvm-profdata (usually in llvm or llvm-tools-preview)
#
# For native CPU optimization (use in addition to PGO):
#   export RUSTFLAGS="-C target-cpu=native"
#
# Combined PGO + native (what this script does):
#   export RUSTFLAGS="-C target-cpu=native -C profile-use=/tmp/pgo-data/merged.profdata"

set -e

# Configuration
PGO_DATA_DIR="/tmp/pgo-data"
MERGED_PROFDATA="$PGO_DATA_DIR/merged.profdata"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "\n${BLUE}==>${NC} ${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}Warning:${NC} $1"
}

print_error() {
    echo -e "${RED}Error:${NC} $1"
}

# Check for llvm-profdata
check_dependencies() {
    print_step "Checking dependencies..."

    # Try to find llvm-profdata
    if command -v llvm-profdata &> /dev/null; then
        LLVM_PROFDATA="llvm-profdata"
    elif command -v llvm-profdata-17 &> /dev/null; then
        LLVM_PROFDATA="llvm-profdata-17"
    elif command -v llvm-profdata-16 &> /dev/null; then
        LLVM_PROFDATA="llvm-profdata-16"
    elif command -v llvm-profdata-15 &> /dev/null; then
        LLVM_PROFDATA="llvm-profdata-15"
    elif command -v llvm-profdata-14 &> /dev/null; then
        LLVM_PROFDATA="llvm-profdata-14"
    else
        # Try rustup component
        print_warning "llvm-profdata not found in PATH, attempting to use rustup component..."
        if rustup component add llvm-tools-preview &> /dev/null; then
            # Find the llvm-profdata from rustup
            RUSTUP_LLVM=$(rustup which rustc 2>/dev/null | xargs dirname)/../lib/rustlib/$(rustc -vV | grep host | cut -d' ' -f2)/bin/llvm-profdata
            if [ -x "$RUSTUP_LLVM" ]; then
                LLVM_PROFDATA="$RUSTUP_LLVM"
            else
                print_error "Could not find llvm-profdata. Please install LLVM tools:"
                echo "  Ubuntu/Debian: sudo apt install llvm"
                echo "  Fedora: sudo dnf install llvm"
                echo "  macOS: brew install llvm"
                echo "  Or: rustup component add llvm-tools-preview"
                exit 1
            fi
        else
            print_error "Could not install llvm-tools-preview via rustup"
            exit 1
        fi
    fi

    echo "Using llvm-profdata: $LLVM_PROFDATA"

    # Check for cargo
    if ! command -v cargo &> /dev/null; then
        print_error "cargo not found. Please install Rust toolchain."
        exit 1
    fi
}

# Clean previous builds and profile data
clean_previous() {
    print_step "Cleaning previous builds and profile data..."

    # Remove old profile data
    rm -rf "$PGO_DATA_DIR"
    mkdir -p "$PGO_DATA_DIR"

    # Clean cargo build artifacts for PGO profiles
    cargo clean --profile release-instrumented 2>/dev/null || true
    cargo clean --profile release-pgo 2>/dev/null || true

    echo "Cleaned profile data directory: $PGO_DATA_DIR"
}

# Build instrumented binary
build_instrumented() {
    print_step "Building instrumented binary (Step 1/3)..."

    export RUSTFLAGS="-C profile-generate=$PGO_DATA_DIR"

    # Build the library and benchmarks with instrumentation
    cargo build --profile release-instrumented
    cargo build --profile release-instrumented --benches

    echo "Instrumented build complete"
}

# Run benchmarks to generate profile data
generate_profile_data() {
    print_step "Running benchmarks to generate profile data (Step 2/3)..."

    # Set RUSTFLAGS for instrumented run
    export RUSTFLAGS="-C profile-generate=$PGO_DATA_DIR"

    # Run the primary benchmarks that exercise hot paths
    # These benchmarks will generate profile data based on real workloads

    echo "Running pq_bench..."
    cargo bench --profile release-instrumented --bench pq_bench -- --noplot 2>/dev/null || \
        cargo bench --profile release-instrumented --bench pq_bench 2>/dev/null || \
        print_warning "pq_bench failed or not found, continuing..."

    echo "Running search_bench..."
    cargo bench --profile release-instrumented --bench search_bench -- --noplot 2>/dev/null || \
        cargo bench --profile release-instrumented --bench search_bench 2>/dev/null || \
        print_warning "search_bench failed or not found, continuing..."

    echo "Running distance_bench..."
    cargo bench --profile release-instrumented --bench distance_bench -- --noplot 2>/dev/null || \
        cargo bench --profile release-instrumented --bench distance_bench 2>/dev/null || \
        print_warning "distance_bench failed or not found, continuing..."

    echo "Running ivf_bench..."
    cargo bench --profile release-instrumented --bench ivf_bench -- --noplot 2>/dev/null || \
        cargo bench --profile release-instrumented --bench ivf_bench 2>/dev/null || \
        print_warning "ivf_bench failed or not found, continuing..."

    echo "Running hnsw_bench..."
    cargo bench --profile release-instrumented --bench hnsw_bench -- --noplot 2>/dev/null || \
        cargo bench --profile release-instrumented --bench hnsw_bench 2>/dev/null || \
        print_warning "hnsw_bench failed or not found, continuing..."

    # Check if any profile data was generated
    PROFRAW_COUNT=$(find "$PGO_DATA_DIR" -name "*.profraw" 2>/dev/null | wc -l)
    if [ "$PROFRAW_COUNT" -eq 0 ]; then
        print_error "No profile data generated. Benchmarks may have failed."
        exit 1
    fi

    echo "Generated $PROFRAW_COUNT profile data files"
}

# Merge profile data
merge_profile_data() {
    print_step "Merging profile data..."

    # Find all .profraw files and merge them
    PROFRAW_FILES=$(find "$PGO_DATA_DIR" -name "*.profraw" -type f)

    if [ -z "$PROFRAW_FILES" ]; then
        print_error "No .profraw files found in $PGO_DATA_DIR"
        exit 1
    fi

    # Merge all profile data into a single file
    $LLVM_PROFDATA merge -o "$MERGED_PROFDATA" $PROFRAW_FILES

    if [ ! -f "$MERGED_PROFDATA" ]; then
        print_error "Failed to create merged profile data"
        exit 1
    fi

    PROFDATA_SIZE=$(du -h "$MERGED_PROFDATA" | cut -f1)
    echo "Merged profile data: $MERGED_PROFDATA ($PROFDATA_SIZE)"
}

# Build optimized binary with profile data
build_optimized() {
    print_step "Building PGO-optimized binary (Step 3/3)..."

    # Build with profile data and native CPU optimizations
    export RUSTFLAGS="-C profile-use=$MERGED_PROFDATA -C target-cpu=native"

    cargo build --profile release-pgo

    echo "PGO-optimized build complete"
}

# Report results
report_results() {
    print_step "Build Summary"

    echo ""
    echo "Profile data location: $MERGED_PROFDATA"
    echo ""

    # Find and report binary sizes
    RELEASE_BIN="$PROJECT_DIR/target/release/libforge_db.rlib"
    PGO_BIN="$PROJECT_DIR/target/release-pgo/libforge_db.rlib"

    if [ -f "$RELEASE_BIN" ] && [ -f "$PGO_BIN" ]; then
        RELEASE_SIZE=$(du -h "$RELEASE_BIN" | cut -f1)
        PGO_SIZE=$(du -h "$PGO_BIN" | cut -f1)
        echo "Binary sizes:"
        echo "  Standard release: $RELEASE_SIZE"
        echo "  PGO-optimized:    $PGO_SIZE"
    fi

    echo ""
    echo -e "${GREEN}PGO build complete!${NC}"
    echo ""
    echo "The optimized library is at:"
    echo "  $PROJECT_DIR/target/release-pgo/"
    echo ""
    echo "To use the PGO-optimized build in your project:"
    echo "  cargo build --profile release-pgo"
    echo ""
    echo "To run benchmarks with PGO optimization:"
    echo "  RUSTFLAGS=\"-C profile-use=$MERGED_PROFDATA -C target-cpu=native\" \\"
    echo "    cargo bench --profile release-pgo"
    echo ""
    echo "Expected improvement: 10-20% overall performance gain"
}

# Main execution
main() {
    echo "=============================================="
    echo " forge-db Profile-Guided Optimization Build"
    echo "=============================================="

    cd "$PROJECT_DIR"

    check_dependencies
    clean_previous
    build_instrumented
    generate_profile_data
    merge_profile_data
    build_optimized
    report_results
}

main "$@"
