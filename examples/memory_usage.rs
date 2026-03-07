//! Memory usage tracking for different index types.
//!
//! Reports approximate memory consumption for BruteForce, HNSW, and IVF-PQ
//! indices across different dataset sizes and dimensions.
//!
//! Run with: cargo run --example memory_usage --release

use forge_db::distance::DistanceMetric;
use forge_db::index::brute_force::BruteForceIndex;
use forge_db::index::hnsw::HNSWIndex;
use forge_db::index::ivf_pq::IVFPQIndex;
use forge_db::Vector;
use std::time::Instant;

fn rss_bytes() -> usize {
    // Read current RSS from /proc/self/stat on Linux or use sysctl on macOS
    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        let pid = std::process::id();
        let output = Command::new("ps")
            .args(["-o", "rss=", "-p", &pid.to_string()])
            .output()
            .ok();
        if let Some(out) = output {
            let s = String::from_utf8_lossy(&out.stdout);
            return s.trim().parse::<usize>().unwrap_or(0) * 1024; // ps reports in KB
        }
        0
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if line.starts_with("VmRSS:") {
                    let kb: usize = line
                        .split_whitespace()
                        .nth(1)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    return kb * 1024;
                }
            }
        }
        0
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux")))]
    {
        0
    }
}

fn format_bytes(bytes: usize) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    } else if bytes >= 1024 {
        format!("{:.2} KB", bytes as f64 / 1024.0)
    } else {
        format!("{} B", bytes)
    }
}

fn raw_data_size(n: usize, dim: usize) -> usize {
    n * dim * std::mem::size_of::<f32>() + n * std::mem::size_of::<u64>()
}

fn main() {
    println!("=== Memory Usage Tracking ===\n");
    println!(
        "{:<20} | {:>8} | {:>6} | {:>12} | {:>12} | {:>10} | {:>10}",
        "Index Type", "N", "Dim", "Raw Data", "RSS Delta", "Overhead", "Build Time"
    );
    println!("{}", "-".repeat(95));

    for &(n, dim) in &[(10_000, 128), (50_000, 128), (10_000, 768)] {
        let vectors: Vec<Vector> = (0..n).map(|i| Vector::random(i as u64, dim)).collect();
        let raw = raw_data_size(n, dim);

        // BruteForce
        {
            let before = rss_bytes();
            let start = Instant::now();
            let mut index = BruteForceIndex::new(DistanceMetric::Euclidean);
            for v in &vectors {
                index.add(v.clone());
            }
            let build_time = start.elapsed();
            let after = rss_bytes();
            let delta = after.saturating_sub(before);
            let overhead = if delta > raw {
                format!("{:.1}x", delta as f64 / raw as f64)
            } else {
                "~1.0x".to_string()
            };

            println!(
                "{:<20} | {:>8} | {:>6} | {:>12} | {:>12} | {:>10} | {:>8.1}ms",
                "BruteForce",
                n,
                dim,
                format_bytes(raw),
                format_bytes(delta),
                overhead,
                build_time.as_secs_f64() * 1000.0
            );
        }

        // HNSW
        {
            let before = rss_bytes();
            let start = Instant::now();
            let _index = HNSWIndex::build(vectors.clone(), 16, 200, DistanceMetric::Euclidean);
            let build_time = start.elapsed();
            let after = rss_bytes();
            let delta = after.saturating_sub(before);
            let overhead = if delta > raw {
                format!("{:.1}x", delta as f64 / raw as f64)
            } else {
                "~1.0x".to_string()
            };

            println!(
                "{:<20} | {:>8} | {:>6} | {:>12} | {:>12} | {:>10} | {:>8.1}ms",
                "HNSW (M=16)",
                n,
                dim,
                format_bytes(raw),
                format_bytes(delta),
                overhead,
                build_time.as_secs_f64() * 1000.0
            );
        }

        // IVF-PQ (only for reasonable sizes)
        if n >= 10_000 && dim <= 128 {
            let n_clusters = (n / 100).min(256);
            let before = rss_bytes();
            let start = Instant::now();
            let _index =
                IVFPQIndex::build(vectors.clone(), n_clusters, 8, DistanceMetric::Euclidean);
            let build_time = start.elapsed();
            let after = rss_bytes();
            let delta = after.saturating_sub(before);
            let compressed_size = n * 8; // 8 subvectors, 1 byte each
            let ratio = raw as f64 / compressed_size as f64;

            println!(
                "{:<20} | {:>8} | {:>6} | {:>12} | {:>12} | {:>10} | {:>8.1}ms",
                format!("IVF-PQ (c={},s=8)", n_clusters),
                n,
                dim,
                format_bytes(raw),
                format_bytes(delta),
                format!("{:.1}x comp", ratio),
                build_time.as_secs_f64() * 1000.0
            );
        }

        println!();
    }

    println!("Notes:");
    println!("  - Raw Data = N * dim * 4 bytes (f32) + N * 8 bytes (u64 id)");
    println!("  - RSS Delta = resident set size change (may include allocator overhead)");
    println!("  - Overhead = RSS Delta / Raw Data size");
    println!("  - IVF-PQ comp = raw data / compressed codes size");
}
