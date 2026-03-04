use std::path::PathBuf;

use anyhow::Context;
use tracing::info;

use forge_server::{grpc, http, state};

/// Command-line arguments (minimal — use the config file for everything else).
#[derive(Debug)]
struct Args {
    config_path: Option<PathBuf>,
}

fn parse_args() -> Args {
    let mut args = std::env::args().skip(1);
    let mut config_path = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--config" | "-c" => {
                config_path = args.next().map(PathBuf::from);
            }
            _ => {}
        }
    }

    Args { config_path }
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = parse_args();

    // Load config
    let config = match args.config_path {
        Some(ref path) => {
            forge_db::ForgeConfig::from_file(path)
                .with_context(|| format!("loading config from {}", path.display()))?
        }
        None => {
            // Try default locations
            let candidates = ["forge.toml", "/etc/forge-db/forge.toml", "./forge.toml"];
            let mut loaded = None;
            for candidate in &candidates {
                if std::path::Path::new(candidate).exists() {
                    loaded = Some(
                        forge_db::ForgeConfig::from_file(candidate)
                            .with_context(|| format!("loading config from {candidate}"))?,
                    );
                    break;
                }
            }
            loaded.unwrap_or_default()
        }
    };

    config
        .validate()
        .context("configuration validation failed")?;

    // Initialize tracing
    init_tracing(&config.log);

    info!(
        grpc_addr = %config.server.grpc_addr,
        http_addr = %config.server.http_addr,
        data_dir = %config.data_dir.display(),
        "forge-server starting"
    );

    // Create data directory
    std::fs::create_dir_all(&config.data_dir)
        .with_context(|| format!("creating data dir {}", config.data_dir.display()))?;

    // Build shared application state
    let state = state::AppState::new(config.clone())
        .await
        .context("initializing application state")?;

    let state = std::sync::Arc::new(state);

    // Start Prometheus metrics
    let metrics_addr: std::net::SocketAddr = "0.0.0.0:9090".parse().unwrap();
    let metrics_handle = metrics_exporter_prometheus::PrometheusBuilder::new()
        .with_http_listener(metrics_addr)
        .install_recorder()
        .context("installing Prometheus recorder")?;
    drop(metrics_handle); // keep alive via process

    info!(addr = %metrics_addr, "Prometheus metrics endpoint");

    // Launch gRPC server
    let grpc_addr: std::net::SocketAddr = config
        .server
        .grpc_addr
        .parse()
        .context("parsing gRPC bind address")?;

    let http_addr: std::net::SocketAddr = config
        .server
        .http_addr
        .parse()
        .context("parsing HTTP bind address")?;

    let grpc_task = {
        let state = state.clone();
        let cfg = config.clone();
        tokio::spawn(async move {
            grpc::serve(grpc_addr, state, cfg)
                .await
                .context("gRPC server error")
        })
    };

    let http_task = {
        let state = state.clone();
        let cfg = config.clone();
        tokio::spawn(async move {
            http::serve(http_addr, state, cfg)
                .await
                .context("HTTP server error")
        })
    };

    // Wait for shutdown signal
    tokio::select! {
        res = grpc_task => {
            if let Ok(Err(e)) = res {
                tracing::error!(error = %e, "gRPC server exited with error");
            }
        }
        res = http_task => {
            if let Ok(Err(e)) = res {
                tracing::error!(error = %e, "HTTP server exited with error");
            }
        }
        _ = shutdown_signal() => {
            info!("shutdown signal received, stopping servers");
        }
    }

    info!("forge-server stopped");
    Ok(())
}

/// Wait for SIGINT or SIGTERM.
async fn shutdown_signal() {
    use tokio::signal;

    let ctrl_c = async {
        signal::ctrl_c().await.expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {}
        _ = terminate => {}
    }
}

fn init_tracing(log: &forge_db::config::LogConfig) {
    use tracing_subscriber::{EnvFilter, fmt};

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&log.level));

    match log.format {
        forge_db::config::LogFormat::Json => {
            fmt()
                .json()
                .with_env_filter(filter)
                .with_target(true)
                .with_current_span(log.include_spans)
                .init();
        }
        forge_db::config::LogFormat::Compact => {
            fmt()
                .compact()
                .with_env_filter(filter)
                .init();
        }
        forge_db::config::LogFormat::Pretty => {
            fmt()
                .pretty()
                .with_env_filter(filter)
                .init();
        }
    }
}
