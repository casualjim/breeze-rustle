use breeze::cli::{Cli, Commands};
use tokio::signal;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

/// Create a cancellation token that triggers on Ctrl+C or SIGTERM
fn create_shutdown_token() -> (CancellationToken, tokio::task::JoinHandle<()>) {
  let token = CancellationToken::new();
  let token_clone = token.clone();

  let handle = tokio::spawn(async move {
    let ctrl_c = async {
      signal::ctrl_c()
        .await
        .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
      signal::unix::signal(signal::unix::SignalKind::terminate())
        .expect("failed to install signal handler")
        .recv()
        .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
      _ = ctrl_c => {},
      _ = terminate => {},
    }

    info!("Shutdown signal received");
    token_clone.cancel();
  });

  (token, handle)
}

fn main() {
  rustls::crypto::aws_lc_rs::default_provider()
    .install_default()
    .expect("Failed to install AWS LC crypto provider");

  #[cfg(feature = "local-embeddings")]
  {
    breeze::ensure_ort_initialized().expect("Failed to initialize ONNX runtime");
  }
  let _log_guard = breeze::init_logging(env!("CARGO_PKG_NAME"));

  let rt = tokio::runtime::Builder::new_multi_thread()
    .enable_all()
    .build()
    .expect("Failed to create Tokio runtime");
  rt.block_on(async_main());

  // Ensure all tasks are completed before exiting
  rt.shutdown_timeout(std::time::Duration::from_secs(10));
  debug!("Runtime shutdown complete");
}

async fn async_main() {
  let cli = Cli::parse();

  // Create shutdown token for graceful shutdown
  let (shutdown_token, signal_handle) = create_shutdown_token();

  // Load configuration using config-rs
  let config = match breeze::Config::load(cli.config.clone()) {
    Ok((cfg, loaded_path)) => {
      if let Some(path) = loaded_path {
        info!("Loaded configuration from: {}", path.display());
      } else {
        info!("Loaded configuration from defaults and environment (no config file found)");
      }
      cfg
    }
    Err(e) => {
      error!("Failed to load configuration: {}", e);
      std::process::exit(1);
    }
  };

  match cli.command {
    Commands::Project(cmd) => match breeze::App::new(config, shutdown_token.clone()).await {
      Ok(app) => {
        if let Err(exit_code) = breeze::handlers::handle_project_commands(app, cmd).await {
          std::process::exit(exit_code);
        }
      }
      Err(e) => {
        error!("Failed to initialize app: {}", e);
        std::process::exit(1);
      }
    },

    Commands::Task(cmd) => match breeze::App::new(config, shutdown_token.clone()).await {
      Ok(app) => {
        if let Err(exit_code) = breeze::handlers::handle_task_commands(app, cmd).await {
          std::process::exit(exit_code);
        }
      }
      Err(e) => {
        error!("Failed to initialize app: {}", e);
        std::process::exit(1);
      }
    },

    Commands::Search {
      query,
      limit,
      chunks_per_file,
      languages,
      granularity,
      node_types,
      node_name_pattern,
      parent_context_pattern,
      scope_depth,
      has_definitions,
      has_references,
      path,
    } => match breeze::App::new(config, shutdown_token.clone()).await {
      Ok(app) => {
        if let Err(exit_code) = breeze::handlers::handle_search_command(
          app,
          breeze::handlers::SearchCommandArgs {
            query,
            limit,
            chunks_per_file,
            languages,
            granularity,
            node_types,
            node_name_pattern,
            parent_context_pattern,
            scope_depth,
            has_definitions,
            has_references,
            path,
          },
        )
        .await
        {
          std::process::exit(exit_code);
        }
      }
      Err(e) => {
        error!("Failed to initialize app: {}", e);
        std::process::exit(1);
      }
    },

    Commands::Init { force } => {
      if let Err(exit_code) = breeze::handlers::handle_init_command(force).await {
        std::process::exit(exit_code);
      }
    }

    Commands::Config { defaults } => {
      if let Err(exit_code) = breeze::handlers::handle_config_command(&config, defaults).await {
        std::process::exit(exit_code);
      }
    }

    #[cfg(feature = "perfprofiling")]
    Commands::Debug { command } => {
      if let Err(exit_code) = breeze::handlers::handle_debug_command(command).await {
        std::process::exit(exit_code);
      }
    }

    Commands::Serve => {
      if let Err(exit_code) = breeze::handlers::handle_serve_command(config, shutdown_token).await {
        std::process::exit(exit_code);
      }
    }
  }

  // Abort the signal handler task to prevent it from keeping the runtime alive
  signal_handle.abort();

  debug!("async_main completed");
}
