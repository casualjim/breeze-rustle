#[cfg(feature = "perfprofiling")]
use breeze::cli::DebugCommands;
use breeze::cli::{Cli, Commands, McpMode};
use tracing::{error, info, warn};

fn main() {
  breeze::ensure_ort_initialized().expect("Failed to initialize ONNX runtime");
  let _log_guard = breeze::init_logging(env!("CARGO_PKG_NAME"));

  let rt = tokio::runtime::Builder::new_multi_thread()
    .enable_all()
    .build()
    .expect("Failed to create Tokio runtime");
  rt.block_on(async_main());
  // Ensure all tasks are completed before exiting
  rt.shutdown_timeout(std::time::Duration::from_secs(10));
}

async fn async_main() {
  let cli = Cli::parse();

  // Load configuration
  let mut config = if let Some(config_path) = &cli.config {
    // User specified a config path explicitly
    match breeze::Config::from_file(config_path) {
      Ok(cfg) => {
        info!("Loaded configuration from: {}", config_path.display());
        cfg
      }
      Err(e) => {
        warn!(
          "Failed to load config from {}: {}",
          config_path.display(),
          e
        );
        info!("Using default configuration");
        breeze::Config::default()
      }
    }
  } else {
    // No config specified, try default location
    match breeze::Config::default_config_path() {
      Ok(default_path) if default_path.exists() => match breeze::Config::from_file(&default_path) {
        Ok(cfg) => {
          info!("Loaded configuration from: {}", default_path.display());
          cfg
        }
        Err(e) => {
          warn!(
            "Failed to load config from {}: {}",
            default_path.display(),
            e
          );
          info!("Using default configuration");
          breeze::Config::default()
        }
      },
      _ => {
        // No config file found, use defaults
        let mut cfg = breeze::Config::default();
        cfg.apply_env_overrides();
        cfg
      }
    }
  };

  match cli.command {
    Commands::Index {
      path,
      database,
      embedding_provider,
      model,
      voyage_api_key,
      voyage_tier,
      voyage_model,
      max_chunk_size,
      max_file_size,
      max_parallel_files,
    } => {
      // Apply CLI/env overrides (clap has already handled env vars and parsing)
      if let Some(db) = database {
        config.indexer.database_path = db;
      }

      if let Some(provider) = embedding_provider {
        config.indexer.embedding_provider = provider;
      }

      if let Some(m) = model {
        config.indexer.model = m;
      }

      // Ensure voyage config exists if using voyage provider
      if config.indexer.embedding_provider == breeze::config::EmbeddingProvider::Voyage
        && config.indexer.voyage.is_none()
      {
        error!(
          "Voyage embedding provider selected but no voyage configuration found. Please ensure your config file has a [voyage] section or set the BREEZE_VOYAGE_API_KEY environment variable."
        );
        std::process::exit(1);
      }

      // Handle voyage-specific overrides
      let api_key = voyage_api_key.or_else(|| std::env::var("BREEZE_VOYAGE_API_KEY").ok());

      if let Some(api_key) = api_key {
        if let Some(ref mut voyage) = config.indexer.voyage {
          voyage.api_key = api_key;
        }
      }

      if let Some(tier) = voyage_tier {
        if let Some(ref mut voyage) = config.indexer.voyage {
          voyage.tier = tier;
        }
      }

      if let Some(model) = voyage_model {
        if let Some(ref mut voyage) = config.indexer.voyage {
          voyage.model = model;
        }
      }

      if let Some(size) = max_chunk_size {
        config.indexer.max_chunk_size = size;
      }
      if let Some(size) = max_file_size {
        config.indexer.max_file_size = Some(size);
      }
      if let Some(parallel) = max_parallel_files {
        config.indexer.max_parallel_files = parallel;
      }

      info!("Starting indexing of: {}", path.display());
      info!("Using configuration: {:?}", config);

      match breeze::App::new(config).await {
        Ok(app) => match app.index(&path).await {
          Ok(_) => {
            info!("Indexing completed successfully!");
          }
          Err(e) => {
            error!("Indexing failed: {}", e);
            std::process::exit(1);
          }
        },
        Err(e) => {
          error!("Failed to initialize app: {}", e);
          std::process::exit(1);
        }
      }
    }

    Commands::Search {
      query,
      database,
      limit,
      full,
    } => {
      // Apply CLI overrides
      if let Some(db) = database {
        config.indexer.database_path = db;
      }

      info!("Starting search for: \"{}\"", query);
      info!("Using configuration: {:?}", config);

      match breeze::App::new(config).await {
        Ok(app) => match app.search(&query, limit).await {
          Ok(results) => {
            if results.is_empty() {
              println!("No results found for query: \"{}\"", query);
            } else {
              println!("\nFound {} results for query: \"{}\"", results.len(), query);
              println!("{}", breeze::cli::format_results(&results, full));
            }
          }
          Err(e) => {
            error!("Search failed: {}", e);
            std::process::exit(1);
          }
        },
        Err(e) => {
          error!("Failed to initialize app: {}", e);
          std::process::exit(1);
        }
      }
    }

    Commands::Init { force } => {
      match breeze::Config::default_config_path() {
        Ok(config_path) => {
          // Check if config already exists
          if config_path.exists() && !force {
            error!(
              "Configuration file already exists at: {}\nUse --force to overwrite",
              config_path.display()
            );
            std::process::exit(1);
          }

          // Create config directory if it doesn't exist
          if let Some(parent) = config_path.parent() {
            if let Err(e) = std::fs::create_dir_all(parent) {
              error!("Failed to create config directory: {}", e);
              std::process::exit(1);
            }
          }

          // Write the commented config file
          let config_content = breeze::Config::generate_commented_config();
          if let Err(e) = std::fs::write(&config_path, config_content) {
            error!("Failed to write config file: {}", e);
            std::process::exit(1);
          }

          println!(
            "✅ Configuration file created at: {}",
            config_path.display()
          );
          println!("\nYou can now:");
          println!("  1. Edit the config file to customize settings");
          println!("  2. Run 'breeze index /path/to/code' to start indexing");
          println!("\nThe config file is heavily commented to help you get started!");
        }
        Err(e) => {
          error!("Failed to determine config path: {}", e);
          std::process::exit(1);
        }
      }
    }

    Commands::Config { defaults } => {
      if defaults {
        // Show the default configuration
        println!(
          "{}",
          toml::to_string_pretty(&breeze::Config::default()).unwrap()
        );
      } else {
        // Show the actual loaded configuration (which already factors in the config file)
        println!("{}", toml::to_string_pretty(&config).unwrap());
      }
    }

    #[cfg(feature = "perfprofiling")]
    Commands::Debug { command } => match command {
      DebugCommands::ChunkDirectory {
        path,
        max_chunk_size,
        max_file_size,
        max_parallel,
        tokenizer,
      } => {
        use breeze_chunkers::{Tokenizer, WalkOptions, walk_project};
        use futures_util::StreamExt;
        use std::collections::BTreeMap;
        use std::sync::Arc;
        use tokio::sync::Mutex;

        info!("Starting chunk analysis of: {}", path.display());

        // Parse tokenizer string
        let tokenizer_obj = if tokenizer == "characters" {
          Tokenizer::Characters
        } else if let Some(encoding) = tokenizer.strip_prefix("tiktoken:") {
          Tokenizer::Tiktoken(encoding.to_string())
        } else if let Some(model) = tokenizer.strip_prefix("hf:") {
          Tokenizer::HuggingFace(model.to_string())
        } else {
          error!(
            "Invalid tokenizer format: {}. Use 'characters', 'tiktoken:cl100k_base', or 'hf:org/repo'",
            tokenizer
          );
          std::process::exit(1);
        };

        let start = std::time::Instant::now();

        let chunker = walk_project(
          &path,
          WalkOptions {
            max_chunk_size,
            tokenizer: tokenizer_obj,
            max_parallel,
            max_file_size,
            large_file_threads: 4,
          },
        );

        let file_chunk_counts = Arc::new(Mutex::new(BTreeMap::<String, usize>::new()));
        #[cfg(feature = "perfprofiling")]
        let file_start_times = Arc::new(Mutex::new(BTreeMap::<String, std::time::Instant>::new()));
        let total_chunks = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Process chunks concurrently with a buffer
        let concurrent_limit = max_parallel * 2; // Process more chunks than walker threads

        chunker
          .map(|chunk_result| {
            let file_chunk_counts = Arc::clone(&file_chunk_counts);
            #[cfg(feature = "perfprofiling")]
            let file_start_times = Arc::clone(&file_start_times);
            let total_chunks = Arc::clone(&total_chunks);

            async move {
              match chunk_result {
                Ok(project_chunk) => {
                  let file_path = project_chunk.file_path.clone();

                  // Track the first time we see a chunk from this file
                  #[cfg(feature = "perfprofiling")]
                  {
                    let mut start_times = file_start_times.lock().await;
                    start_times
                      .entry(file_path.clone())
                      .or_insert_with(std::time::Instant::now);
                  }

                  // Check if this is an EOF chunk
                  if let breeze_chunkers::Chunk::EndOfFile {
                    file_path: eof_path,
                    ..
                  } = &project_chunk.chunk
                  {
                    // Record the file processing time
                    #[cfg(feature = "perfprofiling")]
                    {
                      let start_time = {
                        let mut start_times = file_start_times.lock().await;
                        start_times.remove(eof_path)
                      };

                      if let Some(start_time) = start_time {
                        let duration = start_time.elapsed();
                        let file_size = tokio::fs::metadata(eof_path)
                          .await
                          .map(|m| m.len())
                          .unwrap_or(0);

                        // Detect language from the file path
                        let language = if let Ok(Some(detection)) =
                          hyperpolyglot::detect(std::path::Path::new(eof_path))
                        {
                          detection.language().to_string()
                        } else {
                          "unknown".to_string()
                        };

                        // Record to performance tracker
                        breeze_chunkers::performance::get_tracker().record_file_processing(
                          eof_path.clone(),
                          language,
                          file_size,
                          duration,
                          "chunking_complete",
                        );
                      }
                    }
                  }

                  // Update counts
                  {
                    let mut counts = file_chunk_counts.lock().await;
                    *counts.entry(file_path).or_insert(0) += 1;
                  }

                  let count = total_chunks.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                  if count % 1000 == 0 {
                    info!("Processed {} chunks so far...", count);
                  }
                }
                Err(e) => {
                  error!("Error processing chunk: {}", e);
                }
              }
            }
          })
          .buffer_unordered(concurrent_limit)
          .collect::<Vec<_>>()
          .await;

        let elapsed = start.elapsed();

        // Get final values
        let final_counts = file_chunk_counts.lock().await;
        let final_total = total_chunks.load(std::sync::atomic::Ordering::Relaxed);

        // Print summary statistics
        println!("\n=== Chunking Summary ===");
        println!("Directory: {}", path.display());
        println!("Total files processed: {}", final_counts.len());
        println!("Total chunks: {}", final_total);
        println!("Time elapsed: {:.2}s", elapsed.as_secs_f64());
        println!(
          "Chunks per second: {:.0}",
          final_total as f64 / elapsed.as_secs_f64()
        );

        if !final_counts.is_empty() {
          let avg_chunks_per_file = final_total as f64 / final_counts.len() as f64;
          println!("Average chunks per file: {:.1}", avg_chunks_per_file);
        }
      }
    },

    Commands::Mcp { mode, port, host } => {
      use breeze::mcp::BreezeMcpServer;

      info!("Starting MCP server with configuration: {:?}", config);

      match BreezeMcpServer::new(config).await {
        Ok(server) => match mode {
          McpMode::Stdio => {
            info!("Running MCP server in STDIO mode");
            if let Err(e) = server.run_stdio().await {
              error!("MCP server error: {}", e);
              std::process::exit(1);
            }
          }
          McpMode::Http => {
            info!("Running MCP server in HTTP mode on {}:{}", host, port);
            if let Err(e) = server.run_http(&host, port).await {
              error!("MCP server error: {}", e);
              std::process::exit(1);
            }
          }
        },
        Err(e) => {
          error!("Failed to initialize MCP server: {}", e);
          std::process::exit(1);
        }
      }
    }
  }
}
