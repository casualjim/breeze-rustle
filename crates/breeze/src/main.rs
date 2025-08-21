use breeze::app::SearchRequest;
#[cfg(feature = "perfprofiling")]
use breeze::cli::DebugCommands;
use breeze::cli::{Cli, Commands, ProjectCommands, SearchGranularity, TaskCommands};
use std::path::PathBuf;
use syntastica::highlight;
use syntastica::language_set::SupportedLanguage;
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
      Ok(app) => match cmd {
        ProjectCommands::Create {
          name,
          directory,
          description,
        } => {
          match app
            .create_project(name, directory.to_string_lossy().to_string(), description)
            .await
          {
            Ok(project) => {
              println!("Project created successfully!");
              println!("ID: {}", project.id);
              println!("Name: {}", project.name);
              println!("Directory: {}", project.directory);
              if let Some(desc) = &project.description {
                println!("Description: {}", desc);
              }
            }
            Err(e) => {
              error!("Failed to create project: {}", e);
              std::process::exit(1);
            }
          }
        }
        ProjectCommands::List => match app.list_projects().await {
          Ok(projects) => {
            if projects.is_empty() {
              println!("No projects found.");
            } else {
              println!("Projects:");
              for project in projects {
                println!("\n  ID: {}", project.id);
                println!("  Name: {}", project.name);
                println!("  Directory: {}", project.directory);
                if let Some(desc) = &project.description {
                  println!("  Description: {}", desc);
                }
              }
            }
          }
          Err(e) => {
            error!("Failed to list projects: {}", e);
            std::process::exit(1);
          }
        },
        ProjectCommands::Show { id } => match app.get_project(&id).await {
          Ok(project) => {
            println!("ID: {}", project.id);
            println!("Name: {}", project.name);
            println!("Directory: {}", project.directory);
            if let Some(desc) = &project.description {
              println!("Description: {}", desc);
            }
            println!("Created: {}", project.created_at);
            println!("Updated: {}", project.updated_at);
          }
          Err(e) => {
            error!("Failed to get project: {}", e);
            std::process::exit(1);
          }
        },
        ProjectCommands::Update {
          id,
          name,
          description,
        } => match app.update_project(&id, name, description).await {
          Ok(project) => {
            println!("Project updated successfully!");
            println!("ID: {}", project.id);
            println!("Name: {}", project.name);
            if let Some(desc) = &project.description {
              println!("Description: {}", desc);
            }
          }
          Err(e) => {
            error!("Failed to update project: {}", e);
            std::process::exit(1);
          }
        },
        ProjectCommands::Delete { id } => match app.delete_project(&id).await {
          Ok(_) => {
            println!("Project deleted successfully!");
          }
          Err(e) => {
            error!("Failed to delete project: {}", e);
            std::process::exit(1);
          }
        },
        ProjectCommands::Index { id } => match app.index_project(&id).await {
          Ok(resp) => {
            println!("Indexing task submitted successfully!");
            println!("Task ID: {}", resp.task_id);
            println!("Status: {}", resp.status);
          }
          Err(e) => {
            error!("Failed to index project: {}", e);
            std::process::exit(1);
          }
        },
      },
      Err(e) => {
        error!("Failed to initialize app: {}", e);
        std::process::exit(1);
      }
    },

    Commands::Task(cmd) => match breeze::App::new(config, shutdown_token.clone()).await {
      Ok(app) => match cmd {
        TaskCommands::Show { id } => match app.get_task(&id).await {
          Ok(task) => {
            println!("Task ID: {}", task.id);
            println!("Project ID: {}", task.project_id);
            println!("Path: {}", task.path);
            println!("Status: {}", task.status);
            println!("Created: {}", task.created_at);
            if let Some(started) = task.started_at {
              println!("Started: {}", started);
            }
            if let Some(completed) = task.completed_at {
              println!("Completed: {}", completed);
            }
            if let Some(error) = &task.error {
              println!("Error: {}", error);
            }
            if let Some(files) = task.files_indexed {
              println!("Files indexed: {}", files);
            }
          }
          Err(e) => {
            error!("Failed to get task: {}", e);
            std::process::exit(1);
          }
        },
        TaskCommands::List { limit } => match app.list_tasks(limit).await {
          Ok(tasks) => {
            if tasks.is_empty() {
              println!("No tasks found.");
            } else {
              println!("Tasks:");
              for task in tasks {
                println!("\n  ID: {}", task.id);
                println!("  Project: {}", task.project_id);
                println!("  Status: {}", task.status);
                println!("  Created: {}", task.created_at);
                if let Some(files) = task.files_indexed {
                  println!("  Files indexed: {}", files);
                }
              }
            }
          }
          Err(e) => {
            error!("Failed to list tasks: {}", e);
            std::process::exit(1);
          }
        },
      },
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
    } => {
      info!("Starting search for: \"{}\"", query);

      match breeze::App::new(config, shutdown_token.clone()).await {
        Ok(app) => {
          let req = SearchRequest {
            project_id: None, // No project ID for CLI search
            path: path.map(|p| p.display().to_string()),
            query,
            limit,
            chunks_per_file,
            languages,
            granularity: granularity
              .map(|g| match g {
                SearchGranularity::Document => breeze_server::types::SearchGranularity::Document,
                SearchGranularity::Chunk => breeze_server::types::SearchGranularity::Chunk,
              })
              .or(Some(breeze_server::types::SearchGranularity::Chunk)),
            node_types,
            node_name_pattern,
            parent_context_pattern,
            scope_depth,
            has_definitions,
            has_references,
          };

          match app.search(req).await {
            Ok(results) => {
              if results.is_empty() {
                println!("No results found.");
              } else {
                let theme = syntastica_themes::catppuccin::mocha();
                let mut renderer = syntastica::renderer::TerminalRenderer::default();
                let mut language_set = syntastica_parsers::LanguageSetImpl::default();
                language_set.preload_all().unwrap();
                println!("\nFound {} results:", results.len());
                for (idx, result) in results.iter().enumerate() {
                  println!("\n[{}] {}", idx + 1, result.file_path);
                  println!(
                    "   Score: {:.4} | Chunks: {}",
                    result.relevance_score, result.chunk_count
                  );

                  if !result.chunks.is_empty() {
                    println!("   Found {} relevant chunks:", result.chunks.len());
                    for (chunk_idx, chunk) in result.chunks.iter().enumerate() {
                      println!(
                        "\n   Chunk {} (lines {}-{}, score: {:.4}):",
                        chunk_idx + 1,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.relevance_score
                      );

                      // Determine language from file extension
                      let language = syntastica_parsers::Lang::for_name(
                        &chunk.language.to_lowercase(),
                        &mut language_set,
                      )
                      .ok();

                      if let Some(language) = language {
                        // Preload the language to ensure it's ready for highlighting
                        // language_set.preload(&[language]).unwrap();
                        let highlighted = highlight(
                          &chunk.content,
                          language,
                          &mut language_set,
                          &mut renderer,
                          theme.clone(),
                        )
                        .unwrap();

                        // Highlight content using syntastica
                        // Print highlighted lines with indentation
                        for line in highlighted.lines() {
                          println!("      {}", line);
                        }
                      } else {
                        println!("{}", &chunk.content);
                      }
                    }
                  }
                  println!("{}", "-".repeat(80));
                }
              }
            }
            Err(e) => {
              error!("Search failed: {}", e);
              std::process::exit(1);
            }
          }
        }
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
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
              error!("Failed to create config directory: {}", e);
              std::process::exit(1);
            }
          }

          // Write the commented config file
          let config_content = breeze::Config::generate_commented_config();
          if let Err(e) = tokio::fs::write(&config_path, config_content).await {
            error!("Failed to write config file: {}", e);
            std::process::exit(1);
          }

          println!(
            "✅ Configuration file created at: {}",
            config_path.display()
          );
          println!("\nYou can now:");
          println!("  1. Edit the config file to customize settings");
          println!("  2. Run 'breeze serve' to start the API server");
          println!("  3. Create a project with 'breeze project create <name> <path>'");
          println!("  4. Index it with 'breeze project index <project-id>'");
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
            ..Default::default()
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

    Commands::Serve => {
      info!(
        "Starting Breeze API server on ports HTTP:{} HTTPS:{}",
        config.server.ports.http, config.server.ports.https
      );

      // Additional startup configuration details
      // Max file size (configured)
      if let Some(size) = config.indexer.limits.file_size.as_deref() {
        info!("Max file size: {} bytes", size);
      } else {
        info!("Max file size: 5MB");
      }

      // Max chunk size (configured)
      match config.indexer.limits.max_chunk_size {
        Some(sz) => info!("Max chunk size (configured): {} tokens", sz),
        None => info!("Max chunk size (configured): not set"),
      }

      // Embedding provider/model/dimensions (configured)
      let provider_name = config.embeddings.provider.clone();
      let (model_name, embedding_dim) = match provider_name.as_str() {
        "local" => {
          let model = config
            .embeddings
            .local
            .as_ref()
            .map(|l| l.model.clone())
            .unwrap_or_else(|| "(default) BAAI/bge-small-en-v1.5".to_string());
          let dim = config
            .embeddings
            .local
            .as_ref()
            .and_then(|l| l.embedding_dim)
            .map(|d| d.to_string())
            .unwrap_or_else(|| "unknown".to_string());
          (model, dim)
        }
        "voyage" => {
          let model = config
            .embeddings
            .voyage
            .as_ref()
            .map(|v| v.model.clone())
            .unwrap_or_else(|| "(unset)".to_string());
          let dim = config
            .embeddings
            .voyage
            .as_ref()
            .and_then(|v| v.embedding_dim)
            .map(|d| d.to_string())
            .unwrap_or_else(|| "unknown".to_string());
          (model, dim)
        }
        other => {
          // OpenAI-like (custom) provider
          if let Some(p) = config.embeddings.providers.get(other) {
            let dim = p
              .embedding_dim
              .map(|d| d.to_string())
              .unwrap_or_else(|| "unknown".to_string());
            (p.model.clone(), dim)
          } else {
            ("(unset)".to_string(), "unknown".to_string())
          }
        }
      };
      info!("Embedding provider: {}", provider_name);
      info!("Embedding model: {}", model_name);
      info!("Embedding dimensions: {}", embedding_dim);

      // Convert indexer config
      let indexer_config = match config.to_indexer_config() {
        Ok(cfg) => cfg,
        Err(e) => {
          error!("Failed to convert indexer config: {}", e);
          std::process::exit(1);
        }
      };

      // Log effective (optimal) chunk size based on provider/tier
      let effective_chunk = indexer_config.optimal_chunk_size();
      info!("Max chunk size (effective): {} tokens", effective_chunk);

      // Create server config from breeze config
      let server_config = breeze_server::Config {
        tls_enabled: !config.server.tls.disabled,
        domains: config
          .server
          .tls
          .letsencrypt
          .as_ref()
          .map(|le| le.domains.clone())
          .unwrap_or_default(),
        email: config
          .server
          .tls
          .letsencrypt
          .as_ref()
          .map(|le| le.emails.clone())
          .unwrap_or_default(),
        cache: config
          .server
          .tls
          .letsencrypt
          .as_ref()
          .map(|le| le.cert_dir.clone()),
        production: config
          .server
          .tls
          .letsencrypt
          .as_ref()
          .map(|le| le.production)
          .unwrap_or(false),
        tls_key: config
          .server
          .tls
          .keypair
          .as_ref()
          .map(|kp| PathBuf::from(&kp.tls_key)),
        tls_cert: config
          .server
          .tls
          .keypair
          .as_ref()
          .map(|kp| PathBuf::from(&kp.tls_cert)),
        https_port: config.server.ports.https,
        http_port: config.server.ports.http,
        indexer: indexer_config,
      };

      // Run the server
      match breeze_server::run(server_config, Some(shutdown_token)).await {
        Ok(_) => {
          debug!("Breeze server stopped successfully");
        }
        Err(e) => {
          error!("Server error: {}", e);
          std::process::exit(1);
        }
      }
    }
  }

  // Abort the signal handler task to prevent it from keeping the runtime alive
  signal_handle.abort();

  debug!("async_main completed");
}
