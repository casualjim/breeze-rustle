use breeze::cli::{Cli, Commands, DebugCommands};
use tracing::{error, info, warn};

#[tokio::main]
async fn main() {
  let cli = Cli::parse();

  let _log_guard = breeze::init_logging(env!("CARGO_PKG_NAME"));

  // Load configuration
  let mut config = if let Some(config_path) = &cli.config {
    match breeze::Config::from_file(config_path) {
      Ok(cfg) => {
        info!("Loaded configuration from: {}", config_path.display());
        cfg
      }
      Err(e) => {
        warn!("Failed to load config from {}: {}", config_path.display(), e);
        info!("Using default configuration");
        breeze::Config::default()
      }
    }
  } else {
    breeze::Config::default()
  };

  match cli.command {
    Commands::Index {
      path,
      database,
      model,
      device,
      max_chunk_size,
      max_file_size,
      max_parallel_files,
      batch_size,
    } => {
      // Apply CLI overrides
      if let Some(db) = database {
        config.database_path = db;
      }
      if let Some(m) = model {
        config.model = m;
      }
      if let Some(d) = device {
        config.device = d;
      }
      if let Some(size) = max_chunk_size {
        config.max_chunk_size = size;
      }
      if let Some(size) = max_file_size {
        config.max_file_size = Some(size);
      }
      if let Some(parallel) = max_parallel_files {
        config.max_parallel_files = parallel;
      }
      if let Some(batch) = batch_size {
        config.batch_size = batch;
      }

      info!("Starting indexing of: {}", path.display());
      info!("Using configuration: {:?}", config);

      match breeze::App::new(config).await {
        Ok(app) => match app.index(&path).await {
          Ok(_) => info!("Indexing completed successfully!"),
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
      query: _,
      database,
      limit: _,
      full: _,
    } => {
      // Apply CLI overrides
      if let Some(db) = database {
        config.database_path = db;
      }

      error!("Search functionality not yet implemented");
      std::process::exit(1);
    }

    Commands::Config { defaults } => {
      let cfg = if defaults {
        breeze::Config::default()
      } else {
        config
      };

      println!("{}", toml::to_string_pretty(&cfg).unwrap());
    }

    Commands::Debug { command } => match command {
      DebugCommands::ChunkDirectory {
        path,
        max_chunk_size,
        max_file_size,
        max_parallel,
        tokenizer,
      } => {
        use breeze_chunkers::{walk_project, WalkOptions, Tokenizer};
        use std::collections::BTreeMap;
        use futures_util::StreamExt;
        use std::sync::Arc;
        use tokio::sync::Mutex;

        info!("Starting chunk analysis of: {}", path.display());

        // Parse tokenizer string
        let tokenizer_obj = if tokenizer == "characters" {
          Tokenizer::Characters
        } else if let Some(_model) = tokenizer.strip_prefix("tiktoken:") {
          // For now, tiktoken only supports cl100k_base
          Tokenizer::Tiktoken
        } else if let Some(model) = tokenizer.strip_prefix("hf:") {
          Tokenizer::HuggingFace(model.to_string())
        } else {
          error!("Invalid tokenizer format: {}. Use 'characters', 'tiktoken:cl100k_base', or 'hf:org/repo'", tokenizer);
          std::process::exit(1);
        };

        let start = std::time::Instant::now();

        let mut chunker = walk_project(&path, WalkOptions {
            max_chunk_size,
            tokenizer: tokenizer_obj,
            max_parallel,
            max_file_size,
        });

        let file_chunk_counts = Arc::new(Mutex::new(BTreeMap::<String, usize>::new()));
        let file_start_times = Arc::new(Mutex::new(BTreeMap::<String, std::time::Instant>::new()));
        let total_chunks = Arc::new(std::sync::atomic::AtomicUsize::new(0));

        // Process chunks concurrently with a buffer
        let concurrent_limit = max_parallel * 2; // Process more chunks than walker threads

        chunker
            .map(|chunk_result| {
                let file_chunk_counts = Arc::clone(&file_chunk_counts);
                let file_start_times = Arc::clone(&file_start_times);
                let total_chunks = Arc::clone(&total_chunks);

                async move {
                    match chunk_result {
                        Ok(project_chunk) => {
                            let file_path = project_chunk.file_path.clone();

                            // Track the first time we see a chunk from this file
                            {
                                let mut start_times = file_start_times.lock().await;
                                start_times.entry(file_path.clone()).or_insert_with(std::time::Instant::now);
                            }

                            // Check if this is an EOF chunk
                            if let breeze_chunkers::Chunk::EndOfFile { file_path: eof_path, .. } = &project_chunk.chunk {
                                // Record the file processing time
                                let start_time = {
                                    let mut start_times = file_start_times.lock().await;
                                    start_times.remove(eof_path)
                                };

                                if let Some(start_time) = start_time {
                                    let duration = start_time.elapsed();
                                    let file_size = tokio::fs::metadata(eof_path).await
                                        .map(|m| m.len())
                                        .unwrap_or(0);

                                    // Detect language from the file path
                                    let language = if let Ok(Some(detection)) = hyperpolyglot::detect(std::path::Path::new(eof_path)) {
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
                                        "chunking_complete"
                                    );
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
        println!("Chunks per second: {:.0}", final_total as f64 / elapsed.as_secs_f64());

        if final_counts.len() > 0 {
          let avg_chunks_per_file = final_total as f64 / final_counts.len() as f64;
          println!("Average chunks per file: {:.1}", avg_chunks_per_file);
        }
      }
    },
  }
}
