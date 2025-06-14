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
        use std::collections::HashMap;
        use futures_util::StreamExt;

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

        let mut file_chunk_counts: HashMap<String, usize> = HashMap::new();
        let mut total_chunks = 0;

        while let Some(chunk) = chunker.next().await {
            match chunk {
                Ok(chunk) => {
                    let file_path = chunk.file_path.clone();
                    *file_chunk_counts.entry(file_path).or_insert(0) += 1;
                    total_chunks += 1;
                    if total_chunks % 1000 == 0 {
                        info!("Processed {} chunks so far...", total_chunks);
                    }
                }
                Err(e) => {
                    error!("Error processing chunk: {}", e);
                }
            }
        }

        let elapsed = start.elapsed();

        // Print summary statistics
        println!("\n=== Chunking Summary ===");
        println!("Directory: {}", path.display());
        println!("Total files processed: {}", file_chunk_counts.len());
        println!("Total chunks: {}", total_chunks);
        println!("Time elapsed: {:.2}s", elapsed.as_secs_f64());
        println!("Chunks per second: {:.0}", total_chunks as f64 / elapsed.as_secs_f64());

        if file_chunk_counts.len() > 0 {
          let avg_chunks_per_file = total_chunks as f64 / file_chunk_counts.len() as f64;
          println!("Average chunks per file: {:.1}", avg_chunks_per_file);
        }
      }
    },
  }
}
