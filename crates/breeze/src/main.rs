use std::path::Path;
use tracing::{error, info};

#[tokio::main]
async fn main() {
  let _log_guard = breeze::init_logging(env!("CARGO_PKG_NAME"));

  info!("Starting Breeze indexer");

  let config = breeze::Config::default();
  info!("Loaded configuration: {:?}", config);

  match breeze::App::new(config).await {
    Ok(app) => {
      let index_path = Path::new("/Users/ivan/github/kuzudb/kuzu");
      info!("Indexing directory: {}", index_path.display());

      match app.index(index_path).await {
        Ok(_) => info!("Indexing completed successfully!"),
        Err(e) => {
          error!("Indexing failed: {}", e);
          std::process::exit(1);
        }
      }
    }
    Err(e) => {
      error!("Failed to initialize app: {}", e);
      std::process::exit(1);
    }
  }
  // let mut chunker = breeze_chunkers::walk_project("/Users/ivan/github/kuzudb/kuzu", WalkOptions {
  //     max_chunk_size: 2048,
  //     tokenizer: breeze_chunkers::Tokenizer::default(),
  //     max_parallel: 16,
  //     max_file_size: Some(1024 * 1024 * 5), // 5MB
  // });

  // let mut file_chunk_counts: HashMap<String, usize> = HashMap::new();
  // let mut total_chunks = 0;

  // while let Some(chunk) = chunker.next().await {
  //     match chunk {
  //         Ok(chunk) => {
  //             let file_path = chunk.file_path.clone();
  //             *file_chunk_counts.entry(file_path).or_insert(0) += 1;
  //             total_chunks += 1;
  //             if total_chunks % 1000 == 0 {
  //                 println!("Processed {} chunks so far...", total_chunks);
  //             }
  //             // println!("Chunk: {:?}", chunk);
  //         }
  //         Err(e) => {
  //             eprintln!("Error processing chunk: {}", e);
  //         }
  //     }
  // }

  // // Print summary statistics
  // println!("\n=== Summary ===");
  // println!("Total files processed: {}", file_chunk_counts.len());
  // println!("Total chunks: {}", total_chunks);
  // println!("\nChunks per file:");
}
