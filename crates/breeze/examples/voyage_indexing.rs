use breeze::{App, Config, EmbeddingProvider};
use std::env;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
  // Initialize logging
  tracing_subscriber::fmt::init();

  // Check for API key
  if std::env::var("BREEZE_VOYAGE_API_KEY")
    .or_else(|_| std::env::var("VOYAGE_API_KEY"))
    .is_err()
  {
    eprintln!("Error: BREEZE_VOYAGE_API_KEY or VOYAGE_API_KEY environment variable not set");
    eprintln!();
    eprintln!("To run this example:");
    eprintln!("  export BREEZE_VOYAGE_API_KEY=your-api-key");
    eprintln!("  cargo run --example voyage_indexing /path/to/code");
    return Ok(());
  }

  // Get path from command line
  let args: Vec<String> = env::args().collect();
  if args.len() < 2 {
    eprintln!("Usage: {} <path-to-index>", args[0]);
    return Ok(());
  }
  let path = Path::new(&args[1]);

  // Create config with Voyage provider
  let config = Config {
    embedding_provider: EmbeddingProvider::Voyage,
    ..Default::default()
  };

  // Voyage configuration will be loaded from environment variables:
  // - BREEZE_VOYAGE_API_KEY || VOYAGE_API_KEY (required)
  // - BREEZE_VOYAGE_TIER (optional, defaults to "free")
  // - BREEZE_VOYAGE_MODEL (optional, defaults to "voyage-code-3")

  // The max_chunk_size will be automatically adjusted based on the model's context length
  // For voyage-code-3, this will be around 28,800 tokens (90% of 32k)

  println!("Indexing {} with Voyage AI embeddings...", path.display());
  println!("Provider: {:?}", config.embedding_provider);

  // Create app and index
  let app = App::new(config).await?;
  let count = app.index(path).await.map_err(|e| e.to_string())?;

  println!("Successfully indexed {} documents", count);

  Ok(())
}
