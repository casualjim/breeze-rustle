use anyhow::Result;
use breeze::aiproviders::voyage::{
  self, Config, EmbeddingModel, EmbeddingRequest, Tier, VoyageClient,
};
use std::env;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<()> {
  // Initialize logging
  tracing_subscriber::fmt::init();

  // Get API key from environment
  let api_key = match env::var("VOYAGE_API_KEY") {
    Ok(key) => key,
    Err(_) => {
      eprintln!("Skipping example - VOYAGE_API_KEY not set");
      eprintln!("To run this example:");
      eprintln!("  export VOYAGE_API_KEY=your-api-key");
      eprintln!("  cargo run --example voyage_embeddings");
      return Ok(());
    }
  };

  // Create configuration
  let config = Config::new(api_key, Tier::Tier1, EmbeddingModel::VoyageCode3);

  // Create client
  let client = voyage::new_client(config)?;

  // Example texts to embed
  let texts = vec![
    r#"def hello_world():
    """A simple hello world function."""
    print("Hello, World!")
    return 42"#,
    r#"fn main() {
    println!("Hello, World!");
    let result = calculate_sum(5, 7);
    println!("Sum: {}", result);
}"#,
    "The quick brown fox jumps over the lazy dog",
  ];

  // Create embedding request
  let request = EmbeddingRequest {
    input: texts.iter().map(|s| s.to_string()).collect(),
    model: EmbeddingModel::VoyageCode3.api_name().to_string(),
    input_type: Some("document".to_string()),
    output_dimension: None,
    truncation: None,
  };

  // Estimate tokens (rough estimate: ~4 chars per token)
  let estimated_tokens: u32 = texts.iter().map(|t| (t.len() / 4) as u32).sum();

  // Get embeddings
  println!(
    "Embedding {} texts (estimated {} tokens)...",
    texts.len(),
    estimated_tokens
  );
  let response = client.embed(request, estimated_tokens).await?;

  // Print results
  println!("\nEmbedding Results:");
  println!("Model: {}", response.model);
  println!("Number of embeddings: {}", response.data.len());
  println!("Total tokens used: {}", response.usage.total_tokens);

  for (idx, embedding_data) in response.data.iter().enumerate() {
    println!(
      "Text {}: embedding dimension = {}, first 5 values = {:?}",
      idx,
      embedding_data.embedding.len(),
      &embedding_data.embedding[..5.min(embedding_data.embedding.len())]
    );
  }

  Ok(())
}
