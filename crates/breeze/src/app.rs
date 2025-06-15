use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument};
use embed_anything::embeddings::embed::{Embedder, EmbedderBuilder};
use embed_anything::embeddings::local::text_embedding::ONNXModel;

use crate::config::Config;
use crate::indexer::Indexer;
use crate::models::CodeDocument;

pub struct App {
  config: Config,
  embedder: Arc<Embedder>,
  embedding_dim: usize,
  table: Arc<RwLock<lancedb::Table>>,
}

impl App {
  /// Create a new App instance with the given configuration
  #[instrument(skip(config), fields(database_path = %config.database_path.display(), model = %config.model, table_name = %config.table_name))]
  pub async fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
    info!("Initializing Breeze app");

    // Create LanceDB connection
    debug!(
      "Connecting to LanceDB at: {}",
      config.database_path.display()
    );
    let connection = lancedb::connect(
      config
        .database_path
        .to_str()
        .ok_or("Invalid database path")?,
    )
    .execute()
    .await?;
    info!("Set up LanceDB connection");

    // Load embed_anything embedder
    info!(
      "Loading embed_anything embedder with AllMiniLML6V2 model"
    );

    let embedder = EmbedderBuilder::new()
      .model_architecture("bert")
      .onnx_model_id(Some(ONNXModel::AllMiniLML6V2))
      .from_pretrained_onnx()
      .map_err(|e| format!("Failed to create embedder: {}", e))?;
    
    // Get embedding dimension by embedding a test string
    let test_embeddings = embedder
      .embed(&["test"], None, None)
      .await
      .map_err(|e| format!("Failed to get embedding dimension: {}", e))?;
    
    let embedding_dim = match test_embeddings.first() {
      Some(embed_anything::embeddings::embed::EmbeddingResult::DenseVector(vec)) => vec.len(),
      _ => return Err("Failed to determine embedding dimension".into()),
    };

    info!("Embedder loaded successfully, dimension: {}", embedding_dim);
    
    let embedder = Arc::new(embedder);

    // Ensure table exists
    debug!(
      "Ensuring table '{}' exists with embedding dimension {}",
      config.table_name, embedding_dim
    );
    let table = CodeDocument::ensure_table(&connection, &config.table_name, embedding_dim).await?;
    info!("Table ready");

    Ok(Self {
      config: config.clone(),
      embedder,
      embedding_dim,
      table: Arc::new(RwLock::new(table)),
    })
  }

  /// Index a repository at the given path
  pub async fn index(
    &self,
    path: &Path,
  ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    // Create an indexer with our resources
    let indexer = Indexer::new(&self.config, self.embedder.clone(), self.embedding_dim, self.table.clone());

    // Run the indexing
    indexer.index(path).await
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::tempdir;

  #[tokio::test]
  #[ignore] // Ignore for now since we need a real model
  async fn test_app_index() {
    // Create test config
    let mut config = Config::default();

    // Use temp directory for database
    let temp_db = tempdir().unwrap();
    config.database_path = temp_db.path().to_path_buf();

    // Create app
    let app = App::new(config).await.unwrap();

    // Create test repository
    let test_repo = tempdir().unwrap();
    let test_file = test_repo.path().join("main.py");
    std::fs::write(&test_file, "def hello_world():\n    print('Hello, world!')").unwrap();

    // Index the repository
    let count = app.index(test_repo.path()).await.unwrap();
    assert_eq!(count, 1);
  }
}
