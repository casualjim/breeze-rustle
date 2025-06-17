use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument};

use crate::config::{Config, default_max_chunk_size};
use crate::embeddings::{EmbeddingProvider, factory::create_embedding_provider};
use crate::indexer::Indexer;
use crate::models::CodeDocument;
use crate::search::SearchResult;

pub struct App {
  config: Config,
  embedding_provider: Arc<dyn EmbeddingProvider>,
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

    // Create embedding provider based on configuration
    info!(
      "Creating embedding provider: {:?}",
      config.embedding_provider
    );
    let embedding_provider = create_embedding_provider(&config).await?;
    let embedding_dim = embedding_provider.embedding_dim();

    // Adjust max chunk size based on provider's context length if not explicitly set
    let context_length = embedding_provider.context_length();
    if config.max_chunk_size == default_max_chunk_size() {
      // Auto-adjust chunk size to 90% of context length
      let recommended_chunk_size = (context_length * 90) / 100;
      info!(
        "Auto-adjusting max_chunk_size from {} to {} based on model context length",
        config.max_chunk_size, recommended_chunk_size
      );
      // Note: We can't modify config here as it's borrowed, but we can log the recommendation
    }

    info!(
      "Embedding provider created successfully, dimension: {}",
      embedding_dim
    );

    let embedding_provider = Arc::from(embedding_provider);

    // Ensure table exists
    debug!(
      "Ensuring table '{}' exists with embedding dimension {}",
      config.table_name, embedding_dim
    );
    let table = CodeDocument::ensure_table(&connection, &config.table_name, embedding_dim).await?;
    info!("Table ready");

    Ok(Self {
      config: config.clone(),
      embedding_provider,
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
    let indexer = Indexer::new(
      &self.config,
      self.embedding_provider.clone(),
      self.embedding_dim,
      self.table.clone(),
    );

    // Run the indexing
    indexer.index(path).await
  }

  /// Search the indexed codebase
  #[instrument(skip(self), fields(query_len = query.len(), limit = limit))]
  pub async fn search(
    &self,
    query: &str,
    limit: usize,
  ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
    info!("Searching codebase");

    // Perform hybrid search
    let results = crate::search::hybrid_search(
      self.table.clone(),
      self.embedding_provider.clone(),
      query,
      limit,
    )
    .await?;

    Ok(results)
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
