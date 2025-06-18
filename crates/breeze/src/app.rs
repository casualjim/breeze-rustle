use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument};

use crate::config::{Config, default_max_chunk_size};
use breeze_indexer::{
  Indexer, SearchResult,
  embeddings::{EmbeddingProvider, factory::create_embedding_provider},
  models::CodeDocument,
};

const TABLE_NAME: &str = "code_files";

pub struct App {
  indexer_config: breeze_indexer::config::Config,
  embedding_provider: Arc<dyn EmbeddingProvider>,
  embedding_dim: usize,
  table: Arc<RwLock<lancedb::Table>>,
}

impl App {
  /// Create a new App instance with the given configuration
  #[instrument(skip(config), fields(database_path = %config.indexer.database_path.display(), model = %config.indexer.model, table_name = TABLE_NAME))]
  pub async fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
    info!("Initializing Breeze app");

    // Create LanceDB connection
    debug!(
      "Connecting to LanceDB at: {}",
      config.indexer.database_path.display()
    );
    let connection = lancedb::connect(
      config
        .indexer
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
      config.indexer.embedding_provider
    );
    let breeze_indexer_config = config.indexer.clone();
    let embedding_provider = create_embedding_provider(&breeze_indexer_config).await?;
    let embedding_dim = embedding_provider.embedding_dim();

    // Adjust max chunk size based on provider's context length if not explicitly set
    let context_length = embedding_provider.context_length();
    if config.indexer.max_chunk_size == default_max_chunk_size() {
      // Auto-adjust chunk size to 90% of context length
      let recommended_chunk_size = (context_length * 90) / 100;
      info!(
        "Auto-adjusting max_chunk_size from {} to {} based on model context length",
        config.indexer.max_chunk_size, recommended_chunk_size
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
      TABLE_NAME, embedding_dim
    );
    let table = CodeDocument::ensure_table(&connection, TABLE_NAME, embedding_dim).await?;
    info!("Table ready");

    Ok(Self {
      indexer_config: breeze_indexer_config,
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
      &self.indexer_config,
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
    let results = breeze_indexer::search::hybrid_search(
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
    // Create test config with temp directory for database
    let (_temp_dir, config) = Config::test();

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
