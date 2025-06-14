use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, instrument};

use crate::config::Config;
use crate::embeddings::{
  loader::{dtype_for_device, load_tei_embedder},
  tei::TEIEmbedder,
};
use crate::indexer::Indexer;
use crate::models::CodeDocument;

pub struct App {
  config: Config,
  embedder: TEIEmbedder,
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

    // Load TEI embedder
    let dtype = dtype_for_device(&config.device);
    info!(
      "Loading TEI embedder: {} with dtype: {}",
      config.model, dtype
    );

    let embedder = load_tei_embedder(&config.model, dtype, None).await?;
    let embedding_dim = embedder.embedding_dim();

    info!("Embedder loaded successfully, dimension: {}", embedding_dim);

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
      table: Arc::new(RwLock::new(table)),
    })
  }

  /// Index a repository at the given path
  pub async fn index(
    &self,
    path: &Path,
  ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    // Create an indexer with our resources
    let indexer = Indexer::new(&self.config, &self.embedder, self.table.clone());

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
