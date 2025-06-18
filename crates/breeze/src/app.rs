use std::path::Path;
use tracing::{info, instrument};

use crate::config::Config;
use breeze_indexer::{Indexer, SearchResult};

pub struct App {
  indexer: Indexer,
}

impl App {
  /// Create a new App instance with the given configuration
  #[instrument(skip(config), fields(database_path = %config.indexer.database_path.display(), model = %config.indexer.model))]
  pub async fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
    info!("Initializing Breeze app");

    // Convert the config to breeze_indexer::Config
    let indexer_config = config.indexer.clone();
    
    // Create the indexer using the facade
    let indexer = Indexer::new(indexer_config).await
      .map_err(|e| format!("Failed to create indexer: {}", e))?;

    Ok(Self { indexer })
  }

  /// Index a repository at the given path
  pub async fn index(
    &self,
    path: &Path,
  ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    // Use the facade to index the project
    self.indexer.index_project(path).await
      .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
  }

  /// Search the indexed codebase
  #[instrument(skip(self), fields(query_len = query.len(), limit = limit))]
  pub async fn search(
    &self,
    query: &str,
    limit: usize,
  ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error + Send + Sync>> {
    info!("Searching codebase");

    // Use the facade to search
    self.indexer.search(query, limit).await
      .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
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
