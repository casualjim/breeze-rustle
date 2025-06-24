use std::path::Path;
use tokio_util::sync::CancellationToken;
use tracing::{info, instrument};

use crate::config::Config;
use breeze_indexer::{Indexer, SearchOptions, SearchResult};

pub struct App {
  indexer: Indexer,
}

impl App {
  /// Create a new App instance with the given configuration
  #[instrument(skip(config, shutdown_token), fields(database_path = %config.db_dir.display()))]
  pub async fn new(
    config: Config,
    shutdown_token: CancellationToken,
  ) -> Result<Self, Box<dyn std::error::Error>> {
    info!("Initializing Breeze app");

    // Convert the config to breeze_indexer::Config
    let indexer_config = config.to_indexer_config()?;

    // Create the indexer using the facade
    let indexer = Indexer::new(indexer_config, shutdown_token)
      .await
      .map_err(|e| format!("Failed to create indexer: {}", e))?;

    Ok(Self { indexer })
  }

  /// Index a repository at the given path
  pub async fn index(
    &self,
    path: &Path,
  ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    // Look up existing project or create a new one
    let project = match self.indexer.project_manager().find_by_path(path).await? {
      Some(existing) => existing,
      None => {
        // Create a project for this path
        let project_name = path
          .file_name()
          .and_then(|n| n.to_str())
          .unwrap_or("project")
          .to_string();

        self
          .indexer
          .project_manager()
          .create_project(project_name, path.to_string_lossy().to_string(), None)
          .await?
      }
    };

    // Use the facade to index the project
    self
      .indexer
      .index_project(project.id)
      .await
      .map(|task_id| task_id.to_string())
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

    let options = SearchOptions {
      file_limit: limit,
      ..Default::default()
    };

    // Use the facade to search
    self
      .indexer
      .search(query, options)
      .await
      .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)
  }
}

#[cfg(test)]
mod tests {

  use super::*;
  use tempfile::tempdir;

  #[tokio::test]
  async fn test_app_index() {
    // Create test config with temp directory for database
    let (_temp_dir, indexer_config) = breeze_indexer::Config::test();
    let config = Config {
      db_dir: indexer_config.database_path.clone(),
      ..Default::default()
    };

    // Create app
    let shutdown_token = CancellationToken::new();
    let app = App::new(config, shutdown_token).await.unwrap();

    // Create test repository
    let test_repo = tempdir().unwrap();
    let test_file = test_repo.path().join("main.py");
    tokio::fs::write(&test_file, "def hello_world():\n    print('Hello, world!')")
      .await
      .unwrap();

    // Index the repository
    let task_id = app.index(test_repo.path()).await.unwrap();
    assert!(!task_id.is_empty());
  }
}
