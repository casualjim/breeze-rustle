use anyhow::Result;
use dashmap::DashMap;
use lancedb::Table;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use uuid::Uuid;

use crate::{
  Config, SearchOptions, SearchResult,
  bulk_indexer::BulkIndexer,
  embeddings::{EmbeddingError, EmbeddingProvider, factory::create_embedding_provider},
  file_watcher::ProjectWatcher,
  hybrid_search,
  models::{CodeDocument, FileChange, FileOperation, Project},
  project_manager::ProjectManager,
  rescan_worker,
  task_manager::TaskManager,
};

/// Error type for public API
#[derive(Debug, thiserror::Error)]
pub enum IndexerError {
  #[error("Configuration error: {0}")]
  Config(String),

  #[error("Storage error: {0}")]
  Storage(#[from] lancedb::Error),

  #[error("Database error: {0}")]
  Database(String),

  #[error("Embedding provider error: {0}")]
  Embedding(#[from] EmbeddingError),

  #[error("IO error")]
  Io(#[from] std::io::Error),

  #[error("Chunker error: {0}")]
  Chunker(#[from] breeze_chunkers::ChunkError),

  #[error("Project not found: {0}")]
  ProjectNotFound(Uuid),

  #[error("File {file} outside project directory {project_dir}")]
  FileOutsideProject { file: String, project_dir: String },

  #[error("Search error: {0}")]
  Search(String),

  #[error("Path must be absolute: {0}")]
  PathNotAbsolute(String),

  #[error("Project already exists for directory: {directory} (existing project ID: {existing_id})")]
  ProjectAlreadyExists {
    directory: String,
    existing_id: Uuid,
  },

  #[error("Arrow conversion error: {0}")]
  Arrow(#[from] arrow::error::ArrowError),

  #[error("Task execution error: {0}")]
  Task(String),

  #[error("Serialization error: {0}")]
  Serialization(#[from] serde_json::Error),

  #[error("File watcher error: {0}")]
  FileWatcher(#[from] notify::Error),
}

/// Main public interface for indexing and searching code
pub struct Indexer {
  config: Arc<Config>,
  embedding_provider: Arc<dyn EmbeddingProvider>,
  table: Arc<RwLock<Table>>,
  chunk_table: Arc<RwLock<Table>>,
  task_manager: Arc<TaskManager>,
  project_manager: Arc<ProjectManager>,
  active_watchers: DashMap<Uuid, Arc<ProjectWatcher>>,
  shutdown_token: CancellationToken,
}

impl Indexer {
  /// Create a new Indexer from configuration
  pub async fn new(
    config: Config,
    shutdown_token: CancellationToken,
  ) -> Result<Self, IndexerError> {
    // Initialize embedding provider
    let embedding_provider = create_embedding_provider(&config).await?;

    let embedding_dim = embedding_provider.embedding_dim();

    // Initialize LanceDB connection
    let db_path = &config.database_path;

    let connection = lancedb::connect(
      db_path
        .to_str()
        .ok_or_else(|| IndexerError::Config("Invalid database path".to_string()))?,
    )
    .execute()
    .await?;

    // Ensure tables exist
    let table = CodeDocument::ensure_table(&connection, "code_embeddings", embedding_dim).await?;
    let chunk_table =
      crate::models::CodeChunk::ensure_table(&connection, "code_chunks", embedding_dim).await?;

    let task_table = crate::models::IndexTask::ensure_table(&connection, "index_tasks").await?;

    // Ensure projects table exists
    let project_table = Project::ensure_table(&connection, "projects").await?;

    // Ensure failed batches table exists
    let failed_batches_table =
      crate::models::FailedEmbeddingBatch::ensure_table(&connection, "failed_embedding_batches")
        .await?;

    let project_table = Arc::new(RwLock::new(project_table));
    let task_table = Arc::new(RwLock::new(task_table));
    let failed_batches_table = Arc::new(RwLock::new(failed_batches_table));
    let table = Arc::new(RwLock::new(table));
    let chunk_table = Arc::new(RwLock::new(chunk_table));
    let config = Arc::new(config);
    let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::from(embedding_provider);

    // Create task manager first
    let task_manager = Arc::new(TaskManager::new(
      task_table,
      failed_batches_table,
      project_table.clone(),
      BulkIndexer::new(
        config.clone(),
        embedding_provider.clone(),
        embedding_dim,
        table.clone(),
        chunk_table.clone(),
        project_table.clone(),
      ),
    ));

    // Then create project manager with task manager reference
    let project_manager = Arc::new(ProjectManager::new(project_table, task_manager.clone()));

    Ok(Self {
      config,
      embedding_provider,
      table,
      chunk_table,
      task_manager,
      project_manager,
      active_watchers: DashMap::new(),
      shutdown_token,
    })
  }

  /// Get the task manager
  pub fn task_manager(&self) -> Arc<TaskManager> {
    self.task_manager.clone()
  }

  /// Get the project manager
  pub fn project_manager(&self) -> Arc<ProjectManager> {
    self.project_manager.clone()
  }

  /// Index a single file by submitting a partial index task
  pub async fn index_file(&self, project_id: Uuid, file_path: &Path) -> Result<Uuid, IndexerError> {
    info!(path = %file_path.display(), project_id = %project_id, "Submitting single file index task");

    // Get the project to validate the file belongs to it
    let project = self
      .project_manager
      .get_project(project_id)
      .await?
      .ok_or(IndexerError::ProjectNotFound(project_id))?;

    let project_dir = Path::new(&project.directory);

    // Canonicalize paths to resolve .. and symlinks
    let canonical_project = project_dir.canonicalize()?;
    let canonical_file = file_path.canonicalize()?;

    // Ensure the file is within the project directory
    if !canonical_file.starts_with(&canonical_project) {
      return Err(IndexerError::FileOutsideProject {
        file: file_path.display().to_string(),
        project_dir: project.directory.clone(),
      });
    }

    // Check that file exists is handled by canonicalize() which returns an error if file doesn't exist

    let mut changes = BTreeSet::new();
    let operation = FileOperation::Update;

    changes.insert(FileChange {
      path: canonical_file.clone(),
      operation,
    });

    // Submit partial index task
    let task_id = self
      .task_manager
      .submit_task(
        project_id,
        &canonical_project,
        crate::models::TaskType::PartialUpdate { changes },
      )
      .await?;

    info!(task_id = %task_id, path = %file_path.display(), "Submitted partial index task for file");
    Ok(task_id)
  }

  /// Index an entire project - always submits to task queue
  pub async fn index_project(&self, project_id: Uuid) -> Result<Uuid, IndexerError> {
    // Get the project to find its directory
    let project = self
      .project_manager
      .get_project(project_id)
      .await?
      .ok_or(IndexerError::ProjectNotFound(project_id))?;

    info!(project_id = %project_id, path = %project.directory, "Submitting project indexing task");

    // Check if project already has an active task
    if self.task_manager.has_active_task(project_id).await? {
      return Err(IndexerError::Config(
        "Project already has an active indexing task".to_string(),
      ));
    }

    let task_id = self
      .task_manager
      .submit_task(
        project_id,
        Path::new(&project.directory),
        crate::models::TaskType::FullIndex,
      )
      .await?;

    // Disabled automatic file watching to prevent excessive file handle usage
    // File watching can be enabled per-project as needed
    // if let Err(e) = self.start_file_watcher(project_id).await {
    //   error!(project_id = %project_id, error = %e, "Failed to start file watcher");
    //   // Don't fail the indexing task if watcher fails to start
    // }

    Ok(task_id)
  }

  /// Start file watching for a project
  async fn start_file_watcher(&self, project_id: Uuid) -> Result<(), IndexerError> {
    // Check if watcher already exists
    if self.active_watchers.contains_key(&project_id) {
      info!(project_id = %project_id, "File watcher already active");
      return Ok(());
    }

    // Get the project to find its directory
    let project = self
      .project_manager
      .get_project(project_id)
      .await?
      .ok_or(IndexerError::ProjectNotFound(project_id))?;

    info!(project_id = %project_id, path = %project.directory, "Starting file watcher");

    // Create and start the watcher
    let watcher = Arc::new(
      ProjectWatcher::new(
        project_id,
        &project.directory,
        self.task_manager.clone(),
        self.config.max_file_size,
      )
      .await?,
    );

    // Store the watcher
    self.active_watchers.insert(project_id, watcher.clone());

    // Spawn a task to handle watcher shutdown
    let shutdown_token = self.shutdown_token.clone();
    let active_watchers = self.active_watchers.clone();
    tokio::spawn(async move {
      watcher.wait_for_shutdown(shutdown_token).await;
      active_watchers.remove(&project_id);
    });

    Ok(())
  }

  /// Start file watchers for all existing projects
  pub async fn start_all_project_watchers(&self) -> Result<(), IndexerError> {
    let projects = self.project_manager.list_projects().await?;

    info!("Starting file watchers for {} projects", projects.len());

    for project in projects {
      if let Err(e) = self.start_file_watcher(project.id).await {
        error!(project_id = %project.id, error = %e, "Failed to start file watcher");
        // Continue with other projects even if one fails
      }
    }

    Ok(())
  }

  /// Stop all file watchers
  pub fn stop_all_watchers(&self) {
    self.active_watchers.clear();
    self.shutdown_token.cancel();
  }

  /// Search the indexed code
  pub async fn search(
    &self,
    query: &str,
    options: SearchOptions,
  ) -> Result<Vec<SearchResult>, IndexerError> {
    hybrid_search(
      self.table.clone(),
      self.chunk_table.clone(),
      self.embedding_provider.clone(),
      query,
      options,
      None, // No project filter in this method
    )
    .await
    .map_err(|e| IndexerError::Search(e.to_string()))
  } // This closing brace belongs to the `search` function

  /// Start the indexer
  pub async fn start(&self) -> Result<(), IndexerError> {
    // If already cancelled, it means it was stopped, so we can't restart.
    // If not cancelled, but already running, it's idempotent.
    if self.shutdown_token.is_cancelled() {
      return Ok(());
    }

    // self.start_all_project_watchers().await?;

    // Start worker task
    let worker_shutdown = self.shutdown_token.clone();
    let tm = self.task_manager.clone();
    tokio::spawn(async move {
      if let Err(e) = tm.run_worker(worker_shutdown).await {
        error!("Task worker error: {}", e);
      }
    });

    // Start rescan worker
    let rescan_shutdown = self.shutdown_token.clone();
    rescan_worker::start_rescan_worker(
      self.project_manager.clone(),
      self.task_manager.clone(),
      Duration::from_secs(300), // Rescan every 5 minutes
      rescan_shutdown,
    );

    Ok(())
  }

  /// Stop the indexer
  pub fn stop(&self) {
    // self.stop_all_watchers();
    self.shutdown_token.cancel();
  }
} // This closing brace belongs to the `impl Indexer` block

#[cfg(test)]
mod tests {
  use super::*;
  use crate::Config;
  use std::path::PathBuf;
  use tempfile::TempDir;
  use tokio_util::sync::CancellationToken;
  use uuid::Uuid;

  // Test setup helper
  async fn setup_indexer() -> (Indexer, TempDir, Uuid) {
    let (tempdir, config) = Config::test();
    let shutdown_token = CancellationToken::new();
    let indexer = Indexer::new(config, shutdown_token).await.unwrap();

    // Add a dummy project to ensure watchers are started
    let project_dir = tempdir.path().join("dummy_project");
    tokio::fs::create_dir_all(&project_dir).await.unwrap();
    let project = indexer
      .project_manager
      .create_project(
        "dummy_project_name".to_string(),
        project_dir.to_str().unwrap().to_string(),
        None,
        None,
      )
      .await
      .unwrap();

    (indexer, tempdir, project.id)
  }

  #[tokio::test]
  async fn test_start_stop_functionality() {
    let (indexer, _tempdir, project_id) = setup_indexer().await;

    // Test successful start
    assert!(indexer.start().await.is_ok());

    // Verify components started:
    // - Shutdown token is not cancelled
    assert!(!indexer.shutdown_token.is_cancelled());
    // - Project watchers (check if the dummy project watcher is active)
    assert!(indexer.active_watchers.contains_key(&project_id));
    // - Task worker (implicitly started by `start()`)
    // - Rescan worker (implicitly started by `start()`)

    // Test stop functionality
    indexer.stop();

    // Verify components stopped:
    // - Watchers cleaned up
    assert!(indexer.active_watchers.is_empty());
    // - Shutdown token cancelled
    assert!(indexer.shutdown_token.is_cancelled());
  }

  #[tokio::test]
  async fn test_double_start_handling() {
    let (indexer, _tempdir, _project_id) = setup_indexer().await;

    assert!(indexer.start().await.is_ok());
    // Second start should now return an error because the indexer cannot be restarted after being stopped
    // However, if it's not stopped, it should be idempotent.
    // The current implementation of `start` returns an error if `shutdown_token.is_cancelled()`.
    // So, if we call start twice without stopping, the second call should be Ok.
    // If we stop and then start again, it should be an error.
    // The test is for "double start", implying consecutive starts without an intermediate stop.
    // So, the assertion should be `is_ok()`.
    assert!(indexer.start().await.is_ok());
  }

  #[tokio::test]
  async fn test_stop_without_start() {
    let (indexer, _tempdir, _project_id) = setup_indexer().await;
    // Before stopping, the token should not be cancelled
    assert!(!indexer.shutdown_token.is_cancelled());
    indexer.stop(); // Should handle gracefully without panicking
    assert!(indexer.active_watchers.is_empty());
    assert!(indexer.shutdown_token.is_cancelled());
  }

  #[tokio::test]
  async fn test_start_failure_handling() {
    // Setup indexer with invalid config to force new failure
    let mut invalid_config = Config::default();
    invalid_config.database_path = PathBuf::from("/invalid/path");
    let shutdown_token = CancellationToken::new();
    let indexer_result = Indexer::new(invalid_config, shutdown_token).await;
    assert!(indexer_result.is_err());
    assert!(
      matches!(indexer_result.err().unwrap(), IndexerError::Storage(_)),
      "Expected storage error due to invalid path"
    );
  }
}
