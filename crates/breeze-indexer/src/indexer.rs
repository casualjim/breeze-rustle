use anyhow::Result;
use dashmap::DashMap;
use lancedb::Table;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use uuid::Uuid;

use crate::{
  Config, SearchResult,
  bulk_indexer::BulkIndexer,
  embeddings::{EmbeddingError, EmbeddingProvider, factory::create_embedding_provider},
  file_watcher::ProjectWatcher,
  hybrid_search,
  models::{CodeDocument, FileChange, FileOperation, Project},
  project_manager::ProjectManager,
  task_manager::TaskManager,
};

/// Error type for public API
#[derive(Debug, thiserror::Error)]
pub enum IndexerError {
  #[error("Configuration error: {0}")]
  Config(String),

  #[error("Storage error")]
  Storage(#[from] lancedb::Error),

  #[error("Embedding provider error")]
  Embedding(#[from] EmbeddingError),

  #[error("IO error")]
  Io(#[from] std::io::Error),

  #[error("Chunker error")]
  Chunker(#[from] breeze_chunkers::ChunkError),

  #[error("Project not found: {0}")]
  ProjectNotFound(Uuid),

  #[error("File outside project directory")]
  FileOutsideProject { file: String, project_dir: String },

  #[error("Search error: {0}")]
  Search(String),

  #[error("Path must be absolute: {0}")]
  PathNotAbsolute(String),

  #[error("Arrow conversion error")]
  Arrow(#[from] arrow::error::ArrowError),

  #[error("Task execution error: {0}")]
  Task(String),

  #[error("Serialization error")]
  Serialization(#[from] serde_json::Error),

  #[error("File watcher error")]
  FileWatcher(#[from] notify::Error),
}

/// Main public interface for indexing and searching code
pub struct Indexer {
  config: Arc<Config>,
  embedding_provider: Arc<dyn EmbeddingProvider>,
  table: Arc<RwLock<Table>>,
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

    let task_table = crate::models::IndexTask::ensure_table(&connection, "index_tasks").await?;

    // Ensure projects table exists
    let project_table = Project::ensure_table(&connection, "projects").await?;

    let project_table = Arc::new(RwLock::new(project_table));
    let task_table = Arc::new(RwLock::new(task_table));
    let table = Arc::new(RwLock::new(table));
    let config = Arc::new(config);
    let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::from(embedding_provider);
    let project_manager = Arc::new(ProjectManager::new(project_table));
    let task_manager = Arc::new(TaskManager::new(
      task_table,
      BulkIndexer::new(
        config.clone(),
        embedding_provider.clone(),
        embedding_dim,
        table.clone(),
      ),
    ));

    Ok(Self {
      config,
      embedding_provider,
      table,
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
      .submit_task_with_type(
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
      .submit_task(project_id, Path::new(&project.directory))
      .await?;

    // Start file watcher for the project
    if let Err(e) = self.start_file_watcher(project_id).await {
      error!(project_id = %project_id, error = %e, "Failed to start file watcher");
      // Don't fail the indexing task if watcher fails to start
    }

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
  pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>, IndexerError> {
    hybrid_search(
      self.table.clone(),
      self.embedding_provider.clone(),
      query,
      limit,
    )
    .await
    .map_err(|e| IndexerError::Search(e.to_string()))
  }

  /// Find a document by file path
  pub async fn find_by_path(&self, file_path: &str) -> Result<Option<CodeDocument>, IndexerError> {
    use futures::TryStreamExt;
    use lancedb::query::{ExecutableQuery, QueryBase};

    let table = self.table.read().await;

    let mut results = table
      .query()
      .only_if(format!("file_path = '{}'", file_path))
      .limit(1)
      .execute()
      .await?;

    if let Some(batch) = results.try_next().await? {
      Ok(Some(CodeDocument::from_record_batch(&batch, 0)?))
    } else {
      Ok(None)
    }
  }

  /// Find a document by content hash
  pub async fn find_by_hash(
    &self,
    content_hash: &[u8; 32],
  ) -> Result<Option<CodeDocument>, IndexerError> {
    use futures::TryStreamExt;
    use lancedb::query::{ExecutableQuery, QueryBase};

    let table = self.table.read().await;

    // Convert hash to hex string for query
    let hash_hex = hex::encode(content_hash);

    let mut results = table
      .query()
      .only_if(format!("content_hash = X'{}'", hash_hex))
      .limit(1)
      .execute()
      .await?;

    if let Some(batch) = results.try_next().await? {
      Ok(Some(CodeDocument::from_record_batch(&batch, 0)?))
    } else {
      Ok(None)
    }
  }
}
