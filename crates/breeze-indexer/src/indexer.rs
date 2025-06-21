use anyhow::Result;
use breeze_chunkers::{Chunk, Chunker, ChunkerConfig, Tokenizer};
use dashmap::DashMap;
use futures::StreamExt;
use lancedb::Table;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use uuid::Uuid;

use crate::{
  Config, SearchResult,
  bulk_indexer::BulkIndexer,
  embeddings::{EmbeddingProvider, factory::create_embedding_provider},
  file_watcher::ProjectWatcher,
  hybrid_search,
  models::{CodeDocument, Project},
  project_manager::ProjectManager,
  task_manager::TaskManager,
};

/// Error type for public API
#[derive(Debug, thiserror::Error)]
pub enum IndexerError {
  #[error("Configuration error: {0}")]
  Config(String),

  #[error("Storage error: {0}")]
  Storage(String),

  #[error("Embedding error: {0}")]
  Embedding(String),

  #[error("IO error: {0}")]
  IO(String),

  #[error("Chunker error: {0}")]
  Chunker(#[from] breeze_chunkers::ChunkError),

  #[error("io error: {0}")]
  Io(#[from] std::io::Error),
}

/// Main public interface for indexing and searching code
pub struct Indexer {
  config: Arc<Config>,
  embedding_provider: Arc<dyn EmbeddingProvider>,
  table: Arc<RwLock<Table>>,
  task_manager: Arc<TaskManager>,
  project_manager: Arc<ProjectManager>,
  embedding_dim: usize,
  active_watchers: DashMap<Uuid, Arc<ProjectWatcher>>,
  shutdown_token: CancellationToken,
}

impl Indexer {
  /// Create a new Indexer from configuration
  pub async fn new(config: Config, shutdown_token: CancellationToken) -> Result<Self, IndexerError> {
    // Initialize embedding provider
    let embedding_provider = create_embedding_provider(&config)
      .await
      .map_err(|e| IndexerError::Config(format!("Failed to create embedding provider: {}", e)))?;

    let embedding_dim = embedding_provider.embedding_dim();

    // Initialize LanceDB connection
    let db_path = &config.database_path;

    let connection = lancedb::connect(
      db_path
        .to_str()
        .ok_or_else(|| IndexerError::Config("Invalid database path".to_string()))?,
    )
    .execute()
    .await
    .map_err(|e| IndexerError::Storage(format!("Failed to open LanceDB: {}", e)))?;

    // Ensure tables exist
    let table = CodeDocument::ensure_table(&connection, "code_embeddings", embedding_dim)
      .await
      .map_err(|e| IndexerError::Storage(format!("Failed to ensure table: {}", e)))?;

    let task_table = crate::models::IndexTask::ensure_table(&connection, "index_tasks")
      .await
      .map_err(|e| IndexerError::Storage(format!("Failed to ensure task table: {}", e)))?;

    // Ensure projects table exists
    let project_table = Project::ensure_table(&connection, "projects")
      .await
      .map_err(|e| IndexerError::Storage(format!("Failed to ensure projects table: {}", e)))?;

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
      embedding_dim,
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

  /// Index a single file
  pub async fn index_file(
    &self,
    _project_id: Uuid,
    file_path: &Path,
    content: Option<String>,
  ) -> Result<(), IndexerError> {
    info!(path = %file_path.display(), "Indexing single file");

    let tokenizer = if let Some(provider_tokenizer) = self.embedding_provider.tokenizer() {
      Tokenizer::PreloadedHuggingFace(provider_tokenizer)
    } else {
      Tokenizer::Characters
    };

    let chunker = Chunker::new(ChunkerConfig {
      max_chunk_size: self.config.max_chunk_size,
      tokenizer,
    })?;

    // Create a file accumulator
    let mut accumulator =
      crate::pipeline::FileAccumulator::new(file_path.to_string_lossy().to_string());

    // Chunk the file using breeze-chunkers' language detection
    let mut chunk_stream = chunker.chunk_file(file_path, content).await?;
    let mut chunks_to_embed = Vec::new();

    while let Some(chunk_result) = chunk_stream.next().await {
      match chunk_result {
        Ok(chunk) => {
          match &chunk {
            Chunk::Semantic(_) | Chunk::Text(_) => {
              // Collect chunks that need embedding
              chunks_to_embed.push(chunk);
            }
            Chunk::EndOfFile { .. } => {
              // Add EOF chunk directly to accumulator (it has content and hash)
              accumulator.add_chunk(crate::pipeline::EmbeddedChunk {
                chunk,
                embedding: vec![],
              });
            }
          }
        }
        Err(e) => error!("Error chunking file: {}", e),
      }
    }

    if chunks_to_embed.is_empty() {
      info!(path = %file_path.display(), "No embeddable content in file, skipping");
      return Ok(());
    }

    // Embed all chunks at once
    let embeddings = self
      .embedding_provider
      .embed(
        &chunks_to_embed
          .iter()
          .map(|chunk| match chunk {
            Chunk::Semantic(sc) | Chunk::Text(sc) => crate::embeddings::EmbeddingInput {
              text: &sc.text,
              token_count: sc.tokens.as_ref().map(|t| t.len()),
            },
            _ => unreachable!("Only semantic and text chunks should be here"),
          })
          .collect::<Vec<_>>(),
      )
      .await
      .map_err(|e| IndexerError::Embedding(format!("Failed to embed chunks: {}", e)))?;

    // Add embedded chunks to accumulator
    for (chunk, embedding) in chunks_to_embed.into_iter().zip(embeddings.into_iter()) {
      accumulator.add_chunk(crate::pipeline::EmbeddedChunk { chunk, embedding });
    }

    // Build document using the same logic as bulk indexer
    let doc = match crate::document_builder::build_document_from_accumulator(
      accumulator,
      self.embedding_dim,
    )
    .await
    {
      Some(doc) => doc,
      None => {
        info!(path = %file_path.display(), "No document created for file");
        return Ok(());
      }
    };

    // Use the same sink infrastructure as bulk indexer for consistency
    let converter = crate::converter::BufferedRecordBatchConverter::<CodeDocument>::default()
      .with_schema(Arc::new(CodeDocument::schema(self.embedding_dim)));

    let sink = crate::sinks::lancedb_sink::LanceDbSink::new(self.table.clone());

    // Create a single-item stream
    let doc_stream = futures::stream::once(async move { doc });
    let record_batches = converter.convert(Box::pin(doc_stream));

    // Process through sink
    let mut sink_stream = sink.sink(record_batches);
    while sink_stream.next().await.is_some() {
      // Process the single batch
    }

    info!(path = %file_path.display(), "Successfully indexed file");
    Ok(())
  }

  /// Index an entire project - always submits to task queue
  pub async fn index_project(&self, project_id: Uuid) -> Result<String, IndexerError> {
    // Get the project to find its directory
    let project = self
      .project_manager
      .get_project(project_id)
      .await
      .map_err(|e| IndexerError::Storage(format!("Failed to get project: {}", e)))?
      .ok_or_else(|| IndexerError::Storage(format!("Project not found: {}", project_id)))?;

    info!(project_id = %project_id, path = %project.directory, "Submitting project indexing task");

    // Check if project already has an active task
    if self
      .task_manager
      .has_active_task(project_id)
      .await
      .map_err(|e| IndexerError::Storage(format!("Failed to check active tasks: {}", e)))?
    {
      return Err(IndexerError::Storage(
        "Project already has an active indexing task".to_string(),
      ));
    }

    let task_id = self
      .task_manager
      .submit_task(project_id, Path::new(&project.directory))
      .await
      .map_err(|e| IndexerError::Storage(format!("Failed to submit indexing task: {}", e)))?;
    
    // Start file watcher for the project
    if let Err(e) = self.start_file_watcher(project_id).await {
      error!(project_id = %project_id, error = %e, "Failed to start file watcher");
      // Don't fail the indexing task if watcher fails to start
    }
    
    Ok(task_id.to_string())
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
      .await
      .map_err(|e| IndexerError::Storage(format!("Failed to get project: {}", e)))?
      .ok_or_else(|| IndexerError::Storage(format!("Project not found: {}", project_id)))?;

    info!(project_id = %project_id, path = %project.directory, "Starting file watcher");

    // Create and start the watcher
    let watcher = Arc::new(
      ProjectWatcher::new(
        project_id,
        &project.directory,
        self.task_manager.clone(),
        self.config.max_file_size,
      )
      .await
      .map_err(|e| IndexerError::IO(format!("Failed to create file watcher: {}", e)))?,
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
    let projects = self
      .project_manager
      .list_projects()
      .await
      .map_err(|e| IndexerError::Storage(format!("Failed to list projects: {}", e)))?;

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
    .map_err(|e| IndexerError::Storage(format!("Search failed: {}", e)))
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
      .await
      .map_err(|e| IndexerError::Storage(format!("Failed to query by file path: {}", e)))?;

    if let Some(batch) = results
      .try_next()
      .await
      .map_err(|e| IndexerError::Storage(format!("Failed to get result: {}", e)))?
    {
      CodeDocument::from_record_batch(&batch, 0)
        .map(Some)
        .map_err(|e| IndexerError::Storage(format!("Failed to parse document: {}", e)))
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
      .await
      .map_err(|e| IndexerError::Storage(format!("Failed to query by content hash: {}", e)))?;

    if let Some(batch) = results
      .try_next()
      .await
      .map_err(|e| IndexerError::Storage(format!("Failed to get result: {}", e)))?
    {
      CodeDocument::from_record_batch(&batch, 0)
        .map(Some)
        .map_err(|e| IndexerError::Storage(format!("Failed to parse document: {}", e)))
    } else {
      Ok(None)
    }
  }
}
