use anyhow::Result;
use breeze_chunkers::{Chunk, Chunker, ChunkerConfig, Tokenizer};
use futures::StreamExt;
use lancedb::Table;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{error, info};

use crate::{
  Config, SearchResult,
  bulk_indexer::BulkIndexer,
  embeddings::{EmbeddingProvider, factory::create_embedding_provider},
  hybrid_search,
  models::CodeDocument,
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
  embedding_dim: usize,
}

impl Indexer {
  /// Create a new Indexer from configuration
  pub async fn new(config: Config) -> Result<Self, IndexerError> {
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

    let task_table = Arc::new(RwLock::new(task_table));
    let table = Arc::new(RwLock::new(table));
    let config = Arc::new(config);
    let embedding_provider: Arc<dyn EmbeddingProvider> = Arc::from(embedding_provider);
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
      embedding_dim,
    })
  }

  /// Get the task manager
  pub fn task_manager(&self) -> Arc<TaskManager> {
    self.task_manager.clone()
  }

  /// Index a single file
  pub async fn index_file(
    &self,
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
  pub async fn index_project(&self, project_path: &Path) -> Result<String, IndexerError> {
    info!(path = %project_path.display(), "Submitting project indexing task");

    self
      .task_manager
      .submit_task(project_path)
      .await
      .map_err(|e| IndexerError::Storage(format!("Failed to submit indexing task: {}", e)))
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
}
