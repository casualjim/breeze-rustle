use anyhow::Result;
use breeze_chunkers::{Chunk, InnerChunker, Tokenizer};
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
}

/// Main public interface for indexing and searching code
pub struct Indexer {
  config: Arc<Config>,
  embedding_provider: Arc<dyn EmbeddingProvider>,
  table: Arc<RwLock<Table>>,
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

    // Ensure table exists
    let table = CodeDocument::ensure_table(&connection, "code_embeddings", embedding_dim)
      .await
      .map_err(|e| IndexerError::Storage(format!("Failed to ensure table: {}", e)))?;

    Ok(Self {
      config: Arc::new(config),
      embedding_provider: Arc::from(embedding_provider),
      table: Arc::new(RwLock::new(table)),
      embedding_dim,
    })
  }

  /// Index a single file
  pub async fn index_file(
    &self,
    file_path: &Path,
    content: Option<String>,
  ) -> Result<(), IndexerError> {
    info!(path = %file_path.display(), "Indexing single file");

    // Read content if not provided
    let content = match content {
      Some(c) => c,
      None => tokio::fs::read_to_string(file_path)
        .await
        .map_err(|e| IndexerError::IO(format!("Failed to read file: {}", e)))?,
    };

    // Detect language from file extension
    let extension = file_path.extension().and_then(|e| e.to_str()).unwrap_or("");

    let language = match extension {
      "rs" => "rust",
      "py" => "python",
      "js" | "jsx" => "javascript",
      "ts" | "tsx" => "typescript",
      "go" => "go",
      "java" => "java",
      "cpp" | "cc" | "cxx" => "cpp",
      "c" => "c",
      "rb" => "ruby",
      "php" => "php",
      "cs" => "c_sharp",
      "swift" => "swift",
      "kt" => "kotlin",
      "scala" => "scala",
      "sh" | "bash" => "bash",
      "yaml" | "yml" => "yaml",
      "json" => "json",
      "toml" => "toml",
      "md" => "markdown",
      _ => "text",
    };

    let tokenizer = if let Some(provider_tokenizer) = self.embedding_provider.tokenizer() {
      Tokenizer::PreloadedHuggingFace(provider_tokenizer)
    } else {
      Tokenizer::Characters
    };

    let chunker = InnerChunker::new(self.config.optimal_chunk_size(), tokenizer)?;

    // Chunk the file
    let mut chunks = Vec::new();
    let mut chunk_stream = Box::pin(chunker.chunk_code(
      content.clone(),
      language.to_string(),
      Some(file_path.to_string_lossy().to_string()),
    ));

    while let Some(chunk_result) = chunk_stream.next().await {
      match chunk_result {
        Ok(chunk) => chunks.push(chunk),
        Err(e) => error!("Error chunking file: {}", e),
      }
    }

    if chunks.is_empty() {
      return Ok(());
    }

    // Extract text chunks for embedding
    let texts: Vec<&str> = chunks
      .iter()
      .filter_map(|c| match c {
        Chunk::Semantic(sc) | Chunk::Text(sc) => Some(sc.text.as_str()),
        _ => None,
      })
      .collect();

    if texts.is_empty() {
      return Ok(());
    }

    // Embed all chunks at once
    let embeddings = self
      .embedding_provider
      .embed(
        &texts
          .into_iter()
          .map(|t| crate::embeddings::EmbeddingInput {
            text: t,
            token_count: None,
          })
          .collect::<Vec<_>>(),
      )
      .await
      .map_err(|e| IndexerError::Embedding(format!("Failed to embed chunks: {}", e)))?;

    // Aggregate embeddings using weighted average by chunk size
    let mut aggregated_embedding = vec![0.0; self.embedding_dim];
    let mut total_weight = 0.0;

    for (i, chunk) in chunks.iter().enumerate() {
      if let (Chunk::Semantic(sc) | Chunk::Text(sc), Some(embedding)) = (chunk, embeddings.get(i)) {
        let weight = sc.text.len() as f32;
        total_weight += weight;

        for (j, val) in embedding.iter().enumerate() {
          aggregated_embedding[j] += val * weight;
        }
      }
    }

    // Normalize
    for val in &mut aggregated_embedding {
      *val /= total_weight;
    }

    // Create document
    let mut doc = CodeDocument::new(file_path.to_string_lossy().to_string(), content);
    doc.update_embedding(aggregated_embedding);

    // Insert or update in table
    use lancedb::arrow::IntoArrow;
    let arrow_data = doc
      .into_arrow()
      .map_err(|e| IndexerError::Storage(format!("Failed to convert to Arrow: {}", e)))?;

    let table = self.table.write().await;

    // Try to delete existing entry first (update)
    let _ = table
      .delete(&format!("file_path = '{}'", file_path.to_string_lossy()))
      .await;

    // Insert new entry
    table
      .add(arrow_data)
      .execute()
      .await
      .map_err(|e| IndexerError::Storage(format!("Failed to insert document: {}", e)))?;

    info!(path = %file_path.display(), "Successfully indexed file");
    Ok(())
  }

  /// Index an entire project
  pub async fn index_project(
    &self,
    project_path: &Path,
    cancel_token: Option<tokio_util::sync::CancellationToken>,
  ) -> Result<usize, IndexerError> {
    info!(path = %project_path.display(), "Indexing project");

    // Use BulkIndexer for efficient project indexing
    let bulk_indexer = BulkIndexer::new(
      &self.config,
      self.embedding_provider.clone(),
      self.embedding_dim,
      self.table.clone(),
    );

    bulk_indexer
      .index(project_path, cancel_token)
      .await
      .map_err(|e| IndexerError::IO(format!("Failed to index project: {}", e)))
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
