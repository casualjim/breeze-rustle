use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use breeze_chunkers::{Chunk, Tokenizer, WalkOptions, walk_project};
use futures_util::{StreamExt, TryStreamExt};
use lancedb::Table;
use lancedb::query::{ExecutableQuery, QueryBase};
use tokio::sync::{RwLock, mpsc};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::IndexerError;
use crate::config::Config;
use crate::converter::BufferedRecordBatchConverter;
use crate::document_builder::build_document_from_accumulator;
use crate::embeddings::EmbeddingProvider;
use crate::models::CodeDocument;
use crate::pipeline::{
  ChunkBatch, EmbeddedChunk, EmbeddedChunkWithFile, FileAccumulator, PipelineChunk,
  PipelineProjectChunk,
};
use crate::sinks::lancedb_sink::LanceDbSink;

// Type alias for document builder result
type DocumentBuilderResult = (usize, Option<(std::collections::BTreeSet<String>, String)>);

struct DocumentBuilderParams {
  project_id: Uuid,
  embedded_rx: mpsc::Receiver<EmbeddedChunkWithFile>,
  doc_tx: mpsc::Sender<CodeDocument>,
  chunk_tx: mpsc::Sender<crate::models::CodeChunk>,
  embedding_dim: usize,
  stats: IndexingStats,
  cancel_token: CancellationToken,
}

pub struct BulkIndexer {
  config: Arc<Config>,
  embedding_provider: Arc<dyn EmbeddingProvider>,
  embedding_dim: usize,
  table: Arc<RwLock<Table>>,
  chunk_table: Arc<RwLock<Table>>,
  last_optimize_version: Arc<RwLock<u64>>,
}

impl BulkIndexer {
  pub fn new(
    config: Arc<Config>,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    embedding_dim: usize,
    table: Arc<RwLock<Table>>,
    chunk_table: Arc<RwLock<Table>>,
  ) -> Self {
    // Initialize with version 0 - will be updated on first use
    let last_optimize_version = Arc::new(RwLock::new(0));

    // Spawn a task to read the current table version
    let table_clone = table.clone();
    let version_clone = last_optimize_version.clone();
    tokio::spawn(async move {
      let table_guard = table_clone.read().await;
      if let Ok(version) = table_guard.version().await {
        *version_clone.write().await = version;
      }
    });

    Self {
      config,
      embedding_provider,
      embedding_dim,
      table,
      chunk_table,
      last_optimize_version,
    }
  }

  /// Query existing file hashes from the database for work avoidance
  async fn get_existing_file_hashes(
    &self,
    project_id: Uuid,
  ) -> Result<BTreeMap<PathBuf, [u8; 32]>, IndexerError> {
    use arrow::array::*;
    use futures_util::TryStreamExt;

    let table = self.table.read().await;
    let mut hashes = BTreeMap::new();

    // Query only the file_path and content_hash columns for this project
    let mut query = table
      .query()
      .only_if(format!("project_id = '{}'", project_id))
      .select(lancedb::query::Select::columns(&[
        "file_path",
        "content_hash",
      ]))
      .execute()
      .await
      .map_err(|e| IndexerError::Database(e.to_string()))?;

    while let Some(batch) = query
      .try_next()
      .await
      .map_err(|e| IndexerError::Database(e.to_string()))?
    {
      let file_paths = batch
        .column_by_name("file_path")
        .and_then(|col| col.as_any().downcast_ref::<StringArray>())
        .ok_or_else(|| IndexerError::Database("Missing file_path column".to_string()))?;

      let content_hashes = batch
        .column_by_name("content_hash")
        .and_then(|col| col.as_any().downcast_ref::<FixedSizeBinaryArray>())
        .ok_or_else(|| IndexerError::Database("Missing content_hash column".to_string()))?;

      for i in 0..batch.num_rows() {
        let file_path = PathBuf::from(file_paths.value(i));
        let mut hash = [0u8; 32];
        hash.copy_from_slice(content_hashes.value(i));
        hashes.insert(file_path, hash);
      }
    }

    info!("Loaded {} existing file hashes from database", hashes.len());
    Ok(hashes)
  }

  pub async fn index(
    &self,
    project_id: Uuid,
    project_path: &Path,
    cancel_token: Option<CancellationToken>,
  ) -> Result<(usize, Option<(std::collections::BTreeSet<String>, String)>), IndexerError> {
    let start_time = Instant::now();
    info!(project_id = %project_id, path = %project_path.display(), "Starting channel-based indexing");

    // Query existing hashes for work avoidance
    let existing_hashes = self
      .get_existing_file_hashes(project_id)
      .await
      .unwrap_or_else(|e| {
        error!("Failed to get existing hashes: {}", e);
        BTreeMap::new()
      });

    // Setup
    // Get tokenizer from the embedding provider
    let tokenizer = if let Some(provider_tokenizer) = self.embedding_provider.tokenizer() {
      // Use the pre-loaded tokenizer from the provider
      Tokenizer::PreloadedHuggingFace(provider_tokenizer)
    } else {
      // Default to character-based tokenization for local models
      Tokenizer::Characters
    };

    let optimal_chunk_size = self.config.optimal_chunk_size();
    info!(
      chunk_size = optimal_chunk_size,
      "Using optimal chunk size based on embedding provider configuration"
    );

    let walk_options = WalkOptions {
      max_chunk_size: optimal_chunk_size,
      tokenizer,
      max_parallel: self.config.max_parallel_files,
      max_file_size: self.config.max_file_size,
      large_file_threads: self.config.large_file_threads.unwrap_or(4),
      existing_hashes,
    };

    let chunk_stream = walk_project(project_path.to_path_buf(), walk_options);
    // Use same batch size as local models for consistency
    let result = self
      .index_stream(project_id, chunk_stream, 256, cancel_token)
      .await?;

    let elapsed = start_time.elapsed();
    info!(
      elapsed = %humantime::format_duration(elapsed),
      documents_written = result.0,
      "Indexing completed"
    );

    Ok(result)
  }

  /// Process file changes including expanding directories to individual files for deletion
  pub async fn index_file_changes(
    &self,
    project_id: Uuid,
    project_root: impl AsRef<Path>,
    changes: &std::collections::BTreeSet<crate::models::FileChange>,
    cancel_token: Option<CancellationToken>,
  ) -> Result<(usize, Option<(std::collections::BTreeSet<String>, String)>), IndexerError> {
    use crate::models::FileOperation;
    use arrow::array::StringArray;
    use futures::stream;

    let project_root = project_root.as_ref();
    let mut file_paths = Vec::new();

    // Process each change to expand directories to file paths
    for change in changes {
      debug!(
        "Processing change: {:?} for path: {}",
        change.operation,
        change.path.display()
      );
      match change.operation {
        FileOperation::Delete => {
          // For delete operations, always check if there are files under this path in the database
          // This handles both explicit directory deletions and the case where a directory
          // was deleted and no longer exists on disk
          debug!(
            "Checking if path has files under it: {}",
            change.path.display()
          );
          let dir_str = change.path.to_string_lossy();
          let escaped_path = dir_str.replace("'", "''");

          // Query for all files with this path prefix (could be a directory) for this project
          let query_expr = if escaped_path.ends_with(std::path::MAIN_SEPARATOR) {
            format!(
              "project_id = '{}' AND file_path LIKE '{}%'",
              project_id, escaped_path
            )
          } else {
            // Could be either a file or directory - check for both
            format!(
              "project_id = '{}' AND (file_path = '{}' OR file_path LIKE '{}{}%')",
              project_id,
              escaped_path,
              escaped_path,
              std::path::MAIN_SEPARATOR
            )
          };

          // Get read access to the table
          let table = self.table.read().await;
          let mut query = table
            .query()
            .only_if(&query_expr)
            .select(lancedb::query::Select::columns(&["file_path"]))
            .execute()
            .await
            .map_err(|e| IndexerError::Database(e.to_string()))?;

          // Collect all file paths that match
          let mut found_files = Vec::new();
          while let Some(batch) = query
            .try_next()
            .await
            .map_err(|e| IndexerError::Database(e.to_string()))?
          {
            if let Some(file_paths_array) = batch
              .column_by_name("file_path")
              .and_then(|col| col.as_any().downcast_ref::<StringArray>())
            {
              for i in 0..batch.num_rows() {
                let file_path = file_paths_array.value(i);
                found_files.push(PathBuf::from(file_path));
                debug!(
                  file_path = %file_path,
                  "Found file to delete"
                );
              }
            }
          }

          drop(table); // Release the read lock

          if !found_files.is_empty() {
            // Found files in the database - add them all for deletion
            debug!(
              "Found {} files in DB under path: {}",
              found_files.len(),
              change.path.display()
            );
            for file in found_files {
              file_paths.push(file);
            }
          } else {
            // No files found in DB - just add the original path
            // It might be a file that doesn't exist yet or was already deleted
            debug!(
              "No files found in DB for path: {}, adding as-is",
              change.path.display()
            );
            file_paths.push(change.path.clone());
          }
        }
        _ => {
          // For all other operations (file deletes, adds, updates), use the path as-is
          file_paths.push(change.path.clone());
          debug!(
            file_path = %change.path.display(),
            operation = ?change.operation,
            "Adding file path for processing"
          );
        }
      }
    }

    if file_paths.is_empty() {
      info!("No files to process after expanding directories");
      return Ok((0, None));
    }

    // Create a stream from all file paths
    let file_stream = stream::iter(file_paths);

    // Use the existing index_files method
    self
      .index_files(project_id, project_root, file_stream, cancel_token)
      .await
  }

  pub async fn index_files(
    &self,
    project_id: Uuid,
    project_root: impl AsRef<Path>,
    files: impl futures_util::Stream<Item = std::path::PathBuf> + Send + 'static,
    cancel_token: Option<CancellationToken>,
  ) -> Result<(usize, Option<(std::collections::BTreeSet<String>, String)>), IndexerError> {
    let start_time = Instant::now();
    let project_root = project_root.as_ref().to_path_buf();
    info!(project_id = %project_id, project_root = %project_root.display(), "Starting partial file indexing");

    // Query existing hashes for work avoidance
    let existing_hashes = self
      .get_existing_file_hashes(project_id)
      .await
      .unwrap_or_else(|e| {
        error!("Failed to get existing hashes: {}", e);
        BTreeMap::new()
      });

    // Get tokenizer from the embedding provider
    let tokenizer = if let Some(provider_tokenizer) = self.embedding_provider.tokenizer() {
      // Use the pre-loaded tokenizer from the provider
      Tokenizer::PreloadedHuggingFace(provider_tokenizer)
    } else {
      // Default to character-based tokenization for local models
      Tokenizer::Characters
    };

    let optimal_chunk_size = self.config.optimal_chunk_size();
    info!(
      chunk_size = optimal_chunk_size,
      "Using optimal chunk size based on embedding provider configuration"
    );

    let walk_options = WalkOptions {
      max_chunk_size: optimal_chunk_size,
      tokenizer,
      max_parallel: self.config.max_parallel_files,
      max_file_size: self.config.max_file_size,
      large_file_threads: self.config.large_file_threads.unwrap_or(4),
      existing_hashes,
    };

    // Use the walk_files function from chunkers
    let chunk_stream = breeze_chunkers::walk_files(files, project_root, walk_options);
    // Use same batch size as local models for consistency
    let result = self
      .index_stream(project_id, chunk_stream, 256, cancel_token)
      .await?;

    let elapsed = start_time.elapsed();
    info!(
      elapsed_seconds = elapsed.as_secs_f64(),
      documents_written = result.0,
      "Partial indexing completed"
    );

    Ok(result)
  }

  // Testable pipeline that accepts any stream of chunks
  pub async fn index_stream(
    &self,
    project_id: Uuid,
    chunk_stream: impl futures_util::Stream<
      Item = Result<breeze_chunkers::ProjectChunk, breeze_chunkers::ChunkError>,
    > + Send
    + 'static,
    max_batch_size: usize,
    external_cancel_token: Option<CancellationToken>,
  ) -> Result<(usize, Option<(std::collections::BTreeSet<String>, String)>), IndexerError> {
    let embedding_dim = self.embedding_dim;

    // Create channels with bounded capacity for backpressure
    let (batch_tx, batch_rx) = mpsc::channel::<ChunkBatch>(10);
    let (embedded_tx, embedded_rx) = mpsc::channel::<EmbeddedChunkWithFile>(100);
    let (doc_tx, doc_rx) = mpsc::channel::<CodeDocument>(50);
    let (chunk_tx, chunk_rx) = mpsc::channel::<crate::models::CodeChunk>(100);
    let (delete_tx, delete_rx) = mpsc::channel::<(Uuid, PathBuf)>(25);

    // Progress tracking and cancellation
    let stats = IndexingStats::new();
    let cancel_token = external_cancel_token.unwrap_or_default();

    // Start progress reporter with its own cancellation token
    let progress_cancel_token = CancellationToken::new();
    let progress_handle = self.spawn_progress_reporter(stats.clone(), progress_cancel_token.clone());

    // Start pipeline stages with cancellation support
    // we start this with in the reverse order of execution
    let sink_handle = self.spawn_sink(doc_rx, stats.clone(), embedding_dim);
    let chunk_sink_handle = self.spawn_chunk_sink(chunk_rx, stats.clone(), embedding_dim);
    let delete_handle = self.spawn_delete_handler(project_id, delete_rx, cancel_token.clone());
    let doc_handle = self.spawn_document_builder(DocumentBuilderParams {
      project_id,
      embedded_rx,
      doc_tx,
      chunk_tx,
      embedding_dim,
      stats: stats.clone(),
      cancel_token: cancel_token.clone(),
    });

    // Choose between single worker or multi-worker based on provider type
    let embed_handles = if self.embedding_provider.is_remote() {
      // For remote providers, spawn multiple workers with work-stealing
      info!(
        "Using {} concurrent embedding workers for remote provider",
        self.config.embedding_workers
      );
      self.spawn_remote_embedders(
        batch_rx,
        embedded_tx,
        stats.clone(),
        cancel_token.clone(),
        self.config.embedding_workers,
      )
    } else {
      // For local providers, use single worker
      info!("Using single embedding worker for local provider");
      vec![self.spawn_embedder(batch_rx, embedded_tx, stats.clone(), cancel_token.clone())]
    };

    let stream_processor_params = StreamProcessorParams {
      project_id,
      batch_tx,
      delete_tx,
      max_batch_size,
      stats: stats.clone(),
      cancel_token: cancel_token.clone(),
    };
    let walk_handle = self.spawn_stream_processor(chunk_stream, stream_processor_params);

    // Wait for pipeline completion with proper error handling
    let walk_result = walk_handle.await;
    let doc_result = doc_handle.await;
    // Now wait for sink results
    let sink_result = sink_handle.await;
    let chunk_sink_result = chunk_sink_handle.await;
    let delete_result = delete_handle.await;
    // Wait for all embedding workers
    let embed_results = futures::future::join_all(embed_handles).await;

    // Check all results and report errors, cancelling other tasks on first error
    if let Err(e) = walk_result {
      cancel_token.cancel();
      return Err(IndexerError::Task(format!(
        "Stream processor task failed: {}",
        e
      )));
    }

    // Check embedding worker results
    for (i, result) in embed_results.iter().enumerate() {
      if let Err(e) = result {
        cancel_token.cancel();
        return Err(IndexerError::Task(format!(
          "Embedder task {} failed: {}",
          i, e
        )));
      }
    }

    let doc_result = match doc_result {
      Ok(result) => result,
      Err(e) => {
        cancel_token.cancel();
        return Err(IndexerError::Task(format!(
          "Document builder task failed: {}",
          e
        )));
      }
    };

    if let Err(e) = delete_result {
      cancel_token.cancel();
      return Err(IndexerError::Task(format!(
        "Delete handler task failed: {}",
        e
      )));
    }

    let documents_written = sink_result
      .map_err(|e| IndexerError::Task(format!("Sink task panicked: {}", e)))?
      .map_err(|e| IndexerError::Task(format!("Sink task failed: {}", e)))?;

    let _chunks_written = chunk_sink_result
      .map_err(|e| IndexerError::Task(format!("Chunk sink task panicked: {}", e)))?
      .map_err(|e| IndexerError::Task(format!("Chunk sink task failed: {}", e)))?;

    // Cancel progress reporter
    progress_cancel_token.cancel();
    let _ = progress_handle.await;

    // Report final results
    info!(
      files_chunked = format!(
        "{}/{}",
        stats.files_chunked.load(Ordering::Relaxed),
        stats.files.load(Ordering::Relaxed)
      ),
      files_embedded = format!(
        "{}/{}",
        stats.files_embedded.load(Ordering::Relaxed),
        stats.files.load(Ordering::Relaxed)
      ),
      files_stored = format!(
        "{}/{}",
        stats.files_stored.load(Ordering::Relaxed),
        stats.files.load(Ordering::Relaxed)
      ),
      chunks = format!(
        "{}/{}",
        stats.chunks_processed.load(Ordering::Relaxed),
        stats.chunks.load(Ordering::Relaxed)
      ),
      batches = stats.batches.load(Ordering::Relaxed),
      documents = stats.documents.load(Ordering::Relaxed),
      documents_written,
      "Indexing completed"
    );

    // Return the count and any failures from document builder
    Ok((documents_written, doc_result.1))
  }

  fn spawn_progress_reporter(
    &self,
    stats: IndexingStats,
    cancel_token: CancellationToken,
  ) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
      let start = Instant::now();
      let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
      interval.tick().await; // Skip the immediate tick

      loop {
        tokio::select! {
          _ = cancel_token.cancelled() => break,
          _ = interval.tick() => {
            let elapsed = start.elapsed();
            let files = stats.files.load(Ordering::Relaxed);
            let chunks = stats.chunks.load(Ordering::Relaxed);
            let chunks_processed = stats.chunks_processed.load(Ordering::Relaxed);
            let files_chunked = stats.files_chunked.load(Ordering::Relaxed);
            let files_embedded = stats.files_embedded.load(Ordering::Relaxed);
            let files_stored = stats.files_stored.load(Ordering::Relaxed);

            info!(
              elapsed = %humantime::format_duration(elapsed),
              files_chunked = format!("{}/{}", files_chunked, files),
              files_embedded = format!("{}/{}", files_embedded, files),
              files_stored = format!("{}/{}", files_stored, files),
              chunks = format!("{}/{}", chunks_processed, chunks),
              "Indexing progress"
            );
          }
        }
      }
    })
  }

  fn spawn_stream_processor(
    &self,
    chunk_stream: impl futures_util::Stream<
      Item = Result<breeze_chunkers::ProjectChunk, breeze_chunkers::ChunkError>,
    > + Send
    + 'static,
    params: StreamProcessorParams,
  ) -> tokio::task::JoinHandle<()> {
    tokio::spawn(stream_processor_task(chunk_stream, params))
  }

  fn spawn_embedder(
    &self,
    batch_rx: mpsc::Receiver<ChunkBatch>,
    embedded_tx: mpsc::Sender<EmbeddedChunkWithFile>,
    stats: IndexingStats,
    cancel_token: CancellationToken,
  ) -> tokio::task::JoinHandle<()> {
    let embedding_provider = self.embedding_provider.clone();
    let batching_strategy = self.embedding_provider.create_batching_strategy();
    tokio::spawn(embedder_task(
      embedding_provider,
      batching_strategy,
      batch_rx,
      embedded_tx,
      stats,
      cancel_token,
    ))
  }

  /// Spawn multiple embedding workers with work-stealing for remote providers
  ///
  /// This uses a flume MPMC channel to enable work-stealing behavior.
  /// Workers pull batches from the shared queue as they become available,
  /// naturally balancing load based on their processing speed.
  fn spawn_remote_embedders(
    &self,
    mut batch_rx: mpsc::Receiver<ChunkBatch>,
    embedded_tx: mpsc::Sender<EmbeddedChunkWithFile>,
    stats: IndexingStats,
    cancel_token: CancellationToken,
    num_workers: usize,
  ) -> Vec<tokio::task::JoinHandle<()>> {
    // Create flume channel for work-stealing
    let (work_tx, work_rx) = flume::unbounded::<ChunkBatch>();

    // Spawn a task to forward from mpsc to flume
    let forward_cancel = cancel_token.clone();
    tokio::spawn(async move {
      loop {
        tokio::select! {
          _ = forward_cancel.cancelled() => break,
          batch = batch_rx.recv() => {
            match batch {
              Some(batch) => {
                if work_tx.send(batch).is_err() {
                  break; // Workers have shut down
                }
              }
              None => break, // Channel closed
            }
          }
        }
      }
    });

    // Spawn multiple workers
    let mut handles = Vec::with_capacity(num_workers);
    for i in 0..num_workers {
      let embedding_provider = self.embedding_provider.clone();
      let batching_strategy = self.embedding_provider.create_batching_strategy();
      let work_rx = work_rx.clone();
      let embedded_tx = embedded_tx.clone();
      let stats = stats.clone();
      let cancel_token = cancel_token.clone();

      let handle = tokio::spawn(remote_embedder_worker_task(
        i,
        embedding_provider,
        batching_strategy,
        work_rx,
        embedded_tx,
        stats,
        cancel_token,
      ));
      handles.push(handle);
    }

    handles
  }

  fn spawn_document_builder(
    &self,
    params: DocumentBuilderParams,
  ) -> tokio::task::JoinHandle<DocumentBuilderResult> {
    tokio::spawn(document_builder_task(
      params.project_id,
      params.embedded_rx,
      params.doc_tx,
      params.chunk_tx,
      params.embedding_dim,
      params.stats,
      params.cancel_token,
    ))
  }

  fn spawn_sink(
    &self,
    doc_rx: mpsc::Receiver<CodeDocument>,
    stats: IndexingStats,
    embedding_dim: usize,
  ) -> tokio::task::JoinHandle<Result<usize, IndexerError>> {
    let table = self.table.clone();
    let last_optimize_version = self.last_optimize_version.clone();
    let optimize_threshold = self.config.optimize_threshold;
    tokio::spawn(sink_task(
      doc_rx,
      table,
      embedding_dim,
      last_optimize_version,
      optimize_threshold,
      stats,
    ))
  }

  fn spawn_chunk_sink(
    &self,
    chunk_rx: mpsc::Receiver<crate::models::CodeChunk>,
    _stats: IndexingStats,
    embedding_dim: usize,
  ) -> tokio::task::JoinHandle<Result<usize, IndexerError>> {
    let chunk_table = self.chunk_table.clone();
    let last_optimize_version = self.last_optimize_version.clone();
    let optimize_threshold = self.config.optimize_threshold;
    tokio::spawn(chunk_sink_task(
      chunk_rx,
      chunk_table,
      embedding_dim,
      last_optimize_version,
      optimize_threshold,
    ))
  }

  fn spawn_delete_handler(
    &self,
    project_id: Uuid,
    delete_rx: mpsc::Receiver<(Uuid, PathBuf)>,
    cancel_token: CancellationToken,
  ) -> tokio::task::JoinHandle<()> {
    let table = self.table.clone();
    tokio::spawn(delete_handler_task(
      project_id,
      delete_rx,
      table,
      cancel_token,
    ))
  }
}

#[derive(Clone)]
struct IndexingStats {
  files: Arc<AtomicUsize>,            // Total files discovered
  chunks: Arc<AtomicUsize>,           // Total chunks created
  batches: Arc<AtomicUsize>,          // Total batches processed
  documents: Arc<AtomicUsize>,        // Documents created
  chunks_processed: Arc<AtomicUsize>, // Chunks embedded
  files_chunked: Arc<AtomicUsize>,    // Files that completed chunking
  files_embedded: Arc<AtomicUsize>,   // Files that completed embedding
  files_stored: Arc<AtomicUsize>,     // Files stored to database
}

impl IndexingStats {
  fn new() -> Self {
    Self {
      files: Arc::new(AtomicUsize::new(0)),
      chunks: Arc::new(AtomicUsize::new(0)),
      batches: Arc::new(AtomicUsize::new(0)),
      documents: Arc::new(AtomicUsize::new(0)),
      chunks_processed: Arc::new(AtomicUsize::new(0)),
      files_chunked: Arc::new(AtomicUsize::new(0)),
      files_embedded: Arc::new(AtomicUsize::new(0)),
      files_stored: Arc::new(AtomicUsize::new(0)),
    }
  }
}

struct StreamProcessorParams {
  project_id: Uuid,
  batch_tx: mpsc::Sender<ChunkBatch>,
  delete_tx: mpsc::Sender<(Uuid, PathBuf)>,
  max_batch_size: usize,
  stats: IndexingStats,
  cancel_token: CancellationToken,
}

async fn send_batch(
  tx: &mpsc::Sender<ChunkBatch>,
  buffer: &mut Vec<PipelineProjectChunk>,
  batch_id: usize,
) {
  if !buffer.is_empty() {
    debug!("Sending batch {} with {} chunks", batch_id, buffer.len());
    let batch = ChunkBatch {
      batch_id,
      chunks: std::mem::take(buffer),
    };
    if tx.send(batch).await.is_err() {
      debug!("Receiver dropped, stopping batch sender");
    }
  }
}

// Standalone task functions

struct TaskGuard {
  name: &'static str,
}

impl TaskGuard {
  fn new(name: &'static str) -> Self {
    debug!("{} task started", name);
    Self { name }
  }
}

impl Drop for TaskGuard {
  fn drop(&mut self) {
    debug!("{} task finished", self.name);
  }
}

async fn stream_processor_task(
  chunk_stream: impl futures_util::Stream<
    Item = Result<breeze_chunkers::ProjectChunk, breeze_chunkers::ChunkError>,
  > + Send
  + 'static,
  params: StreamProcessorParams,
) {
  let _guard = TaskGuard::new("Stream processor");
  let mut chunk_stream = Box::pin(chunk_stream);
  let mut batch_buffer = Vec::new();
  let mut regular_chunk_count = 0;
  let mut batch_id = 0;

  loop {
    tokio::select! {
      _ = params.cancel_token.cancelled() => {
        info!("Walker cancelled");
        break;
      }
      result = chunk_stream.next() => {
        match result {
          Some(Ok(project_chunk)) => {
            // Handle Delete chunks separately
            if let Chunk::Delete { file_path } = &project_chunk.chunk {
              // Send delete request immediately
              info!("Stream processor sending delete request for: {}", file_path);
              if let Err(e) = params.delete_tx.send((params.project_id, PathBuf::from(file_path))).await {
                error!("Failed to send delete request for {}: {}", file_path, e);
              }
              continue; // Don't add to batch buffer
            }

            // Convert to PipelineChunk
            if let Some(pipeline_chunk) = PipelineChunk::from_chunk(project_chunk.chunk.clone()) {
              let pipeline_project_chunk = PipelineProjectChunk {
                file_path: project_chunk.file_path,
                chunk: pipeline_chunk,
              };

              // Count files when we see EOF markers
              let is_eof = matches!(pipeline_project_chunk.chunk, PipelineChunk::EndOfFile { .. });

              if is_eof {
                params.stats.files.fetch_add(1, Ordering::Relaxed);
                params.stats.files_chunked.fetch_add(1, Ordering::Relaxed);
              } else {
                params.stats.chunks.fetch_add(1, Ordering::Relaxed);
                regular_chunk_count += 1;
              }

              // Add chunks (including EOF) to batch buffer
              batch_buffer.push(pipeline_project_chunk);
            }

            // Send batch when we have enough REGULAR chunks (not counting EOFs)
            if regular_chunk_count >= params.max_batch_size {
              send_batch(&params.batch_tx, &mut batch_buffer, batch_id).await;
              batch_id += 1;
              regular_chunk_count = 0;
            }
          }
          Some(Err(e)) => error!("Error processing chunk: {}", e),
          None => {
            break;
          }
        }
      }
    }
  }

  // Send remaining chunks
  if !batch_buffer.is_empty() {
    send_batch(&params.batch_tx, &mut batch_buffer, batch_id).await;
  }

  info!(
    files = params.stats.files.load(Ordering::Relaxed),
    chunks = params.stats.chunks.load(Ordering::Relaxed),
    "Stream processor completed"
  );
}

/// Worker task for remote embedding providers that pulls work from a shared queue
///
/// This enables work-stealing behavior where faster workers naturally process more batches
async fn remote_embedder_worker_task(
  worker_id: usize,
  embedding_provider: Arc<dyn EmbeddingProvider>,
  batching_strategy: Box<dyn crate::embeddings::batching::BatchingStrategy>,
  work_rx: flume::Receiver<ChunkBatch>,
  embedded_tx: mpsc::Sender<EmbeddedChunkWithFile>,
  stats: IndexingStats,
  cancel_token: CancellationToken,
) {
  debug!("Remote Embedder Worker {} started", worker_id);

  // Simple structure to hold regular chunks and EOF chunks separately
  struct PendingBatch {
    regular_chunks: Vec<PipelineProjectChunk>,
    eof_chunks: Vec<PipelineProjectChunk>,
  }

  // Keep batches in order - BTreeMap maintains key order
  let mut pending_batches: std::collections::BTreeMap<usize, PendingBatch> =
    std::collections::BTreeMap::new();

  loop {
    // Check for cancellation
    if cancel_token.is_cancelled() {
      info!("Remote embedder worker {} cancelled", worker_id);
      break;
    }

    // Try to get work with a timeout to check cancellation periodically
    match work_rx.recv_timeout(std::time::Duration::from_millis(100)) {
      Ok(batch) => {
        let batch_id = batch.batch_id;

        // Separate EOF markers from regular chunks
        let (eof_chunks, regular_chunks): (Vec<_>, Vec<_>) = batch
          .chunks
          .into_iter()
          .partition(|pc| matches!(pc.chunk, PipelineChunk::EndOfFile { .. }));

        // Store this batch
        pending_batches.insert(
          batch_id,
          PendingBatch {
            regular_chunks,
            eof_chunks,
          },
        );

        // Check if we have enough chunks to process
        let mut chunk_count = 0;
        let mut have_enough = false;
        for pb in pending_batches.values() {
          chunk_count += pb.regular_chunks.len();
          if chunk_count >= batching_strategy.max_batch_size() {
            have_enough = true;
            break;
          }
        }

        // Process if we have enough chunks
        if have_enough {
          // Collect chunks to process while maintaining batch_id association
          let mut chunks_by_batch_id: std::collections::BTreeMap<usize, Vec<PipelineProjectChunk>> =
            std::collections::BTreeMap::new();

          for (bid, pending_batch) in pending_batches.iter_mut() {
            if !pending_batch.regular_chunks.is_empty() {
              // Take all regular chunks from this batch
              let chunks = std::mem::take(&mut pending_batch.regular_chunks);
              chunks_by_batch_id.insert(*bid, chunks);
            }
          }

          // Process chunks through batching strategy with batch IDs
          let mut chunks_with_batch_id = Vec::new();
          for (bid, chunks) in chunks_by_batch_id {
            for chunk in chunks {
              chunks_with_batch_id.push((bid, chunk));
            }
          }

          let embedding_batches = batching_strategy
            .prepare_batches(chunks_with_batch_id)
            .await;

          for embedding_batch in embedding_batches {
            debug!(
              "Worker {} processing batch {} with {} chunks",
              worker_id,
              embedding_batch.batch_id,
              embedding_batch.chunks.len()
            );

            process_embedding_batch(
              embedding_provider.as_ref(),
              embedding_batch.chunks,
              embedding_batch.batch_id,
              &embedded_tx,
              &stats,
            )
            .await;

            // Send EOF chunks if this batch is complete
            for eof_chunk in embedding_batch.eof_chunks {
              if let PipelineChunk::EndOfFile {
                file_path,
                content,
                content_hash,
              } = eof_chunk.chunk
              {
                stats.files_embedded.fetch_add(1, Ordering::Relaxed);
                let eof = EmbeddedChunkWithFile::EndOfFile {
                  batch_id: embedding_batch.batch_id,
                  file_path,
                  content,
                  content_hash,
                };
                if embedded_tx.send(eof).await.is_err() {
                  return;
                }
              }
            }
          }

          // Clean up empty batches
          pending_batches
            .retain(|_, pb| !pb.regular_chunks.is_empty() || !pb.eof_chunks.is_empty());
        }
      }
      Err(flume::RecvTimeoutError::Timeout) => {
        // Continue to check cancellation
        continue;
      }
      Err(flume::RecvTimeoutError::Disconnected) => {
        // Channel closed, process remaining chunks
        if !pending_batches.is_empty() {
          // Collect all remaining chunks with their batch IDs
          let mut chunks_with_batch_id = Vec::new();

          for (bid, pending_batch) in pending_batches.iter_mut() {
            let chunks = std::mem::take(&mut pending_batch.regular_chunks);
            for chunk in chunks {
              chunks_with_batch_id.push((*bid, chunk));
            }
          }

          if !chunks_with_batch_id.is_empty() {
            let embedding_batches = batching_strategy
              .prepare_batches(chunks_with_batch_id)
              .await;

            for embedding_batch in embedding_batches {
              process_embedding_batch(
                embedding_provider.as_ref(),
                embedding_batch.chunks,
                embedding_batch.batch_id,
                &embedded_tx,
                &stats,
              )
              .await;

              // Send EOF chunks
              for eof_chunk in embedding_batch.eof_chunks {
                if let PipelineChunk::EndOfFile {
                  file_path,
                  content,
                  content_hash,
                } = eof_chunk.chunk
                {
                  let eof = EmbeddedChunkWithFile::EndOfFile {
                    batch_id: embedding_batch.batch_id,
                    file_path,
                    content,
                    content_hash,
                  };
                  if embedded_tx.send(eof).await.is_err() {
                    return;
                  }
                }
              }
            }
          }

          // Send any remaining EOF chunks
          for (bid, pending_batch) in pending_batches {
            for eof_chunk in pending_batch.eof_chunks {
              if let PipelineChunk::EndOfFile {
                file_path,
                content,
                content_hash,
              } = eof_chunk.chunk
              {
                stats.files_embedded.fetch_add(1, Ordering::Relaxed);
                let eof = EmbeddedChunkWithFile::EndOfFile {
                  batch_id: bid,
                  file_path,
                  content,
                  content_hash,
                };
                if embedded_tx.send(eof).await.is_err() {
                  return;
                }
              }
            }
          }
        }

        break;
      }
    }
  }

  debug!("Remote embedder worker {} completed", worker_id);
}

async fn embedder_task(
  embedding_provider: Arc<dyn EmbeddingProvider>,
  batching_strategy: Box<dyn crate::embeddings::batching::BatchingStrategy>,
  mut batch_rx: mpsc::Receiver<ChunkBatch>,
  embedded_tx: mpsc::Sender<EmbeddedChunkWithFile>,
  stats: IndexingStats,
  cancel_token: CancellationToken,
) {
  let _guard = TaskGuard::new("Embedder");

  // Simple structure to hold regular chunks and EOF chunks separately
  struct PendingBatch {
    regular_chunks: Vec<PipelineProjectChunk>,
    eof_chunks: Vec<PipelineProjectChunk>,
  }

  // Keep batches in order - BTreeMap maintains key order
  let mut pending_batches: std::collections::BTreeMap<usize, PendingBatch> =
    std::collections::BTreeMap::new();

  loop {
    tokio::select! {
      _ = cancel_token.cancelled() => {
        info!("Embedder cancelled");
        break;
      }
      batch = batch_rx.recv() => {
        match batch {
          Some(batch) => {
            let batch_id = batch.batch_id;

            // Separate EOF markers from regular chunks
            let (eof_chunks, regular_chunks): (Vec<_>, Vec<_>) = batch.chunks.into_iter()
              .partition(|pc| matches!(pc.chunk, PipelineChunk::EndOfFile { .. }));

            // Store this batch
            pending_batches.insert(batch_id, PendingBatch {
              regular_chunks,
              eof_chunks,
            });

            // Check if we have enough chunks to process
            let mut chunk_count = 0;
            let mut have_enough = false;
            for pb in pending_batches.values() {
              chunk_count += pb.regular_chunks.len();
              if chunk_count >= batching_strategy.max_batch_size() {
                have_enough = true;
                break;
              }
            }

            // Process if we have enough chunks
            if have_enough {
              // Collect chunks with their batch IDs
              let mut chunks_with_batch_id = Vec::new();

              for (bid, pending_batch) in pending_batches.iter_mut() {
                if !pending_batch.regular_chunks.is_empty() {
                  let chunks = std::mem::take(&mut pending_batch.regular_chunks);
                  for chunk in chunks {
                    chunks_with_batch_id.push((*bid, chunk));
                  }
                }
              }

              // Process chunks through batching strategy
              let embedding_batches = batching_strategy.prepare_batches(chunks_with_batch_id).await;

              for embedding_batch in embedding_batches {
                debug!(
                  "Processing batch {} with {} chunks",
                  embedding_batch.batch_id,
                  embedding_batch.chunks.len()
                );

                process_embedding_batch(
                  embedding_provider.as_ref(),
                  embedding_batch.chunks,
                  embedding_batch.batch_id,
                  &embedded_tx,
                  &stats,
                )
                .await;

                // Send EOF chunks if this batch is complete
                for eof_chunk in embedding_batch.eof_chunks {
                  if let PipelineChunk::EndOfFile { file_path, content, content_hash } = eof_chunk.chunk {
                    stats.files_embedded.fetch_add(1, Ordering::Relaxed);
                    let eof = EmbeddedChunkWithFile::EndOfFile {
                      batch_id: embedding_batch.batch_id,
                      file_path,
                      content,
                      content_hash,
                    };
                    if embedded_tx.send(eof).await.is_err() {
                      return;
                    }
                  }
                }
              }

              // Clean up empty batches
              pending_batches.retain(|_, pb| !pb.regular_chunks.is_empty() || !pb.eof_chunks.is_empty());
            }
          }
          None => {
            // Channel closed, process remaining chunks
            if !pending_batches.is_empty() {
              // Collect all remaining chunks with their batch IDs
              let mut chunks_with_batch_id = Vec::new();

              for (bid, pending_batch) in pending_batches.iter_mut() {
                let chunks = std::mem::take(&mut pending_batch.regular_chunks);
                for chunk in chunks {
                  chunks_with_batch_id.push((*bid, chunk));
                }
              }

              if !chunks_with_batch_id.is_empty() {
                let embedding_batches = batching_strategy.prepare_batches(chunks_with_batch_id).await;

                for embedding_batch in embedding_batches {
                  process_embedding_batch(
                    embedding_provider.as_ref(),
                    embedding_batch.chunks,
                    embedding_batch.batch_id,
                    &embedded_tx,
                    &stats,
                  ).await;

                  // Send EOF chunks
                  for eof_chunk in embedding_batch.eof_chunks {
                    if let PipelineChunk::EndOfFile { file_path, content, content_hash } = eof_chunk.chunk {
                            let eof = EmbeddedChunkWithFile::EndOfFile {
                        batch_id: embedding_batch.batch_id,
                        file_path,
                        content,
                        content_hash,
                      };
                      if embedded_tx.send(eof).await.is_err() {
                        return;
                      }
                    }
                  }
                }
              }

              // Send any remaining EOF chunks
              for (bid, pending_batch) in pending_batches {
                for eof_chunk in pending_batch.eof_chunks {
                  if let PipelineChunk::EndOfFile { file_path, content, content_hash } = eof_chunk.chunk {
                    stats.files_embedded.fetch_add(1, Ordering::Relaxed);
                    let eof = EmbeddedChunkWithFile::EndOfFile {
                      batch_id: bid,
                      file_path,
                      content,
                      content_hash,
                    };
                    if embedded_tx.send(eof).await.is_err() {
                      return;
                    }
                  }
                }
              }
            }

            break;
          }
        }
      }
    }
  }

  info!(
    batches = stats.batches.load(Ordering::Relaxed),
    "Embedder completed"
  );
}

async fn process_embedding_batch(
  embedding_provider: &dyn EmbeddingProvider,
  batch: Vec<PipelineProjectChunk>,
  batch_id: usize,
  embedded_tx: &mpsc::Sender<EmbeddedChunkWithFile>,
  stats: &IndexingStats,
) {
  if batch.is_empty() {
    return;
  }

  debug!("Processing embedding batch of {} chunks", batch.len());

  // Collect files in this batch for failure tracking
  let mut files_in_batch = std::collections::BTreeSet::new();

  // Create embedding inputs from chunks (without cloning text)
  let mut chunks_to_embed = Vec::new();
  let inputs: Vec<crate::embeddings::EmbeddingInput> = batch
    .iter()
    .enumerate()
    .filter_map(|(idx, pc)| match &pc.chunk {
      PipelineChunk::Semantic(sc) | PipelineChunk::Text(sc) => {
        files_in_batch.insert(pc.file_path.clone());
        let input = crate::embeddings::EmbeddingInput {
          text: &sc.text,
          token_count: sc.tokens.as_ref().map(|t| t.len()),
        };
        chunks_to_embed.push(idx);
        Some(input)
      }
      PipelineChunk::EndOfFile { .. } => None,
    })
    .collect();

  if inputs.is_empty() {
    return;
  }

  match embedding_provider.embed(&inputs).await {
    Ok(embeddings) => {
      debug!("Successfully embedded {} chunks", embeddings.len());
      stats.batches.fetch_add(1, Ordering::Relaxed);
      stats
        .chunks_processed
        .fetch_add(embeddings.len(), Ordering::Relaxed);

      let mut embedding_iter = embeddings.into_iter();
      let mut chunk_idx_iter = chunks_to_embed.into_iter();

      // Consume the batch to avoid cloning
      for (idx, pc) in batch.into_iter().enumerate() {
        if chunk_idx_iter.next() == Some(idx) {
          if let Some(embedding_vec) = embedding_iter.next() {
            let item = EmbeddedChunkWithFile::Embedded {
              batch_id,
              file_path: pc.file_path,
              chunk: Box::new(pc.chunk),
              embedding: embedding_vec,
            };
            if embedded_tx.send(item).await.is_err() {
              error!("Failed to send embedded chunk - receiver dropped");
              return;
            }
          }
        }
      }
    }
    Err(e) => {
      error!("Failed to embed batch of {} chunks: {}", batch.len(), e);

      // Send BatchFailure notification
      let failure = EmbeddedChunkWithFile::BatchFailure {
        batch_id,
        failed_files: files_in_batch,
        error: e.to_string(),
      };

      if embedded_tx.send(failure).await.is_err() {
        error!("Failed to send batch failure notification - receiver dropped");
      }
    }
  }
}

async fn document_builder_task(
  project_id: Uuid,
  mut embedded_rx: mpsc::Receiver<EmbeddedChunkWithFile>,
  doc_tx: mpsc::Sender<CodeDocument>,
  chunk_tx: mpsc::Sender<crate::models::CodeChunk>,
  embedding_dim: usize,
  stats: IndexingStats,
  cancel_token: CancellationToken,
) -> (usize, Option<(std::collections::BTreeSet<String>, String)>) {
  let _guard = TaskGuard::new("Document builder");
  let mut file_accumulators: std::collections::HashMap<String, FileAccumulator> =
    std::collections::HashMap::new();

  // Buffer for batching documents before sending
  let mut document_batch: Vec<CodeDocument> = Vec::with_capacity(100);
  let mut total_files_built = 0u64;

  // Buffer for out-of-order chunks
  let mut pending_chunks: std::collections::BTreeMap<usize, Vec<EmbeddedChunkWithFile>> =
    std::collections::BTreeMap::new();
  let mut next_expected_batch_id = 0;

  // Track failed files
  let mut failed_files: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();

  loop {
    tokio::select! {
      _ = cancel_token.cancelled() => {
        info!("Document builder cancelled");
        break;
      }
      embedded_chunk = embedded_rx.recv() => {
        match embedded_chunk {
          Some(embedded_chunk) => {
            let batch_id = match &embedded_chunk {
              EmbeddedChunkWithFile::Embedded { batch_id, .. } => *batch_id,
              EmbeddedChunkWithFile::EndOfFile { batch_id, .. } => *batch_id,
              EmbeddedChunkWithFile::BatchFailure { batch_id, .. } => *batch_id,
            };

            // Add chunk to pending buffer
            pending_chunks.entry(batch_id)
              .or_default()
              .push(embedded_chunk);

            // Process any chunks that are now in order
            while let Some((&current_batch_id, _)) = pending_chunks.first_key_value() {
              if current_batch_id == next_expected_batch_id {
                // Process this batch
                let batch_chunks = pending_chunks.remove(&current_batch_id).unwrap();

                for embedded_chunk in batch_chunks {
                  match embedded_chunk {
                    EmbeddedChunkWithFile::Embedded { file_path, chunk, embedding, .. } => {
                      // Skip chunks for failed files
                      if failed_files.contains(&file_path) {
                        continue;
                      }

                      // Accumulate chunk
                      let accumulator = file_accumulators.entry(file_path.clone())
                        .or_insert_with(|| FileAccumulator::new(file_path));

                      accumulator.add_chunk(EmbeddedChunk {
                        chunk: *chunk,
                        embedding,
                      });
                    }
                    EmbeddedChunkWithFile::EndOfFile { file_path, content, content_hash, .. } => {
                      // Skip EOF for failed files
                      if failed_files.contains(&file_path) {
                        continue;
                      }

                      // Build document for completed file
                      if let Some(mut accumulator) = file_accumulators.remove(&file_path) {
                        total_files_built += 1;
                        debug!(file_path, total_files_built, batch_id = current_batch_id,
                          "Building document for file: {file_path}");

                        // Add the EOF chunk to the accumulator
                        accumulator.add_chunk(EmbeddedChunk {
                          chunk: PipelineChunk::EndOfFile {
                            file_path: file_path.clone(),
                            content,
                            content_hash
                          },
                          embedding: vec![],
                        });

                        if let Some((doc, chunks)) = build_document_from_accumulator(project_id, accumulator, embedding_dim).await {
                          stats.documents.fetch_add(1, Ordering::Relaxed);
                          stats.files_stored.fetch_add(1, Ordering::Relaxed);

                          // Store chunks to code_chunks table
                          for chunk in chunks {
                            if chunk_tx.send(chunk).await.is_err() {
                              debug!("Chunk receiver dropped");
                              break;
                            }
                          }

                          document_batch.push(doc);

                          // Send batch when we reach 100 documents
                          if document_batch.len() >= 100 {
                            debug!(total_files_built, batch_len=document_batch.len(),
                              "Sending batch of {} documents", document_batch.len());
                            for doc in document_batch.drain(..) {
                              if doc_tx.send(doc).await.is_err() {
                                debug!("Document receiver dropped");
                                // Return early with current stats and failed files
                                let documents_processed = stats.documents.load(Ordering::Relaxed);
                                if failed_files.is_empty() {
                                  return (documents_processed, None);
                                } else {
                                  let error_msg = "Document receiver dropped while processing".to_string();
                                  return (documents_processed, Some((failed_files, error_msg)));
                                }
                              }
                            }
                          }
                        }
                      } else {
                        error!("Received EOF chunk for file without content chunks: {}", file_path);
                      }
                    }
                    EmbeddedChunkWithFile::BatchFailure { failed_files: batch_failed_files, error, .. } => {
                      error!("Batch {} failed: {}", current_batch_id, error);
                      // Mark all files in the failed batch as failed
                      for file in batch_failed_files {
                        failed_files.insert(file);
                      }
                    }
                  }
                }

                next_expected_batch_id += 1;
              } else {
                // We're missing some earlier batch, stop processing
                break;
              }
            }
          }
          None => break,
        }
      }
    }
  }

  // Process any remaining pending chunks that are buffered
  // This handles the case where we have out-of-order chunks still waiting
  for (batch_id, batch_chunks) in pending_chunks {
    debug!(
      "Processing remaining batch {} with {} chunks",
      batch_id,
      batch_chunks.len()
    );
    for embedded_chunk in batch_chunks {
      match embedded_chunk {
        EmbeddedChunkWithFile::Embedded {
          file_path,
          chunk,
          embedding,
          ..
        } => {
          // Skip chunks for failed files
          if failed_files.contains(&file_path) {
            continue;
          }

          // Accumulate chunk
          let accumulator = file_accumulators
            .entry(file_path.clone())
            .or_insert_with(|| FileAccumulator::new(file_path));

          accumulator.add_chunk(EmbeddedChunk {
            chunk: *chunk,
            embedding,
          });
        }
        EmbeddedChunkWithFile::EndOfFile {
          file_path,
          content,
          content_hash,
          ..
        } => {
          // Skip EOF for failed files
          if failed_files.contains(&file_path) {
            continue;
          }

          // Build document for completed file
          if let Some(mut accumulator) = file_accumulators.remove(&file_path) {
            total_files_built += 1;
            debug!(
              file_path,
              total_files_built,
              batch_id,
              "Building document for file from remaining chunks: {file_path}"
            );

            // Add the EOF chunk to the accumulator
            accumulator.add_chunk(EmbeddedChunk {
              chunk: PipelineChunk::EndOfFile {
                file_path: file_path.clone(),
                content,
                content_hash,
              },
              embedding: vec![],
            });

            if let Some((doc, chunks)) =
              build_document_from_accumulator(project_id, accumulator, embedding_dim).await
            {
              stats.documents.fetch_add(1, Ordering::Relaxed);

              // Store chunks to code_chunks table
              for chunk in chunks {
                if chunk_tx.send(chunk).await.is_err() {
                  debug!("Chunk receiver dropped");
                  break;
                }
              }

              document_batch.push(doc);
            }
          } else {
            error!(
              "Received EOF chunk for file without content chunks: {}",
              file_path
            );
          }
        }
        EmbeddedChunkWithFile::BatchFailure {
          failed_files: batch_failed_files,
          error,
          ..
        } => {
          error!("Batch {} failed: {}", batch_id, error);
          // Mark all files in the failed batch as failed
          for file in batch_failed_files {
            failed_files.insert(file);
          }
        }
      }
    }
  }

  // Process remaining files (those without EOF chunks)
  for (file_path, accumulator) in file_accumulators {
    // Skip failed files
    if failed_files.contains(&file_path) {
      continue;
    }

    if let Some((doc, chunks)) =
      build_document_from_accumulator(project_id, accumulator, embedding_dim).await
    {
      stats.documents.fetch_add(1, Ordering::Relaxed);

      // Store chunks to code_chunks table
      for chunk in chunks {
        if chunk_tx.send(chunk).await.is_err() {
          debug!("Chunk receiver dropped");
          break;
        }
      }

      document_batch.push(doc);
    }
  }

  // Send any remaining documents in the batch
  if !document_batch.is_empty() {
    debug!("Sending final batch of {} documents", document_batch.len());
    for doc in document_batch {
      let _ = doc_tx.send(doc).await;
    }
  }

  let documents_processed = stats.documents.load(Ordering::Relaxed);
  info!(
    documents = documents_processed,
    failed_files = failed_files.len(),
    "Document builder completed"
  );

  // Return the count and any failures
  if failed_files.is_empty() {
    (documents_processed, None)
  } else {
    // Aggregate error messages from batch failures
    let error_msg = "Embedding failures occurred during processing".to_string();
    (documents_processed, Some((failed_files, error_msg)))
  }
}

async fn sink_task(
  doc_rx: mpsc::Receiver<CodeDocument>,
  table: Arc<RwLock<Table>>,
  embedding_dim: usize,
  last_optimize_version: Arc<RwLock<u64>>,
  optimize_threshold: u64,
  _stats: IndexingStats,
) -> Result<usize, IndexerError> {
  let _guard = TaskGuard::new("Sink");
  let converter = BufferedRecordBatchConverter::<CodeDocument>::default()
    .with_schema(Arc::new(CodeDocument::schema(embedding_dim)));

  let sink = LanceDbSink::new(table.clone(), last_optimize_version, optimize_threshold);

  let doc_stream = tokio_stream::wrappers::ReceiverStream::new(doc_rx);
  let record_batches = converter.convert(Box::pin(doc_stream));

  let mut sink_stream = sink.sink(record_batches);
  let mut batch_count = 0;

  while let Some(()) = sink_stream.next().await {
    batch_count += 1;
    debug!(batch_number = batch_count, "Written batch to LanceDB");
  }

  // Get the final row count from the table, excluding the dummy document
  let table_guard = table.read().await;
  let count = table_guard
    .count_rows(Some(format!("id != '{}'", Uuid::nil())))
    .await? as usize;
  debug!(
    row_count = count,
    "Final table row count (excluding dummy document)"
  );

  Ok(count)
}

async fn chunk_sink_task(
  chunk_rx: mpsc::Receiver<crate::models::CodeChunk>,
  table: Arc<RwLock<Table>>,
  embedding_dim: usize,
  last_optimize_version: Arc<RwLock<u64>>,
  optimize_threshold: u64,
) -> Result<usize, IndexerError> {
  let _guard = TaskGuard::new("Chunk Sink");
  let converter = BufferedRecordBatchConverter::<crate::models::CodeChunk>::default()
    .with_schema(Arc::new(crate::models::CodeChunk::schema(embedding_dim)));

  let sink = crate::sinks::chunk_sink::ChunkSink::new(
    table.clone(),
    last_optimize_version,
    optimize_threshold,
  );

  let chunk_stream = tokio_stream::wrappers::ReceiverStream::new(chunk_rx);
  let record_batches = converter.convert(Box::pin(chunk_stream));

  let mut sink_stream = sink.sink(record_batches);
  let mut batch_count = 0;

  while let Some(()) = sink_stream.next().await {
    batch_count += 1;
    debug!(batch_number = batch_count, "Written chunk batch to LanceDB");
  }

  // Get the final row count from the table
  let table_guard = table.read().await;
  let count = table_guard.count_rows(None).await? as usize;
  debug!(row_count = count, "Final chunk table row count");

  Ok(count)
}

async fn delete_handler_task(
  project_id: Uuid,
  mut delete_rx: mpsc::Receiver<(Uuid, PathBuf)>,
  table: Arc<RwLock<Table>>,
  cancel_token: CancellationToken,
) {
  let _guard = TaskGuard::new("Delete handler");
  let mut delete_count = 0;

  loop {
    tokio::select! {
      _ = cancel_token.cancelled() => {
        info!("Delete handler cancelled");
        break;
      }
      path = delete_rx.recv() => {
        match path {
          Some((recv_project_id, file_path)) => {
            // Only process deletions for the current project
            if recv_project_id != project_id {
              debug!("Skipping deletion for different project: {} != {}", recv_project_id, project_id);
              continue;
            }

            let path_str = file_path.to_string_lossy();

            // Never delete the dummy document
            if path_str == "__lancedb_dummy__.txt" {
              debug!("Skipping deletion of dummy document");
              continue;
            }

            let escaped_path = path_str.to_string().replace("'", "''");
            let delete_expr = format!("project_id = '{}' AND file_path = '{}'", project_id, escaped_path);

            info!("Delete handler received delete request for: {}", file_path.display());

            let table_guard = table.write().await;
            match table_guard.delete(&delete_expr).await {
              Ok(_) => {
                delete_count += 1;
                info!("Successfully deleted document: {}", file_path.display());
              }
              Err(e) => {
                error!("Failed to delete document {}: {}", file_path.display(), e);
              }
            }
            drop(table_guard); // Release the write lock
          }
          None => break, // Channel closed
        }
      }
    }
  }

  if delete_count > 0 {
    info!(
      "Delete handler completed: {} documents deleted",
      delete_count
    );
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::config::Config;
  use breeze_chunkers::{ChunkError, ChunkMetadata, ProjectChunk, SemanticChunk};
  use futures_util::stream;
  use tempfile::tempdir;
  use tokio_util::sync::CancellationToken;

  // Helper to create test chunks with fake tokens
  fn create_test_chunk(file_path: &str, text: &str, is_eof: bool) -> ProjectChunk {
    use std::collections::HashMap;
    use std::sync::OnceLock;

    // For tests, we'll accumulate content to pass in EOF
    static TEST_CONTENT: OnceLock<std::sync::Mutex<HashMap<String, String>>> = OnceLock::new();
    let test_content = TEST_CONTENT.get_or_init(|| std::sync::Mutex::new(HashMap::new()));

    if is_eof {
      // Get accumulated content for this file
      let content = test_content
        .lock()
        .unwrap()
        .get(file_path)
        .cloned()
        .unwrap_or_default();

      // Compute hash
      let hash = blake3::hash(content.as_bytes());
      let mut content_hash = [0u8; 32];
      content_hash.copy_from_slice(hash.as_bytes());

      // Clear the content for this file after use
      test_content.lock().unwrap().remove(file_path);

      ProjectChunk {
        file_path: file_path.to_string(),
        chunk: Chunk::EndOfFile {
          file_path: file_path.to_string(),
          content,
          content_hash,
        },
      }
    } else {
      // Accumulate content for this file
      test_content
        .lock()
        .unwrap()
        .entry(file_path.to_string())
        .or_default()
        .push_str(text);

      // Create fake tokens - just split by whitespace for tests
      let tokens: Vec<u32> = text.split_whitespace().map(|_| 1234u32).collect();
      ProjectChunk {
        file_path: file_path.to_string(),
        chunk: Chunk::Text(SemanticChunk {
          text: text.to_string(),
          start_byte: 0,
          end_byte: text.len(),
          start_line: 1,
          end_line: 1,
          tokens: Some(tokens),
          metadata: ChunkMetadata {
            node_type: "test".to_string(),
            node_name: None,
            language: "rust".to_string(),
            parent_context: None,
            scope_path: vec![],
            definitions: vec![],
            references: vec![],
          },
        }),
      }
    }
  }

  // Mock embedding provider for tests
  struct MockEmbeddingProvider {
    embedding_dim: usize,
  }

  impl MockEmbeddingProvider {
    fn new(embedding_dim: usize) -> Self {
      Self { embedding_dim }
    }
  }

  #[async_trait::async_trait]
  impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(
      &self,
      inputs: &[crate::embeddings::EmbeddingInput<'_>],
    ) -> crate::embeddings::EmbeddingResult<Vec<Vec<f32>>> {
      Ok(
        inputs
          .iter()
          .map(|input| {
            // Generate a deterministic embedding based on text content
            let mut embedding = vec![0.0; self.embedding_dim];
            let hash = blake3::hash(input.text.as_bytes());
            let hash_bytes = hash.as_bytes();
            for (i, &byte) in hash_bytes.iter().enumerate() {
              if i < self.embedding_dim {
                embedding[i] = (byte as f32) / 255.0;
              }
            }
            embedding
          })
          .collect(),
      )
    }

    fn embedding_dim(&self) -> usize {
      self.embedding_dim
    }

    fn context_length(&self) -> usize {
      8192 // Mock context length
    }

    fn create_batching_strategy(&self) -> Box<dyn crate::embeddings::batching::BatchingStrategy> {
      Box::new(crate::embeddings::batching::LocalBatchingStrategy::new(256))
    }

    fn tokenizer(&self) -> Option<Arc<tokenizers::Tokenizer>> {
      None // Mock provider doesn't use a tokenizer
    }
  }

  // Helper function to create test embedding provider
  async fn create_test_embedding_provider() -> (Arc<dyn EmbeddingProvider>, usize) {
    let embedding_dim = 384; // Standard test embedding dimension
    let provider = MockEmbeddingProvider::new(embedding_dim);
    (Arc::new(provider), embedding_dim)
  }

  #[tokio::test]
  async fn test_pipeline_basic_flow() {
    let _ = tracing_subscriber::fmt()
      .with_env_filter("breeze=debug")
      .try_init();

    let (_temp_dir, config) = Config::test();

    // Create embedding provider for tests
    let (embedding_provider, embedding_dim) = create_test_embedding_provider().await;

    // Create temporary LanceDB
    let temp_db = tempdir().unwrap();
    let connection = lancedb::connect(temp_db.path().to_str().unwrap())
      .execute()
      .await
      .unwrap();
    let table = CodeDocument::ensure_table(&connection, "test", embedding_dim)
      .await
      .unwrap();
    let chunk_table =
      crate::models::CodeChunk::ensure_table(&connection, "test_chunks", embedding_dim)
        .await
        .unwrap();

    let indexer = BulkIndexer::new(
      Arc::new(config),
      embedding_provider,
      embedding_dim,
      Arc::new(RwLock::new(table)),
      Arc::new(RwLock::new(chunk_table)),
    );

    // Create test stream with 2 files
    // Use unique file names to avoid conflicts with concurrent tests
    let file1 = format!("test_basic_flow_file1_{}.rs", std::process::id());
    let file2 = format!("test_basic_flow_file2_{}.rs", std::process::id());

    let chunks = vec![
      Ok(create_test_chunk(&file1, "fn main() {}", false)),
      Ok(create_test_chunk(&file1, "println!(\"Hello\");", false)),
      Ok(create_test_chunk(&file1, "", true)), // EOF
      Ok(create_test_chunk(&file2, "struct Foo;", false)),
      Ok(create_test_chunk(&file2, "", true)), // EOF
    ];

    let chunk_stream = stream::iter(chunks);

    let (count, failures) = indexer
      .index_stream(Uuid::now_v7(), chunk_stream, 2, None)
      .await
      .unwrap();

    assert_eq!(count, 2, "Should have indexed 2 documents");
    assert!(failures.is_none(), "Should have no failures");
  }

  #[tokio::test]
  async fn test_batch_ordering_with_out_of_order_chunks() {
    let embedding_dim = 384;
    let (embedded_tx, embedded_rx) = mpsc::channel(1000);
    let (doc_tx, mut doc_rx) = mpsc::channel(1000);
    let (chunk_tx, _chunk_rx) = mpsc::channel(1000);
    let stats = IndexingStats::new();
    let cancel_token = CancellationToken::new();

    // Spawn document builder task
    let stats_clone = stats.clone();
    let cancel_clone = cancel_token.clone();
    let builder_handle = tokio::spawn(async move {
      document_builder_task(
        Uuid::now_v7(),
        embedded_rx,
        doc_tx,
        chunk_tx,
        embedding_dim,
        stats_clone,
        cancel_clone,
      )
      .await;
    });

    // Send chunks out of order to simulate race condition
    // Batch 1 chunks
    embedded_tx
      .send(EmbeddedChunkWithFile::Embedded {
        batch_id: 1,
        file_path: "file2.rs".to_string(),
        chunk: Box::new(PipelineChunk::Text(SemanticChunk {
          text: "chunk1".to_string(),
          tokens: None,
          start_byte: 0,
          end_byte: 6,
          start_line: 1,
          end_line: 1,
          metadata: ChunkMetadata {
            node_type: "text".to_string(),
            node_name: None,
            language: "rust".to_string(),
            parent_context: None,
            scope_path: vec![],
            definitions: vec![],
            references: vec![],
          },
        })),
        embedding: vec![0.1; embedding_dim],
      })
      .await
      .unwrap();

    // Batch 0 chunks arrive later (out of order)
    embedded_tx
      .send(EmbeddedChunkWithFile::Embedded {
        batch_id: 0,
        file_path: "file1.rs".to_string(),
        chunk: Box::new(PipelineChunk::Text(SemanticChunk {
          text: "chunk1".to_string(),
          tokens: None,
          start_byte: 0,
          end_byte: 6,
          start_line: 1,
          end_line: 1,
          metadata: ChunkMetadata {
            node_type: "text".to_string(),
            node_name: None,
            language: "rust".to_string(),
            parent_context: None,
            scope_path: vec![],
            definitions: vec![],
            references: vec![],
          },
        })),
        embedding: vec![0.2; embedding_dim],
      })
      .await
      .unwrap();

    // Send EOF for file1 (batch 0)
    embedded_tx
      .send(EmbeddedChunkWithFile::EndOfFile {
        batch_id: 0,
        file_path: "file1.rs".to_string(),
        content: "chunk1".to_string(),
        content_hash: [0u8; 32],
      })
      .await
      .unwrap();

    // Send EOF for file2 (batch 1)
    embedded_tx
      .send(EmbeddedChunkWithFile::EndOfFile {
        batch_id: 1,
        file_path: "file2.rs".to_string(),
        content: "chunk1".to_string(),
        content_hash: [1u8; 32],
      })
      .await
      .unwrap();

    // Close channel
    drop(embedded_tx);

    // Wait for builder to finish
    builder_handle.await.unwrap();

    // Collect all documents
    let mut documents_received: Vec<CodeDocument> = Vec::new();
    while let Ok(doc) = doc_rx.try_recv() {
      documents_received.push(doc);
    }

    // Both files should be processed successfully
    assert_eq!(documents_received.len(), 2, "Should have 2 documents");

    // Verify files were processed in correct order despite out-of-order arrival
    let file_names: Vec<&str> = documents_received
      .iter()
      .map(|doc| doc.file_path.as_str())
      .collect();

    assert!(file_names.contains(&"file1.rs"));
    assert!(file_names.contains(&"file2.rs"));
  }

  #[tokio::test]
  async fn test_pipeline_error_handling() {
    let (_temp_dir, config) = Config::test();
    // Create embedding provider for tests
    let (embedding_provider, embedding_dim) = create_test_embedding_provider().await;

    let temp_db = tempdir().unwrap();
    let connection = lancedb::connect(temp_db.path().to_str().unwrap())
      .execute()
      .await
      .unwrap();
    let table = CodeDocument::ensure_table(&connection, "test", embedding_dim)
      .await
      .unwrap();
    let chunk_table =
      crate::models::CodeChunk::ensure_table(&connection, "test_chunks", embedding_dim)
        .await
        .unwrap();

    let indexer = BulkIndexer::new(
      Arc::new(config),
      embedding_provider,
      embedding_dim,
      Arc::new(RwLock::new(table)),
      Arc::new(RwLock::new(chunk_table)),
    );

    // Stream with an error in the middle
    let chunks = vec![
      Ok(create_test_chunk("file1.rs", "valid", false)),
      Ok(create_test_chunk("file1.rs", "", true)),
      Err(ChunkError::ParseError("simulated error".to_string())),
      Ok(create_test_chunk("file2.rs", "still valid", false)),
      Ok(create_test_chunk("file2.rs", "", true)),
    ];

    let chunk_stream = stream::iter(chunks);
    let (count, failures) = indexer
      .index_stream(Uuid::now_v7(), chunk_stream, 10, None)
      .await
      .unwrap();

    // Should still process valid files despite error
    assert_eq!(count, 2, "Should have indexed 2 documents despite error");
    assert!(failures.is_none(), "Should have no embedding failures");
  }

  #[tokio::test]
  async fn test_pipeline_cancellation() {
    // Create a stream that sends a few chunks then blocks forever
    let (tx, rx) = tokio::sync::mpsc::unbounded_channel();

    // Send a few chunks
    tx.send(Ok(create_test_chunk("file1.rs", "content", false)))
      .unwrap();
    tx.send(Ok(create_test_chunk("file1.rs", "", true)))
      .unwrap();
    // Don't close the channel - it will block forever waiting for more

    let chunk_stream = tokio_stream::wrappers::UnboundedReceiverStream::new(rx);

    // Start indexing in a task
    let index_handle = tokio::spawn(async move {
      let (_temp_dir, config) = Config::test();

      // Create embedding provider for tests
      let (embedding_provider, embedding_dim) = create_test_embedding_provider().await;

      let temp_db = tempdir().unwrap();
      let connection = lancedb::connect(temp_db.path().to_str().unwrap())
        .execute()
        .await
        .unwrap();
      let table = CodeDocument::ensure_table(&connection, "test", embedding_dim)
        .await
        .unwrap();
      let chunk_table =
        crate::models::CodeChunk::ensure_table(&connection, "test_chunks", embedding_dim)
          .await
          .unwrap();

      let indexer = BulkIndexer::new(
        Arc::new(config),
        embedding_provider,
        embedding_dim,
        Arc::new(RwLock::new(table)),
        Arc::new(RwLock::new(chunk_table)),
      );
      indexer
        .index_stream(Uuid::now_v7(), chunk_stream, 5, None)
        .await
    });

    // Give it time to process the first chunks
    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;

    // Cancel the task
    index_handle.abort();

    // Should be cancelled
    assert!(index_handle.await.is_err());
  }

  #[tokio::test]
  async fn test_pipeline_batch_accumulation() {
    let (_temp_dir, config) = Config::test();
    // Create embedding provider for tests
    let (embedding_provider, embedding_dim) = create_test_embedding_provider().await;

    let temp_db = tempdir().unwrap();
    let connection = lancedb::connect(temp_db.path().to_str().unwrap())
      .execute()
      .await
      .unwrap();
    let table = CodeDocument::ensure_table(&connection, "test", embedding_dim)
      .await
      .unwrap();
    let chunk_table =
      crate::models::CodeChunk::ensure_table(&connection, "test_chunks", embedding_dim)
        .await
        .unwrap();

    let indexer = BulkIndexer::new(
      Arc::new(config),
      embedding_provider,
      embedding_dim,
      Arc::new(RwLock::new(table)),
      Arc::new(RwLock::new(chunk_table)),
    );

    // Create many chunks for one file to test batching
    let mut chunks = vec![];
    for i in 0..10 {
      chunks.push(Ok(create_test_chunk(
        "bigfile.rs",
        &format!("chunk {}", i),
        false,
      )));
    }
    chunks.push(Ok(create_test_chunk("bigfile.rs", "", true)));

    let chunk_stream = stream::iter(chunks);
    let (count, failures) = indexer
      .index_stream(Uuid::now_v7(), chunk_stream, 3, None)
      .await
      .unwrap(); // batch size 3

    assert_eq!(count, 1, "Should have indexed 1 document with many chunks");
    assert!(failures.is_none(), "Should have no failures");
  }

  #[tokio::test]
  async fn test_pipeline_empty_stream() {
    let (_temp_dir, config) = Config::test();
    // Create embedding provider for tests
    let (embedding_provider, embedding_dim) = create_test_embedding_provider().await;

    let temp_db = tempdir().unwrap();
    let connection = lancedb::connect(temp_db.path().to_str().unwrap())
      .execute()
      .await
      .unwrap();
    let table = CodeDocument::ensure_table(&connection, "test", embedding_dim)
      .await
      .unwrap();
    let chunk_table =
      crate::models::CodeChunk::ensure_table(&connection, "test_chunks", embedding_dim)
        .await
        .unwrap();

    let indexer = BulkIndexer::new(
      Arc::new(config),
      embedding_provider,
      embedding_dim,
      Arc::new(RwLock::new(table)),
      Arc::new(RwLock::new(chunk_table)),
    );

    let chunk_stream = stream::iter(vec![] as Vec<Result<ProjectChunk, ChunkError>>);
    let (count, failures) = indexer
      .index_stream(Uuid::now_v7(), chunk_stream, 10, None)
      .await
      .unwrap();

    assert_eq!(count, 0, "Should handle empty stream gracefully");
    assert!(failures.is_none(), "Should have no failures");
  }

  #[tokio::test]
  async fn test_pipeline_eof_only_files() {
    let (_temp_dir, config) = Config::test();
    // Create embedding provider for tests
    let (embedding_provider, embedding_dim) = create_test_embedding_provider().await;

    let temp_db = tempdir().unwrap();
    let connection = lancedb::connect(temp_db.path().to_str().unwrap())
      .execute()
      .await
      .unwrap();
    let table = CodeDocument::ensure_table(&connection, "test", embedding_dim)
      .await
      .unwrap();
    let chunk_table =
      crate::models::CodeChunk::ensure_table(&connection, "test_chunks", embedding_dim)
        .await
        .unwrap();

    let indexer = BulkIndexer::new(
      Arc::new(config),
      embedding_provider,
      embedding_dim,
      Arc::new(RwLock::new(table)),
      Arc::new(RwLock::new(chunk_table)),
    );

    // Files with only EOF markers (empty files)
    let chunks = vec![
      Ok(create_test_chunk("empty1.rs", "", true)),
      Ok(create_test_chunk("empty2.rs", "", true)),
    ];

    let chunk_stream = stream::iter(chunks);
    let (count, failures) = indexer
      .index_stream(Uuid::now_v7(), chunk_stream, 10, None)
      .await
      .unwrap();

    // Empty files should not create documents
    assert_eq!(count, 0, "Should not index empty files");
    assert!(failures.is_none(), "Should have no failures");
  }

  #[tokio::test]
  async fn test_pipeline_single_file_multiple_chunks() {
    let (_temp_dir, config) = Config::test();
    // Create embedding provider for tests
    let (embedding_provider, embedding_dim) = create_test_embedding_provider().await;

    let temp_db = tempdir().unwrap();
    let connection = lancedb::connect(temp_db.path().to_str().unwrap())
      .execute()
      .await
      .unwrap();
    let table = CodeDocument::ensure_table(&connection, "test", embedding_dim)
      .await
      .unwrap();
    let chunk_table =
      crate::models::CodeChunk::ensure_table(&connection, "test_chunks", embedding_dim)
        .await
        .unwrap();

    let indexer = BulkIndexer::new(
      Arc::new(config),
      embedding_provider,
      embedding_dim,
      Arc::new(RwLock::new(table)),
      Arc::new(RwLock::new(chunk_table)),
    );

    // Single file with multiple chunks
    let chunks = vec![
      Ok(create_test_chunk("single.rs", "fn one() {}", false)),
      Ok(create_test_chunk("single.rs", "fn two() {}", false)),
      Ok(create_test_chunk("single.rs", "fn three() {}", false)),
      Ok(create_test_chunk("single.rs", "", true)), // EOF
    ];

    let chunk_stream = stream::iter(chunks);
    let (count, failures) = indexer
      .index_stream(Uuid::now_v7(), chunk_stream, 2, None)
      .await
      .unwrap(); // batch size 2

    assert_eq!(count, 1, "Should have indexed 1 document with 3 chunks");
    assert!(failures.is_none(), "Should have no failures");

    // Verify chunks were stored (excluding dummy chunk)
    let chunk_table = connection
      .open_table("test_chunks")
      .execute()
      .await
      .unwrap();
    let mut chunk_stream = chunk_table
      .query()
      .only_if(format!("id != '{}'", Uuid::nil()))
      .execute()
      .await
      .unwrap();
    let mut chunk_results = Vec::new();
    while let Some(batch) = chunk_stream.next().await {
      chunk_results.push(batch.unwrap());
    }

    // Count non-dummy chunks
    let chunk_count = chunk_table
      .count_rows(Some(format!("id != '{}'", Uuid::nil())))
      .await
      .unwrap();
    assert_eq!(chunk_count, 3, "Should have stored 3 chunks");

    // Verify chunk content
    for batch in &chunk_results {
      if batch.num_rows() > 0 {
        let file_path_col = batch
          .column_by_name("file_path")
          .unwrap()
          .as_any()
          .downcast_ref::<arrow::array::StringArray>()
          .unwrap();

        for i in 0..batch.num_rows() {
          let file_path = file_path_col.value(i);
          assert_eq!(
            file_path, "single.rs",
            "All chunks should be from single.rs"
          );
        }
      }
    }
  }

  #[tokio::test]
  async fn test_pipeline_document_batching() {
    let _ = tracing_subscriber::fmt()
      .with_env_filter("breeze=debug")
      .try_init();

    let (_temp_dir, config) = Config::test();
    // Create embedding provider for tests
    let (embedding_provider, embedding_dim) = create_test_embedding_provider().await;

    let temp_db = tempdir().unwrap();
    let connection = lancedb::connect(temp_db.path().to_str().unwrap())
      .execute()
      .await
      .unwrap();
    let table = CodeDocument::ensure_table(&connection, "test", embedding_dim)
      .await
      .unwrap();
    let chunk_table =
      crate::models::CodeChunk::ensure_table(&connection, "test_chunks", embedding_dim)
        .await
        .unwrap();

    let indexer = BulkIndexer::new(
      Arc::new(config),
      embedding_provider,
      embedding_dim,
      Arc::new(RwLock::new(table)),
      Arc::new(RwLock::new(chunk_table)),
    );

    // Create 101 files to test the 100 document batch limit
    // Use unique file names to avoid conflicts with concurrent tests
    let test_id = std::process::id();
    let mut chunks = Vec::new();
    for i in 0..101 {
      let file_name = format!("test_batch_file{}_{}.rs", i, test_id);
      chunks.push(Ok(create_test_chunk(
        &file_name,
        &format!("fn file{}() {{}}", i),
        false,
      )));
      chunks.push(Ok(create_test_chunk(&file_name, "", true))); // EOF
    }

    let chunk_stream = stream::iter(chunks);
    let (count, failures) = indexer
      .index_stream(Uuid::now_v7(), chunk_stream, 10, None)
      .await
      .unwrap();

    assert_eq!(count, 101, "Should have indexed 101 documents");
    assert!(failures.is_none(), "Should have no failures");

    // Verify chunks were stored (101 files × 1 chunk each, excluding dummy)
    let chunk_table = connection
      .open_table("test_chunks")
      .execute()
      .await
      .unwrap();
    let chunk_count = chunk_table
      .count_rows(Some(format!("id != '{}'", Uuid::nil())))
      .await
      .unwrap();
    assert_eq!(chunk_count, 101, "Should have stored 101 chunks");
  }

  // Integration test with real embedder (keep existing test)
  #[tokio::test]
  async fn test_indexer_integration() {
    let _ = tracing_subscriber::fmt()
      .with_env_filter("breeze=debug,breeze_chunkers=debug")
      .try_init();

    let (_temp_dir, config) = Config::test();

    info!("Loading embedder model: {}", config.model);
    // Create embedding provider for tests
    let (embedding_provider, embedding_dim) = create_test_embedding_provider().await;
    info!("Embedder loaded, embedding dimension: {}", embedding_dim);

    let temp_db = tempdir().unwrap();
    let connection = lancedb::connect(temp_db.path().to_str().unwrap())
      .execute()
      .await
      .unwrap();

    let table = CodeDocument::ensure_table(&connection, "test_table", embedding_dim)
      .await
      .unwrap();
    let chunk_table =
      crate::models::CodeChunk::ensure_table(&connection, "test_chunks", embedding_dim)
        .await
        .unwrap();

    let indexer = BulkIndexer::new(
      Arc::new(config),
      embedding_provider,
      embedding_dim,
      Arc::new(RwLock::new(table)),
      Arc::new(RwLock::new(chunk_table)),
    );

    let test_dir = tempdir().unwrap();
    let test_file = test_dir.path().join("test.py");
    let test_content = r#"
def hello():
    """Say hello to the world."""
    print('Hello, world!')

def goodbye():
    """Say goodbye."""
    print('Goodbye!')
"#;
    std::fs::write(&test_file, test_content).unwrap();
    info!("Created test file: {}", test_file.display());

    info!("Starting indexing...");
    let (count, failures) = indexer
      .index(Uuid::now_v7(), test_dir.path(), None)
      .await
      .unwrap();
    info!("Indexing completed, document count: {}", count);
    assert_eq!(count, 1);
    assert!(failures.is_none(), "Should have no failures");
  }

  #[tokio::test]
  async fn test_eof_chunk_loss_on_embedding_failure() {
    let _ = tracing_subscriber::fmt()
      .with_env_filter("breeze=debug")
      .try_init();

    // Mock embedding provider that fails on specific batches
    struct FailingEmbeddingProvider {
      fail_batch_count: std::sync::atomic::AtomicUsize,
      fail_every_n: usize,
      embedding_dim: usize,
    }

    impl FailingEmbeddingProvider {
      fn new(fail_every_n: usize, embedding_dim: usize) -> Self {
        Self {
          fail_batch_count: std::sync::atomic::AtomicUsize::new(0),
          fail_every_n,
          embedding_dim,
        }
      }
    }

    #[async_trait::async_trait]
    impl EmbeddingProvider for FailingEmbeddingProvider {
      async fn embed(
        &self,
        inputs: &[crate::embeddings::EmbeddingInput<'_>],
      ) -> crate::embeddings::EmbeddingResult<Vec<Vec<f32>>> {
        let batch_num = self
          .fail_batch_count
          .fetch_add(1, std::sync::atomic::Ordering::SeqCst);

        // Fail every nth batch
        if (batch_num + 1) % self.fail_every_n == 0 {
          Err(crate::embeddings::EmbeddingError::EmbeddingFailed(
            "Mock embedding failure".to_string(),
          ))
        } else {
          // Return dummy embeddings
          Ok(
            inputs
              .iter()
              .map(|_| vec![0.0; self.embedding_dim])
              .collect(),
          )
        }
      }

      fn embedding_dim(&self) -> usize {
        self.embedding_dim
      }

      fn context_length(&self) -> usize {
        8192
      }

      fn create_batching_strategy(&self) -> Box<dyn crate::embeddings::batching::BatchingStrategy> {
        Box::new(crate::embeddings::batching::LocalBatchingStrategy::new(3)) // Small batch size
      }

      fn tokenizer(&self) -> Option<Arc<tokenizers::Tokenizer>> {
        None
      }
    }

    // Create test setup
    let embedding_dim = 384;
    let cancel_token = CancellationToken::new();
    let stats = IndexingStats::new();

    // Set up channels
    let (batch_tx, batch_rx) = mpsc::channel(100);
    let (embedded_tx, mut embedded_rx) = mpsc::channel(100);

    // Create test data with multiple files
    let test_files = vec![
      ("file1.rs", vec!["chunk1_1", "chunk1_2"], "file1 content"),
      (
        "file2.rs",
        vec!["chunk2_1", "chunk2_2", "chunk2_3"],
        "file2 content",
      ),
      ("file3.rs", vec!["chunk3_1"], "file3 content"),
      ("file4.rs", vec!["chunk4_1", "chunk4_2"], "file4 content"),
      (
        "file5.rs",
        vec!["chunk5_1", "chunk5_2", "chunk5_3", "chunk5_4"],
        "file5 content",
      ),
    ];

    // Create chunks
    let mut all_chunks = Vec::new();

    for (file_path, chunks, _) in &test_files {
      for chunk_text in chunks {
        let chunk = PipelineProjectChunk {
          file_path: file_path.to_string(),
          chunk: PipelineChunk::Text(SemanticChunk {
            text: chunk_text.to_string(),
            start_byte: 0,
            end_byte: chunk_text.len(),
            start_line: 1,
            end_line: 1,
            tokens: Some(vec![1234; 5]), // Mock tokens
            metadata: ChunkMetadata {
              node_type: "test".to_string(),
              node_name: None,
              language: "rust".to_string(),
              parent_context: None,
              scope_path: vec![],
              definitions: vec![],
              references: vec![],
            },
          }),
        };
        all_chunks.push(chunk);
      }
    }

    // Add EOF chunks
    for (file_path, _, content) in &test_files {
      let hash = blake3::hash(content.as_bytes());
      let mut content_hash = [0u8; 32];
      content_hash.copy_from_slice(hash.as_bytes());

      let eof = PipelineProjectChunk {
        file_path: file_path.to_string(),
        chunk: PipelineChunk::EndOfFile {
          file_path: file_path.to_string(),
          content: content.to_string(),
          content_hash,
        },
      };
      all_chunks.push(eof);
    }

    // Create batches manually (simulating what stream_processor would do)
    let batch_size = 3;
    for (batch_id, chunk_batch) in all_chunks.chunks(batch_size).enumerate() {
      let batch = ChunkBatch {
        batch_id,
        chunks: chunk_batch.to_vec(),
      };
      batch_tx.send(batch).await.unwrap();
    }

    drop(batch_tx); // Close the channel

    // Create failing embedding provider
    let embedding_provider = Arc::new(FailingEmbeddingProvider::new(3, embedding_dim)); // Fail every 3rd batch
    let batching_strategy = embedding_provider.create_batching_strategy();

    // Start embedder task
    let embedder_handle = tokio::spawn({
      let embedded_tx = embedded_tx.clone();
      let stats = stats.clone();
      let cancel_token = cancel_token.clone();
      let embedding_provider = embedding_provider.clone();

      async move {
        embedder_task(
          embedding_provider,
          batching_strategy,
          batch_rx,
          embedded_tx,
          stats,
          cancel_token,
        )
        .await;
      }
    });

    drop(embedded_tx); // Close the embedded channel after embedder starts

    // Collect results
    let mut file_chunks: std::collections::HashMap<String, Vec<String>> =
      std::collections::HashMap::new();
    let mut eof_received: std::collections::HashMap<String, bool> =
      std::collections::HashMap::new();

    // Initialize tracking maps
    for (file_path, _, _) in &test_files {
      file_chunks.insert(file_path.to_string(), Vec::new());
      eof_received.insert(file_path.to_string(), false);
    }

    while let Some(chunk) = embedded_rx.recv().await {
      match chunk {
        EmbeddedChunkWithFile::Embedded {
          file_path, chunk, ..
        } => match chunk.as_ref() {
          PipelineChunk::Text(sc) | PipelineChunk::Semantic(sc) => {
            file_chunks
              .get_mut(&file_path)
              .unwrap()
              .push(sc.text.clone());
          }
          _ => {}
        },
        EmbeddedChunkWithFile::EndOfFile { file_path, .. } => {
          eof_received.insert(file_path, true);
        }
        EmbeddedChunkWithFile::BatchFailure { .. } => {
          // Ignore for this test
        }
      }
    }

    embedder_handle.await.unwrap();

    // Check results
    info!("File chunks received:");
    for (file, chunks) in &file_chunks {
      info!("  {}: {} chunks", file, chunks.len());
    }

    info!("EOF chunks received:");
    for (file, received) in &eof_received {
      info!("  {}: {}", file, if *received { "YES" } else { "NO" });
    }

    // Verify that all files received their EOF chunks (because EOF chunks are sent separately)
    let files_missing_eof: Vec<_> = eof_received
      .iter()
      .filter(|(_, received)| !**received)
      .map(|(file, _)| file.clone())
      .collect();

    info!("Files missing EOF: {:?}", files_missing_eof);

    // With the fix, no files should be missing EOF chunks
    assert!(
      files_missing_eof.is_empty(),
      "No files should be missing EOF chunks with the new implementation"
    );

    // Check that some files didn't receive all their content chunks due to batch failure
    let incomplete_files: Vec<_> = test_files
      .iter()
      .filter(|(file_path, expected_chunks, _)| {
        let received = file_chunks.get(*file_path).unwrap();
        received.len() < expected_chunks.len()
      })
      .map(|(file_path, _, _)| file_path.to_string())
      .collect();

    info!("Files with incomplete chunks: {:?}", incomplete_files);
    assert!(
      !incomplete_files.is_empty(),
      "Expected some files to have incomplete chunks due to embedding failures"
    );

    // Verify that file4 specifically lost its chunks (it was in batch 2 which failed)
    assert_eq!(
      file_chunks.get("file4.rs").unwrap().len(),
      0,
      "file4.rs should have lost all its chunks because they were in the failed batch"
    );
  }

  #[tokio::test]
  async fn test_document_builder_handles_batch_failures() {
    let embedding_dim = 384;
    let (embedded_tx, embedded_rx) = mpsc::channel(100);
    let (doc_tx, mut doc_rx) = mpsc::channel(100);
    let (chunk_tx, _chunk_rx) = mpsc::channel(100);
    let stats = IndexingStats::new();
    let cancel_token = CancellationToken::new();
    let project_id = Uuid::now_v7();

    // Spawn document builder task
    let builder_handle = tokio::spawn(async move {
      document_builder_task(
        project_id,
        embedded_rx,
        doc_tx,
        chunk_tx,
        embedding_dim,
        stats,
        cancel_token,
      )
      .await;
    });

    // Send chunks for file1 (successful)
    embedded_tx
      .send(EmbeddedChunkWithFile::Embedded {
        batch_id: 0,
        file_path: "file1.rs".to_string(),
        chunk: Box::new(PipelineChunk::Text(SemanticChunk {
          text: "content1".to_string(),
          tokens: Some(vec![1, 2, 3]),
          start_byte: 0,
          end_byte: 8,
          start_line: 1,
          end_line: 1,
          metadata: ChunkMetadata {
            node_type: "text".to_string(),
            node_name: None,
            language: "rust".to_string(),
            parent_context: None,
            scope_path: vec![],
            definitions: vec![],
            references: vec![],
          },
        })),
        embedding: vec![0.1; embedding_dim],
      })
      .await
      .unwrap();

    // Send EOF for file1
    embedded_tx
      .send(EmbeddedChunkWithFile::EndOfFile {
        batch_id: 0,
        file_path: "file1.rs".to_string(),
        content: "content1".to_string(),
        content_hash: [1u8; 32],
      })
      .await
      .unwrap();

    // Send batch failure notification for batch 1
    let mut failed_files = std::collections::BTreeSet::new();
    failed_files.insert("file2.rs".to_string());
    failed_files.insert("file3.rs".to_string());

    embedded_tx
      .send(EmbeddedChunkWithFile::BatchFailure {
        batch_id: 1,
        failed_files: failed_files.clone(),
        error: "Mock embedding failure".to_string(),
      })
      .await
      .unwrap();

    // Send EOF for file2 (should be ignored because it's in failed_files)
    embedded_tx
      .send(EmbeddedChunkWithFile::EndOfFile {
        batch_id: 1,
        file_path: "file2.rs".to_string(),
        content: "content2".to_string(),
        content_hash: [2u8; 32],
      })
      .await
      .unwrap();

    // Send chunks for file3 (should be ignored because it's in failed_files)
    embedded_tx
      .send(EmbeddedChunkWithFile::Embedded {
        batch_id: 1,
        file_path: "file3.rs".to_string(),
        chunk: Box::new(PipelineChunk::Text(SemanticChunk {
          text: "content3".to_string(),
          tokens: Some(vec![4, 5, 6]),
          start_byte: 0,
          end_byte: 8,
          start_line: 1,
          end_line: 1,
          metadata: ChunkMetadata {
            node_type: "text".to_string(),
            node_name: None,
            language: "rust".to_string(),
            parent_context: None,
            scope_path: vec![],
            definitions: vec![],
            references: vec![],
          },
        })),
        embedding: vec![0.3; embedding_dim],
      })
      .await
      .unwrap();

    // Send EOF for file3 (should be ignored)
    embedded_tx
      .send(EmbeddedChunkWithFile::EndOfFile {
        batch_id: 1,
        file_path: "file3.rs".to_string(),
        content: "content3".to_string(),
        content_hash: [3u8; 32],
      })
      .await
      .unwrap();

    // Close channel
    drop(embedded_tx);

    // Wait for builder to finish
    builder_handle.await.unwrap();

    // Collect all documents
    let mut documents_received: Vec<CodeDocument> = Vec::new();
    while let Ok(doc) = doc_rx.try_recv() {
      documents_received.push(doc);
    }

    // Only file1 should have been processed
    assert_eq!(documents_received.len(), 1, "Should have only 1 document");
    assert_eq!(documents_received[0].file_path, "file1.rs");

    // Verify the document has correct content
    assert_eq!(documents_received[0].content, "content1");
  }
}
