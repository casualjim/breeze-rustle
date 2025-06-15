use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use breeze_chunkers::{Chunk, Tokenizer, WalkOptions, walk_project};
use futures_util::StreamExt;
use lancedb::Table;
use tokio::sync::{RwLock, mpsc};
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

use crate::config::Config;
use crate::converter::BufferedRecordBatchConverter;
use crate::document_builder::build_document_from_accumulator;
use crate::embeddings::EmbeddingProvider;
use crate::models::CodeDocument;
use crate::pipeline::{ChunkBatch, EmbeddedChunk, EmbeddedChunkWithFile, FileAccumulator};
use crate::sinks::lancedb_sink::LanceDbSink;

pub struct Indexer<'a> {
  config: &'a Config,
  embedding_provider: Arc<dyn EmbeddingProvider>,
  embedding_dim: usize,
  table: Arc<RwLock<Table>>,
}

impl<'a> Indexer<'a> {
  pub fn new(
    config: &'a Config,
    embedding_provider: Arc<dyn EmbeddingProvider>,
    embedding_dim: usize,
    table: Arc<RwLock<Table>>,
  ) -> Self {
    Self {
      config,
      embedding_provider,
      embedding_dim,
      table,
    }
  }

  pub async fn index(
    &self,
    path: &Path,
  ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let start_time = Instant::now();
    info!(path = %path.display(), "Starting channel-based indexing");

    // Setup
    // Get tokenizer from the embedding provider
    let tokenizer = if let Some(provider_tokenizer) = self.embedding_provider.tokenizer() {
      // Use the pre-loaded tokenizer from the provider
      Tokenizer::PreloadedHuggingFace(provider_tokenizer)
    } else {
      // Default to character-based tokenization for local models
      Tokenizer::Characters
    };

    let walk_options = WalkOptions {
      max_chunk_size: self.config.max_chunk_size,
      tokenizer,
      max_parallel: self.config.max_parallel_files,
      max_file_size: self.config.max_file_size,
    };

    let chunk_stream = walk_project(path.to_path_buf(), walk_options);
    let result = self.index_stream(chunk_stream, 256).await?;

    let elapsed = start_time.elapsed();
    info!(
      elapsed_seconds = elapsed.as_secs_f64(),
      documents_written = result,
      "Indexing completed"
    );

    Ok(result)
  }

  // Testable pipeline that accepts any stream of chunks
  pub async fn index_stream(
    &self,
    chunk_stream: impl futures_util::Stream<
      Item = Result<breeze_chunkers::ProjectChunk, breeze_chunkers::ChunkError>,
    > + Send
    + 'static,
    max_batch_size: usize,
  ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
    let embedding_dim = self.embedding_dim;

    // Create channels with bounded capacity for backpressure
    let (batch_tx, batch_rx) = mpsc::channel::<ChunkBatch>(10);
    let (embedded_tx, embedded_rx) = mpsc::channel::<EmbeddedChunkWithFile>(100);
    let (doc_tx, doc_rx) = mpsc::channel::<CodeDocument>(50);

    // Progress tracking and cancellation
    let stats = IndexingStats::new();
    let cancel_token = CancellationToken::new();

    // Start pipeline stages with cancellation support
    // we start this with in the reverse order of execution
    let sink_handle = self.spawn_sink(doc_rx, stats.clone(), embedding_dim);
    let doc_handle = self.spawn_document_builder(
      embedded_rx,
      doc_tx,
      embedding_dim,
      stats.clone(),
      cancel_token.clone(),
    );
    let embed_handle =
      self.spawn_embedder(batch_rx, embedded_tx, stats.clone(), cancel_token.clone());
    let walk_handle = self.spawn_stream_processor(
      chunk_stream,
      batch_tx,
      max_batch_size,
      stats.clone(),
      cancel_token.clone(),
    );

    // Wait for pipeline completion with proper error handling
    let (walk_result, embed_result, doc_result, sink_result) =
      tokio::join!(walk_handle, embed_handle, doc_handle, sink_handle);

    // Check all results and report errors, cancelling other tasks on first error
    if let Err(e) = walk_result {
      cancel_token.cancel();
      return Err(format!("Stream processor task failed: {}", e).into());
    }
    if let Err(e) = embed_result {
      cancel_token.cancel();
      return Err(format!("Embedder task failed: {}", e).into());
    }
    if let Err(e) = doc_result {
      cancel_token.cancel();
      return Err(format!("Document builder task failed: {}", e).into());
    }

    let documents_written = sink_result
      .map_err(|e| format!("Sink task panicked: {}", e))?
      .map_err(|e| format!("Sink task failed: {}", e))?;

    // Report results
    info!(
      files = stats.files.load(Ordering::Relaxed),
      chunks = stats.chunks.load(Ordering::Relaxed),
      batches = stats.batches.load(Ordering::Relaxed),
      documents = stats.documents.load(Ordering::Relaxed),
      documents_written,
      "Indexing completed"
    );

    Ok(documents_written)
  }

  fn spawn_stream_processor(
    &self,
    chunk_stream: impl futures_util::Stream<
      Item = Result<breeze_chunkers::ProjectChunk, breeze_chunkers::ChunkError>,
    > + Send
    + 'static,
    batch_tx: mpsc::Sender<ChunkBatch>,
    max_batch_size: usize,
    stats: IndexingStats,
    cancel_token: CancellationToken,
  ) -> tokio::task::JoinHandle<()> {
    tokio::spawn(stream_processor_task(
      chunk_stream,
      batch_tx,
      max_batch_size,
      stats,
      cancel_token,
    ))
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

  fn spawn_document_builder(
    &self,
    embedded_rx: mpsc::Receiver<EmbeddedChunkWithFile>,
    doc_tx: mpsc::Sender<CodeDocument>,
    embedding_dim: usize,
    stats: IndexingStats,
    cancel_token: CancellationToken,
  ) -> tokio::task::JoinHandle<()> {
    tokio::spawn(document_builder_task(
      embedded_rx,
      doc_tx,
      embedding_dim,
      stats,
      cancel_token,
    ))
  }

  fn spawn_sink(
    &self,
    doc_rx: mpsc::Receiver<CodeDocument>,
    stats: IndexingStats,
    embedding_dim: usize,
  ) -> tokio::task::JoinHandle<Result<usize, Box<dyn std::error::Error + Send + Sync>>> {
    let table = self.table.clone();
    tokio::spawn(sink_task(doc_rx, table, embedding_dim, stats))
  }
}

#[derive(Clone)]
struct IndexingStats {
  files: Arc<AtomicUsize>,
  chunks: Arc<AtomicUsize>,
  batches: Arc<AtomicUsize>,
  documents: Arc<AtomicUsize>,
}

impl IndexingStats {
  fn new() -> Self {
    Self {
      files: Arc::new(AtomicUsize::new(0)),
      chunks: Arc::new(AtomicUsize::new(0)),
      batches: Arc::new(AtomicUsize::new(0)),
      documents: Arc::new(AtomicUsize::new(0)),
    }
  }
}

async fn send_batch(
  tx: &mpsc::Sender<ChunkBatch>,
  buffer: &mut Vec<breeze_chunkers::ProjectChunk>,
) {
  if !buffer.is_empty() {
    debug!("Sending batch with {} chunks", buffer.len());
    let batch = ChunkBatch {
      chunks: buffer.clone(),
    };
    if tx.send(batch).await.is_err() {
      debug!("Receiver dropped, stopping batch sender");
    }
    buffer.clear();
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
  batch_tx: mpsc::Sender<ChunkBatch>,
  max_batch_size: usize,
  stats: IndexingStats,
  cancel_token: CancellationToken,
) {
  let _guard = TaskGuard::new("Stream processor");
  let mut chunk_stream = Box::pin(chunk_stream);
  let mut batch_buffer = Vec::new();
  let mut regular_chunk_count = 0;

  loop {
    tokio::select! {
      _ = cancel_token.cancelled() => {
        info!("Walker cancelled");
        break;
      }
      result = chunk_stream.next() => {
        match result {
          Some(Ok(project_chunk)) => {
            // Count files when we see EOF markers
            let is_eof = matches!(project_chunk.chunk, Chunk::EndOfFile { .. });
            if is_eof {
              stats.files.fetch_add(1, Ordering::Relaxed);
            } else {
              stats.chunks.fetch_add(1, Ordering::Relaxed);
              regular_chunk_count += 1;
            }

            // Add all chunks (including EOF) to batch buffer
            batch_buffer.push(project_chunk);

            // Send batch when we have enough REGULAR chunks (not counting EOFs)
            if regular_chunk_count >= max_batch_size {
              send_batch(&batch_tx, &mut batch_buffer).await;
              regular_chunk_count = 0;
            }
          }
          Some(Err(e)) => error!("Error processing chunk: {}", e),
          None => break,
        }
      }
    }
  }

  // Send remaining chunks
  if !batch_buffer.is_empty() {
    send_batch(&batch_tx, &mut batch_buffer).await;
  }

  info!(
    files = stats.files.load(Ordering::Relaxed),
    chunks = stats.chunks.load(Ordering::Relaxed),
    "Stream processor completed"
  );
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

  // Buffer for accumulating chunks to process
  let mut pending_chunks: Vec<breeze_chunkers::ProjectChunk> = Vec::new();
  // Buffer for EOF chunks to send after processing their associated content
  let mut pending_eofs: Vec<breeze_chunkers::ProjectChunk> = Vec::new();

  // Running totals for logging
  let mut total_chunks_processed = 0usize;
  let mut total_files_processed = 0usize;

  loop {
    tokio::select! {
      _ = cancel_token.cancelled() => {
        info!("Embedder cancelled");
        break;
      }
      batch = batch_rx.recv() => {
        match batch {
          Some(batch) => {
            // First, process any pending EOF chunks from previous iteration
            for pc in pending_eofs.drain(..) {
              if let Chunk::EndOfFile { .. } = pc.chunk {
                total_files_processed += 1;
                let eof = EmbeddedChunkWithFile {
                  file_path: pc.file_path.clone(),
                  chunk: pc.chunk,
                  embedding: vec![],
                };
                if embedded_tx.send(eof).await.is_err() {
                  return; // Channel closed, safe to exit
                }
              }
            }

            // Separate EOF markers from regular chunks
            let (eof_chunks, regular_chunks): (Vec<_>, Vec<_>) = batch.chunks.into_iter()
              .partition(|pc| matches!(pc.chunk, Chunk::EndOfFile { .. }));

            // Add regular chunks to pending
            pending_chunks.extend(regular_chunks);

            // Store EOF chunks for next iteration
            pending_eofs.extend(eof_chunks);

            // Use batching strategy to prepare chunks
            if pending_chunks.len() >= batching_strategy.max_batch_size() ||
               (batching_strategy.is_token_aware() && should_process_token_batch(&pending_chunks)) {
              // Get chunks to process
              let chunks_to_batch = pending_chunks.clone();
              pending_chunks.clear();

              // Let the batching strategy prepare the batches
              let embedding_batches = batching_strategy.prepare_batches(
                chunks_to_batch
              ).await;

              // Process each batch
              for embedding_batch in embedding_batches {
                total_chunks_processed += embedding_batch.chunks.len();
                process_embedding_batch(
                  embedding_provider.as_ref(),
                  embedding_batch.chunks,
                  &embedded_tx,
                  &stats,
                  total_chunks_processed,
                  total_files_processed
                ).await;
              }
            }
          }
          None => {
            // Channel closed, process remaining chunks
            if !pending_chunks.is_empty() {
              let embedding_batches = batching_strategy.prepare_batches(
                pending_chunks
              ).await;

              for embedding_batch in embedding_batches {
                total_chunks_processed += embedding_batch.chunks.len();
                process_embedding_batch(
                  embedding_provider.as_ref(),
                  embedding_batch.chunks,
                  &embedded_tx,
                  &stats,
                  total_chunks_processed,
                  total_files_processed
                ).await;
              }
            }

            // Send any remaining EOF chunks
            for pc in pending_eofs {
              if let Chunk::EndOfFile { .. } = pc.chunk {
                total_files_processed += 1;
                let eof = EmbeddedChunkWithFile {
                  file_path: pc.file_path.clone(),
                  chunk: pc.chunk,
                  embedding: vec![],
                };
                let _ = embedded_tx.send(eof).await;
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

// Helper function to check if we should process based on token count
fn should_process_token_batch(chunks: &[breeze_chunkers::ProjectChunk]) -> bool {
  // For now, simple heuristic - process if we have a reasonable amount of chunks
  // The actual token counting should come from the walker
  chunks.len() >= 10
}

async fn process_embedding_batch(
  embedding_provider: &dyn EmbeddingProvider,
  batch: Vec<breeze_chunkers::ProjectChunk>,
  embedded_tx: &mpsc::Sender<EmbeddedChunkWithFile>,
  stats: &IndexingStats,
  total_chunks_processed: usize,
  total_files_processed: usize,
) {
  if batch.is_empty() {
    return;
  }

  debug!(
    "Processing embedding batch of {} chunks (total chunks: {}, files: {})",
    batch.len(),
    total_chunks_processed,
    total_files_processed
  );
  // Create embedding inputs from chunks (without cloning text)
  let mut chunks_to_embed = Vec::new();
  let inputs: Vec<crate::embeddings::EmbeddingInput> = batch
    .iter()
    .enumerate()
    .filter_map(|(idx, pc)| match &pc.chunk {
      Chunk::Semantic(sc) | Chunk::Text(sc) => {
        let input = crate::embeddings::EmbeddingInput {
          text: &sc.text,
          token_count: sc.tokens.as_ref().map(|t| t.len()),
        };
        chunks_to_embed.push(idx);
        Some(input)
      }
      Chunk::EndOfFile { .. } => None,
    })
    .collect();

  if inputs.is_empty() {
    return;
  }

  match embedding_provider.embed(&inputs).await {
    Ok(embeddings) => {
      debug!("Successfully embedded {} chunks", embeddings.len());
      stats.batches.fetch_add(1, Ordering::Relaxed);

      let mut embedding_iter = embeddings.into_iter();
      let mut chunk_idx_iter = chunks_to_embed.into_iter();

      // Consume the batch to avoid cloning
      for (idx, pc) in batch.into_iter().enumerate() {
        if chunk_idx_iter.next() == Some(idx) {
          if let Some(embedding_vec) = embedding_iter.next() {
            let item = EmbeddedChunkWithFile {
              file_path: pc.file_path,
              chunk: pc.chunk,
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
      // Continue processing other batches
    }
  }
}

async fn document_builder_task(
  mut embedded_rx: mpsc::Receiver<EmbeddedChunkWithFile>,
  doc_tx: mpsc::Sender<CodeDocument>,
  embedding_dim: usize,
  stats: IndexingStats,
  cancel_token: CancellationToken,
) {
  let _guard = TaskGuard::new("Document builder");
  let mut file_accumulators: std::collections::HashMap<String, FileAccumulator> =
    std::collections::HashMap::new();

  // Buffer for batching documents before sending
  let mut document_batch: Vec<CodeDocument> = Vec::with_capacity(100);
  let mut total_files_built = 0u64;

  loop {
    tokio::select! {
      _ = cancel_token.cancelled() => {
        info!("Document builder cancelled");
        break;
      }
      embedded_chunk = embedded_rx.recv() => {
        match embedded_chunk {
          Some(embedded_chunk) => {
            let file_path = embedded_chunk.file_path.clone();

            if matches!(embedded_chunk.chunk, Chunk::EndOfFile { .. }) {
              // Build document for completed file
              if let Some(mut accumulator) = file_accumulators.remove(&file_path) {
                total_files_built += 1;
                debug!(file_path, total_files_built, "Building document for file: {file_path}");
                // Add the EOF chunk to the accumulator
                accumulator.add_chunk(EmbeddedChunk {
                  chunk: embedded_chunk.chunk,
                  embedding: embedded_chunk.embedding,
                });
                if let Some(doc) = build_document_from_accumulator(accumulator, embedding_dim).await {
                  stats.documents.fetch_add(1, Ordering::Relaxed);
                  document_batch.push(doc);

                  // Send batch when we reach 100 documents
                  if document_batch.len() >= 100 {
                    debug!(total_files_built, batch_len=document_batch.len(), "Sending batch of {} documents", document_batch.len());
                    for doc in document_batch.drain(..) {
                      if doc_tx.send(doc).await.is_err() {
                        debug!("Document receiver dropped");
                        return;
                      }
                    }
                  }
                }
              }
            } else {
              // Accumulate chunk
              let accumulator = file_accumulators.entry(file_path)
                .or_insert_with(|| FileAccumulator::new(embedded_chunk.file_path.clone()));

              accumulator.add_chunk(EmbeddedChunk {
                chunk: embedded_chunk.chunk,
                embedding: embedded_chunk.embedding,
              });
            }
          }
          None => break,
        }
      }
    }
  }

  // Process remaining files
  for (_, accumulator) in file_accumulators {
    if let Some(doc) = build_document_from_accumulator(accumulator, embedding_dim).await {
      stats.documents.fetch_add(1, Ordering::Relaxed);
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

  info!(
    documents = stats.documents.load(Ordering::Relaxed),
    "Document builder completed"
  );
}

async fn sink_task(
  doc_rx: mpsc::Receiver<CodeDocument>,
  table: Arc<RwLock<Table>>,
  embedding_dim: usize,
  _stats: IndexingStats,
) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
  let _guard = TaskGuard::new("Sink");
  let converter = BufferedRecordBatchConverter::<CodeDocument>::default()
    .with_schema(Arc::new(CodeDocument::schema(embedding_dim)));

  let sink = LanceDbSink::new(table.clone());

  let doc_stream = tokio_stream::wrappers::ReceiverStream::new(doc_rx);
  let record_batches = converter.convert(Box::pin(doc_stream));

  let mut sink_stream = sink.sink(record_batches);
  let mut batch_count = 0;

  while let Some(()) = sink_stream.next().await {
    batch_count += 1;
    debug!(batch_number = batch_count, "Written batch to LanceDB");
  }

  // Get the final row count from the table
  let table_guard = table.read().await;
  let count = table_guard.count_rows(None).await? as usize;
  debug!(row_count = count, "Final table row count");

  Ok(count)
}

#[cfg(test)]
mod tests {
  use super::*;
  use breeze_chunkers::{ChunkError, ChunkMetadata, ProjectChunk, SemanticChunk};
  use futures_util::stream;
  use tempfile::tempdir;

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

  // Helper function to create test embedding provider
  async fn create_test_embedding_provider() -> (Arc<dyn EmbeddingProvider>, usize) {
    use crate::embeddings::local::LocalEmbeddingProvider;

    let provider = LocalEmbeddingProvider::new(
      "test-model".to_string(),
      2, // small batch size for tests
    )
    .await
    .unwrap();

    let dim = provider.embedding_dim();
    (Arc::new(provider), dim)
  }

  #[tokio::test]
  async fn test_pipeline_basic_flow() {
    let _ = tracing_subscriber::fmt()
      .with_env_filter("breeze=debug")
      .try_init();

    let config = Config::test();

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

    let indexer = Indexer::new(
      &config,
      embedding_provider,
      embedding_dim,
      Arc::new(RwLock::new(table)),
    );

    // Create test stream with 2 files
    let chunks = vec![
      Ok(create_test_chunk("file1.rs", "fn main() {}", false)),
      Ok(create_test_chunk("file1.rs", "println!(\"Hello\");", false)),
      Ok(create_test_chunk("file1.rs", "", true)), // EOF
      Ok(create_test_chunk("file2.rs", "struct Foo;", false)),
      Ok(create_test_chunk("file2.rs", "", true)), // EOF
    ];

    let chunk_stream = stream::iter(chunks);

    let count = indexer.index_stream(chunk_stream, 2).await.unwrap();

    assert_eq!(count, 2, "Should have indexed 2 documents");
  }

  #[tokio::test]
  async fn test_pipeline_error_handling() {
    let config = Config::test();
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

    let indexer = Indexer::new(
      &config,
      embedding_provider,
      embedding_dim,
      Arc::new(RwLock::new(table)),
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
    let count = indexer.index_stream(chunk_stream, 10).await.unwrap();

    // Should still process valid files despite error
    assert_eq!(count, 2, "Should have indexed 2 documents despite error");
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
      let config = Config::test();

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

      let indexer = Indexer::new(
        &config,
        embedding_provider,
        embedding_dim,
        Arc::new(RwLock::new(table)),
      );
      indexer.index_stream(chunk_stream, 5).await
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
    let config = Config::test();
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

    let indexer = Indexer::new(
      &config,
      embedding_provider,
      embedding_dim,
      Arc::new(RwLock::new(table)),
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
    let count = indexer.index_stream(chunk_stream, 3).await.unwrap(); // batch size 3

    assert_eq!(count, 1, "Should have indexed 1 document with many chunks");
  }

  #[tokio::test]
  async fn test_pipeline_empty_stream() {
    let config = Config::test();
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

    let indexer = Indexer::new(
      &config,
      embedding_provider,
      embedding_dim,
      Arc::new(RwLock::new(table)),
    );

    let chunk_stream = stream::iter(vec![] as Vec<Result<ProjectChunk, ChunkError>>);
    let count = indexer.index_stream(chunk_stream, 10).await.unwrap();

    assert_eq!(count, 0, "Should handle empty stream gracefully");
  }

  #[tokio::test]
  async fn test_pipeline_eof_only_files() {
    let config = Config::test();
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

    let indexer = Indexer::new(
      &config,
      embedding_provider,
      embedding_dim,
      Arc::new(RwLock::new(table)),
    );

    // Files with only EOF markers (empty files)
    let chunks = vec![
      Ok(create_test_chunk("empty1.rs", "", true)),
      Ok(create_test_chunk("empty2.rs", "", true)),
    ];

    let chunk_stream = stream::iter(chunks);
    let count = indexer.index_stream(chunk_stream, 10).await.unwrap();

    // Empty files should not create documents
    assert_eq!(count, 0, "Should not index empty files");
  }

  #[tokio::test]
  async fn test_pipeline_single_file_multiple_chunks() {
    let config = Config::test();
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

    let indexer = Indexer::new(
      &config,
      embedding_provider,
      embedding_dim,
      Arc::new(RwLock::new(table)),
    );

    // Single file with multiple chunks
    let chunks = vec![
      Ok(create_test_chunk("single.rs", "fn one() {}", false)),
      Ok(create_test_chunk("single.rs", "fn two() {}", false)),
      Ok(create_test_chunk("single.rs", "fn three() {}", false)),
      Ok(create_test_chunk("single.rs", "", true)), // EOF
    ];

    let chunk_stream = stream::iter(chunks);
    let count = indexer.index_stream(chunk_stream, 2).await.unwrap(); // batch size 2

    assert_eq!(count, 1, "Should have indexed 1 document with 3 chunks");
  }

  #[tokio::test]
  async fn test_pipeline_document_batching() {
    let _ = tracing_subscriber::fmt()
      .with_env_filter("breeze=debug")
      .try_init();

    let config = Config::test();
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

    let indexer = Indexer::new(
      &config,
      embedding_provider,
      embedding_dim,
      Arc::new(RwLock::new(table)),
    );

    // Create 101 files to test the 100 document batch limit
    let mut chunks = Vec::new();
    for i in 0..101 {
      chunks.push(Ok(create_test_chunk(
        &format!("file{}.rs", i),
        &format!("fn file{}() {{}}", i),
        false,
      )));
      chunks.push(Ok(create_test_chunk(&format!("file{}.rs", i), "", true))); // EOF
    }

    let chunk_stream = stream::iter(chunks);
    let count = indexer.index_stream(chunk_stream, 10).await.unwrap();

    assert_eq!(count, 101, "Should have indexed 101 documents");
  }

  // Integration test with real embedder (keep existing test)
  #[tokio::test]
  async fn test_indexer_integration() {
    let _ = tracing_subscriber::fmt()
      .with_env_filter("breeze=debug,breeze_chunkers=debug")
      .try_init();

    let config = Config::test();

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

    let indexer = Indexer::new(
      &config,
      embedding_provider,
      embedding_dim,
      Arc::new(RwLock::new(table)),
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
    let count = indexer.index(test_dir.path()).await.unwrap();
    info!("Indexing completed, document count: {}", count);
    assert_eq!(count, 1);
  }
}
