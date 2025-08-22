use std::collections::{BTreeMap, BTreeSet, HashMap};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::IndexerError;
use crate::bulk_indexer::IndexingStats;
use crate::models::{ChunkMetadataUpdate, CodeChunk, CodeDocument};
use crate::pipeline::{EmbeddedChunk, EmbeddedChunkWithFile, FileAccumulator, PipelineChunk, ReplaceFileChunks, ReplaceFileChunksSender};

/// Context for document building operations
pub(crate) struct DocumentBuildContext<'a> {
  pub doc_tx: &'a mpsc::Sender<CodeDocument>,
  pub chunks_replace_tx: &'a ReplaceFileChunksSender,
  pub stats: &'a IndexingStats,
  pub cancel_token: &'a CancellationToken,
  pub batch_size: usize,
}

/// Stream-specific document builder that maintains ordering for a single stream
struct StreamDocumentBuilder {
  project_id: Uuid,
  embedding_dim: usize,
  file_accumulators: BTreeMap<String, FileAccumulator>,
  failed_files: BTreeSet<String>,
  total_files_built: u64,
}

impl StreamDocumentBuilder {
  fn new(project_id: Uuid, embedding_dim: usize) -> Self {
    Self {
      project_id,
      embedding_dim,
      file_accumulators: BTreeMap::new(),
      failed_files: BTreeSet::new(),
      total_files_built: 0,
    }
  }

  async fn add_chunk(
    &mut self,
    embedded_chunk: EmbeddedChunkWithFile,
    document_batch: &mut Vec<CodeDocument>,
    ctx: &DocumentBuildContext<'_>,
  ) -> Result<(), IndexerError> {
    // Process chunk immediately without buffering
    match embedded_chunk {
      EmbeddedChunkWithFile::Embedded {
        file_path,
        chunk,
        embedding,
        batch_id,
        ..
      } => {
        debug!(
          "Received embedded chunk for {} (batch {})",
          file_path, batch_id
        );

        // Skip chunks for failed files
        if self.failed_files.contains(&file_path) {
          return Ok(());
        }

        // Get or create accumulator for this file
        let accumulator = self
          .file_accumulators
          .entry(file_path.clone())
          .or_insert_with(|| FileAccumulator::new(file_path.clone()));

        // Add the chunk
        accumulator.add_chunk(EmbeddedChunk {
          chunk: *chunk,
          embedding,
        });

        // Check if file is complete
        if accumulator.is_complete() {
          debug!(
            "File {} is complete with {} chunks",
            accumulator.file_path, accumulator.received_content_chunks
          );
          self
            .try_build_document(file_path, document_batch, ctx)
            .await?;
        }
      }
      EmbeddedChunkWithFile::EndOfFile {
        file_path,
        content,
        content_hash,
        expected_chunks,
        batch_id,
        ..
      } => {
        debug!(
          "Received EOF for {} expecting {} chunks (batch {})",
          file_path, expected_chunks, batch_id
        );

        // Skip EOF for failed files
        if self.failed_files.contains(&file_path) {
          return Ok(());
        }

        // Get or create accumulator
        let accumulator = self
          .file_accumulators
          .entry(file_path.clone())
          .or_insert_with(|| FileAccumulator::new(file_path.clone()));

        // Add the EOF chunk
        accumulator.add_chunk(EmbeddedChunk {
          chunk: PipelineChunk::EndOfFile {
            file_path: file_path.clone(),
            content,
            content_hash,
            expected_chunks,
          },
          embedding: vec![],
        });

        // Check if file is complete
        if accumulator.is_complete() {
          debug!(
            "File {} is complete after EOF with {} chunks",
            accumulator.file_path, accumulator.received_content_chunks
          );
          self
            .try_build_document(file_path, document_batch, ctx)
            .await?;
        } else {
          debug!(
            "File {} not yet complete: {} of {} chunks received",
            file_path, accumulator.received_content_chunks, expected_chunks
          );
        }
      }
      EmbeddedChunkWithFile::BatchFailure {
        failed_files: batch_failed_files,
        error,
        batch_id,
        ..
      } => {
        error!("Batch {} failed: {}", batch_id, error);
        for file in batch_failed_files {
          self.failed_files.insert(file.clone());
          // Remove accumulator for failed file
          self.file_accumulators.remove(&file);
        }
      }
    }

    Ok(())
  }

  async fn try_build_document(
    &mut self,
    file_path: String,
    document_batch: &mut Vec<CodeDocument>,
    ctx: &DocumentBuildContext<'_>,
  ) -> Result<(), IndexerError> {
    if let Some(accumulator) = self.file_accumulators.remove(&file_path) {
      self.total_files_built += 1;
      debug!(
        file_path,
        total_files_built = self.total_files_built,
        chunks = accumulator.received_content_chunks,
        "Building document for file: {file_path}"
      );

      // Build the document
      let result = build_document_from_accumulator(
        self.project_id,
        accumulator,
        self.embedding_dim,
        document_batch,
        ctx,
      )
      .await;

      if let Err(e) = result {
        error!("Failed to build document for {}: {}", file_path, e);
        self.failed_files.insert(file_path);
      }
    }
    Ok(())
  }

  async fn flush_remaining(&mut self) -> Result<(), IndexerError> {
    // Handle any incomplete files
    let incomplete_files: Vec<String> = self.file_accumulators.keys().cloned().collect();

    for file_path in incomplete_files {
      let accumulator = self.file_accumulators.get(&file_path).unwrap();

      if accumulator.has_eof {
        error!(
          "File {} incomplete at shutdown: {} of {} chunks received",
          file_path,
          accumulator.received_content_chunks,
          accumulator.expected_chunks.unwrap_or(0)
        );
      } else {
        error!(
          "File {} missing EOF marker, received {} chunks",
          file_path, accumulator.received_content_chunks
        );
      }

      // Add to failed files
      self.failed_files.insert(file_path);
    }

    Ok(())
  }
}

/// Build a document from accumulated file chunks using weighted average
/// Returns both the document and the individual chunks for storage
  pub(crate) async fn build_document_from_accumulator(
  project_id: Uuid,
  accumulator: FileAccumulator,
  embedding_dim: usize,
  document_batch: &mut Vec<CodeDocument>,
  ctx: &DocumentBuildContext<'_>,
) -> Result<(), IndexerError> {
  if accumulator.embedded_chunks.is_empty() {
    return Ok(());
  }

  let file_path = accumulator.file_path;
  let embedded_chunks = accumulator.embedded_chunks;

  // Calculate weights based on token counts
  let mut weights = Vec::new();
  let mut total_weight = 0.0;

  for embedded_chunk in &embedded_chunks {
    match &embedded_chunk.chunk {
      PipelineChunk::Semantic(sc) | PipelineChunk::Text(sc) => {
        // Use actual token count if available, otherwise estimate
        let token_count = sc
          .tokens
          .as_ref()
          .map(|tokens| tokens.len() as f32)
          .unwrap_or_else(|| (sc.text.len() as f32 / 4.0).max(1.0));
        weights.push(token_count);
        total_weight += token_count;
      }
      PipelineChunk::EndOfFile { .. } => continue, // Skip EOF markers
    }
  }

  // Normalize weights
  for weight in &mut weights {
    *weight /= total_weight;
  }

  // Compute weighted average embedding
  let mut aggregated_embedding = vec![0.0; embedding_dim];

  for (i, embedded_chunk) in embedded_chunks
    .iter()
    .filter(|ec| !matches!(ec.chunk, PipelineChunk::EndOfFile { .. }))
    .enumerate()
  {
    let weight = weights[i];
    for (j, &value) in embedded_chunk.embedding.iter().enumerate() {
      if j < embedding_dim {
        aggregated_embedding[j] += value * weight;
      }
    }
  }

  debug!(
      file_path = %file_path,
      num_chunks = embedded_chunks.len(),
      total_tokens_approx = total_weight as u64,
      "Aggregated embeddings for file"
  );

  // Check if we have an EOF chunk with content
  let (content, content_hash) = if let Some(eof_chunk) = embedded_chunks
    .iter()
    .find(|ec| matches!(ec.chunk, PipelineChunk::EndOfFile { .. }))
  {
    if let PipelineChunk::EndOfFile {
      ref content,
      ref content_hash,
      ..
    } = eof_chunk.chunk
    {
      (content.clone(), *content_hash)
    } else {
      // This shouldn't happen but handle gracefully
      error!("EOF chunk found but missing content for {}", file_path);
      return Err(IndexerError::Task(format!(
        "EOF chunk found but missing content for {}",
        file_path
      )));
    }
  } else {
    // No EOF chunk found - this shouldn't happen in normal flow
    error!("No EOF chunk found for {}", file_path);
    return Err(IndexerError::Task(format!(
      "No EOF chunk found for {}",
      file_path
    )));
  };

  // Create document with content from EOF chunk
  let mut doc = CodeDocument::new(project_id, file_path.clone(), content);

  // Extract languages and create chunks
  let mut language_counts: HashMap<String, usize> = HashMap::new();
  let mut code_chunks = Vec::new();
  let doc_id = doc.id;

  for embedded_chunk in &embedded_chunks {
    match &embedded_chunk.chunk {
      PipelineChunk::Semantic(sc) | PipelineChunk::Text(sc) => {
        // Count language occurrences
        *language_counts
          .entry(sc.metadata.language.clone())
          .or_insert(0) += sc
          .tokens
          .as_ref()
          .map(|t| t.len())
          .unwrap_or(sc.text.len() / 4);

        // Create CodeChunk
        let mut chunk = CodeChunk::builder()
          .file_id(doc_id)
          .project_id(project_id)
          .file_path(file_path.clone())
          .content(sc.text.clone())
          .start_byte(sc.start_byte)
          .end_byte(sc.end_byte)
          .start_line(sc.start_line)
          .end_line(sc.end_line)
          .build();

        // Update chunk with embedding and metadata
        chunk.update_embedding(embedded_chunk.embedding.clone());
        chunk.update_metadata(
          ChunkMetadataUpdate::builder()
            .node_type(sc.metadata.node_type.clone())
            .node_name(sc.metadata.node_name.clone())
            .language(sc.metadata.language.clone())
            .parent_context(sc.metadata.parent_context.clone())
            .scope_path(sc.metadata.scope_path.clone())
            .definitions(sc.metadata.definitions.clone())
            .references(sc.metadata.references.clone())
            .build(),
        );

        code_chunks.push(chunk);
      }
      PipelineChunk::EndOfFile { .. } => continue,
    }
  }

  // Set document languages and primary language
  let mut languages: Vec<(String, usize)> = language_counts.into_iter().collect();
  languages.sort_by(|a, b| b.1.cmp(&a.1)); // Sort by count descending

  doc.languages = languages.iter().map(|(lang, _)| lang.clone()).collect();
  doc.primary_language = languages.first().map(|(lang, _)| lang.clone());
  doc.chunk_count = code_chunks.len() as u32;

  doc.update_embedding(aggregated_embedding);
  doc.update_content_hash(content_hash);

  // Send a single ReplaceFileChunks message (single path, no per-chunk streaming)
  let replace = ReplaceFileChunks {
    project_id,
    file_path: file_path.clone(),
    chunks: code_chunks,
  };
  if ctx.chunks_replace_tx.send(replace).await.is_err() {
    return Err(IndexerError::Task("ReplaceFileChunks receiver dropped".into()));
  }

  // Batch documents before sending
  document_batch.push(doc);
  debug!(
    "Added document to batch. Current batch size: {}, File: {}",
    document_batch.len(),
    file_path
  );

  if document_batch.len() >= ctx.batch_size || ctx.cancel_token.is_cancelled() {
    info!(
      "Flushing document batch: {} documents (triggered by: {})",
      document_batch.len(),
      if ctx.cancel_token.is_cancelled() {
        "cancellation"
      } else {
        "batch size"
      }
    );
    for doc in document_batch.drain(..) {
      if ctx.doc_tx.send(doc).await.is_err() {
        return Err(IndexerError::Task("Document receiver dropped".into()));
      }
    }
  }

  ctx.stats.increment_files_stored();
  Ok(())
}

pub(crate) struct DocumentBuilderParams {
  pub project_id: Uuid,
  pub embedded_rx: mpsc::Receiver<EmbeddedChunkWithFile>,
  pub doc_tx: mpsc::Sender<CodeDocument>,
  pub chunks_replace_tx: ReplaceFileChunksSender,
  pub embedding_dim: usize,
  pub stats: IndexingStats,
  pub cancel_token: CancellationToken,
  pub batch_size: usize,
}

/// Document builder task that uses StreamDocumentBuilder
pub(crate) async fn document_builder_task(
  params: DocumentBuilderParams,
) -> (usize, Option<(BTreeSet<String>, String)>) {
  let DocumentBuilderParams {
    project_id,
    mut embedded_rx,
    doc_tx,
    chunk_tx,
    embedding_dim,
    stats,
    cancel_token,
    batch_size,
  } = params;
  let mut builder = StreamDocumentBuilder::new(project_id, embedding_dim);
  let mut document_batch: Vec<CodeDocument> = Vec::with_capacity(100);

  let ctx = DocumentBuildContext {
    doc_tx: &doc_tx,
    chunks_replace_tx: &chunks_replace_tx,
    stats: &stats,
    cancel_token: &cancel_token,
    batch_size,
  };

  loop {
    tokio::select! {
      _ = cancel_token.cancelled() => {
        info!("Document builder cancelled");
        break;
      }
      embedded_chunk = embedded_rx.recv() => {
        match embedded_chunk {
          Some(embedded_chunk) => {
            let chunk_info = match &embedded_chunk {
              EmbeddedChunkWithFile::Embedded { batch_id, file_path, .. } => {
                format!("Embedded chunk - batch: {}, file: {}", batch_id, file_path)
              }
              EmbeddedChunkWithFile::EndOfFile { batch_id, file_path, .. } => {
                format!("EOF - batch: {}, file: {}", batch_id, file_path)
              }
              EmbeddedChunkWithFile::BatchFailure { batch_id, .. } => {
                format!("Batch failure - batch: {}", batch_id)
              }
            };
            debug!("Document builder received: {}", chunk_info);

            if let Err(e) = builder.add_chunk(
              embedded_chunk,
              &mut document_batch,
              &ctx,
            ).await {
              error!("Failed to process chunk: {}", e);
              // Stop processing on error - likely means receiver dropped
              break;
            }
          }
          None => {
            debug!("Document builder channel closed");
            break;
          }
        }
      }
    }
  }

  // Flush any remaining documents as a batch
  if !document_batch.is_empty() {
    info!(
      "Final flush: sending batch of {} documents",
      document_batch.len()
    );
    // Send all documents at once to ensure they're batched together
    for doc in document_batch.drain(..) {
      if doc_tx.send(doc).await.is_err() {
        error!("Failed to send final documents - receiver dropped");
        break;
      }
    }
  }

  // Process any remaining pending chunks
  if let Err(e) = builder.flush_remaining().await {
    error!("Failed to flush remaining documents: {}", e);
  }

  // Get results from builder
  let total_files = builder.total_files_built;
  let failed_files = builder.failed_files;

  info!(
    total_files = total_files,
    files_stored = stats.get_files_stored(),
    "Document builder completed"
  );

  if failed_files.is_empty() {
    (total_files as usize, None)
  } else {
    let error_summary = format!("{} files failed", failed_files.len());
    (total_files as usize, Some((failed_files, error_summary)))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::pipeline::{EmbeddedChunk, PipelineChunk};
  use breeze_chunkers::{ChunkMetadata, SemanticChunk};
  use std::io::Write;
  use tempfile::NamedTempFile;

  fn create_test_chunk(text: &str, embedding: Vec<f32>) -> EmbeddedChunk {
    EmbeddedChunk {
      chunk: PipelineChunk::Text(SemanticChunk {
        text: text.to_string(),
        start_byte: 0,
        end_byte: text.len(),
        start_line: 1,
        end_line: 1,
        tokens: None,
        metadata: ChunkMetadata {
          node_type: "text".to_string(),
          node_name: None,
          language: "text".to_string(),
          parent_context: None,
          scope_path: vec![],
          definitions: vec![],
          references: vec![],
        },
      }),
      embedding,
    }
  }

  #[tokio::test]
  async fn test_weighted_average_builder() {
    // Create a temporary file
    let mut temp_file = NamedTempFile::new().unwrap();
    let content = "short\nthis is a longer chunk\nmedium chunk";
    write!(temp_file, "{}", content).unwrap();
    let file_path = temp_file.path().to_string_lossy().to_string();

    // Create test accumulator
    let mut accumulator = FileAccumulator::new(file_path.clone());

    // Add chunks
    accumulator.add_chunk(create_test_chunk("short", vec![1.0, 0.0, 0.0])); // weight ~1.25
    accumulator.add_chunk(create_test_chunk(
      "this is a longer chunk",
      vec![0.0, 1.0, 0.0],
    )); // weight ~5.5
    accumulator.add_chunk(create_test_chunk("medium chunk", vec![0.0, 0.0, 1.0])); // weight ~3.0

    // Add EOF chunk with content
    let hash = blake3::hash(content.as_bytes());
    let mut content_hash = [0u8; 32];
    content_hash.copy_from_slice(hash.as_bytes());
    accumulator.add_chunk(EmbeddedChunk {
      chunk: PipelineChunk::EndOfFile {
        file_path: file_path.clone(),
        content: content.to_string(),
        content_hash,
        expected_chunks: 3, // We added 3 chunks above
      },
      embedding: vec![],
    });

    let project_id = uuid::Uuid::now_v7();
    let (doc_tx, _doc_rx) = mpsc::channel(1);
    let (chunks_replace_tx, mut chunks_replace_rx) = mpsc::channel(10);
    let stats = IndexingStats::new();
    let cancel_token = CancellationToken::new();
    let mut document_batch = Vec::new();

    let ctx = DocumentBuildContext {
      doc_tx: &doc_tx,
      chunks_replace_tx: &chunks_replace_tx,
      stats: &stats,
      cancel_token: &cancel_token,
      batch_size: 100,
    };

    build_document_from_accumulator(project_id, accumulator, 3, &mut document_batch, &ctx)
      .await
      .unwrap();

    // Get the document from the batch
    assert_eq!(document_batch.len(), 1);
    let doc = document_batch.pop().unwrap();

    // Collect replace messages
    drop(chunks_replace_tx);
    let mut replaces = Vec::new();
    while let Some(rep) = chunks_replace_rx.recv().await {
      replaces.push(rep);
    }

    // Verify document properties
    assert_eq!(doc.file_path, file_path);
    assert_eq!(doc.content, content);
    assert_eq!(doc.content_embedding.len(), 3);
    assert_eq!(doc.chunk_count, 3);
    assert_eq!(doc.languages, vec!["text"]);
    assert_eq!(doc.primary_language, Some("text".to_string()));

    // Verify replaces
    assert_eq!(replaces.len(), 1);
    assert_eq!(replaces[0].chunks.len(), 3);

    // The weighted average should favor the longer chunk
    assert!(doc.content_embedding[0] < 0.2); // Should be small
    assert!(doc.content_embedding[1] > 0.5); // Should be largest
    assert!(doc.content_embedding[2] > 0.2 && doc.content_embedding[2] < 0.4); // Should be medium

    // Sum should be approximately 1.0
    let sum: f32 = doc.content_embedding.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
  }

  #[tokio::test]
  async fn test_empty_chunks() {
    let project_id = uuid::Uuid::now_v7();
    let accumulator = FileAccumulator::new("empty.txt".to_string());
    let (doc_tx, _doc_rx) = mpsc::channel(1);
    let (chunks_replace_tx, _chunks_replace_rx) = mpsc::channel(10);
    let stats = IndexingStats::new();
    let cancel_token = CancellationToken::new();
    let mut document_batch = Vec::new();

    let ctx = DocumentBuildContext {
      doc_tx: &doc_tx,
      chunks_replace_tx: &chunks_replace_tx,
      stats: &stats,
      cancel_token: &cancel_token,
      batch_size: 100,
    };

    // Should return error for empty accumulator
    let _result =
      build_document_from_accumulator(project_id, accumulator, 3, &mut document_batch, &ctx).await;

    // Empty accumulator should not produce any documents
    assert_eq!(document_batch.len(), 0);
  }
}
