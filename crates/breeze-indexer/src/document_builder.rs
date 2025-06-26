use std::collections::{BTreeMap, BTreeSet, HashMap};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::IndexerError;
use crate::bulk_indexer::IndexingStats;
use crate::models::{ChunkMetadataUpdate, CodeChunk, CodeDocument};
use crate::pipeline::{EmbeddedChunk, EmbeddedChunkWithFile, FileAccumulator, PipelineChunk};
use breeze_chunkers::ChunkStream;

/// Context for document building operations
pub(crate) struct DocumentBuildContext<'a> {
  pub doc_tx: &'a mpsc::Sender<CodeDocument>,
  pub chunk_tx: &'a mpsc::Sender<CodeChunk>,
  pub stats: &'a IndexingStats,
  pub cancel_token: &'a CancellationToken,
}

/// Stream-specific document builder that maintains ordering for a single stream
struct StreamDocumentBuilder {
  project_id: Uuid,
  embedding_dim: usize,
  file_accumulators: BTreeMap<String, FileAccumulator>,
  pending_chunks: BTreeMap<usize, Vec<EmbeddedChunkWithFile>>,
  next_expected_batch_id: usize,
  failed_files: BTreeSet<String>,
  total_files_built: u64,
}

impl StreamDocumentBuilder {
  fn new(project_id: Uuid, embedding_dim: usize) -> Self {
    Self {
      project_id,
      embedding_dim,
      file_accumulators: BTreeMap::new(),
      pending_chunks: BTreeMap::new(),
      next_expected_batch_id: 0,
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
    let batch_id = match &embedded_chunk {
      EmbeddedChunkWithFile::Embedded { batch_id, .. } => *batch_id,
      EmbeddedChunkWithFile::EndOfFile { batch_id, .. } => *batch_id,
      EmbeddedChunkWithFile::BatchFailure { batch_id, .. } => *batch_id,
    };

    // Add chunk to pending buffer
    self
      .pending_chunks
      .entry(batch_id)
      .or_default()
      .push(embedded_chunk);

    // Process any chunks that are now in order
    while let Some((&current_batch_id, _)) = self.pending_chunks.first_key_value() {
      if current_batch_id == self.next_expected_batch_id {
        // Process this batch
        let batch_chunks = self.pending_chunks.remove(&current_batch_id).unwrap();

        for embedded_chunk in batch_chunks {
          self
            .process_chunk(embedded_chunk, current_batch_id, document_batch, ctx)
            .await?;
        }

        self.next_expected_batch_id += 1;
      } else {
        // We're missing some earlier batch, stop processing
        break;
      }
    }

    Ok(())
  }

  async fn process_chunk(
    &mut self,
    embedded_chunk: EmbeddedChunkWithFile,
    batch_id: usize,
    document_batch: &mut Vec<CodeDocument>,
    ctx: &DocumentBuildContext<'_>,
  ) -> Result<(), IndexerError> {
    match embedded_chunk {
      EmbeddedChunkWithFile::Embedded {
        file_path,
        chunk,
        embedding,
        ..
      } => {
        // Skip chunks for failed files
        if self.failed_files.contains(&file_path) {
          return Ok(());
        }

        // Accumulate chunk
        let accumulator = self
          .file_accumulators
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
        if self.failed_files.contains(&file_path) {
          return Ok(());
        }

        // Build document for completed file
        if let Some(mut accumulator) = self.file_accumulators.remove(&file_path) {
          self.total_files_built += 1;
          debug!(
            file_path,
            total_files_built = self.total_files_built,
            batch_id,
            "Building document for file: {file_path}"
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
        for file in batch_failed_files {
          self.failed_files.insert(file);
        }
      }
    }

    Ok(())
  }

  async fn flush_remaining(
    &mut self,
    document_batch: &mut Vec<CodeDocument>,
    ctx: &DocumentBuildContext<'_>,
  ) -> Result<(), IndexerError> {
    // Process any remaining pending chunks
    for (batch_id, batch_chunks) in std::mem::take(&mut self.pending_chunks) {
      debug!(
        "Processing remaining batch {} with {} chunks",
        batch_id,
        batch_chunks.len()
      );
      for embedded_chunk in batch_chunks {
        self
          .process_chunk(embedded_chunk, batch_id, document_batch, ctx)
          .await?;
      }
    }

    Ok(())
  }
}

/// Dual-stream document builder that routes chunks to appropriate stream builders
pub(crate) struct DualStreamDocumentBuilder {
  regular_builder: StreamDocumentBuilder,
  large_file_builder: StreamDocumentBuilder,
}

impl DualStreamDocumentBuilder {
  pub fn new(project_id: Uuid, embedding_dim: usize) -> Self {
    Self {
      regular_builder: StreamDocumentBuilder::new(project_id, embedding_dim),
      large_file_builder: StreamDocumentBuilder::new(project_id, embedding_dim),
    }
  }

  pub async fn add_chunk(
    &mut self,
    embedded_chunk: EmbeddedChunkWithFile,
    document_batch: &mut Vec<CodeDocument>,
    ctx: &DocumentBuildContext<'_>,
  ) -> Result<(), IndexerError> {
    // Route to appropriate builder based on stream
    let stream = match &embedded_chunk {
      EmbeddedChunkWithFile::Embedded { stream, .. } => *stream,
      EmbeddedChunkWithFile::EndOfFile { stream, .. } => *stream,
      EmbeddedChunkWithFile::BatchFailure { stream, .. } => *stream,
    };

    match stream {
      ChunkStream::Regular => {
        self
          .regular_builder
          .add_chunk(embedded_chunk, document_batch, ctx)
          .await
      }
      ChunkStream::LargeFile => {
        self
          .large_file_builder
          .add_chunk(embedded_chunk, document_batch, ctx)
          .await
      }
    }
  }

  pub async fn flush_remaining(
    &mut self,
    document_batch: &mut Vec<CodeDocument>,
    ctx: &DocumentBuildContext<'_>,
  ) -> Result<(u64, BTreeSet<String>), IndexerError> {
    // Flush both builders
    self
      .regular_builder
      .flush_remaining(document_batch, ctx)
      .await?;

    self
      .large_file_builder
      .flush_remaining(document_batch, ctx)
      .await?;

    // Combine results
    let total_files =
      self.regular_builder.total_files_built + self.large_file_builder.total_files_built;
    let mut all_failed_files = self.regular_builder.failed_files.clone();
    all_failed_files.extend(self.large_file_builder.failed_files.clone());

    Ok((total_files, all_failed_files))
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

  for (i, embedded_chunk) in embedded_chunks.iter().enumerate() {
    if matches!(embedded_chunk.chunk, PipelineChunk::EndOfFile { .. }) {
      continue;
    }
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

  // Send chunks
  for chunk in code_chunks {
    if ctx.chunk_tx.send(chunk).await.is_err() {
      return Err(IndexerError::Task("Chunk receiver dropped".into()));
    }
  }

  // Batch documents before sending
  document_batch.push(doc);
  if document_batch.len() >= 100 || ctx.cancel_token.is_cancelled() {
    for doc in document_batch.drain(..) {
      if ctx.doc_tx.send(doc).await.is_err() {
        return Err(IndexerError::Task("Document receiver dropped".into()));
      }
    }
  }

  ctx.stats.increment_files_stored();
  Ok(())
}

/// Document builder task that uses DualStreamDocumentBuilder
pub(crate) async fn document_builder_task(
  project_id: Uuid,
  mut embedded_rx: mpsc::Receiver<EmbeddedChunkWithFile>,
  doc_tx: mpsc::Sender<CodeDocument>,
  chunk_tx: mpsc::Sender<CodeChunk>,
  embedding_dim: usize,
  stats: IndexingStats,
  cancel_token: CancellationToken,
) -> (usize, Option<(BTreeSet<String>, String)>) {
  let mut builder = DualStreamDocumentBuilder::new(project_id, embedding_dim);
  let mut document_batch: Vec<CodeDocument> = Vec::with_capacity(100);

  let ctx = DocumentBuildContext {
    doc_tx: &doc_tx,
    chunk_tx: &chunk_tx,
    stats: &stats,
    cancel_token: &cancel_token,
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
          None => break,
        }
      }
    }
  }

  // Flush any remaining documents
  if !document_batch.is_empty() {
    for doc in document_batch {
      if doc_tx.send(doc).await.is_err() {
        error!("Failed to send final documents - receiver dropped");
        break;
      }
    }
  }

  // Process any remaining pending chunks
  let mut final_batch = Vec::new();
  match builder.flush_remaining(&mut final_batch, &ctx).await {
    Ok((total_files, failed_files)) => {
      // Send final batch
      for doc in final_batch {
        if doc_tx.send(doc).await.is_err() {
          error!("Failed to send final documents - receiver dropped");
          break;
        }
      }

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
    Err(e) => {
      error!("Failed to flush remaining documents: {}", e);
      (0, Some((BTreeSet::new(), e.to_string())))
    }
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
      },
      embedding: vec![],
    });

    let project_id = uuid::Uuid::now_v7();
    let (doc_tx, _doc_rx) = mpsc::channel(1);
    let (chunk_tx, mut chunk_rx) = mpsc::channel(10);
    let stats = IndexingStats::new();
    let cancel_token = CancellationToken::new();
    let mut document_batch = Vec::new();

    let ctx = DocumentBuildContext {
      doc_tx: &doc_tx,
      chunk_tx: &chunk_tx,
      stats: &stats,
      cancel_token: &cancel_token,
    };

    build_document_from_accumulator(project_id, accumulator, 3, &mut document_batch, &ctx)
      .await
      .unwrap();

    // Get the document from the batch
    assert_eq!(document_batch.len(), 1);
    let doc = document_batch.pop().unwrap();

    // Collect chunks
    drop(chunk_tx);
    let mut chunks = Vec::new();
    while let Some(chunk) = chunk_rx.recv().await {
      chunks.push(chunk);
    }

    // Verify document properties
    assert_eq!(doc.file_path, file_path);
    assert_eq!(doc.content, content);
    assert_eq!(doc.content_embedding.len(), 3);
    assert_eq!(doc.chunk_count, 3);
    assert_eq!(doc.languages, vec!["text"]);
    assert_eq!(doc.primary_language, Some("text".to_string()));

    // Verify chunks
    assert_eq!(chunks.len(), 3);

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
    let (chunk_tx, _chunk_rx) = mpsc::channel(10);
    let stats = IndexingStats::new();
    let cancel_token = CancellationToken::new();
    let mut document_batch = Vec::new();

    let ctx = DocumentBuildContext {
      doc_tx: &doc_tx,
      chunk_tx: &chunk_tx,
      stats: &stats,
      cancel_token: &cancel_token,
    };

    // Should return error for empty accumulator
    let _result =
      build_document_from_accumulator(project_id, accumulator, 3, &mut document_batch, &ctx).await;

    // Empty accumulator should not produce any documents
    assert_eq!(document_batch.len(), 0);
  }
}
