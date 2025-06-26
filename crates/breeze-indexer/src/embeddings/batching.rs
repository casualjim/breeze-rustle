use crate::pipeline::{ChunkBatch, PipelineChunk, PipelineProjectChunk};
use async_trait::async_trait;

use breeze_chunkers::ChunkStream;

/// Represents a batch of chunks ready for embedding
#[derive(Debug, Clone)]
pub struct EmbeddingBatch {
  pub batch_id: usize,
  pub chunks: Vec<PipelineProjectChunk>,
  pub eof_chunks: Vec<PipelineProjectChunk>,
  pub stream: ChunkStream,
}

/// Strategy for batching chunks for embedding
#[async_trait]
pub trait BatchingStrategy: Send + Sync {
  /// Prepare batches for embedding
  /// Takes batches and reorganizes them based on the strategy
  async fn prepare_batches(&self, batches: Vec<ChunkBatch>) -> Vec<EmbeddingBatch>;

  /// Maximum items per batch
  fn max_batch_size(&self) -> usize;
}

/// Simple batching strategy for local providers
#[cfg(any(feature = "local-embeddings", test))]
pub struct LocalBatchingStrategy {
  batch_size: usize,
}

#[cfg(any(feature = "local-embeddings", test))]
impl LocalBatchingStrategy {
  pub fn new(batch_size: usize) -> Self {
    Self { batch_size }
  }
}

#[cfg(any(feature = "local-embeddings", test))]
#[async_trait]
impl BatchingStrategy for LocalBatchingStrategy {
  async fn prepare_batches(&self, batches: Vec<ChunkBatch>) -> Vec<EmbeddingBatch> {
    let mut result = Vec::new();

    for batch in batches {
      let ChunkBatch {
        batch_id,
        chunks,
        stream,
      } = batch;

      // Separate regular chunks from EOF chunks
      let mut regular_chunks = Vec::new();
      let mut eof_chunks = Vec::new();

      for chunk in chunks {
        match &chunk.chunk {
          PipelineChunk::EndOfFile { .. } => eof_chunks.push(chunk),
          _ => regular_chunks.push(chunk),
        }
      }

      // Split regular chunks into smaller batches respecting batch_size
      for chunk_batch in regular_chunks.chunks(self.batch_size) {
        result.push(EmbeddingBatch {
          batch_id,
          chunks: chunk_batch.to_vec(),
          eof_chunks: Vec::new(), // EOF chunks only go with the last batch
          stream,
        });
      }

      // Add EOF chunks to the last batch of this batch_id
      if !eof_chunks.is_empty() {
        if let Some(last_batch) = result.last_mut() {
          if last_batch.batch_id == batch_id && last_batch.stream == stream {
            last_batch.eof_chunks = eof_chunks;
          } else {
            // Create a new batch just for EOF chunks if no regular chunks exist
            result.push(EmbeddingBatch {
              batch_id,
              chunks: Vec::new(),
              eof_chunks,
              stream,
            });
          }
        } else {
          // No regular chunks, create batch with just EOF chunks
          result.push(EmbeddingBatch {
            batch_id,
            chunks: Vec::new(),
            eof_chunks,
            stream,
          });
        }
      }
    }

    result
  }

  fn max_batch_size(&self) -> usize {
    self.batch_size
  }
}

/// Token-aware batching strategy for API providers
pub struct TokenAwareBatchingStrategy {
  max_tokens_per_batch: usize,
  max_items_per_batch: usize,
}

impl TokenAwareBatchingStrategy {
  pub fn new(max_tokens_per_batch: usize, max_items_per_batch: usize) -> Self {
    tracing::debug!(
      tokens_per_batch = max_tokens_per_batch,
      items_per_batch = max_items_per_batch,
      "Creating TokenAwareBatchingStrategy"
    );
    Self {
      max_tokens_per_batch,
      max_items_per_batch,
    }
  }
}

#[async_trait]
impl BatchingStrategy for TokenAwareBatchingStrategy {
  async fn prepare_batches(&self, batches: Vec<ChunkBatch>) -> Vec<EmbeddingBatch> {
    let mut result = Vec::new();

    for batch in batches {
      let ChunkBatch {
        batch_id,
        chunks,
        stream,
      } = batch;

      // Separate regular chunks from EOF chunks
      let mut regular_chunks = Vec::new();
      let mut eof_chunks = Vec::new();

      for chunk in chunks {
        match &chunk.chunk {
          PipelineChunk::EndOfFile { .. } => eof_chunks.push(chunk),
          _ => regular_chunks.push(chunk),
        }
      }

      let mut current_batch = Vec::new();
      let mut current_tokens = 0;

      for chunk in regular_chunks {
        // Use pre-computed token count if available, otherwise estimate
        let chunk_tokens = match &chunk.chunk {
          PipelineChunk::Semantic(sc) | PipelineChunk::Text(sc) => {
            if let Some(ref tokens) = sc.tokens {
              // Use pre-computed token count
              tokens.len()
            } else {
              // Estimate ~4 chars per token
              sc.text.len() / 4
            }
          }
          PipelineChunk::EndOfFile { .. } => 0,
        };

        // Check if adding this chunk would exceed limits
        if !current_batch.is_empty()
          && (current_batch.len() >= self.max_items_per_batch
            || current_tokens + chunk_tokens > self.max_tokens_per_batch)
        {
          // Save current batch
          result.push(EmbeddingBatch {
            batch_id,
            chunks: current_batch,
            eof_chunks: Vec::new(),
            stream,
          });
          current_batch = Vec::new();
          current_tokens = 0;
        }

        // Add chunk to current batch
        current_tokens += chunk_tokens;
        current_batch.push(chunk);
      }

      // Add final batch if not empty
      if !current_batch.is_empty() {
        result.push(EmbeddingBatch {
          batch_id,
          chunks: current_batch,
          eof_chunks: eof_chunks.clone(),
          stream,
        });
      } else if !eof_chunks.is_empty() {
        // No regular chunks, create batch with just EOF chunks
        result.push(EmbeddingBatch {
          batch_id,
          chunks: Vec::new(),
          eof_chunks,
          stream,
        });
      } else {
        // Add EOF chunks to the last batch of this batch_id and stream
        if let Some(last_batch) = result
          .iter_mut()
          .rev()
          .find(|b| b.batch_id == batch_id && b.stream == stream)
        {
          last_batch.eof_chunks = eof_chunks;
        }
      }
    }

    result
  }

  fn max_batch_size(&self) -> usize {
    self.max_items_per_batch
  }
}
