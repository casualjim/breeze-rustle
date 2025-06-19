use async_trait::async_trait;
use breeze_chunkers::ProjectChunk;

/// Represents a batch of chunks ready for embedding
#[derive(Debug, Clone)]
pub struct EmbeddingBatch {
  pub batch_id: usize,
  pub chunks: Vec<ProjectChunk>,
  pub eof_chunks: Vec<ProjectChunk>,
}

/// Strategy for batching chunks for embedding
#[async_trait]
pub trait BatchingStrategy: Send + Sync {
  /// Prepare chunks for batching
  /// Takes chunks with their batch IDs and groups them appropriately
  async fn prepare_batches(&self, chunks: Vec<(usize, ProjectChunk)>) -> Vec<EmbeddingBatch>;

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
  async fn prepare_batches(&self, chunks: Vec<(usize, ProjectChunk)>) -> Vec<EmbeddingBatch> {
    use std::collections::BTreeMap;

    // Group chunks by batch_id, separating regular chunks from EOF chunks
    let mut batches: BTreeMap<usize, (Vec<ProjectChunk>, Vec<ProjectChunk>)> = BTreeMap::new();

    for (batch_id, chunk) in chunks {
      let (regular, eof) = batches.entry(batch_id).or_insert((Vec::new(), Vec::new()));

      match &chunk.chunk {
        breeze_chunkers::Chunk::EndOfFile { .. } => eof.push(chunk),
        _ => regular.push(chunk),
      }
    }

    // Process each batch_id's chunks
    let mut result = Vec::new();
    for (batch_id, (regular_chunks, eof_chunks)) in batches {
      // Split regular chunks into smaller batches respecting batch_size
      for chunk_batch in regular_chunks.chunks(self.batch_size) {
        result.push(EmbeddingBatch {
          batch_id,
          chunks: chunk_batch.to_vec(),
          eof_chunks: Vec::new(), // EOF chunks only go with the last batch
        });
      }

      // Add EOF chunks to the last batch of this batch_id
      if !eof_chunks.is_empty() {
        if let Some(last_batch) = result.last_mut() {
          if last_batch.batch_id == batch_id {
            last_batch.eof_chunks = eof_chunks;
          } else {
            // Create a new batch just for EOF chunks if no regular chunks exist
            result.push(EmbeddingBatch {
              batch_id,
              chunks: Vec::new(),
              eof_chunks,
            });
          }
        } else {
          // No regular chunks, create batch with just EOF chunks
          result.push(EmbeddingBatch {
            batch_id,
            chunks: Vec::new(),
            eof_chunks,
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
  async fn prepare_batches(&self, chunks: Vec<(usize, ProjectChunk)>) -> Vec<EmbeddingBatch> {
    use std::collections::BTreeMap;

    // Group chunks by batch_id, separating regular chunks from EOF chunks
    let mut batches: BTreeMap<usize, (Vec<ProjectChunk>, Vec<ProjectChunk>)> = BTreeMap::new();

    for (batch_id, chunk) in chunks {
      let (regular, eof) = batches.entry(batch_id).or_insert((Vec::new(), Vec::new()));

      match &chunk.chunk {
        breeze_chunkers::Chunk::EndOfFile { .. } => eof.push(chunk),
        _ => regular.push(chunk),
      }
    }

    // Process each batch_id's chunks
    let mut result = Vec::new();
    for (batch_id, (regular_chunks, eof_chunks)) in batches {
      let mut current_batch = Vec::new();
      let mut current_tokens = 0;

      for chunk in regular_chunks {
        // Use pre-computed token count if available, otherwise estimate
        let chunk_tokens = match &chunk.chunk {
          breeze_chunkers::Chunk::Semantic(sc) | breeze_chunkers::Chunk::Text(sc) => {
            if let Some(ref tokens) = sc.tokens {
              // Use pre-computed token count
              tokens.len()
            } else {
              // Estimate ~4 chars per token
              sc.text.len() / 4
            }
          }
          breeze_chunkers::Chunk::EndOfFile { .. } => 0,
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
        });
      } else if !eof_chunks.is_empty() {
        // No regular chunks, create batch with just EOF chunks
        result.push(EmbeddingBatch {
          batch_id,
          chunks: Vec::new(),
          eof_chunks,
        });
      } else {
        // Add EOF chunks to the last batch of this batch_id
        if let Some(last_batch) = result.iter_mut().rev().find(|b| b.batch_id == batch_id) {
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
