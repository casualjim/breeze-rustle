use async_trait::async_trait;
use breeze_chunkers::ProjectChunk;

/// Represents a batch of chunks ready for embedding
#[derive(Debug, Clone)]
pub struct EmbeddingBatch {
  pub chunks: Vec<ProjectChunk>,
}

/// Strategy for batching chunks for embedding
#[async_trait]
pub trait BatchingStrategy: Send + Sync {
  /// Prepare chunks for batching
  async fn prepare_batches(&self, chunks: Vec<ProjectChunk>) -> Vec<EmbeddingBatch>;

  /// Maximum items per batch
  fn max_batch_size(&self) -> usize;

  /// Whether this strategy considers token counts
  fn is_token_aware(&self) -> bool;
}

/// Simple batching strategy for local providers
pub struct LocalBatchingStrategy {
  batch_size: usize,
}

impl LocalBatchingStrategy {
  pub fn new(batch_size: usize) -> Self {
    Self { batch_size }
  }
}

#[async_trait]
impl BatchingStrategy for LocalBatchingStrategy {
  async fn prepare_batches(&self, chunks: Vec<ProjectChunk>) -> Vec<EmbeddingBatch> {
    chunks
      .chunks(self.batch_size)
      .map(|chunk_batch| EmbeddingBatch {
        chunks: chunk_batch.to_vec(),
      })
      .collect()
  }

  fn max_batch_size(&self) -> usize {
    self.batch_size
  }

  fn is_token_aware(&self) -> bool {
    false
  }
}

/// Token-aware batching strategy for API providers
pub struct TokenAwareBatchingStrategy {
  max_tokens_per_batch: usize,
  max_items_per_batch: usize,
}

impl TokenAwareBatchingStrategy {
  pub fn new(max_tokens_per_batch: usize, max_items_per_batch: usize) -> Self {
    Self {
      max_tokens_per_batch,
      max_items_per_batch,
    }
  }
}

#[async_trait]
impl BatchingStrategy for TokenAwareBatchingStrategy {
  async fn prepare_batches(&self, chunks: Vec<ProjectChunk>) -> Vec<EmbeddingBatch> {
    let mut batches = Vec::new();
    let mut current_batch = Vec::new();
    let mut current_tokens = 0;

    for chunk in chunks {
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
        batches.push(EmbeddingBatch {
          chunks: current_batch,
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
      batches.push(EmbeddingBatch {
        chunks: current_batch,
      });
    }

    batches
  }

  fn max_batch_size(&self) -> usize {
    self.max_items_per_batch
  }

  fn is_token_aware(&self) -> bool {
    true
  }
}
