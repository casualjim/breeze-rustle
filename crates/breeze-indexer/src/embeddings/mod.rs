use async_trait::async_trait;
use std::sync::Arc;

pub mod batching;
pub mod error;
pub mod factory;
#[cfg(feature = "local-embeddings")]
pub mod local;
pub mod openailike;
pub mod voyage;

pub use error::{EmbeddingError, EmbeddingResult};

/// Input for embedding a single item (borrows text to avoid cloning)
#[derive(Debug)]
pub struct EmbeddingInput<'a> {
  /// The text to embed
  pub text: &'a str,
  /// Pre-computed token count (if available)
  pub token_count: Option<usize>,
}

/// Main trait for embedding providers
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
  /// Embed a batch of inputs
  async fn embed(&self, inputs: &[EmbeddingInput<'_>]) -> EmbeddingResult<Vec<Vec<f32>>>;

  /// Get the embedding dimension for this provider's model
  fn embedding_dim(&self) -> usize;

  /// Get the maximum context length in tokens
  fn context_length(&self) -> usize;

  /// Create the appropriate batching strategy for this provider
  fn create_batching_strategy(&self) -> Box<dyn batching::BatchingStrategy>;

  /// Get the HuggingFace tokenizer for this provider (if applicable)
  fn tokenizer(&self) -> Option<Arc<tokenizers::Tokenizer>>;

  /// Whether this provider is remote (network-based) or local
  fn is_remote(&self) -> bool {
    false // Default to local
  }
}
