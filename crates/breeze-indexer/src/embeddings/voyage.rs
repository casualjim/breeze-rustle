use async_trait::async_trait;
use std::sync::Arc;
use tokenizers::Tokenizer;

use super::{
  EmbeddingProvider,
  batching::{BatchingStrategy, TokenAwareBatchingStrategy},
};
use crate::aiproviders::voyage::{
  Config as VoyageClientConfig, EmbeddingModel, EmbeddingRequest, VoyageClient, new_client,
};
use crate::config::VoyageConfig;

/// Voyage AI embedding provider
pub struct VoyageEmbeddingProvider {
  client: Arc<dyn VoyageClient>,
  model: EmbeddingModel,
  tokenizer: Arc<Tokenizer>,
}

impl VoyageEmbeddingProvider {
  pub async fn new(
    config: &VoyageConfig,
    _worker_count: usize,
    _chunk_size: usize,
  ) -> super::EmbeddingResult<Self> {
    let client_config = VoyageClientConfig::new(config.api_key.clone(), config.tier, config.model);

    let client = new_client(client_config)?;

    // Load HuggingFace tokenizer for this model
    let repo_id = format!("voyageai/{}", config.model.api_name());
    tracing::debug!("Loading tokenizer for Voyage model from: {}", repo_id);
    let tokenizer = Tokenizer::from_pretrained(&repo_id, None).map_err(|e| {
      super::EmbeddingError::TokenizationError(format!(
        "Failed to load tokenizer for {}: {}",
        repo_id, e
      ))
    })?;
    tracing::info!("Successfully loaded Voyage tokenizer");

    Ok(Self {
      client: Arc::new(client),
      model: config.model,
      tokenizer: Arc::new(tokenizer),
    })
  }
}

#[async_trait]
impl EmbeddingProvider for VoyageEmbeddingProvider {
  async fn embed(
    &self,
    inputs: &[super::EmbeddingInput<'_>],
  ) -> super::EmbeddingResult<Vec<Vec<f32>>> {
    // Prepare request
    let request = EmbeddingRequest {
      input: inputs.iter().map(|input| input.text.to_string()).collect(),
      model: self.model.api_name().to_string(),
      input_type: Some("document".to_string()),
      output_dimension: None,
      truncation: Some(true),
    };

    // Calculate token counts - use pre-computed if available (from indexing),
    // otherwise tokenize on the fly (for search queries)
    let estimated_tokens: u32 = inputs
      .iter()
      .map(|input| {
        if let Some(count) = input.token_count {
          count as u32
        } else {
          // This happens for search queries - tokenize on the fly
          match self.tokenizer.encode(input.text, false) {
            Ok(encoding) => encoding.len() as u32,
            Err(e) => {
              tracing::warn!("Failed to tokenize input: {}", e);
              // Rough estimate: ~4 chars per token
              (input.text.len() / 4) as u32
            }
          }
        }
      })
      .sum();

    // Make API call
    let response = self
      .client
      .embed(request, estimated_tokens)
      .await
      .map_err(|e| super::EmbeddingError::ApiError(e.to_string()))?;

    // Log usage information
    tracing::debug!(
      "Voyage API call completed. Model: {}, Total tokens used: {}, Estimated tokens: {}",
      response.model,
      response.usage.total_tokens,
      estimated_tokens
    );

    // Extract embeddings in order
    let mut embeddings = vec![vec![]; response.data.len()];
    for data in response.data {
      if data.index < embeddings.len() {
        embeddings[data.index] = data.embedding;
      }
    }

    Ok(embeddings)
  }

  fn embedding_dim(&self) -> usize {
    self.model.dimensions()
  }

  fn context_length(&self) -> usize {
    self.model.context_length()
  }

  fn create_batching_strategy(&self) -> Box<dyn BatchingStrategy> {
    // Use conservative batch sizes to avoid rate limits
    // Python uses 80% safety margin on these limits
    const MAX_TOKENS_PER_BATCH: usize = 96_000;
    const MAX_TEXTS_PER_BATCH: usize = 100;

    Box::new(TokenAwareBatchingStrategy::new(
      MAX_TOKENS_PER_BATCH,
      MAX_TEXTS_PER_BATCH,
    ))
  }

  fn tokenizer(&self) -> Option<Arc<tokenizers::Tokenizer>> {
    Some(self.tokenizer.clone())
  }

  fn is_remote(&self) -> bool {
    true
  }
}
