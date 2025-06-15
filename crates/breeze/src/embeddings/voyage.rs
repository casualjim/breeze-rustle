use async_trait::async_trait;
use std::sync::Arc;
use tokenizers::Tokenizer;

use super::{
  EmbeddingProvider,
  batching::{BatchingStrategy, TokenAwareBatchingStrategy},
};
use crate::aiproviders::voyage::{
  Config as VoyageClientConfig, EmbeddingModel, EmbeddingRequest, Tier, VoyageClient, new_client,
};
use crate::config::VoyageConfig;

/// Voyage AI embedding provider
pub struct VoyageEmbeddingProvider {
  client: Arc<dyn VoyageClient>,
  model: EmbeddingModel,
  tokenizer: Arc<Tokenizer>,
  tier: Tier,
}

impl VoyageEmbeddingProvider {
  pub async fn new(config: &VoyageConfig) -> Result<Self, Box<dyn std::error::Error>> {
    let client_config = VoyageClientConfig::new(config.api_key.clone(), config.tier, config.model);

    let client = new_client(client_config)?;

    // Load HuggingFace tokenizer for this model
    let repo_id = format!("voyageai/{}", config.model.api_name());
    let tokenizer = Tokenizer::from_pretrained(&repo_id, None)
      .map_err(|e| format!("Failed to load tokenizer for {}: {}", repo_id, e))?;

    Ok(Self {
      client: Arc::new(client),
      model: config.model,
      tokenizer: Arc::new(tokenizer),
      tier: config.tier,
    })
  }
}

#[async_trait]
impl EmbeddingProvider for VoyageEmbeddingProvider {
  async fn embed(
    &self,
    inputs: &[super::EmbeddingInput<'_>],
  ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
    // Prepare request
    let request = EmbeddingRequest {
      input: inputs.iter().map(|input| input.text.to_string()).collect(),
      model: self.model.api_name().to_string(),
      input_type: Some("document".to_string()),
      output_dimension: None,
      truncation: Some(true),
    };

    // Use pre-computed token counts if available, otherwise count tokens
    let estimated_tokens: u32 = inputs
      .iter()
      .map(|input| {
        if let Some(count) = input.token_count {
          count as u32
        } else {
          // Count tokens using the tokenizer
          self
            .tokenizer
            .encode(input.text, false)
            .map(|encoding| encoding.len() as u32)
            .unwrap_or_else(|_| (input.text.len() / 4) as u32) // Fallback estimate
        }
      })
      .sum();

    // Make API call
    let response = self
      .client
      .embed(request, estimated_tokens)
      .await
      .map_err(|e| Box::new(e) as Box<dyn std::error::Error + Send + Sync>)?;

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
    // Calculate safe token limit (90% of tier limit divided by concurrent requests)
    let tokens_per_minute = self.tier.safe_tokens_per_minute(self.model);
    let max_concurrent = 4; // Could be configurable
    let max_tokens_per_batch = tokens_per_minute / (max_concurrent * 60); // Per second rate

    Box::new(TokenAwareBatchingStrategy::new(
      max_tokens_per_batch as usize,
      self.model.max_batch_size(),
    ))
  }

  fn tokenizer(&self) -> Option<Arc<tokenizers::Tokenizer>> {
    Some(self.tokenizer.clone())
  }
}
