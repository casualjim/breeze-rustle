use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;
use tiktoken_rs::{CoreBPE, cl100k_base, o200k_base, p50k_base, r50k_base};
use tokenizers::Tokenizer;
use tracing::debug;

use super::{
  EmbeddingInput, EmbeddingProvider,
  batching::{BatchingStrategy, TokenAwareBatchingStrategy},
};
use crate::config::OpenAILikeConfig;
use crate::reqwestx::api_client::{ApiClient, ApiClientConfig};

#[derive(Debug, Serialize)]
struct EmbeddingRequest {
  input: Vec<String>,
  model: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  encoding_format: Option<String>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
  data: Vec<EmbeddingData>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingData {
  index: usize,
  embedding: Vec<f32>,
}

/// Tokenizer wrapper that supports both tiktoken and HuggingFace
enum TokenizerWrapper {
  Tiktoken(CoreBPE),
  HuggingFace(Arc<Tokenizer>),
}

impl TokenizerWrapper {
  /// Count tokens in text
  fn count_tokens(&self, text: &str) -> Result<usize, tokenizers::Error> {
    match self {
      Self::Tiktoken(encoder) => Ok(encoder.encode_ordinary(text).len()),
      Self::HuggingFace(tokenizer) => tokenizer.encode(text, false).map(|encoding| encoding.len()),
    }
  }
}

/// OpenAI-compatible embedding provider
pub struct OpenAILikeEmbeddingProvider {
  client: ApiClient,
  model: String,
  tokenizer: TokenizerWrapper,
  embedding_dim: usize,
  context_length: usize,
  max_batch_size: usize,
}

impl OpenAILikeEmbeddingProvider {
  pub async fn new(config: &OpenAILikeConfig) -> super::EmbeddingResult<Self> {
    // Create API client configuration
    let api_config = ApiClientConfig {
      base_url: config.api_base.clone(),
      api_key: config.api_key.clone(),
      max_concurrent_requests: config.max_concurrent_requests,
      max_requests_per_minute: config.requests_per_minute as usize,
      max_tokens_per_minute: config.tokens_per_minute as usize,
      max_retries: 3,
      timeout: Duration::from_secs(30),
    };

    let client = ApiClient::new(api_config)?;

    // Load tokenizer based on config
    let tokenizer = match &config.tokenizer {
      breeze_chunkers::Tokenizer::Tiktoken(encoding) => {
        let encoder = match encoding.as_str() {
          "cl100k_base" => cl100k_base()?,
          "p50k_base" => p50k_base()?,
          "r50k_base" => r50k_base()?,
          "o200k_base" => o200k_base()?,
          _ => {
            return Err(super::EmbeddingError::InvalidConfig(format!(
              "Unknown tiktoken encoding: {}",
              encoding
            )));
          }
        };
        TokenizerWrapper::Tiktoken(encoder)
      }
      breeze_chunkers::Tokenizer::PreloadedTiktoken(tiktoken) => TokenizerWrapper::Tiktoken(
        std::sync::Arc::try_unwrap(tiktoken.clone()).unwrap_or_else(|arc| (*arc).clone()),
      ),
      breeze_chunkers::Tokenizer::HuggingFace(model_id) => {
        let tokenizer = Tokenizer::from_pretrained(model_id, None).map_err(|e| {
          super::EmbeddingError::ModelLoadFailed(format!(
            "Failed to load tokenizer {}: {}",
            model_id, e
          ))
        })?;
        TokenizerWrapper::HuggingFace(Arc::new(tokenizer))
      }
      breeze_chunkers::Tokenizer::PreloadedHuggingFace(tokenizer) => {
        TokenizerWrapper::HuggingFace(tokenizer.clone())
      }
      breeze_chunkers::Tokenizer::Characters => {
        return Err(super::EmbeddingError::InvalidConfig(
          "Character tokenizer not supported for embeddings API".to_string(),
        ));
      }
    };

    Ok(Self {
      client,
      model: config.model.clone(),
      tokenizer,
      embedding_dim: config.embedding_dim,
      context_length: config.context_length,
      max_batch_size: config.max_batch_size,
    })
  }
}

#[async_trait]
impl EmbeddingProvider for OpenAILikeEmbeddingProvider {
  async fn embed(&self, inputs: &[EmbeddingInput<'_>]) -> super::EmbeddingResult<Vec<Vec<f32>>> {
    let request = EmbeddingRequest {
      input: inputs.iter().map(|input| input.text.to_string()).collect(),
      model: self.model.clone(),
      encoding_format: Some("float".to_string()),
    };

    // Calculate total tokens for rate limiting
    let estimated_tokens: u32 = inputs
      .iter()
      .map(|input| {
        if let Some(count) = input.token_count {
          count as u32
        } else {
          // Count tokens using our tokenizer
          self
            .tokenizer
            .count_tokens(input.text)
            .unwrap_or_else(|_| (input.text.len() / 4)) as u32
        }
      })
      .sum();

    debug!(
      "Embedding {} texts with ~{} tokens",
      inputs.len(),
      estimated_tokens
    );

    // Make request with token count
    let embedding_response: EmbeddingResponse = self
      .client
      .post_json("/embeddings", &request, estimated_tokens)
      .await
      .map_err(|e| super::EmbeddingError::ApiError(e.to_string()))?;

    // Extract embeddings in order
    let mut embeddings = vec![vec![]; embedding_response.data.len()];
    for data in embedding_response.data {
      if data.index < embeddings.len() {
        embeddings[data.index] = data.embedding;
      }
    }

    Ok(embeddings)
  }

  fn embedding_dim(&self) -> usize {
    self.embedding_dim
  }

  fn context_length(&self) -> usize {
    self.context_length
  }

  fn create_batching_strategy(&self) -> Box<dyn BatchingStrategy> {
    // Use configured max_batch_size and context_length
    // The context_length already represents the model's token limit
    Box::new(TokenAwareBatchingStrategy::new(
      self.context_length,
      self.max_batch_size,
    ))
  }

  fn tokenizer(&self) -> Option<Arc<tokenizers::Tokenizer>> {
    // Return HuggingFace tokenizer if available
    match &self.tokenizer {
      TokenizerWrapper::HuggingFace(tokenizer) => Some(tokenizer.clone()),
      TokenizerWrapper::Tiktoken(_) => None, // Tiktoken doesn't implement the trait
    }
  }

  fn is_remote(&self) -> bool {
    true
  }
}
