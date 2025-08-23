use std::time::Duration;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use tracing::debug;

use super::error::{ApiErrorResponse, Error, ErrorCode};
use super::models::EmbeddingModel;
use super::types::Tier;
use crate::reqwestx::api_client::{ApiClient, ApiClientConfig};

/// Voyage API base URL
const API_BASE: &str = "https://api.voyageai.com/v1";

/// Request payload for embeddings API
#[derive(Debug, Serialize)]
pub struct EmbeddingRequest {
  pub input: Vec<String>,
  pub model: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub input_type: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub output_dimension: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub truncation: Option<bool>,
}

/// Response from embeddings API
#[derive(Debug, Deserialize)]
pub struct EmbeddingResponse {
  pub object: String,
  pub data: Vec<EmbeddingData>,
  pub model: String,
  pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct EmbeddingData {
  pub object: String,
  pub embedding: Vec<f32>,
  pub index: usize,
}

#[derive(Debug, Deserialize)]
pub struct Usage {
  pub total_tokens: usize,
}

/// Request payload for reranking API
#[derive(Debug, Serialize)]
pub struct RerankRequest {
  pub query: String,
  pub documents: Vec<String>,
  pub model: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub top_k: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub truncation: Option<bool>,
}

/// Response from reranking API
#[derive(Debug, Deserialize)]
pub struct RerankResponse {
  pub object: String,
  pub data: Vec<RerankData>,
  pub model: String,
  pub usage: Usage,
}

#[derive(Debug, Deserialize)]
pub struct RerankData {
  pub index: usize,
  pub relevance_score: f32,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub document: Option<String>,
}

/// Configuration for Voyage client
#[derive(Debug, Clone)]
pub struct Config {
  pub api_key: String,
  pub tier: Tier,
  pub model: EmbeddingModel,
  pub timeout: Duration,
  pub max_concurrent_requests: Option<usize>,
}

impl Config {
  /// Create a new configuration
  pub fn new(api_key: String, tier: Tier, model: EmbeddingModel) -> Self {
    Self {
      api_key,
      tier,
      model,
      timeout: Duration::from_secs(90),
      max_concurrent_requests: None,
    }
  }

  /// Set custom timeout
  pub fn with_timeout(mut self, timeout: Duration) -> Self {
    self.timeout = timeout;
    self
  }

  /// Set max concurrent requests (e.g., based on number of embed workers)
  pub fn with_max_concurrent_requests(mut self, max_concurrent_requests: usize) -> Self {
    self.max_concurrent_requests = Some(max_concurrent_requests);
    self
  }
}

/// Voyage AI API client
pub struct Client {
  client: ApiClient,
}

impl Client {
  /// Create a new Voyage client with rate limiting
  pub fn new(config: Config) -> Result<Self> {
    let api_config = ApiClientConfig {
      base_url: API_BASE.to_string(),
      api_key: Some(config.api_key.clone()),
      max_concurrent_requests: config.max_concurrent_requests.unwrap_or(50),
      max_requests_per_minute: config.tier.safe_requests_per_minute() as usize,
      max_tokens_per_minute: config.tier.safe_tokens_per_minute(config.model) as usize,
      max_retries: 3,
      timeout: config.timeout,
    };

    let client = ApiClient::new(api_config).context("Failed to create API client")?;

    Ok(Self { client })
  }

  /// Generate embeddings for the given texts
  pub async fn embed(
    &self,
    request: EmbeddingRequest,
    estimated_tokens: u32,
  ) -> Result<EmbeddingResponse, Error> {
    debug!(
      "Embedding {} texts with ~{} tokens",
      request.input.len(),
      estimated_tokens
    );

    self
      .client
      .post_json("/embeddings", &request, estimated_tokens)
      .await
      .map_err(|e| {
        let error_str = e.to_string();

        // Try to extract HTTP status code from error message
        if let Some(status_match) = error_str.split("status ").nth(1)
          && let Some(status_str) = status_match.split(':').next()
          && let Ok(status) = status_str.parse::<u16>()
        {
          // Try to parse the error body
          if let Some(api_error) = error_str.split(": ").last()
            && let Ok(api_err) = serde_json::from_str::<ApiErrorResponse>(api_error)
          {
            let code = ErrorCode::from_status(status);
            return Error::Api {
              status,
              code,
              message: api_err.error.message,
            };
          }

          // Return generic API error for known status codes
          let code = ErrorCode::from_status(status);
          return Error::Api {
            status,
            code,
            message: error_str,
          };
        }

        // Convert to anyhow error
        Error::Other(anyhow::Error::msg(error_str))
      })
  }

  /// Rerank documents based on relevance to a query
  pub async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, Error> {
    debug!("Reranking {} documents", request.documents.len());

    // Reranking typically uses minimal tokens
    self
      .client
      .post_json("/rerank", &request, 100)
      .await
      .map_err(|e| {
        let error_str = e.to_string();

        // Try to extract HTTP status code from error message
        if let Some(status_match) = error_str.split("status ").nth(1)
          && let Some(status_str) = status_match.split(':').next()
          && let Ok(status) = status_str.parse::<u16>()
        {
          // Try to parse the error body
          if let Some(api_error) = error_str.split(": ").last()
            && let Ok(api_err) = serde_json::from_str::<ApiErrorResponse>(api_error)
          {
            let code = ErrorCode::from_status(status);
            return Error::Api {
              status,
              code,
              message: api_err.error.message,
            };
          }

          // Return generic API error for known status codes
          let code = ErrorCode::from_status(status);
          return Error::Api {
            status,
            code,
            message: error_str,
          };
        }

        // Convert to anyhow error
        Error::Other(anyhow::Error::msg(error_str))
      })
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_config_creation() {
    let config = Config::new(
      "test_key".to_string(),
      Tier::Free,
      EmbeddingModel::VoyageCode3,
    )
    .with_timeout(Duration::from_secs(60));

    assert_eq!(config.api_key, "test_key");
    assert_eq!(config.tier, Tier::Free);
    assert_eq!(config.timeout, Duration::from_secs(60));
  }

  #[tokio::test]
  async fn test_client_creation() {
    let config = Config::new(
      "test_key".to_string(),
      Tier::Free,
      EmbeddingModel::VoyageCode3,
    );
    let client = Client::new(config);
    assert!(client.is_ok());
  }
}
