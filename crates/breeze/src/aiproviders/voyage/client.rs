use std::time::Duration;

use anyhow::{Context, Result};
use reqwest::header::{AUTHORIZATION, CONTENT_TYPE, HeaderMap, HeaderValue};
use reqwest_middleware::{ClientBuilder, ClientWithMiddleware};
use reqwest_retry::{RetryTransientMiddleware, policies::ExponentialBackoff};
use serde::{Deserialize, Serialize};
use tracing::debug;

use super::error::{ApiErrorResponse, Error, ErrorCode};
use super::middleware::{RateLimitMiddleware, RateLimiterConfig, TokenCount};
use super::models::EmbeddingModel;
use super::types::Tier;

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
}

impl Config {
  /// Create a new configuration
  pub fn new(api_key: String, tier: Tier, model: EmbeddingModel) -> Self {
    Self {
      api_key,
      tier,
      model,
      timeout: Duration::from_secs(30),
    }
  }

  /// Set custom timeout
  pub fn with_timeout(mut self, timeout: Duration) -> Self {
    self.timeout = timeout;
    self
  }
}

/// Voyage AI API client
pub struct Client {
  client: ClientWithMiddleware,
}

impl Client {
  /// Create a new Voyage client with rate limiting
  pub fn new(config: Config) -> Result<Self> {
    // Create base reqwest client
    let mut headers = HeaderMap::new();
    headers.insert(
      AUTHORIZATION,
      HeaderValue::from_str(&format!("Bearer {}", config.api_key))
        .context("Invalid API key format")?,
    );
    headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));

    let reqwest_client = reqwest::Client::builder()
      .default_headers(headers)
      .timeout(config.timeout)
      .build()
      .context("Failed to create HTTP client")?;

    // Create retry policy with exponential backoff
    let retry_policy = ExponentialBackoff::builder()
      .base(2)
      .retry_bounds(Duration::from_secs(1), Duration::from_secs(60))
      .build_with_max_retries(3);

    // Build middleware client with rate limiting and retry
    let rate_limiter_config = RateLimiterConfig {
      tier: config.tier,
      model: config.model,
    };

    let client = ClientBuilder::new(reqwest_client)
      .with(RateLimitMiddleware::new(rate_limiter_config))
      .with(RetryTransientMiddleware::new_with_policy(retry_policy))
      .build();

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

    // Create request with token count in extensions
    let response = self
      .client
      .post(format!("{}/embeddings", API_BASE))
      .with_extension(TokenCount(estimated_tokens))
      .json(&request)
      .send()
      .await?;

    self.handle_response(response).await
  }

  /// Rerank documents based on relevance to a query
  pub async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, Error> {
    debug!("Reranking {} documents", request.documents.len());

    let response = self
      .client
      .post(format!("{}/rerank", API_BASE))
      .with_extension(TokenCount(0))
      .json(&request)
      .send()
      .await?;

    self.handle_response(response).await
  }

  /// Handle API response and errors
  async fn handle_response<T: for<'de> Deserialize<'de>>(
    &self,
    response: reqwest::Response,
  ) -> Result<T, Error> {
    if !response.status().is_success() {
      let status = response.status();
      let code = ErrorCode::from_status(status.as_u16());

      let error_text = response.text().await.unwrap_or_default();

      // Try to parse as API error response
      let message = if let Ok(api_error) = serde_json::from_str::<ApiErrorResponse>(&error_text) {
        api_error.error.message
      } else {
        error_text
      };

      return Err(Error::Api {
        status: status.as_u16(),
        code,
        message,
      });
    }

    response.json().await.map_err(Into::into)
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
