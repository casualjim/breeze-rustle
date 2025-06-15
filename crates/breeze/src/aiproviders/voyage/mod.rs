mod client;
pub mod error;
mod middleware;
pub mod models;
pub mod types;

use async_trait::async_trait;
pub use error::{Error, ErrorCode};
pub use models::{EmbeddingModel, RerankingModel};
pub use types::{EmbeddingResult, FileContent, Tier};

// Re-export only what's needed for the public API
pub use client::{Config, EmbeddingRequest, EmbeddingResponse, RerankRequest, RerankResponse};

/// Create a new Voyage AI client
pub fn new_client(config: Config) -> Result<impl VoyageClient, anyhow::Error> {
  client::Client::new(config)
}

/// Trait for Voyage AI client operations
#[async_trait]
pub trait VoyageClient: Send + Sync {
  async fn embed(
    &self,
    request: EmbeddingRequest,
    estimated_tokens: u32,
  ) -> Result<EmbeddingResponse, Error>;
  async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, Error>;
}

#[async_trait]
impl VoyageClient for client::Client {
  async fn embed(
    &self,
    request: EmbeddingRequest,
    estimated_tokens: u32,
  ) -> Result<EmbeddingResponse, Error> {
    self.embed(request, estimated_tokens).await
  }

  async fn rerank(&self, request: RerankRequest) -> Result<RerankResponse, Error> {
    self.rerank(request).await
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_voyage_config() {
    let config = Config::new(
      "test_key".to_string(),
      Tier::Free,
      EmbeddingModel::VoyageCode3,
    )
    .with_timeout(std::time::Duration::from_secs(60));

    assert_eq!(config.api_key, "test_key");
    assert_eq!(config.tier, Tier::Free);
    assert_eq!(config.model, EmbeddingModel::VoyageCode3);
  }

  #[test]
  fn test_tier_limits() {
    // Test tier limits with model-specific values
    let model = EmbeddingModel::VoyageCode3;
    assert_eq!(Tier::Free.tokens_per_minute(model), 3_000_000);
    assert_eq!(Tier::Free.safe_tokens_per_minute(model), 2_700_000); // 90% of 3M

    assert_eq!(Tier::Tier2.requests_per_minute(), 4_000);
    assert_eq!(Tier::Tier2.safe_requests_per_minute(), 3_600); // 90% of 4k
  }

  #[tokio::test]
  async fn test_voyage_client_real_api() {
    use client::Client;

    // This test requires VOYAGE_API_KEY to be set
    let api_key = match std::env::var("VOYAGE_API_KEY") {
      Ok(key) => key,
      Err(_) => {
        eprintln!("Skipping Voyage API test - VOYAGE_API_KEY not set");
        return;
      }
    };

    let config = Config::new(api_key, Tier::Free, EmbeddingModel::VoyageCode3);
    let client = Client::new(config).unwrap();

    let request = EmbeddingRequest {
      input: vec!["Hello, world!".to_string()],
      model: "voyage-code-3".to_string(),
      input_type: Some("document".to_string()),
      output_dimension: None,
      truncation: None,
    };

    let result = client.embed(request, 3).await.unwrap(); // ~3 tokens

    assert_eq!(result.data.len(), 1);
    assert_eq!(result.data[0].embedding.len(), 1024); // voyage-code-3 default is 1024 dimensions
  }
}
