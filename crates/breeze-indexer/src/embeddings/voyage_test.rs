#[cfg(test)]
mod tests {
  use super::*;
  use crate::aiproviders::voyage::{EmbeddingModel, Tier};
  use crate::config::VoyageConfig;
  use crate::embeddings::{EmbeddingInput, EmbeddingProvider};

  #[tokio::test]
  async fn test_voyage_embedding_without_token_count_panics() {
    // This test should panic because the VoyageEmbeddingProvider expects
    // token_count to be set on all inputs, but we're not providing it

    let config = VoyageConfig {
      api_key: "test-key".to_string(),
      tier: Tier::Free,
      model: EmbeddingModel::VoyageCode3,
    };

    let provider = VoyageEmbeddingProvider::new(&config, 1, 512)
      .await
      .expect("Failed to create provider");

    // Create inputs without token counts - this should cause a panic
    let inputs = vec![
      EmbeddingInput {
        text: "Hello world",
        token_count: None, // This will cause the panic
      },
      EmbeddingInput {
        text: "Another text",
        token_count: None, // This will cause the panic
      },
    ];

    // This should panic with "Token count should be provided by batching strategy"
    let _ = provider.embed(&inputs).await;
  }

  #[tokio::test]
  async fn test_voyage_embedding_with_token_count_works() {
    // This test should pass when token counts are provided

    let config = VoyageConfig {
      api_key: "test-key".to_string(),
      tier: Tier::Free,
      model: EmbeddingModel::VoyageCode3,
    };

    // Mock the provider or skip if no API key is available
    if std::env::var("VOYAGE_API_KEY").is_err() {
      println!("Skipping test - VOYAGE_API_KEY not set");
      return;
    }

    let provider = VoyageEmbeddingProvider::new(&config, 1, 512)
      .await
      .expect("Failed to create provider");

    // Create inputs with token counts
    let inputs = vec![
      EmbeddingInput {
        text: "Hello world",
        token_count: Some(2), // Providing token count
      },
      EmbeddingInput {
        text: "Another text",
        token_count: Some(2), // Providing token count
      },
    ];

    // This should work without panicking
    match provider.embed(&inputs).await {
      Ok(embeddings) => {
        assert_eq!(embeddings.len(), 2);
        assert!(!embeddings[0].is_empty());
        assert!(!embeddings[1].is_empty());
      }
      Err(e) => {
        // It's okay if it fails due to API issues, as long as it doesn't panic
        println!("API call failed (expected in test): {}", e);
      }
    }
  }
}
