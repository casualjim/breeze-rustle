use super::{EmbeddingProvider, local::LocalEmbeddingProvider, voyage::VoyageEmbeddingProvider};
use crate::config::{Config, EmbeddingProvider as EmbeddingProviderType};

/// Create an embedding provider based on configuration
pub async fn create_embedding_provider(
  config: &Config,
) -> Result<Box<dyn EmbeddingProvider>, Box<dyn std::error::Error>> {
  match config.embedding_provider {
    EmbeddingProviderType::Local => {
      let provider = LocalEmbeddingProvider::new(config.model.clone(), config.batch_size).await?;
      Ok(Box::new(provider))
    }
    EmbeddingProviderType::Voyage => {
      let voyage_config = config
        .voyage
        .as_ref()
        .ok_or("Voyage configuration required for Voyage provider")?;

      // Check that API key is provided
      if voyage_config.api_key.is_empty() {
        return Err("Voyage API key is required. Set BREEZE_VOYAGE_API_KEY or VOYAGE_API_KEY environment variable".into());
      }

      let provider = VoyageEmbeddingProvider::new(voyage_config).await?;
      Ok(Box::new(provider))
    }
  }
}
