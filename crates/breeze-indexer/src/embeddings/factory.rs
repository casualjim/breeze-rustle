#[cfg(feature = "local-embeddings")]
use super::local::LocalEmbeddingProvider;
use super::{
  EmbeddingError, EmbeddingResult, EmbeddingProvider, openailike::OpenAILikeEmbeddingProvider, voyage::VoyageEmbeddingProvider,
};
use crate::config::{Config, EmbeddingProvider as EmbeddingProviderType};

/// Create an embedding provider based on configuration
pub async fn create_embedding_provider(
  config: &Config,
) -> EmbeddingResult<Box<dyn EmbeddingProvider>> {
  match &config.embedding_provider {
    EmbeddingProviderType::Local => {
      #[cfg(feature = "local-embeddings")]
      {
        let provider = LocalEmbeddingProvider::new(config.model.clone()).await?;
        Ok(Box::new(provider))
      }
      #[cfg(not(feature = "local-embeddings"))]
      {
        Err(EmbeddingError::ProviderNotAvailable("Local embeddings support not enabled. Enable the 'local-embeddings' feature to use local embedding models.".to_string()))
      }
    }
    EmbeddingProviderType::Voyage => {
      let voyage_config = config
        .voyage
        .as_ref()
        .ok_or_else(|| EmbeddingError::InvalidConfig("Voyage configuration required for Voyage provider".to_string()))?;

      // Check that API key is provided
      if voyage_config.api_key.is_empty() {
        return Err(EmbeddingError::InvalidConfig("Voyage API key is required. Set BREEZE_VOYAGE_API_KEY or VOYAGE_API_KEY environment variable".to_string()));
      }

      let chunk_size = config.optimal_chunk_size();
      let provider =
        VoyageEmbeddingProvider::new(voyage_config, config.embedding_workers, chunk_size).await?;
      Ok(Box::new(provider))
    }
    EmbeddingProviderType::OpenAILike(provider_name) => {
      let openai_config = config.openai_providers.get(provider_name).ok_or_else(|| {
        EmbeddingError::InvalidConfig(format!(
          "OpenAI provider '{}' not found in openai_providers",
          provider_name
        ))
      })?;

      let provider = OpenAILikeEmbeddingProvider::new(openai_config).await?;
      Ok(Box::new(provider))
    }
  }
}
