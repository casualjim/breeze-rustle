use super::{
  EmbeddingProvider, local::LocalEmbeddingProvider, openailike::OpenAILikeEmbeddingProvider,
  voyage::VoyageEmbeddingProvider,
};
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

      let chunk_size = config.optimal_chunk_size();
      let provider =
        VoyageEmbeddingProvider::new(voyage_config, config.embedding_workers, chunk_size).await?;
      Ok(Box::new(provider))
    }
    EmbeddingProviderType::OpenAILike => {
      let provider_name = config
        .openai_provider
        .as_ref()
        .ok_or("OpenAI provider name must be specified with openai_provider")?;

      let openai_config = config.openai_providers.get(provider_name).ok_or_else(|| {
        format!(
          "OpenAI provider '{}' not found in openai_providers",
          provider_name
        )
      })?;

      let provider = OpenAILikeEmbeddingProvider::new(openai_config).await?;
      Ok(Box::new(provider))
    }
  }
}
