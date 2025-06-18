use serde::{Deserialize, Serialize};
use std::path::PathBuf;

// Re-export for convenience
pub use breeze_indexer::EmbeddingProvider;

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct Config {
  /// Indexer configuration (embedding provider, model, etc.)
  #[serde(flatten)]
  pub indexer: breeze_indexer::Config,
}

pub fn default_max_chunk_size() -> usize {
  512
}

impl Config {
  /// Get the default config directory path
  pub fn default_config_dir() -> anyhow::Result<PathBuf> {
    let config_dir = dirs::preference_dir()
      .ok_or_else(|| anyhow::anyhow!("Could not determine preferences directory"))?
      .join(env!("CARGO_PKG_NAME"));
    Ok(config_dir)
  }

  /// Get the default config file path
  pub fn default_config_path() -> anyhow::Result<PathBuf> {
    Ok(Self::default_config_dir()?.join("config.toml"))
  }

  /// Get the default data directory path
  pub fn default_data_dir() -> anyhow::Result<PathBuf> {
    let data_dir = dirs::data_dir()
      .ok_or_else(|| anyhow::anyhow!("Could not determine data directory"))?
      .join(env!("CARGO_PKG_NAME"));
    Ok(data_dir)
  }

  pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self> {
    let content = std::fs::read_to_string(path)?;
    let mut config: Self = toml::from_str(&content)?;

    // Apply environment variable overrides for API keys
    config.apply_env_overrides();

    Ok(config)
  }

  /// Apply environment variable overrides for API keys and other sensitive settings
  pub fn apply_env_overrides(&mut self) {
    // Handle Voyage API key from environment
    if self.indexer.embedding_provider == EmbeddingProvider::Voyage {
      if let Some(ref mut voyage) = self.indexer.voyage {
        if voyage.api_key.is_empty() {
          if let Ok(api_key) =
            std::env::var("VOYAGE_API_KEY").or_else(|_| std::env::var("BREEZE_VOYAGE_API_KEY"))
          {
            voyage.api_key = api_key;
          }
        }
      }
    }

    // Handle OpenAI-like API keys from environment
    if matches!(
      self.indexer.embedding_provider,
      EmbeddingProvider::OpenAILike(_)
    ) {
      for (provider_name, provider_config) in &mut self.indexer.openai_providers {
        if provider_config.api_key.is_none() {
          // Try provider-specific env var first, then generic OPENAI_API_KEY
          let env_var_name = format!("{}_API_KEY", provider_name.to_uppercase());
          if let Ok(api_key) =
            std::env::var(&env_var_name).or_else(|_| std::env::var("OPENAI_API_KEY"))
          {
            provider_config.api_key = Some(api_key);
          }
        }
      }
    }
  }

  pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> anyhow::Result<()> {
    let content = toml::to_string_pretty(self)?;
    std::fs::write(path, content)?;
    Ok(())
  }

  /// Load config from default location or return default
  pub fn load_from_default_or_fallback() -> Self {
    let mut config = match Self::default_config_path() {
      Ok(default_path) if default_path.exists() => {
        Self::from_file(&default_path).unwrap_or_else(|_| Self::default())
      }
      _ => Self::default(),
    };

    // Apply env overrides even if using default config
    config.apply_env_overrides();
    config
  }

  /// Generate a default configuration with helpful comments
  pub fn generate_commented_config() -> String {
    format!(
      r#"# Breeze Configuration File
#
# This configuration file controls how breeze indexes and searches your codebase.
# You can override any of these settings via command-line arguments or environment variables.
#
# Command-line arguments take precedence over environment variables, which take precedence
# over this config file.

# Embedding provider to use: "local", "voyage", or "openailike"
# - "local": Uses local ONNX models (no API required, runs on your machine)
# - "voyage": Uses Voyage AI's API (requires API key)
# - "openailike": Uses any OpenAI-compatible API (configurable endpoints)
# Default: "local"
# Environment variable: BREEZE_EMBEDDING_PROVIDER
embedding_provider = "local"

# Model to use for local embeddings
# This should be a HuggingFace model ID that has ONNX exports available
# Popular options:
# - "sentence-transformers/all-MiniLM-L6-v2" (default, fast, good quality)
# - "BAAI/bge-small-en-v1.5" (small, efficient)
# - "thenlper/gte-large" (larger, better quality)
# Default: "sentence-transformers/all-MiniLM-L6-v2"
# Environment variable: BREEZE_MODEL
model = "sentence-transformers/all-MiniLM-L6-v2"

# Maximum chunk size in tokens
# Files are split into chunks no larger than this to fit within embedding model limits
# Smaller chunks = more granular search, but more storage and slower indexing
# Larger chunks = better context, but might miss specific details
# Default: 512
# Environment variable: BREEZE_MAX_CHUNK_SIZE
max_chunk_size = 512

# Maximum file size in bytes to process
# Files larger than this are skipped during indexing
# Set to null to process files of any size
# Default: 5242880 (5MB)
# Environment variable: BREEZE_MAX_FILE_SIZE
max_file_size = 5242880

# Number of files to process in parallel during indexing
# Higher values speed up indexing but use more memory
# Default: number of CPU cores
# Environment variable: BREEZE_MAX_PARALLEL_FILES
max_parallel_files = {}

# Number of threads dedicated to processing large files
# Large files are processed in parallel chunks using this many threads
# Default: 4
large_file_threads = 4

# Number of concurrent workers for remote embedding providers
# Only applies to voyage and openailike providers
# Default: 4
# Environment variable: BREEZE_EMBEDDING_WORKERS
embedding_workers = 4

# ============================================================================
# Voyage AI Configuration
# ============================================================================
# Uncomment and configure this section if using embedding_provider = "voyage"

# [voyage]
# # Your Voyage AI API key
# # Get one at: https://dash.voyageai.com
# # Can also be set via VOYAGE_API_KEY or BREEZE_VOYAGE_API_KEY environment variable
# api_key = "your-api-key-here"
#
# # Subscription tier: "free", "tier1", "tier2", or "tier3"
# # This affects rate limits and chunk size optimization
# # Default: "free"
# # Environment variable: BREEZE_VOYAGE_TIER
# tier = "free"
#
# # Voyage model to use
# # Options:
# # - "voyage-3" (best general-purpose)
# # - "voyage-3-lite" (faster, lower cost)
# # - "voyage-code-3" (optimized for code, recommended)
# # - "voyage-finance-2" (finance documents)
# # - "voyage-law-2" (legal documents)
# # - "voyage-multilingual-2" (multiple languages)
# # Default: "voyage-code-3"
# # Environment variable: BREEZE_VOYAGE_MODEL
# model = "voyage-code-3"

# ============================================================================
# OpenAI-like Provider Configuration
# ============================================================================
# Configure any OpenAI-compatible embedding API (OpenAI, llama.cpp, Ollama, etc.)
# You can define multiple providers and switch between them

# # Example: OpenAI
# [openai_providers.openai]
# api_base = "https://api.openai.com/v1"
# api_key = "sk-..."  # Or set via OPENAI_API_KEY environment variable
# model = "text-embedding-3-small"
# embedding_dim = 1536
# context_length = 8191
# max_batch_size = 2048
# requests_per_minute = 3000
# tokens_per_minute = 1000000
# max_concurrent_requests = 50
#
# [openai_providers.openai.tokenizer]
# type = "tiktoken"
# encoding = "cl100k_base"

# # Example: Local llama.cpp server
# [openai_providers.local]
# api_base = "http://localhost:8080/v1"
# # No API key needed for local server
# model = "nomic-embed-text"
# embedding_dim = 768
# context_length = 8192
# max_batch_size = 512
# requests_per_minute = 10000  # High limit for local server
# tokens_per_minute = 10000000
# max_concurrent_requests = 50
#
# [openai_providers.local.tokenizer]
# type = "huggingface"
# model_id = "nomic-ai/nomic-embed-text-v1.5"

# # Select which provider to use (must match a key in openai_providers)
# # Only needed when embedding_provider = "openailike"
# openai_provider = "openai"
"#,
      num_cpus::get()
    )
  }

  /// Create a config suitable for tests with a small, fast model
  /// Returns both a TempDir (for cleanup) and a Config with database_path inside that tempdir
  #[cfg(test)]
  pub fn test() -> (tempfile::TempDir, Self) {
    let (temp_dir, indexer_config) = breeze_indexer::Config::test();
    let config = Self {
      indexer: indexer_config,
    };
    (temp_dir, config)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::tempdir;

  #[test]
  fn test_default_config() {
    let config = Config::default();
    assert_eq!(
      config.indexer.model,
      "sentence-transformers/all-MiniLM-L6-v2"
    );
    assert_eq!(config.indexer.max_chunk_size, 512);
  }

  #[test]
  fn test_config_serialization() {
    let config = Config::default();
    let toml_str = toml::to_string(&config).unwrap();
    let parsed: Config = toml::from_str(&toml_str).unwrap();

    assert_eq!(config.indexer.model, parsed.indexer.model);
  }

  #[test]
  fn test_config_file_operations() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("test_config.toml");

    let config = Config::default();
    config.save_to_file(&config_path).unwrap();

    let loaded = Config::from_file(&config_path).unwrap();
    assert_eq!(config.indexer.model, loaded.indexer.model);
  }
}
