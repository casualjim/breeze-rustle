use crate::aiproviders::voyage::{EmbeddingModel as VoyageModel, Tier as VoyageTier};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProvider {
  Local,
  Voyage,
  #[serde(rename = "openailike")]
  OpenAILike(String),
}

impl FromStr for EmbeddingProvider {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "local" => Ok(EmbeddingProvider::Local),
      "voyage" => Ok(EmbeddingProvider::Voyage),
      _ => {
        // If it's not local or voyage, assume it's an OpenAI-like provider
        Ok(EmbeddingProvider::OpenAILike(s.to_string()))
      }
    }
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VoyageConfig {
  /// API key for Voyage AI
  pub api_key: String,

  /// Subscription tier
  #[serde(default = "default_voyage_tier")]
  pub tier: VoyageTier,

  /// Model to use
  #[serde(default = "default_voyage_model")]
  pub model: VoyageModel,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAILikeConfig {
  /// API base URL (e.g., "https://api.openai.com/v1" or "http://localhost:8080/v1")
  pub api_base: String,

  /// Optional API key for authentication
  #[serde(skip_serializing_if = "Option::is_none")]
  pub api_key: Option<String>,

  /// Model name to use (e.g., "text-embedding-ada-002")
  pub model: String,

  /// Embedding dimension
  pub embedding_dim: usize,

  /// Maximum context length in tokens
  pub context_length: usize,

  /// Maximum batch size for API calls
  pub max_batch_size: usize,

  /// Tokenizer configuration
  pub tokenizer: breeze_chunkers::Tokenizer,

  /// Rate limiting: requests per minute
  pub requests_per_minute: u32,

  /// Rate limiting: tokens per minute
  pub tokens_per_minute: u32,

  /// Maximum concurrent requests (defaults to 50)
  #[serde(default = "default_max_concurrent_requests")]
  pub max_concurrent_requests: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
  /// LanceDB database path
  #[serde(default = "default_database_path")]
  pub database_path: PathBuf,

  /// Embedding provider to use
  pub embedding_provider: EmbeddingProvider,

  /// Model to use for embeddings (for local provider)
  pub model: String,

  /// Voyage-specific configuration
  #[serde(skip_serializing_if = "Option::is_none")]
  pub voyage: Option<VoyageConfig>,

  /// OpenAI-like API configurations (keyed by provider name)
  #[serde(default)]
  pub openai_providers: std::collections::HashMap<String, OpenAILikeConfig>,

  /// Selected OpenAI-like provider (must match a key in openai_providers)
  #[serde(skip_serializing_if = "Option::is_none")]
  pub openai_provider: Option<String>,

  /// Maximum chunk size in tokens
  pub max_chunk_size: usize,

  /// Maximum file size in bytes
  pub max_file_size: Option<u64>,

  /// Number of files to process in parallel
  pub max_parallel_files: usize,

  /// Number of threads dedicated to processing large files
  pub large_file_threads: Option<usize>,

  /// Number of concurrent embedding workers for remote providers
  pub embedding_workers: usize,
}

impl Default for Config {
  fn default() -> Self {
    Self {
      database_path: default_database_path(),
      embedding_provider: EmbeddingProvider::Local,
      model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
      voyage: None,
      openai_providers: std::collections::HashMap::new(),
      openai_provider: None,
      max_chunk_size: 512,
      max_file_size: Some(5 * 1024 * 1024), // 5MB
      max_parallel_files: num_cpus::get(),
      large_file_threads: Some(4),
      embedding_workers: 4,
    }
  }
}

impl Config {
  /// Calculate optimal chunk size based on embedding provider and tier
  pub fn optimal_chunk_size(&self) -> usize {
    match &self.embedding_provider {
      EmbeddingProvider::Voyage => {
        if let Some(voyage_config) = &self.voyage {
          // Get model context length
          let context_length = voyage_config.model.context_length();

          // Base chunk size on tier and model
          let chunk_size = match voyage_config.tier {
            VoyageTier::Free => {
              // Conservative for free tier: aim for ~2k tokens per chunk
              // This allows good batching with 128 max batch size
              2048.min(context_length / 8)
            }
            VoyageTier::Tier1 => {
              // Moderate for Tier 1: ~4k tokens per chunk
              4096.min(context_length / 6)
            }
            VoyageTier::Tier2 => {
              // Larger for Tier 2: ~8k tokens per chunk
              8192.min(context_length / 4)
            }
            VoyageTier::Tier3 => {
              // Maximum efficiency for Tier 3: ~16k tokens per chunk
              // But cap at 80% of context length for safety
              16384.min((context_length * 4) / 5)
            }
          };

          // Allow user override if explicitly set
          if self.max_chunk_size != 512 {
            // 512 is the default, so if it's different, user set it
            chunk_size.min(self.max_chunk_size)
          } else {
            chunk_size
          }
        } else {
          self.max_chunk_size
        }
      }
      _ => {
        // For local and OpenAI-like providers, use configured max_chunk_size
        self.max_chunk_size
      }
    }
  }

  /// Test configuration for unit tests
  /// Returns both a TempDir (for cleanup) and a Config with database_path inside that tempdir
  #[cfg(any(test, feature = "testing"))]
  pub fn test() -> (tempfile::TempDir, Self) {
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let database_path = temp_dir.path().join("test_embeddings.db");

    let config = Self {
      database_path,
      embedding_provider: EmbeddingProvider::Local,
      model: "test-model".to_string(),
      voyage: None,
      openai_providers: std::collections::HashMap::new(),
      openai_provider: None,
      max_chunk_size: 512,
      max_file_size: Some(10 * 1024 * 1024),
      max_parallel_files: 4,
      large_file_threads: Some(4),
      embedding_workers: 3,
    };

    (temp_dir, config)
  }
}

fn default_database_path() -> PathBuf {
  // Default to ./data/embeddings.db
  PathBuf::from("./data/embeddings.db")
}

fn default_voyage_tier() -> VoyageTier {
  VoyageTier::Free
}

fn default_voyage_model() -> VoyageModel {
  VoyageModel::VoyageCode3
}

fn default_max_concurrent_requests() -> usize {
  50
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_default_config() {
    let (_temp_dir, config) = Config::test();
    assert_eq!(config.model, "test-model");
    assert_eq!(config.max_chunk_size, 512);
    assert_eq!(config.embedding_provider, EmbeddingProvider::Local);
  }

  #[test]
  fn test_config_serialization() {
    let (_temp_dir, config) = Config::test();
    let toml_str = toml::to_string(&config).unwrap();
    let parsed: Config = toml::from_str(&toml_str).unwrap();

    assert_eq!(config.model, parsed.model);
    assert_eq!(config.max_chunk_size, parsed.max_chunk_size);
    assert_eq!(config.embedding_provider, parsed.embedding_provider);
  }

  #[test]
  fn test_optimal_chunk_size_voyage_tiers() {
    // Test Free tier
    let (_temp_dir, test_config) = Config::test();
    let mut config = Config {
      embedding_provider: EmbeddingProvider::Voyage,
      voyage: Some(VoyageConfig {
        api_key: "test".to_string(),
        tier: VoyageTier::Free,
        model: VoyageModel::VoyageCode3,
      }),
      max_chunk_size: 512, // Use default to trigger calculation
      ..test_config
    };
    assert_eq!(
      config.optimal_chunk_size(),
      2048,
      "Free tier should use 2k chunks"
    );

    // Test Tier 1
    config.voyage.as_mut().unwrap().tier = VoyageTier::Tier1;
    assert_eq!(
      config.optimal_chunk_size(),
      4096,
      "Tier 1 should use 4k chunks"
    );

    // Test Tier 2
    config.voyage.as_mut().unwrap().tier = VoyageTier::Tier2;
    assert_eq!(
      config.optimal_chunk_size(),
      8000,
      "Tier 2 should use 8k chunks"
    );

    // Test Tier 3
    config.voyage.as_mut().unwrap().tier = VoyageTier::Tier3;
    assert_eq!(
      config.optimal_chunk_size(),
      16384,
      "Tier 3 should use 16k chunks"
    );

    // Test with voyage-law-2 (16k context)
    config.voyage.as_mut().unwrap().model = VoyageModel::VoyageLaw2;
    config.voyage.as_mut().unwrap().tier = VoyageTier::Tier3;
    assert_eq!(
      config.optimal_chunk_size(),
      12800,
      "Tier 3 with 16k model should cap at 80% of context"
    );
  }

  #[test]
  fn test_optimal_chunk_size_user_override() {
    let (_temp_dir, test_config) = Config::test();
    let config = Config {
      embedding_provider: EmbeddingProvider::Voyage,
      voyage: Some(VoyageConfig {
        api_key: "test".to_string(),
        tier: VoyageTier::Free,
        model: VoyageModel::VoyageCode3,
      }),
      max_chunk_size: 1000, // User-specified value
      ..test_config
    };
    assert_eq!(
      config.optimal_chunk_size(),
      1000,
      "Should respect user override"
    );
  }

  #[test]
  fn test_optimal_chunk_size_local_provider() {
    let (_temp_dir, test_config) = Config::test();
    let config = Config {
      embedding_provider: EmbeddingProvider::Local,
      max_chunk_size: 1024,
      ..test_config
    };
    assert_eq!(
      config.optimal_chunk_size(),
      1024,
      "Local provider should use configured size"
    );
  }

  #[test]
  fn test_optimal_chunk_size_openai_like_provider() {
    let (_temp_dir, test_config) = Config::test();
    let config = Config {
      embedding_provider: EmbeddingProvider::OpenAILike("test".to_string()),
      max_chunk_size: 2048,
      ..test_config
    };
    assert_eq!(
      config.optimal_chunk_size(),
      2048,
      "OpenAI-like provider should use configured size"
    );
  }

  #[test]
  fn test_embedding_provider_from_str() {
    assert_eq!(
      EmbeddingProvider::from_str("local").unwrap(),
      EmbeddingProvider::Local
    );
    assert_eq!(
      EmbeddingProvider::from_str("voyage").unwrap(),
      EmbeddingProvider::Voyage
    );
    assert_eq!(
      EmbeddingProvider::from_str("openai").unwrap(),
      EmbeddingProvider::OpenAILike("openai".to_string())
    );
    assert_eq!(
      EmbeddingProvider::from_str("custom-provider").unwrap(),
      EmbeddingProvider::OpenAILike("custom-provider".to_string())
    );
  }
}
