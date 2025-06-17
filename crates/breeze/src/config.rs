use crate::aiproviders::voyage::{EmbeddingModel as VoyageModel, Tier};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProvider {
  Local,
  Voyage,
  OpenAILike,
}

impl FromStr for EmbeddingProvider {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "local" => Ok(EmbeddingProvider::Local),
      "voyage" => Ok(EmbeddingProvider::Voyage),
      "openailike" | "openai-like" | "openai_like" => Ok(EmbeddingProvider::OpenAILike),
      _ => Err(format!(
        "Invalid embedding provider: {}. Use 'local', 'voyage', or 'openailike'",
        s
      )),
    }
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VoyageConfig {
  /// API key for Voyage AI
  pub api_key: String,

  /// Subscription tier
  #[serde(default = "default_voyage_tier")]
  pub tier: Tier,

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

fn default_max_concurrent_requests() -> usize {
  50
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
  /// LanceDB database path
  #[serde(default = "default_database_path")]
  pub database_path: PathBuf,

  /// Table name in LanceDB
  #[serde(default = "default_table_name")]
  pub table_name: String,

  /// Embedding provider to use
  #[serde(default = "default_embedding_provider")]
  pub embedding_provider: EmbeddingProvider,

  /// Model to use for embeddings (for local provider)
  #[serde(default = "default_model")]
  pub model: String,

  /// Voyage-specific configuration
  #[serde(skip_serializing_if = "Option::is_none")]
  pub voyage: Option<VoyageConfig>,

  /// OpenAI-like API configurations (keyed by provider name)
  #[serde(default, skip_serializing_if = "std::collections::HashMap::is_empty")]
  pub openai_providers: std::collections::HashMap<String, OpenAILikeConfig>,

  /// Selected OpenAI-like provider (must match a key in openai_providers)
  #[serde(skip_serializing_if = "Option::is_none")]
  pub openai_provider: Option<String>,

  /// Maximum chunk size in tokens
  #[serde(default = "default_max_chunk_size")]
  pub max_chunk_size: usize,

  /// Maximum file size in bytes
  #[serde(default = "default_max_file_size")]
  pub max_file_size: Option<u64>,

  /// Number of files to process in parallel
  #[serde(default = "default_max_parallel_files")]
  pub max_parallel_files: usize,

  /// Batch size for embedding operations
  #[serde(default = "default_batch_size")]
  pub batch_size: usize,

  /// Number of threads dedicated to processing large files
  #[serde(default = "default_large_file_threads")]
  pub large_file_threads: Option<usize>,

  /// Number of concurrent embedding workers for remote providers
  #[serde(default = "default_embedding_workers")]
  pub embedding_workers: usize,
}

fn default_database_path() -> PathBuf {
  Config::default_data_dir()
    .unwrap_or_else(|_| PathBuf::from("./data"))
    .join("embeddings.db")
}

fn default_table_name() -> String {
  "code_files".to_string()
}

fn default_embedding_provider() -> EmbeddingProvider {
  EmbeddingProvider::Local
}

fn default_model() -> String {
  "sentence-transformers/all-MiniLM-L6-v2".to_string()
}

fn default_voyage_tier() -> Tier {
  Tier::Free
}

fn default_voyage_model() -> VoyageModel {
  VoyageModel::VoyageCode3
}

pub fn default_max_chunk_size() -> usize {
  512
}

fn default_max_file_size() -> Option<u64> {
  Some(5 * 1024 * 1024) // 5MB
}

fn default_max_parallel_files() -> usize {
  num_cpus::get() // Use number of available CPU cores
}

fn default_batch_size() -> usize {
  50 // Default batch size for embedding operations
}

fn default_large_file_threads() -> Option<usize> {
  Some(4) // Default 4 threads for large files
}

fn default_embedding_workers() -> usize {
  4 // Default 4 workers for remote embedding providers
}

impl Default for Config {
  fn default() -> Self {
    Self {
      database_path: default_database_path(),
      table_name: default_table_name(),
      embedding_provider: default_embedding_provider(),
      model: default_model(),
      voyage: None,
      openai_providers: std::collections::HashMap::new(),
      openai_provider: None,
      max_chunk_size: default_max_chunk_size(),
      max_file_size: default_max_file_size(),
      max_parallel_files: default_max_parallel_files(),
      batch_size: default_batch_size(),
      large_file_threads: default_large_file_threads(),
      embedding_workers: default_embedding_workers(),
    }
  }
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
    let config: Self = toml::from_str(&content)?;
    Ok(config)
  }

  pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> anyhow::Result<()> {
    let content = toml::to_string_pretty(self)?;
    std::fs::write(path, content)?;
    Ok(())
  }

  /// Load config from default location or return default
  pub fn load_from_default_or_fallback() -> Self {
    match Self::default_config_path() {
      Ok(default_path) if default_path.exists() => {
        Self::from_file(&default_path).unwrap_or_else(|_| Self::default())
      }
      _ => Self::default(),
    }
  }

  /// Generate a default configuration with helpful comments
  pub fn generate_commented_config() -> String {
    let data_dir = Self::default_data_dir()
      .unwrap_or_else(|_| PathBuf::from("./data"))
      .display()
      .to_string();

    format!(
      r#"# Breeze Configuration File
#
# This configuration file controls how breeze indexes and searches your codebase.
# You can override any of these settings via command-line arguments or environment variables.
#
# Command-line arguments take precedence over environment variables, which take precedence
# over this config file.

# Path to the LanceDB database where embeddings are stored
# Default: "./embeddings.db"
# Environment variable: BREEZE_DATABASE_PATH
database_path = "{}/embeddings.db"

# Table name in the LanceDB database
# Default: "code_files"
table_name = "code_files"

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

# Batch size for embedding operations
# How many chunks to embed at once (for providers that support batching)
# Default: 50
# Environment variable: BREEZE_BATCH_SIZE
batch_size = 50

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
      data_dir,
      num_cpus::get()
    )
  }

  /// Calculate optimal chunk size based on embedding provider and tier
  pub fn optimal_chunk_size(&self) -> usize {
    match self.embedding_provider {
      EmbeddingProvider::Voyage => {
        if let Some(voyage_config) = &self.voyage {
          // Get model context length
          let context_length = voyage_config.model.context_length();

          // Base chunk size on tier and model
          let chunk_size = match voyage_config.tier {
            Tier::Free => {
              // Conservative for free tier: aim for ~2k tokens per chunk
              // This allows good batching with 128 max batch size
              2048.min(context_length / 8)
            }
            Tier::Tier1 => {
              // Moderate for Tier 1: ~4k tokens per chunk
              4096.min(context_length / 6)
            }
            Tier::Tier2 => {
              // Larger for Tier 2: ~8k tokens per chunk
              8192.min(context_length / 4)
            }
            Tier::Tier3 => {
              // Maximum efficiency for Tier 3: ~16k tokens per chunk
              // But cap at 80% of context length for safety
              16384.min((context_length * 4) / 5)
            }
          };

          // Override with user-specified value if it's not the default
          if self.max_chunk_size != default_max_chunk_size() {
            self.max_chunk_size
          } else {
            chunk_size
          }
        } else {
          self.max_chunk_size
        }
      }
      EmbeddingProvider::Local => {
        // For local models, use the configured value
        self.max_chunk_size
      }
      EmbeddingProvider::OpenAILike => {
        // For OpenAI-like providers, use the configured value
        // The actual limits depend on the specific provider
        self.max_chunk_size
      }
    }
  }

  /// Create a config suitable for tests with a small, fast model
  #[cfg(test)]
  pub fn test() -> Self {
    Self {
      database_path: PathBuf::from("./test_db"),
      table_name: "test_documents".to_string(),
      embedding_provider: EmbeddingProvider::Local,
      model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
      voyage: None,
      openai_providers: std::collections::HashMap::new(),
      openai_provider: None,
      max_chunk_size: 512,
      max_file_size: Some(1024 * 1024), // 1MB
      max_parallel_files: 2,
      batch_size: 2,               // Small batch size for tests
      large_file_threads: Some(2), // Small number for tests
      embedding_workers: 1,        // Small number for tests
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::tempdir;

  #[test]
  fn test_default_config() {
    let config = Config::default();
    assert_eq!(config.model, "sentence-transformers/all-MiniLM-L6-v2");
    assert_eq!(config.max_chunk_size, 512);
  }

  #[test]
  fn test_config_serialization() {
    let config = Config::default();
    let toml_str = toml::to_string(&config).unwrap();
    let parsed: Config = toml::from_str(&toml_str).unwrap();

    assert_eq!(config.model, parsed.model);
    assert_eq!(config.table_name, parsed.table_name);
  }

  #[test]
  fn test_config_file_operations() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("test_config.toml");

    let config = Config::default();
    config.save_to_file(&config_path).unwrap();

    let loaded = Config::from_file(&config_path).unwrap();
    assert_eq!(config.table_name, loaded.table_name);
  }

  #[test]
  fn test_optimal_chunk_size_voyage_tiers() {
    // Test Free tier
    let mut config = Config {
      embedding_provider: EmbeddingProvider::Voyage,
      voyage: Some(VoyageConfig {
        api_key: "test".to_string(),
        tier: Tier::Free,
        model: VoyageModel::VoyageCode3,
      }),
      max_chunk_size: default_max_chunk_size(), // Use default to trigger calculation
      ..Default::default()
    };
    assert_eq!(
      config.optimal_chunk_size(),
      2048,
      "Free tier should use 2k chunks"
    );

    // Test Tier 1
    config.voyage.as_mut().unwrap().tier = Tier::Tier1;
    assert_eq!(
      config.optimal_chunk_size(),
      4096,
      "Tier 1 should use 4k chunks"
    );

    // Test Tier 2
    config.voyage.as_mut().unwrap().tier = Tier::Tier2;
    assert_eq!(
      config.optimal_chunk_size(),
      8000,
      "Tier 2 should use 8k chunks"
    );

    // Test Tier 3
    config.voyage.as_mut().unwrap().tier = Tier::Tier3;
    assert_eq!(
      config.optimal_chunk_size(),
      16384,
      "Tier 3 should use 16k chunks"
    );

    // Test with voyage-law-2 (16k context)
    config.voyage.as_mut().unwrap().model = VoyageModel::VoyageLaw2;
    config.voyage.as_mut().unwrap().tier = Tier::Tier3;
    assert_eq!(
      config.optimal_chunk_size(),
      12800,
      "Tier 3 with 16k model should cap at 80% of context"
    );
  }

  #[test]
  fn test_optimal_chunk_size_user_override() {
    let config = Config {
      embedding_provider: EmbeddingProvider::Voyage,
      voyage: Some(VoyageConfig {
        api_key: "test".to_string(),
        tier: Tier::Free,
        model: VoyageModel::VoyageCode3,
      }),
      max_chunk_size: 1000, // User-specified value
      ..Default::default()
    };
    assert_eq!(
      config.optimal_chunk_size(),
      1000,
      "Should respect user override"
    );
  }

  #[test]
  fn test_optimal_chunk_size_local_provider() {
    let config = Config {
      embedding_provider: EmbeddingProvider::Local,
      max_chunk_size: 1024,
      ..Default::default()
    };
    assert_eq!(
      config.optimal_chunk_size(),
      1024,
      "Local provider should use configured size"
    );
  }
}
