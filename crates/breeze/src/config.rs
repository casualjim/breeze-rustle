use crate::aiproviders::voyage::{EmbeddingModel as VoyageModel, Tier};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::str::FromStr;

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum EmbeddingProvider {
  Local,
  Voyage,
}

impl FromStr for EmbeddingProvider {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "local" => Ok(EmbeddingProvider::Local),
      "voyage" => Ok(EmbeddingProvider::Voyage),
      _ => Err(format!(
        "Invalid embedding provider: {}. Use 'local' or 'voyage'",
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
}

fn default_database_path() -> PathBuf {
  PathBuf::from("./embeddings.db")
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

impl Default for Config {
  fn default() -> Self {
    Self {
      database_path: default_database_path(),
      table_name: default_table_name(),
      embedding_provider: default_embedding_provider(),
      model: default_model(),
      voyage: None,
      max_chunk_size: default_max_chunk_size(),
      max_file_size: default_max_file_size(),
      max_parallel_files: default_max_parallel_files(),
      batch_size: default_batch_size(),
    }
  }
}

impl Config {
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

  /// Create a config suitable for tests with a small, fast model
  #[cfg(test)]
  pub fn test() -> Self {
    Self {
      database_path: PathBuf::from("./test_db"),
      table_name: "test_documents".to_string(),
      embedding_provider: EmbeddingProvider::Local,
      model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
      voyage: None,
      max_chunk_size: 512,
      max_file_size: Some(1024 * 1024), // 1MB
      max_parallel_files: 2,
      batch_size: 2, // Small batch size for tests
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
}
