use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
  /// LanceDB database path
  #[serde(default = "default_database_path")]
  pub database_path: PathBuf,

  /// Table name in LanceDB
  #[serde(default = "default_table_name")]
  pub table_name: String,

  /// Model to use for embeddings
  #[serde(default = "default_model")]
  pub model: String,

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

fn default_model() -> String {
  "ibm-granite/granite-embedding-125m-english".to_string()
}

fn default_max_chunk_size() -> usize {
  512
}

fn default_max_file_size() -> Option<u64> {
  Some(5 * 1024 * 1024) // 5MB
}

fn default_max_parallel_files() -> usize {
  num_cpus::get() // Use number of available CPU cores
}

fn default_batch_size() -> usize {
  50  // Default batch size for embedding operations
}


impl Default for Config {
  fn default() -> Self {
    let mut config = Self {
      database_path: default_database_path(),
      table_name: default_table_name(),
      model: default_model(),
      max_chunk_size: default_max_chunk_size(),
      max_file_size: default_max_file_size(),
      max_parallel_files: default_max_parallel_files(),
      batch_size: default_batch_size(),
    };
    config.apply_env_overrides();
    config
  }
}

impl Config {
  pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self> {
    let content = std::fs::read_to_string(path)?;
    let mut config: Self = toml::from_str(&content)?;
    config.apply_env_overrides();
    Ok(config)
  }

  /// Apply environment variable overrides to the configuration
  ///
  /// Environment variables:
  /// - BREEZE_DATABASE_PATH: Override database path
  /// - BREEZE_MODEL: Override model name
  /// - BREEZE_MAX_CHUNK_SIZE: Override max chunk size
  /// - BREEZE_MAX_FILE_SIZE: Override max file size in bytes
  /// - BREEZE_MAX_PARALLEL_FILES: Override max parallel files
  /// - BREEZE_BATCH_SIZE: Override batch size for embeddings
  pub fn apply_env_overrides(&mut self) {
    if let Ok(path) = std::env::var("BREEZE_DATABASE_PATH") {
      self.database_path = PathBuf::from(path);
    }

    if let Ok(model) = std::env::var("BREEZE_MODEL") {
      self.model = model;
    }

    if let Ok(size) = std::env::var("BREEZE_MAX_CHUNK_SIZE") {
      if let Ok(parsed) = size.parse::<usize>() {
        self.max_chunk_size = parsed;
      }
    }

    if let Ok(size) = std::env::var("BREEZE_MAX_FILE_SIZE") {
      if let Ok(parsed) = size.parse::<u64>() {
        self.max_file_size = Some(parsed);
      }
    }

    if let Ok(parallel) = std::env::var("BREEZE_MAX_PARALLEL_FILES") {
      if let Ok(parsed) = parallel.parse::<usize>() {
        self.max_parallel_files = parsed;
      }
    }

    if let Ok(batch) = std::env::var("BREEZE_BATCH_SIZE") {
      if let Ok(parsed) = batch.parse::<usize>() {
        self.batch_size = parsed;
      }
    }
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
      model: "sentence-transformers/all-MiniLM-L6-v2".to_string(),
      max_chunk_size: 512,
      max_file_size: Some(1024 * 1024), // 1MB
      max_parallel_files: 2,
      batch_size: 2,  // Small batch size for tests
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
    assert_eq!(config.model, "ibm-granite/granite-embedding-125m-english");
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
