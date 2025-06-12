use candle_core::{backend::BackendDevice, Device};
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

  /// Device to run model on ("cpu", "mps", "cuda")
  #[serde(default = "default_device")]
  pub device: String,

  /// Maximum chunk size in tokens
  #[serde(default = "default_max_chunk_size")]
  pub max_chunk_size: usize,

  /// Maximum file size in bytes
  #[serde(default = "default_max_file_size")]
  pub max_file_size: Option<u64>,

  /// Number of files to process in parallel
  #[serde(default = "default_max_parallel_files")]
  pub max_parallel_files: usize,
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

fn default_device() -> String {
  if cfg!(feature = "cuda") {
    return "cuda".to_string()
  } else if cfg!(target_os = "macos") && cfg!(feature = "metal") {
    // Use MPS on macOS with Metal feature enabled
    return "mps".to_string()
  }
  // Default to CPU on macOS without Metal
  "cpu".to_string()
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

impl Default for Config {
  fn default() -> Self {
    Self {
      database_path: default_database_path(),
      table_name: default_table_name(),
      model: default_model(),
      device: default_device(),
      max_chunk_size: default_max_chunk_size(),
      max_file_size: default_max_file_size(),
      max_parallel_files: default_max_parallel_files(),
    }
  }
}

impl Config {
  pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> anyhow::Result<Self> {
    let content = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&content)?)
  }

  pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> anyhow::Result<()> {
    let content = toml::to_string_pretty(self)?;
    std::fs::write(path, content)?;
    Ok(())
  }

  pub fn get_device(&self, idx: usize) -> Device {
    match self.device.as_str()  {
      "cpu" => Device::Cpu,
      "mps" => Device::Metal(candle_core::MetalDevice::new(idx).expect("Failed to create MPS device")),
      "cuda" => Device::Cuda(candle_core::CudaDevice::new(idx).expect("Failed to create CUDA device")),
      _ => panic!("Unsupported device type: {}", self.device),
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
