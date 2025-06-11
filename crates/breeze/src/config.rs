use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use anyhow::{Context, Result};

use crate::embeddings::EmbeddingError;
// TODO: Import these once providers are implemented
// use crate::embeddings::providers::{HttpEmbeddingProvider, LocalEmbeddingProvider};
// use crate::embeddings::providers::http::HttpProviderConfig;
// use crate::embeddings::{EmbeddingProvider};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub repository: RepositoryConfig,
    pub embedding: EmbeddingConfig,
    pub storage: StorageConfig,
    pub processing: ProcessingConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RepositoryConfig {
    pub path: PathBuf,
    pub max_file_size: Option<u64>, // In bytes
    pub exclude_patterns: Vec<String>,
    pub include_patterns: Option<Vec<String>>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingConfig {
    pub provider: ProviderConfig,
    pub chunking: ChunkingConfig,
    pub processing: EmbeddingProcessingConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ProviderConfig {
    OpenAI {
        model: String,
        api_key: Option<String>, // If None, read from env
    },
    Voyage {
        model: String,
        api_key: Option<String>, // If None, read from env
    },
    Local {
        model: String,
        device: String, // "cpu", "mps", "cuda"
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChunkingConfig {
    pub max_chunk_size: usize,
    pub chunk_overlap: f32, // 0.0 to 1.0
    pub use_semantic_chunking: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingProcessingConfig {
    pub max_concurrent_requests: usize,
    pub retry_attempts: usize,
    pub batch_timeout_seconds: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StorageConfig {
    pub database_path: PathBuf,
    pub table_name: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProcessingConfig {
    pub max_parallel_files: usize,
    pub progress_report_interval: usize, // Report progress every N files
}

impl Default for Config {
    fn default() -> Self {
        Self {
            repository: RepositoryConfig {
                path: PathBuf::from("."),
                max_file_size: Some(5 * 1024 * 1024), // 5MB
                exclude_patterns: vec![
                    "*.git*".to_string(),
                    "node_modules".to_string(),
                    "target".to_string(),
                    "build".to_string(),
                    "dist".to_string(),
                    "*.pyc".to_string(),
                    "*.so".to_string(),
                    "*.dylib".to_string(),
                    "*.dll".to_string(),
                ],
                include_patterns: None,
            },
            embedding: EmbeddingConfig {
                provider: ProviderConfig::OpenAI {
                    model: "text-embedding-3-small".to_string(),
                    api_key: None,
                },
                chunking: ChunkingConfig {
                    max_chunk_size: 2048,
                    chunk_overlap: 0.1,
                    use_semantic_chunking: true,
                },
                processing: EmbeddingProcessingConfig {
                    max_concurrent_requests: 10,
                    retry_attempts: 3,
                    batch_timeout_seconds: 5,
                },
            },
            storage: StorageConfig {
                database_path: PathBuf::from("./embeddings.db"),
                table_name: "code_embeddings".to_string(),
            },
            processing: ProcessingConfig {
                max_parallel_files: 16,
                progress_report_interval: 100,
            },
        }
    }
}

impl Config {
    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .context("Failed to read config file")?;
        
        toml::from_str(&content)
            .context("Failed to parse config file")
    }
    
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let content = toml::to_string_pretty(self)
            .context("Failed to serialize config")?;
        
        std::fs::write(path, content)
            .context("Failed to write config file")
    }
    
    // TODO: Implement once providers are ready
    // /// Create an embedding provider from the configuration
    // pub async fn create_embedding_provider(&self) -> Result<Box<dyn EmbeddingProvider>, EmbeddingError> {
    //     match &self.embedding.provider {
    //         ProviderConfig::OpenAI { model, api_key } => {
    //             let api_key = api_key.clone()
    //                 .or_else(|| std::env::var("OPENAI_API_KEY").ok())
    //                 .ok_or_else(|| EmbeddingError::ConfigError(
    //                     "OpenAI API key not found in config or OPENAI_API_KEY environment variable".to_string()
    //                 ))?;
    //             
    //             let config = HttpProviderConfig::openai(model, api_key);
    //             let provider = HttpEmbeddingProvider::new(config)?;
    //             Ok(Box::new(provider))
    //         }
    //         
    //         ProviderConfig::Voyage { model, api_key } => {
    //             let api_key = api_key.clone()
    //                 .or_else(|| std::env::var("VOYAGE_API_KEY").ok())
    //                 .ok_or_else(|| EmbeddingError::ConfigError(
    //                     "Voyage API key not found in config or VOYAGE_API_KEY environment variable".to_string()
    //                 ))?;
    //             
    //             let config = HttpProviderConfig::voyage(model, api_key);
    //             let provider = HttpEmbeddingProvider::new(config)?;
    //             Ok(Box::new(provider))
    //         }
    //         
    //         ProviderConfig::Local { model, device } => {
    //             let provider = LocalEmbeddingProvider::new(model.clone(), device.clone())?;
    //             Ok(Box::new(provider))
    //         }
    //     }
    // }
    
    /// Validate the configuration
    pub fn validate(&self) -> Result<(), EmbeddingError> {
        // Check repository path exists
        if !self.repository.path.exists() {
            return Err(EmbeddingError::ConfigError(
                format!("Repository path does not exist: {}", self.repository.path.display())
            ));
        }
        
        // Validate chunking parameters
        if self.embedding.chunking.chunk_overlap < 0.0 || self.embedding.chunking.chunk_overlap > 1.0 {
            return Err(EmbeddingError::ConfigError(
                "Chunk overlap must be between 0.0 and 1.0".to_string()
            ));
        }
        
        if self.embedding.chunking.max_chunk_size == 0 {
            return Err(EmbeddingError::ConfigError(
                "Max chunk size must be greater than 0".to_string()
            ));
        }
        
        // Validate processing parameters
        if self.embedding.processing.max_concurrent_requests == 0 {
            return Err(EmbeddingError::ConfigError(
                "Max concurrent requests must be greater than 0".to_string()
            ));
        }
        
        if self.processing.max_parallel_files == 0 {
            return Err(EmbeddingError::ConfigError(
                "Max parallel files must be greater than 0".to_string()
            ));
        }
        
        Ok(())
    }
}

/// CLI configuration that can override file-based config
#[derive(Debug, Clone)]
pub struct CliConfig {
    pub repository_path: Option<PathBuf>,
    pub provider_override: Option<ProviderConfig>,
    pub output_path: Option<PathBuf>,
    pub verbose: bool,
    pub dry_run: bool,
}

impl CliConfig {
    pub fn merge_with_config(&self, mut config: Config) -> Config {
        if let Some(repo_path) = &self.repository_path {
            config.repository.path = repo_path.clone();
        }
        
        if let Some(provider) = &self.provider_override {
            config.embedding.provider = provider.clone();
        }
        
        if let Some(output_path) = &self.output_path {
            config.storage.database_path = output_path.clone();
        }
        
        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert!(config.validate().is_ok());
    }
    
    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml_str = toml::to_string(&config).unwrap();
        let parsed: Config = toml::from_str(&toml_str).unwrap();
        
        // Check some key fields
        assert_eq!(config.embedding.chunking.max_chunk_size, parsed.embedding.chunking.max_chunk_size);
        assert_eq!(config.storage.table_name, parsed.storage.table_name);
    }
    
    #[test]
    fn test_config_file_operations() {
        let dir = tempdir().unwrap();
        let config_path = dir.path().join("test_config.toml");
        
        let config = Config::default();
        config.save_to_file(&config_path).unwrap();
        
        let loaded = Config::from_file(&config_path).unwrap();
        assert_eq!(config.storage.table_name, loaded.storage.table_name);
    }
    
    #[test]
    fn test_config_validation() {
        let mut config = Config::default();
        
        // Valid config should pass
        assert!(config.validate().is_ok());
        
        // Invalid chunk overlap
        config.embedding.chunking.chunk_overlap = 1.5;
        assert!(config.validate().is_err());
        
        config.embedding.chunking.chunk_overlap = 0.1;
        
        // Invalid chunk size
        config.embedding.chunking.max_chunk_size = 0;
        assert!(config.validate().is_err());
    }
}