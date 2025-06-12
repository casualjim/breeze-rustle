pub mod loader;
pub mod tei;

use thiserror::Error;

#[derive(Error, Debug)]
pub enum EmbeddingError {
  #[error("Model load error: {0}")]
  ModelLoadError(String),
  #[error("API error: {0}")]
  ApiError(String),
  #[error("Configuration error: {0}")]
  ConfigError(String),
  #[error("Inference error: {0}")]
  InferenceError(String),
}