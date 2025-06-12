pub mod loader;
pub mod models;
pub mod sentence_transformer;

use std::fmt;

#[derive(Debug)]
pub enum EmbeddingError {
  ModelLoadError(String),
  InferenceError(String),
  ApiError(String),
  ConfigError(String),
}

impl fmt::Display for EmbeddingError {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      Self::ModelLoadError(msg) => write!(f, "Model load error: {}", msg),
      Self::InferenceError(msg) => write!(f, "Inference error: {}", msg),
      Self::ApiError(msg) => write!(f, "API error: {}", msg),
      Self::ConfigError(msg) => write!(f, "Config error: {}", msg),
    }
  }
}

impl std::error::Error for EmbeddingError {}
