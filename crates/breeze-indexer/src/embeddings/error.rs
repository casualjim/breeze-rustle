use std::io;

#[derive(Debug, thiserror::Error)]
pub enum EmbeddingError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Model loading failed: {0}")]
    ModelLoadFailed(String),

    #[error("API error: {0}")]
    ApiError(String),

    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    #[error("Tokenization error: {0}")]
    TokenizationError(String),

    #[error("IO error")]
    Io(#[from] io::Error),

    #[error("HTTP request failed")]
    Http(#[from] reqwest::Error),

    #[error("Context length exceeded: input tokens {input} > max {max}")]
    ContextLengthExceeded { input: usize, max: usize },

    #[error("Batch size exceeded: {0} items")]
    BatchSizeExceeded(usize),

    #[error("Provider not available: {0}")]
    ProviderNotAvailable(String),
}

pub type EmbeddingResult<T> = Result<T, EmbeddingError>;