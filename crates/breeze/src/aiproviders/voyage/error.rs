use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use thiserror::Error;

/// Voyage AI error codes from their documentation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ErrorCode {
    InvalidAuthentication,
    IncorrectApiKey,
    RateLimitExceeded,
    InvalidRequest,
    ResourceNotFound,
    ServerError,
    PermissionDenied,
    BadGateway,
    ServiceUnavailable,
    GatewayTimeout,
}

impl ErrorCode {
    /// Parse from HTTP status code
    pub fn from_status(status: u16) -> Self {
        match status {
            401 => Self::InvalidAuthentication,
            403 => Self::PermissionDenied,
            404 => Self::ResourceNotFound,
            429 => Self::RateLimitExceeded,
            502 => Self::BadGateway,
            503 => Self::ServiceUnavailable,
            504 => Self::GatewayTimeout,
            500..=599 => Self::ServerError,
            _ => Self::InvalidRequest,
        }
    }

    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            Self::RateLimitExceeded
                | Self::BadGateway
                | Self::ServiceUnavailable
                | Self::GatewayTimeout
                | Self::ServerError
        )
    }
}

/// Voyage AI API errors
#[derive(Error, Debug)]
pub enum Error {
    #[error("API error {status}: {message}")]
    Api {
        status: u16,
        code: ErrorCode,
        message: String,
    },

    #[error("Network error: {0}")]
    Network(#[from] reqwest::Error),

    #[error("Middleware error: {0}")]
    Middleware(#[from] reqwest_middleware::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Rate limit error: {0}")]
    RateLimit(String),

    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl Error {
    /// Check if this error is retryable
    pub fn is_retryable(&self) -> bool {
        match self {
            Self::Api { code, .. } => code.is_retryable(),
            Self::Network(e) => {
                // Network timeouts and connection errors are retryable
                e.is_timeout() || e.is_connect()
            }
            Self::Middleware(e) => {
                // Middleware errors can contain network errors
                if let Some(source) = e.source() {
                    // Try to downcast to reqwest error
                    if let Some(req_err) = source.downcast_ref::<reqwest::Error>() {
                        req_err.is_timeout() || req_err.is_connect()
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
            Self::RateLimit(_) => true,
            _ => false,
        }
    }
}

/// API error response structure
#[derive(Debug, Deserialize)]
pub struct ApiErrorResponse {
    pub error: ApiErrorDetail,
}

#[derive(Debug, Deserialize)]
pub struct ApiErrorDetail {
    pub message: String,
    pub r#type: Option<String>,
    pub code: Option<String>,
}