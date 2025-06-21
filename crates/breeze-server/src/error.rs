use axum::{
  Json,
  http::StatusCode,
  response::{IntoResponse, Response},
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::error::Error as StdError;
use uuid::Uuid;

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
pub struct ErrorResponse {
  pub error: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub code: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
  #[error("Project not found: {0}")]
  ProjectNotFound(Uuid),

  #[error("Task not found: {0}")]
  TaskNotFound(Uuid),

  #[error("File not found: {0}")]
  FileNotFound(String),

  #[error("Internal error: {0}")]
  Internal(String),
}

impl IntoResponse for ApiError {
  fn into_response(self) -> Response {
    let (status, code) = match &self {
      ApiError::ProjectNotFound(_) => (StatusCode::NOT_FOUND, "PROJECT_NOT_FOUND"),
      ApiError::TaskNotFound(_) => (StatusCode::NOT_FOUND, "TASK_NOT_FOUND"),
      ApiError::FileNotFound(_) => (StatusCode::NOT_FOUND, "FILE_NOT_FOUND"),
      ApiError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "INTERNAL_ERROR"),
    };

    let body = Json(ErrorResponse {
      error: self.to_string(),
      code: Some(code.to_string()),
    });

    (status, body).into_response()
  }
}

// Convenience conversions
impl From<breeze_indexer::IndexerError> for ApiError {
  fn from(err: breeze_indexer::IndexerError) -> Self {
    ApiError::Internal(err.to_string())
  }
}

impl From<std::io::Error> for ApiError {
  fn from(err: std::io::Error) -> Self {
    match err.kind() {
      std::io::ErrorKind::NotFound => ApiError::FileNotFound(err.to_string()),
      _ => ApiError::Internal(err.to_string()),
    }
  }
}

impl From<Box<dyn StdError + Send + Sync>> for ApiError {
  fn from(err: Box<dyn StdError + Send + Sync>) -> Self {
    ApiError::Internal(err.to_string())
  }
}
