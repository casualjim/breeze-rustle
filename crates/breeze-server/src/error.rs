use axum::{
  Json,
  http::StatusCode,
  response::{IntoResponse, Response},
};
use breeze_indexer::{EmbeddingError, IndexerError};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
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
  #[error("{0}")]
  IndexerError(#[from] breeze_indexer::IndexerError),
}

impl IntoResponse for ApiError {
  fn into_response(self) -> Response {
    let (status, code) = match &self {
      ApiError::ProjectNotFound(_) => (StatusCode::NOT_FOUND, "PROJECT_NOT_FOUND"),
      ApiError::TaskNotFound(_) => (StatusCode::NOT_FOUND, "TASK_NOT_FOUND"),
      ApiError::IndexerError(err) => match err {
        // 404 Not Found
        IndexerError::ProjectNotFound(_) => (StatusCode::NOT_FOUND, "PROJECT_NOT_FOUND"),

        // 400 Bad Request
        IndexerError::FileOutsideProject { .. } => {
          (StatusCode::BAD_REQUEST, "FILE_OUTSIDE_PROJECT")
        }
        IndexerError::PathNotAbsolute(_) => (StatusCode::BAD_REQUEST, "PATH_NOT_ABSOLUTE"),

        // 422 Unprocessable Entity - validation/configuration errors
        IndexerError::Config(_) => (StatusCode::UNPROCESSABLE_ENTITY, "INVALID_CONFIGURATION"),

        // 409 Conflict
        IndexerError::Task(msg) if msg.contains("already has an active indexing task") => {
          (StatusCode::CONFLICT, "TASK_ALREADY_RUNNING")
        }

        // IO errors
        IndexerError::Io(io_err) => match io_err.kind() {
          std::io::ErrorKind::NotFound => (StatusCode::NOT_FOUND, "FILE_NOT_FOUND"),
          std::io::ErrorKind::PermissionDenied => (StatusCode::FORBIDDEN, "PERMISSION_DENIED"),
          _ => (StatusCode::INTERNAL_SERVER_ERROR, "IO_ERROR"),
        },

        // Embedding errors
        IndexerError::Embedding(embed_err) => match embed_err {
          EmbeddingError::InvalidConfig(_) => {
            (StatusCode::UNPROCESSABLE_ENTITY, "INVALID_EMBEDDING_CONFIG")
          }
          EmbeddingError::ContextLengthExceeded { .. } => {
            (StatusCode::PAYLOAD_TOO_LARGE, "CONTEXT_LENGTH_EXCEEDED")
          }
          EmbeddingError::BatchSizeExceeded(_) => {
            (StatusCode::PAYLOAD_TOO_LARGE, "BATCH_SIZE_EXCEEDED")
          }
          EmbeddingError::ProviderNotAvailable(_) => (
            StatusCode::SERVICE_UNAVAILABLE,
            "EMBEDDING_PROVIDER_UNAVAILABLE",
          ),
          EmbeddingError::ModelNotFound(_) => (StatusCode::NOT_FOUND, "MODEL_NOT_FOUND"),
          EmbeddingError::ModelLoadFailed(_) => {
            (StatusCode::SERVICE_UNAVAILABLE, "MODEL_LOAD_FAILED")
          }
          EmbeddingError::Http(_) => (StatusCode::BAD_GATEWAY, "EXTERNAL_API_ERROR"),
          EmbeddingError::ApiError(_) => (StatusCode::BAD_GATEWAY, "EMBEDDING_API_ERROR"),
          EmbeddingError::OperationNotSupported(_) => {
            (StatusCode::NOT_IMPLEMENTED, "OPERATION_NOT_SUPPORTED")
          }
          _ => (StatusCode::INTERNAL_SERVER_ERROR, "EMBEDDING_ERROR"),
        },

        // True internal errors
        IndexerError::Storage(_) => (StatusCode::INTERNAL_SERVER_ERROR, "STORAGE_ERROR"),
        IndexerError::Chunker(_) => (StatusCode::INTERNAL_SERVER_ERROR, "CHUNKER_ERROR"),
        IndexerError::Search(_) => (StatusCode::INTERNAL_SERVER_ERROR, "SEARCH_ERROR"),
        IndexerError::Arrow(_) => (StatusCode::INTERNAL_SERVER_ERROR, "ARROW_ERROR"),
        IndexerError::Serialization(_) => {
          (StatusCode::INTERNAL_SERVER_ERROR, "SERIALIZATION_ERROR")
        }
        IndexerError::FileWatcher(_) => (StatusCode::INTERNAL_SERVER_ERROR, "FILE_WATCHER_ERROR"),
        IndexerError::Task(_) => (StatusCode::INTERNAL_SERVER_ERROR, "TASK_ERROR"),
        IndexerError::Database(_) => (StatusCode::INTERNAL_SERVER_ERROR, "DATABASE_ERROR"),
      },
    };

    let body = Json(ErrorResponse {
      error: self.to_string(),
      code: Some(code.to_string()),
    });

    (status, body).into_response()
  }
}
