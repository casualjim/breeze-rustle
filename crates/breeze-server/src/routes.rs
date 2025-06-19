use std::path::PathBuf;

use axum::{
  Router,
  extract::{Json, State},
  http::StatusCode,
  response::IntoResponse,
  routing::post,
};
use serde::{Deserialize, Serialize};
use tracing::{error, info};

use crate::app::{self, AppState};

#[derive(Deserialize)]
struct IndexProjectRequest {
  path: String,
}

#[derive(Deserialize)]
struct IndexFileRequest {
  path: String,
  content: Option<String>,
}

#[derive(Deserialize)]
struct SearchRequest {
  query: String,
  limit: Option<usize>,
}

#[derive(Serialize)]
struct IndexResponse {
  status: String,
  files_indexed: usize,
}

#[derive(Serialize)]
struct ErrorResponse {
  error: String,
}

pub fn router(app: app::AppState) -> Router<AppState> {
  Router::new()
    .route("/api/v1/index/project", post(index_project))
    .route("/api/v1/index/file", post(index_file))
    .route("/api/v1/search", post(search))
    .with_state(app)
}

async fn index_project(
  State(state): State<AppState>,
  Json(req): Json<IndexProjectRequest>,
) -> impl IntoResponse {
  info!(path = %req.path, "Indexing project");

  let project_path = PathBuf::from(&req.path);
  if !project_path.exists() {
    return (
      StatusCode::BAD_REQUEST,
      Json(ErrorResponse {
        error: format!("Project path '{}' does not exist", req.path),
      }),
    )
      .into_response();
  }

  match state
    .indexer
    .index_project(&project_path, Some(state.shutdown_token.clone()))
    .await
  {
    Ok(files_indexed) => {
      info!(files_indexed, "Project indexed successfully");
      (
        StatusCode::OK,
        Json(IndexResponse {
          status: "success".to_string(),
          files_indexed,
        }),
      )
        .into_response()
    }
    Err(e) => {
      error!("Failed to index project: {}", e);
      (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
          error: format!("Failed to index project: {}", e),
        }),
      )
        .into_response()
    }
  }
}

async fn index_file(
  State(state): State<AppState>,
  Json(req): Json<IndexFileRequest>,
) -> impl IntoResponse {
  info!(path = %req.path, "Indexing file");

  let file_path = PathBuf::from(&req.path);

  // If content is not provided, check that file exists
  if req.content.is_none() && !file_path.exists() {
    return (
      StatusCode::BAD_REQUEST,
      Json(ErrorResponse {
        error: format!("File path '{}' does not exist", req.path),
      }),
    )
      .into_response();
  }

  match state.indexer.index_file(&file_path, req.content).await {
    Ok(()) => {
      info!("File indexed successfully");
      (
        StatusCode::OK,
        Json(IndexResponse {
          status: "success".to_string(),
          files_indexed: 1,
        }),
      )
        .into_response()
    }
    Err(e) => {
      error!("Failed to index file: {}", e);
      (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
          error: format!("Failed to index file: {}", e),
        }),
      )
        .into_response()
    }
  }
}

async fn search(
  State(state): State<AppState>,
  Json(req): Json<SearchRequest>,
) -> impl IntoResponse {
  info!(query = %req.query, limit = ?req.limit, "Performing search");

  let limit = req.limit.unwrap_or(10);

  match state.indexer.search(&req.query, limit).await {
    Ok(results) => {
      info!(results = results.len(), "Search completed");
      (StatusCode::OK, Json(results)).into_response()
    }
    Err(e) => {
      error!("Search failed: {}", e);
      (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
          error: format!("Search failed: {}", e),
        }),
      )
        .into_response()
    }
  }
}
