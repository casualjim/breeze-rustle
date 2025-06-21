use std::path::PathBuf;

use axum::{
  Router,
  extract::{Json, Path, Query, State},
  http::StatusCode,
  response::IntoResponse,
  routing::{get, post},
};
use tracing::{error, info};
use uuid::Uuid;

use crate::app::{self, AppState};
use crate::types::*;

pub fn router(app: app::AppState) -> Router<AppState> {
  Router::new()
    // Project endpoints
    .route("/api/v1/projects", post(create_project).get(list_projects))
    .route("/api/v1/projects/:id", get(get_project).put(update_project).delete(delete_project))
    .route("/api/v1/projects/:id/index", post(index_project_by_id))
    .route("/api/v1/projects/:id/index/file", post(index_file))
    // Search endpoint
    .route("/api/v1/search", post(search))
    // Task endpoints
    .route("/api/v1/tasks/:id", get(get_task))
    .route("/api/v1/tasks", get(list_tasks))
    .with_state(app)
}

// Project CRUD handlers
async fn create_project(
  State(state): State<AppState>,
  Json(req): Json<CreateProjectRequest>,
) -> impl IntoResponse {
  info!(name = %req.name, directory = %req.directory, "Creating project");

  match state.indexer.project_manager().create_project(
    req.name,
    req.directory,
    req.description,
  ).await {
    Ok(project) => {
      info!(project_id = %project.id, "Project created");
      (StatusCode::CREATED, Json(project)).into_response()
    }
    Err(e) => {
      error!("Failed to create project: {}", e);
      (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
          error: format!("Failed to create project: {}", e),
        }),
      )
        .into_response()
    }
  }
}

async fn get_project(
  State(state): State<AppState>,
  Path(id): Path<Uuid>,
) -> impl IntoResponse {
  match state.indexer.project_manager().get_project(id).await {
    Ok(Some(project)) => (StatusCode::OK, Json(project)).into_response(),
    Ok(None) => (
      StatusCode::NOT_FOUND,
      Json(ErrorResponse {
        error: format!("Project '{}' not found", id),
      }),
    )
      .into_response(),
    Err(e) => {
      error!("Failed to get project: {}", e);
      (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
          error: format!("Failed to get project: {}", e),
        }),
      )
        .into_response()
    }
  }
}

async fn list_projects(
  State(state): State<AppState>,
) -> impl IntoResponse {
  match state.indexer.project_manager().list_projects().await {
    Ok(projects) => (StatusCode::OK, Json(projects)).into_response(),
    Err(e) => {
      error!("Failed to list projects: {}", e);
      (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
          error: format!("Failed to list projects: {}", e),
        }),
      )
        .into_response()
    }
  }
}

async fn update_project(
  State(state): State<AppState>,
  Path(id): Path<Uuid>,
  Json(req): Json<UpdateProjectRequest>,
) -> impl IntoResponse {
  match state.indexer.project_manager().update_project(
    id,
    req.name,
    req.description,
  ).await {
    Ok(Some(project)) => (StatusCode::OK, Json(project)).into_response(),
    Ok(None) => (
      StatusCode::NOT_FOUND,
      Json(ErrorResponse {
        error: format!("Project '{}' not found", id),
      }),
    )
      .into_response(),
    Err(e) => {
      error!("Failed to update project: {}", e);
      (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
          error: format!("Failed to update project: {}", e),
        }),
      )
        .into_response()
    }
  }
}

async fn delete_project(
  State(state): State<AppState>,
  Path(id): Path<Uuid>,
) -> impl IntoResponse {
  match state.indexer.project_manager().delete_project(id).await {
    Ok(true) => StatusCode::NO_CONTENT.into_response(),
    Ok(false) => (
      StatusCode::NOT_FOUND,
      Json(ErrorResponse {
        error: format!("Project '{}' not found", id),
      }),
    )
      .into_response(),
    Err(e) => {
      error!("Failed to delete project: {}", e);
      (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
          error: format!("Failed to delete project: {}", e),
        }),
      )
        .into_response()
    }
  }
}

async fn index_project_by_id(
  State(state): State<AppState>,
  Path(id): Path<Uuid>,
) -> impl IntoResponse {
  info!(project_id = %id, "Submitting project indexing task");

  match state.indexer.index_project(id).await {
    Ok(task_id) => {
      info!(task_id, project_id = %id, "Indexing task submitted");
      (
        StatusCode::ACCEPTED,
        Json(TaskSubmittedResponse {
          task_id,
          status: "pending".to_string(),
        }),
      )
        .into_response()
    }
    Err(e) => {
      error!("Failed to submit indexing task: {}", e);
      (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
          error: format!("Failed to submit indexing task: {}", e),
        }),
      )
        .into_response()
    }
  }
}

async fn index_file(
  State(state): State<AppState>,
  Path(project_id): Path<Uuid>,
  Json(req): Json<IndexFileRequest>,
) -> impl IntoResponse {
  info!(project_id = %project_id, path = %req.path, "Indexing file");

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

  match state.indexer.index_file(project_id, &file_path, req.content).await {
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

async fn get_task(State(state): State<AppState>, Path(task_id): Path<String>) -> impl IntoResponse {
  let task_uuid = match Uuid::parse_str(&task_id) {
    Ok(uuid) => uuid,
    Err(_) => {
      return (
        StatusCode::BAD_REQUEST,
        Json(ErrorResponse {
          error: format!("Invalid task ID format: {}", task_id),
        }),
      )
        .into_response();
    }
  };

  match state.indexer.task_manager().get_task(&task_uuid).await {
    Ok(Some(task)) => (StatusCode::OK, Json(task)).into_response(),
    Ok(None) => (
      StatusCode::NOT_FOUND,
      Json(ErrorResponse {
        error: format!("Task '{}' not found", task_id),
      }),
    )
      .into_response(),
    Err(e) => {
      error!("Failed to get task: {}", e);
      (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
          error: format!("Failed to get task: {}", e),
        }),
      )
        .into_response()
    }
  }
}

async fn list_tasks(
  State(state): State<AppState>,
  Query(query): Query<ListTasksQuery>,
) -> impl IntoResponse {
  let limit = query.limit.unwrap_or(20).min(100); // Cap at 100

  match state.indexer.task_manager().list_tasks(limit).await {
    Ok(tasks) => (StatusCode::OK, Json(tasks)).into_response(),
    Err(e) => {
      error!("Failed to list tasks: {}", e);
      (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(ErrorResponse {
          error: format!("Failed to list tasks: {}", e),
        }),
      )
        .into_response()
    }
  }
}
