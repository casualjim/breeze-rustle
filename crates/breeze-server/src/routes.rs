use std::path::PathBuf;

use aide::{
  axum::{
    ApiRouter, IntoApiResponse,
    routing::{get_with, post_with},
  },
  openapi::{Info, OpenApi},
};

use axum::{
  extract::{Path, Query, State},
  http::StatusCode,
  response::{IntoResponse, Json},
};

use tracing::info;
use uuid::Uuid;

use crate::error::ApiError;
use crate::types::*;
use crate::{
  app::{self, AppState},
  error::ErrorResponse,
};

pub fn openapi() -> OpenApi {
  OpenApi {
    info: Info {
      title: "Breeze Indexer API".to_string(),
      description: Some("High-performance semantic code indexing and search API".to_string()),
      version: "1.0.0".to_string(),
      ..Default::default()
    },
    ..Default::default()
  }
}

pub fn router(_app: app::AppState) -> ApiRouter<AppState> {
  ApiRouter::new()
    // Project endpoints
    .api_route(
      "/api/v1/projects",
      post_with(create_project, |op| {
        op.summary("Create a new project")
          .description("Creates a new project with the specified name and directory")
          .tag("Projects")
          .response::<201, Json<Project>>()
          .response::<400, Json<ErrorResponse>>()
      })
      .get_with(list_projects, |op| {
        op.summary("List projects")
          .description("Returns a list of all projects in the system")
          .tag("Projects")
          .response::<200, Json<Vec<Project>>>()
          .response::<500, Json<ErrorResponse>>()
      }),
    )
  .api_route("/api/v1/projects/:id",
      get_with(get_project, |op| op
          .summary("Get project details")
          .description("Retrieves detailed information about a specific project")
          .tag("Projects")
          .response::<200, Json<Project>>()
          .response::<404, Json<ErrorResponse>>()
          .response::<500, Json<ErrorResponse>>()
      )
      .put_with(update_project, |op| op
          .summary("Update project")
          .description("Updates the name and/or description of an existing project")
          .tag("Projects")
          .response::<200, Json<Project>>()
          .response::<404, Json<ErrorResponse>>()
          .response::<500, Json<ErrorResponse>>()
      )
      .delete_with(delete_project, |op| op
          .summary("Delete project")
          .description("Deletes a project and all associated data")
          .tag("Projects")
          .response::<204, ()>()
          .response::<404, Json<ErrorResponse>>()
          .response::<500, Json<ErrorResponse>>()
      )
  )
  .api_route("/api/v1/projects/:id/index",
      post_with(index_project_by_id, |op| op
          .summary("Index project")
          .description("Submits a background task to index all files in a project")
          .tag("Indexing")
          .response::<202, Json<TaskSubmittedResponse>>()
          .response::<500, Json<ErrorResponse>>()
      )
  )
  .api_route("/api/v1/projects/:id/files",
      post_with(index_file, |op| op
          .summary("Index file")
          .description("Submits a background task to index a single file within a project. Can provide content directly or read from filesystem.")
          .tag("Indexing")
          .response::<202, Json<TaskSubmittedResponse>>()
          .response::<400, Json<ErrorResponse>>()
          .response::<404, Json<ErrorResponse>>()
          .response::<500, Json<ErrorResponse>>()
      )
  )
  // Search endpoint
  .api_route("/api/v1/search",
      post_with(search, |op| op
          .summary("Search code")
          .description("Performs semantic search across all indexed content")
          .tag("Search")
          .response::<200, Json<Vec<SearchResult>>>()
          .response::<500, Json<ErrorResponse>>()
      )
  )
  // Task endpoints
  .api_route("/api/v1/tasks/:id",
      get_with(get_task, |op| op
          .summary("Get task")
          .description("Retrieves the status and details of a background task")
          .tag("Tasks")
          .response::<200, Json<Task>>()
          .response::<400, Json<ErrorResponse>>()
          .response::<404, Json<ErrorResponse>>()
          .response::<500, Json<ErrorResponse>>()
      )
  )
  .api_route("/api/v1/tasks",
      get_with(list_tasks, |op| op
          .summary("List tasks")
          .description("Returns a list of recent background tasks")
          .tag("Tasks")
          .response::<200, Json<Vec<Task>>>()
          .response::<500, Json<ErrorResponse>>()
      )
  )
}

// Project CRUD handlers
#[axum::debug_handler]
async fn create_project(
  State(state): State<AppState>,
  Json(req): Json<CreateProjectRequest>,
) -> impl IntoApiResponse {
  info!(name = %req.name, directory = %req.directory, "Creating project");

  match state
    .indexer
    .project_manager()
    .create_project(req.name, req.directory, req.description)
    .await
  {
    Ok(project) => {
      info!(project_id = %project.id, "Project created");
      (StatusCode::CREATED, Json(Into::<Project>::into(project))).into_response()
    }
    Err(e) => ApiError::from(e).into_response(),
  }
}

async fn get_project(State(state): State<AppState>, Path(id): Path<Uuid>) -> impl IntoApiResponse {
  match state.indexer.project_manager().get_project(id).await {
    Ok(Some(project)) => Json(Into::<Project>::into(project)).into_response(),
    Ok(None) => ApiError::ProjectNotFound(id).into_response(),
    Err(e) => ApiError::from(e).into_response(),
  }
}

async fn list_projects(State(state): State<AppState>) -> impl IntoApiResponse {
  match state.indexer.project_manager().list_projects().await {
    Ok(projects) => Json(
      projects
        .into_iter()
        .map(Into::into)
        .collect::<Vec<Project>>(),
    )
    .into_response(),
    Err(e) => ApiError::from(e).into_response(),
  }
}

async fn update_project(
  State(state): State<AppState>,
  Path(id): Path<Uuid>,
  Json(req): Json<UpdateProjectRequest>,
) -> impl IntoApiResponse {
  match state
    .indexer
    .project_manager()
    .update_project(id, req.name, req.description)
    .await
  {
    Ok(Some(project)) => Json(Into::<Project>::into(project)).into_response(),
    Ok(None) => ApiError::ProjectNotFound(id).into_response(),
    Err(e) => ApiError::from(e).into_response(),
  }
}

async fn delete_project(
  State(state): State<AppState>,
  Path(id): Path<Uuid>,
) -> impl IntoApiResponse {
  match state.indexer.project_manager().delete_project(id).await {
    Ok(true) => StatusCode::NO_CONTENT.into_response(),
    Ok(false) => ApiError::ProjectNotFound(id).into_response(),
    Err(e) => ApiError::from(e).into_response(),
  }
}

async fn index_project_by_id(
  State(state): State<AppState>,
  Path(id): Path<Uuid>,
) -> impl IntoApiResponse {
  info!(project_id = %id, "Submitting project indexing task");

  match state.indexer.index_project(id).await {
    Ok(task_id) => {
      info!(%task_id, project_id = %id, "Indexing task submitted");
      (
        StatusCode::ACCEPTED,
        Json(TaskSubmittedResponse {
          task_id,
          status: TaskStatus::Pending,
        }),
      )
        .into_response()
    }
    Err(e) => ApiError::from(e).into_response(),
  }
}

async fn index_file(
  State(state): State<AppState>,
  Path(project_id): Path<Uuid>,
  Json(req): Json<IndexFileRequest>,
) -> impl IntoApiResponse {
  info!(project_id = %project_id, path = %req.path, "Submitting file index task");

  let file_path = PathBuf::from(&req.path);

  match state.indexer.index_file(project_id, &file_path).await {
    Ok(task_id) => {
      info!(task_id = %task_id, "File index task submitted successfully");
      (
        StatusCode::ACCEPTED,
        Json(TaskSubmittedResponse {
          task_id,
          status: TaskStatus::Pending,
        }),
      )
        .into_response()
    }
    Err(e) => ApiError::from(e).into_response(),
  }
}

async fn search(
  State(state): State<AppState>,
  Json(req): Json<SearchRequest>,
) -> impl IntoApiResponse {
  info!(query = %req.query, limit = ?req.limit, "Performing search");

  let limit = req.limit.unwrap_or(10);

  match state.indexer.search(&req.query, limit).await {
    Ok(results) => {
      info!(results = results.len(), "Search completed");
      Json(
        results
          .into_iter()
          .map(Into::into)
          .collect::<Vec<SearchResult>>(),
      )
      .into_response()
    }
    Err(e) => ApiError::from(e).into_response(),
  }
}

async fn get_task(
  State(state): State<AppState>,
  Path(task_id): Path<Uuid>,
) -> impl IntoApiResponse {
  let result = state
    .indexer
    .task_manager()
    .get_task(&task_id)
    .await
    .map_err(ApiError::from)
    .and_then(|task| {
      task
        .map(Into::<Task>::into)
        .ok_or(ApiError::TaskNotFound(task_id))
    });

  match result {
    Ok(task) => Json(task).into_response(),
    Err(e) => e.into_response(),
  }
}

async fn list_tasks(
  State(state): State<AppState>,
  Query(query): Query<ListTasksQuery>,
) -> impl IntoApiResponse {
  let limit = query.limit.unwrap_or(20).min(100); // Cap at 100

  match state.indexer.task_manager().list_tasks(limit).await {
    Ok(tasks) => Json(tasks.into_iter().map(Into::into).collect::<Vec<Task>>()).into_response(),
    Err(e) => ApiError::from(e).into_response(),
  }
}
