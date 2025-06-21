use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
use uuid::Uuid;

// Note: These types are shared between the REST API and MCP server
// JsonSchema is used for both MCP and future OpenAPI spec generation

// Shared request types
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct CreateProjectRequest {
  pub name: String,
  pub directory: String,
  pub description: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct UpdateProjectRequest {
  pub name: Option<String>,
  pub description: Option<Option<String>>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct IndexProjectRequest {
  pub project_id: Uuid,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct IndexFileRequest {
  pub path: String,
  pub content: Option<String>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct SearchRequest {
  pub query: String,
  pub limit: Option<usize>,
}

// MCP-specific request types (MCP uses string UUIDs)
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct IndexProjectByIdRequest {
  pub project_id: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct IndexFileByProjectRequest {
  pub project_id: String,
  pub path: String,
  pub content: Option<String>,
}

// Response types
#[derive(Debug, Serialize)]
pub struct IndexResponse {
  pub status: String,
  pub files_indexed: usize,
}

#[derive(Debug, Serialize)]
pub struct TaskSubmittedResponse {
  pub task_id: String,
  pub status: String,
}

#[derive(Debug, Serialize)]
pub struct ErrorResponse {
  pub error: String,
}

// Query parameter types
#[derive(Debug, Deserialize)]
pub struct ListTasksQuery {
  pub limit: Option<usize>,
}