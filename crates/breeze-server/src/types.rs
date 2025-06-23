use std::{collections::BTreeSet, path::PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Project {
  pub id: Uuid,
  pub name: String,
  pub directory: String,
  pub description: Option<String>,
  pub created_at: chrono::NaiveDateTime,
  pub updated_at: chrono::NaiveDateTime,
}

impl From<breeze_indexer::Project> for Project {
  fn from(project: breeze_indexer::Project) -> Self {
    Project {
      id: project.id,
      name: project.name,
      directory: project.directory,
      description: project.description,
      created_at: project.created_at,
      updated_at: project.updated_at,
    }
  }
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
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct SearchRequest {
  pub query: String,
  pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchResult {
  pub id: String,
  pub file_path: String,
  pub content: String,
  pub content_hash: [u8; 32],
  pub relevance_score: f32,
  pub file_size: u64,
  pub last_modified: chrono::NaiveDateTime,
  pub indexed_at: chrono::NaiveDateTime,
}

impl From<breeze_indexer::SearchResult> for SearchResult {
  fn from(result: breeze_indexer::SearchResult) -> Self {
    SearchResult {
      id: result.id.to_string(),
      file_path: result.file_path,
      content: result.content,
      content_hash: result.content_hash,
      relevance_score: result.relevance_score,
      file_size: result.file_size,
      last_modified: result.last_modified,
      indexed_at: result.indexed_at,
    }
  }
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
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct GetProjectRequest {
  pub project_id: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct UpdateProjectByIdRequest {
  pub project_id: String,
  pub name: Option<String>,
  pub description: Option<Option<String>>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct DeleteProjectRequest {
  pub project_id: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct GetTaskRequest {
  pub task_id: String,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct ListTasksRequest {
  pub limit: Option<usize>,
}

// Response types
#[derive(Debug, Serialize, JsonSchema)]
pub struct IndexResponse {
  pub status: String,
  pub files_indexed: usize,
}

#[derive(Debug, Serialize, JsonSchema)]
pub struct TaskSubmittedResponse {
  pub task_id: Uuid,
  pub status: TaskStatus,
}

// Query parameter types
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ListTasksQuery {
  pub limit: Option<usize>,
}

#[derive(
  Debug,
  Clone,
  PartialEq,
  Eq,
  PartialOrd,
  Ord,
  Serialize,
  Deserialize,
  JsonSchema
)]
pub enum FileOperation {
  Add,
  Update,
  Delete,
}

impl From<breeze_indexer::FileOperation> for FileOperation {
  fn from(op: breeze_indexer::FileOperation) -> Self {
    match op {
      breeze_indexer::FileOperation::Add => FileOperation::Add,
      breeze_indexer::FileOperation::Update => FileOperation::Update,
      breeze_indexer::FileOperation::Delete => FileOperation::Delete,
    }
  }
}

#[derive(
  Debug,
  Clone,
  PartialEq,
  Eq,
  PartialOrd,
  Ord,
  Serialize,
  Deserialize,
  JsonSchema
)]
pub struct FileChange {
  pub path: PathBuf,
  pub operation: FileOperation,
}

impl From<breeze_indexer::FileChange> for FileChange {
  fn from(change: breeze_indexer::FileChange) -> Self {
    FileChange {
      path: change.path,
      operation: change.operation.into(),
    }
  }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum TaskType {
  FullIndex,
  PartialUpdate { changes: BTreeSet<FileChange> },
}

impl From<breeze_indexer::TaskType> for TaskType {
  fn from(task_type: breeze_indexer::TaskType) -> Self {
    match task_type {
      breeze_indexer::TaskType::FullIndex => TaskType::FullIndex,
      breeze_indexer::TaskType::PartialUpdate { changes } => TaskType::PartialUpdate {
        changes: changes.into_iter().map(Into::into).collect(),
      },
    }
  }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum TaskStatus {
  Pending,
  Running,
  Completed,
  Failed,
  Merged,
  PartiallyCompleted,
}

impl From<breeze_indexer::TaskStatus> for TaskStatus {
  fn from(status: breeze_indexer::TaskStatus) -> Self {
    match status {
      breeze_indexer::TaskStatus::Pending => TaskStatus::Pending,
      breeze_indexer::TaskStatus::Running => TaskStatus::Running,
      breeze_indexer::TaskStatus::Completed => TaskStatus::Completed,
      breeze_indexer::TaskStatus::Failed => TaskStatus::Failed,
      breeze_indexer::TaskStatus::Merged => TaskStatus::Merged,
      breeze_indexer::TaskStatus::PartiallyCompleted => TaskStatus::PartiallyCompleted,
    }
  }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Task {
  pub id: Uuid,
  pub project_id: Uuid,
  pub path: String,
  pub task_type: TaskType,
  pub status: TaskStatus,
  pub created_at: chrono::NaiveDateTime,
  pub started_at: Option<chrono::NaiveDateTime>,
  pub completed_at: Option<chrono::NaiveDateTime>,
  pub error: Option<String>,
  pub files_indexed: Option<usize>,
  pub merged_into: Option<Uuid>,
}

impl From<breeze_indexer::IndexTask> for Task {
  fn from(task: breeze_indexer::IndexTask) -> Self {
    Task {
      id: task.id,
      project_id: task.project_id,
      path: task.path,
      task_type: task.task_type.into(),
      status: task.status.into(),
      created_at: task.created_at,
      started_at: task.started_at,
      completed_at: task.completed_at,
      error: task.error,
      files_indexed: task.files_indexed,
      merged_into: task.merged_into,
    }
  }
}
