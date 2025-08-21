use std::{collections::BTreeSet, path::PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::duration::HumanDuration;

// Note: These types are shared between the REST API and MCP server
// JsonSchema is used for both MCP and future OpenAPI spec generation

// Shared request types
#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct CreateProjectRequest {
  pub path: String,
  pub name: Option<String>,
  pub description: Option<String>,
  pub rescan_interval: Option<HumanDuration>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(
  title = "CreateProject",
  description = "Create a new project for indexing. Provide the absolute path to the project root."
)]
pub struct CreateProject {
  #[schemars(
    description = "Absolute filesystem path to the project root (e.g., /Users/alice/workspace/myproj)"
  )]
  pub path: String,
  #[schemars(
    description = "Optional display name for the project; defaults to the directory name if omitted"
  )]
  pub name: Option<String>,
  #[schemars(description = "Optional human-friendly description of the project")]
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
  pub rescan_interval: Option<HumanDuration>,
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
      rescan_interval: project.rescan_interval.map(HumanDuration::from),
    }
  }
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
pub struct UpdateProjectRequest {
  pub name: Option<String>,
  pub description: Option<Option<String>>,
  pub rescan_interval: Option<HumanDuration>,
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
  pub project_id: Option<Uuid>,
  pub path: Option<String>,
  pub query: String,
  pub limit: Option<usize>,
  pub chunks_per_file: Option<usize>,
  pub languages: Option<Vec<String>>,
  pub granularity: Option<SearchGranularity>,

  // Semantic filters for chunk mode
  pub node_types: Option<Vec<String>>,
  pub node_name_pattern: Option<String>,
  pub parent_context_pattern: Option<String>,
  pub scope_depth: Option<(usize, usize)>,
  pub has_definitions: Option<Vec<String>>,
  pub has_references: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
#[schemars(
  title = "SimpleSearchRequest",
  description = "Search code using semantic understanding, bounded to a project directory."
)]
pub struct SimpleSearchRequest {
  /// The path to search within
  #[schemars(
    description = "Absolute project path used to bound the search. Provide the full filesystem path (e.g., /Users/alice/workspace/myproj). Relative paths are not accepted."
  )]
  pub path: Option<String>,
  /// The search query
  #[schemars(description = "The semantic search query text")]
  pub query: String,
  /// The maximum number of results to return
  #[schemars(
    description = "Maximum number of files to return (default 10). Must be positive if provided.",
    range(min = 1)
  )]
  pub limit: Option<isize>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum SearchGranularity {
  Document,
  Chunk,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ChunkResult {
  pub content: String,
  pub start_line: usize,
  pub end_line: usize,
  pub start_byte: usize,
  pub end_byte: usize,
  pub relevance_score: f32,

  // Semantic metadata from CodeChunk
  pub node_type: String,
  pub node_name: Option<String>,
  pub language: String,
  pub parent_context: Option<String>,
  pub scope_path: Vec<String>,
  pub definitions: Vec<String>,
  pub references: Vec<String>,
}

impl From<breeze_indexer::ChunkResult> for ChunkResult {
  fn from(chunk: breeze_indexer::ChunkResult) -> Self {
    ChunkResult {
      content: chunk.content,
      start_line: chunk.start_line,
      end_line: chunk.end_line,
      start_byte: chunk.start_byte,
      end_byte: chunk.end_byte,
      relevance_score: chunk.relevance_score,
      node_type: chunk.node_type,
      node_name: chunk.node_name,
      language: chunk.language,
      parent_context: chunk.parent_context,
      scope_path: chunk.scope_path,
      definitions: chunk.definitions,
      references: chunk.references,
    }
  }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SearchResult {
  pub id: String,
  pub file_path: String,
  pub relevance_score: f32,
  pub chunk_count: u32,
  pub chunks: Vec<ChunkResult>,

  // Document-level metadata from CodeDocument
  pub file_size: u64,
  pub last_modified: chrono::NaiveDateTime,
  pub indexed_at: chrono::NaiveDateTime,
  pub languages: Vec<String>,
  pub primary_language: Option<String>,
}

impl From<breeze_indexer::SearchResult> for SearchResult {
  fn from(result: breeze_indexer::SearchResult) -> Self {
    SearchResult {
      id: result.id,
      file_path: result.file_path,
      relevance_score: result.relevance_score,
      chunk_count: result.chunk_count,
      chunks: result.chunks.into_iter().map(ChunkResult::from).collect(),
      file_size: result.file_size,
      last_modified: result.last_modified,
      indexed_at: result.indexed_at,
      languages: result.languages,
      primary_language: result.primary_language,
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
  pub rescan_interval: Option<HumanDuration>,
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

#[derive(Debug, Serialize, Deserialize, JsonSchema)]
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

impl std::fmt::Display for TaskStatus {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      TaskStatus::Pending => write!(f, "Pending"),
      TaskStatus::Running => write!(f, "Running"),
      TaskStatus::Completed => write!(f, "Completed"),
      TaskStatus::Failed => write!(f, "Failed"),
      TaskStatus::Merged => write!(f, "Merged"),
      TaskStatus::PartiallyCompleted => write!(f, "Partially Completed"),
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
