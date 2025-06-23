use std::path::PathBuf;
use std::sync::Arc;

use breeze_indexer::Indexer;
use rmcp::transport::streamable_http_server::{
  StreamableHttpService, session::local::LocalSessionManager,
};
use rmcp::{Error as McpError, RoleServer, ServerHandler, model::*, service::RequestContext, tool};
use serde_json::json;
use tracing::info;
use uuid::Uuid;

use crate::types::*;

#[derive(Clone)]
pub struct BreezeService {
  indexer: Arc<Indexer>,
}

#[tool(tool_box)]
impl BreezeService {
  pub fn new(indexer: Arc<Indexer>) -> Self {
    Self { indexer }
  }

  #[tool(description = "Create a new project for indexing")]
  async fn create_project(
    &self,
    #[tool(aggr)] CreateProjectRequest {
      name,
      directory,
      description,
    }: CreateProjectRequest,
  ) -> Result<CallToolResult, McpError> {
    info!("Creating project: {} at {}", name, directory);

    match self
      .indexer
      .project_manager()
      .create_project(name, directory, description)
      .await
    {
      Ok(project) => Ok(CallToolResult::success(vec![Content::text(format!(
        "Successfully created project '{}' with ID: {}",
        project.name, project.id
      ))])),
      Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
        "Failed to create project: {}",
        e
      ))])),
    }
  }

  #[tool(description = "Search code using semantic understanding")]
  async fn search_code(
    &self,
    #[tool(aggr)] SearchRequest { query, limit }: SearchRequest,
  ) -> Result<CallToolResult, McpError> {
    let limit = limit.unwrap_or(10);

    info!("Searching for '{}' with limit {}", query, limit);

    match self.indexer.search(&query, limit).await {
      Ok(results) => Ok(CallToolResult::success(vec![Content::json(results)?])),
      Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
        "Search failed: {}",
        e
      ))])),
    }
  }

  #[tool(description = "Index a project directory for semantic code search")]
  async fn index_project(
    &self,
    #[tool(aggr)] IndexProjectByIdRequest { project_id }: IndexProjectByIdRequest,
  ) -> Result<CallToolResult, McpError> {
    info!("Indexing project: {}", project_id);

    let project_uuid = match Uuid::parse_str(&project_id) {
      Ok(uuid) => uuid,
      Err(e) => {
        return Ok(CallToolResult::error(vec![Content::text(format!(
          "Invalid project ID: {}",
          e
        ))]));
      }
    };

    match self.indexer.index_project(project_uuid).await {
      Ok(task_id) => Ok(CallToolResult::success(vec![Content::text(format!(
        "Successfully submitted indexing task for project: {}. Task ID: {}",
        project_id, task_id
      ))])),
      Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
        "Failed to submit indexing task: {}",
        e
      ))])),
    }
  }

  #[tool(description = "Index a single file for semantic code search")]
  async fn index_file(
    &self,
    #[tool(aggr)] IndexFileByProjectRequest { project_id, path }: IndexFileByProjectRequest,
  ) -> Result<CallToolResult, McpError> {
    info!("Indexing file: {} for project: {}", path, project_id);

    let project_uuid = match Uuid::parse_str(&project_id) {
      Ok(uuid) => uuid,
      Err(e) => {
        return Ok(CallToolResult::error(vec![Content::text(format!(
          "Invalid project ID: {}",
          e
        ))]));
      }
    };

    let file_path = PathBuf::from(&path);

    match self.indexer.index_file(project_uuid, &file_path).await {
      Ok(id) => Ok(CallToolResult::success(vec![Content::json(json!({
        "message": format!("Scheduled indexing of file task_id={id} path={path}"),
        "task_id": id,
        "project_id": project_uuid.to_string(),
      }))?])),
      Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
        "Indexing failed: {}",
        e
      ))])),
    }
  }

  #[tool(description = "Update an existing project")]
  async fn update_project(
    &self,
    #[tool(aggr)] UpdateProjectByIdRequest {
      project_id,
      name,
      description,
    }: UpdateProjectByIdRequest,
  ) -> Result<CallToolResult, McpError> {
    info!("Updating project: {}", project_id);

    let project_uuid = match Uuid::parse_str(&project_id) {
      Ok(uuid) => uuid,
      Err(e) => {
        return Ok(CallToolResult::error(vec![Content::text(format!(
          "Invalid project ID: {}",
          e
        ))]));
      }
    };

    match self
      .indexer
      .project_manager()
      .update_project(project_uuid, name, description)
      .await
    {
      Ok(Some(project)) => Ok(CallToolResult::success(vec![Content::text(format!(
        "Successfully updated project '{}' ({})",
        project.name, project.id
      ))])),
      Ok(None) => Ok(CallToolResult::error(vec![Content::text(format!(
        "Project not found: {}",
        project_id
      ))])),
      Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
        "Failed to update project: {}",
        e
      ))])),
    }
  }

  #[tool(description = "Delete a project and all associated data")]
  async fn delete_project(
    &self,
    #[tool(aggr)] DeleteProjectRequest { project_id }: DeleteProjectRequest,
  ) -> Result<CallToolResult, McpError> {
    info!("Deleting project: {}", project_id);

    let project_uuid = match Uuid::parse_str(&project_id) {
      Ok(uuid) => uuid,
      Err(e) => {
        return Ok(CallToolResult::error(vec![Content::text(format!(
          "Invalid project ID: {}",
          e
        ))]));
      }
    };

    match self
      .indexer
      .project_manager()
      .delete_project(project_uuid)
      .await
    {
      Ok(true) => Ok(CallToolResult::success(vec![Content::text(format!(
        "Successfully deleted project: {}",
        project_id
      ))])),
      Ok(false) => Ok(CallToolResult::error(vec![Content::text(format!(
        "Project not found: {}",
        project_id
      ))])),
      Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
        "Failed to delete project: {}",
        e
      ))])),
    }
  }
}

#[tool(tool_box)]
impl ServerHandler for BreezeService {
  fn get_info(&self) -> ServerInfo {
    ServerInfo {
            protocol_version: ProtocolVersion::LATEST,
            capabilities: ServerCapabilities::builder()
                .enable_logging()
                .enable_prompts()
                .enable_resources()
                .enable_prompts_list_changed()
                .enable_resources_list_changed()
                .enable_resources_subscribe()
                .enable_tools()
                .build(),
            server_info: Implementation {
                name: "breeze-mcp".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            instructions: Some("Breeze MCP server provides semantic code search and indexing capabilities. Use 'search_code' to search through indexed code and 'index_directory' to index new directories.".to_string()),
        }
  }

  async fn initialize(
    &self,
    _request: InitializeRequestParam,
    context: RequestContext<RoleServer>,
  ) -> Result<InitializeResult, McpError> {
    if let Some(http_request_part) = context.extensions.get::<axum::http::request::Parts>() {
      let initialize_headers = &http_request_part.headers;
      let initialize_uri = &http_request_part.uri;
      info!(?initialize_headers, %initialize_uri, "MCP initialize from http server");
    }
    Ok(self.get_info())
  }
}

/// Create an HTTP streamable MCP service
pub fn create_http_service(indexer: Arc<Indexer>) -> StreamableHttpService<BreezeService> {
  StreamableHttpService::new(
    move || Ok(BreezeService::new(indexer.clone())),
    LocalSessionManager::default().into(),
    Default::default(),
  )
}
