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

  #[tool(description = "Search code using semantic understanding")]
  async fn search_code(
    &self,
    #[tool(aggr)] SearchRequest { query, limit }: SearchRequest,
  ) -> Result<CallToolResult, McpError> {
    let limit = limit.unwrap_or(10);

    info!("Searching for '{}' with limit {}", query, limit);

    match self.indexer.search(&query, limit).await {
      Ok(results) => {
        let mut content = String::new();
        content.push_str(&format!(
          "Found {} results for '{}':\n\n",
          results.len(),
          query
        ));

        for (idx, result) in results.iter().enumerate() {
          content.push_str(&format!(
            "{}. {} (score: {:.3})\n",
            idx + 1,
            result.file_path,
            result.relevance_score
          ));

          // Show first few lines as preview
          let preview_lines: Vec<&str> = result.content.lines().take(3).collect();
          for line in preview_lines {
            content.push_str(&format!("   {}\n", line));
          }

          let total_lines = result.content.lines().count();
          if total_lines > 3 {
            content.push_str(&format!("   ... ({} more lines)\n", total_lines - 3));
          }
          content.push('\n');
        }

        Ok(CallToolResult::success(vec![Content::text(content)]))
      }
      Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
        "Search failed: {}",
        e
      ))])),
    }
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

  #[tool(description = "List all projects")]
  async fn list_projects(&self) -> Result<CallToolResult, McpError> {
    info!("Listing projects");

    match self.indexer.project_manager().list_projects().await {
      Ok(projects) => {
        let mut content = String::from("Projects:\n\n");
        for project in projects {
          content.push_str(&format!("- {} ({})\n", project.name, project.id));
          content.push_str(&format!("  Directory: {}\n", project.directory));
          if let Some(desc) = project.description {
            content.push_str(&format!("  Description: {}\n", desc));
          }
          content.push('\n');
        }
        Ok(CallToolResult::success(vec![Content::text(content)]))
      }
      Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
        "Failed to list projects: {}",
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
