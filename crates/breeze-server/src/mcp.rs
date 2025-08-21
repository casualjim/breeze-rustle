use std::sync::Arc;

use breeze_indexer::Indexer;
use breeze_indexer::SearchGranularity as IndexerSearchGranularity;
use breeze_indexer::SearchOptions;

use rmcp::handler::server::tool::Parameters;
use rmcp::handler::server::tool::ToolRouter;
use rmcp::tool_handler;
use rmcp::tool_router;
use rmcp::transport::streamable_http_server::{
  StreamableHttpService, session::local::LocalSessionManager,
};
use rmcp::{ErrorData, RoleServer, ServerHandler, model::*, service::RequestContext, tool};
use tracing::info;

use crate::types::*;

#[derive(Clone)]
pub struct BreezeService {
  indexer: Arc<Indexer>,
  tool_router: ToolRouter<BreezeService>,
}

#[tool_router]
impl BreezeService {
  pub fn new(indexer: Arc<Indexer>) -> Self {
    Self {
      indexer,
      tool_router: Self::tool_router(),
    }
  }

  #[tool(description = "Create a new project for indexing")]
  async fn create_project(
    &self,
    Parameters(CreateProjectRequest {
      name,
      path,
      description,
      rescan_interval,
    }): Parameters<CreateProjectRequest>,
  ) -> Result<CallToolResult, ErrorData> {
    let project_path = std::path::PathBuf::from(path);
    let name = name.unwrap_or_else(|| {
      project_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string()
    });
    info!("Creating project: {} at {}", name, project_path.display());

    match self
      .indexer
      .project_manager()
      .create_project(
        name,
        project_path.to_string_lossy().to_string(),
        description,
        rescan_interval.map(|v| *v),
      )
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
    Parameters(search_req): Parameters<SimpleSearchRequest>,
  ) -> Result<CallToolResult, ErrorData> {
    let limit = search_req.limit.unwrap_or(10);

    info!("Searching for '{}' with limit {}", search_req.query, limit);

    let options = SearchOptions {
      file_limit: limit,
      chunks_per_file: 3,
      granularity: IndexerSearchGranularity::Chunk,

      ..Default::default()
    };

    match self
      .indexer
      .search(&search_req.query, options, search_req.project_id)
      .await
    {
      Ok(results) => Ok(CallToolResult::success(vec![Content::json(results)?])),
      Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
        "Search failed: {}",
        e
      ))])),
    }
  }
}

#[tool_handler]
impl ServerHandler for BreezeService {
  fn get_info(&self) -> ServerInfo {
    ServerInfo {
            protocol_version: ProtocolVersion::LATEST,
            capabilities: ServerCapabilities::builder()
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
            instructions: Some("Breeze MCP server provides semantic code search and indexing capabilities. Use 'search_code' to search through indexed code and 'create_project' to index new directories.".to_string()),
        }
  }

  async fn initialize(
    &self,
    _request: InitializeRequestParam,
    context: RequestContext<RoleServer>,
  ) -> Result<InitializeResult, rmcp::ErrorData> {
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
