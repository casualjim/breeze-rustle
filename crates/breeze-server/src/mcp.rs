use std::path::PathBuf;
use std::sync::Arc;

use breeze_indexer::Indexer;
use rmcp::transport::sse_server::{SseServer, SseServerConfig};
use rmcp::transport::streamable_http_server::{
  StreamableHttpService, session::local::LocalSessionManager,
};
use rmcp::{
  Error as McpError, RoleServer, ServerHandler, model::*, schemars, service::RequestContext, tool,
};
use tokio_util::sync::CancellationToken;
use tracing::info;

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct SearchCodeRequest {
  pub query: String,
  pub limit: Option<usize>,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct IndexDirectoryRequest {
  pub path: String,
}

#[derive(Debug, serde::Deserialize, schemars::JsonSchema)]
pub struct IndexFileRequest {
  pub path: String,
  pub content: Option<String>,
}

#[derive(Clone)]
pub struct BreezeService {
  indexer: Arc<Indexer>,
  shutdown_token: CancellationToken,
}

#[tool(tool_box)]
impl BreezeService {
  pub fn new(indexer: Arc<Indexer>, shutdown_token: CancellationToken) -> Self {
    Self {
      indexer,
      shutdown_token,
    }
  }

  #[tool(description = "Search code using semantic understanding")]
  async fn search_code(
    &self,
    #[tool(aggr)] SearchCodeRequest { query, limit }: SearchCodeRequest,
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

  #[tool(description = "Index a directory for semantic code search")]
  async fn index_directory(
    &self,
    #[tool(aggr)] IndexDirectoryRequest { path }: IndexDirectoryRequest,
  ) -> Result<CallToolResult, McpError> {
    info!("Indexing directory: {}", path);

    let project_path = PathBuf::from(&path);

    if !project_path.exists() {
      return Ok(CallToolResult::error(vec![Content::text(format!(
        "Directory '{}' does not exist",
        path
      ))]));
    }

    match self
      .indexer
      .index_project(&project_path, Some(self.shutdown_token.clone()))
      .await
    {
      Ok(files_indexed) => Ok(CallToolResult::success(vec![Content::text(format!(
        "Successfully indexed {} files in directory: {}",
        files_indexed, path
      ))])),
      Err(e) => Ok(CallToolResult::error(vec![Content::text(format!(
        "Indexing failed: {}",
        e
      ))])),
    }
  }

  #[tool(description = "Index a single file for semantic code search")]
  async fn index_file(
    &self,
    #[tool(aggr)] IndexFileRequest { path, content }: IndexFileRequest,
  ) -> Result<CallToolResult, McpError> {
    info!("Indexing file: {}", path);

    let file_path = PathBuf::from(&path);

    // If content is not provided, check that file exists
    if content.is_none() && !file_path.exists() {
      return Ok(CallToolResult::error(vec![Content::text(format!(
        "File '{}' does not exist",
        path
      ))]));
    }

    match self.indexer.index_file(&file_path, content).await {
      Ok(()) => Ok(CallToolResult::success(vec![Content::text(format!(
        "Successfully indexed file: {}",
        path
      ))])),
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
pub fn create_http_service(
  indexer: Arc<Indexer>,
  shutdown_token: CancellationToken,
) -> StreamableHttpService<BreezeService> {
  StreamableHttpService::new(
    move || Ok(BreezeService::new(indexer.clone(), shutdown_token.clone())),
    LocalSessionManager::default().into(),
    Default::default(),
  )
}

/// Create an SSE MCP server
pub fn create_sse_server(
  ct: CancellationToken,
  sse_path: String,
  post_path: String,
  bind_address: std::net::SocketAddr,
) -> (SseServer, axum::Router) {
  let config = SseServerConfig {
    bind: bind_address,
    sse_path,
    post_path,
    ct,
    sse_keep_alive: Some(std::time::Duration::from_secs(30)),
  };

  SseServer::new(config)
}

/// Start the SSE service with the BreezeService handler
pub fn start_sse_service(
  sse_server: SseServer,
  indexer: Arc<Indexer>,
  shutdown_token: CancellationToken,
) -> CancellationToken {
  sse_server.with_service(move || BreezeService::new(indexer.clone(), shutdown_token.clone()))
}
