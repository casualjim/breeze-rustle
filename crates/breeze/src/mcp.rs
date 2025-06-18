use async_trait::async_trait;
use mcp_protocol_sdk::{
  core::{
    error::{McpError, McpResult},
    tool::ToolHandler,
  },
  protocol::types::{Content, ToolResult},
  server::{HttpMcpServer, McpServer, mcp_server::ServerConfig},
  transport::{http::HttpServerTransport, stdio::StdioServerTransport},
};
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::{App, Config};

/// MCP server for breeze code search
pub struct BreezeMcpServer {
  app: Arc<Mutex<App>>,
}

impl BreezeMcpServer {
  pub async fn new(config: Config) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
    let app = App::new(config)
      .await
      .map_err(|e| -> Box<dyn std::error::Error + Send + Sync> {
        Box::new(std::io::Error::other(e.to_string()))
      })?;
    Ok(Self {
      app: Arc::new(Mutex::new(app)),
    })
  }

  /// Run the MCP server in stdio mode
  pub async fn run_stdio(self) -> McpResult<()> {
    tracing::info!("Starting breeze MCP server in STDIO mode");

    let config = ServerConfig {
      max_concurrent_requests: 50,
      request_timeout_ms: 30000,
      validate_requests: true,
      enable_logging: true,
    };

    let mut server = McpServer::with_config(
      "breeze-mcp-server".to_string(),
      env!("CARGO_PKG_VERSION").to_string(),
      config,
    );

    // Add tools
    self.register_tools(&mut server).await?;

    // Start with stdio transport
    let transport = StdioServerTransport::new();
    server.start(transport).await?;

    // Keep running until interrupted
    tokio::signal::ctrl_c()
      .await
      .expect("Failed to listen for ctrl+c");
    server.stop().await?;

    Ok(())
  }

  /// Run the MCP server in HTTP mode
  pub async fn run_http(self, host: &str, port: u16) -> McpResult<()> {
    tracing::info!(
      "Starting breeze MCP server in HTTP mode on {}:{}",
      host,
      port
    );

    let mut http_server = HttpMcpServer::new(
      "breeze-mcp-server".to_string(),
      env!("CARGO_PKG_VERSION").to_string(),
    );

    // Get a reference to the underlying server for adding tools
    let server = http_server.server().await;

    // Add tools to the locked server
    {
      let server_guard = server.lock().await;
      self.register_tools_to_guard(&server_guard).await?;
    }

    // Start HTTP server
    tracing::info!("HTTP MCP server endpoints:");
    tracing::info!("  - POST /mcp - JSON-RPC requests");
    tracing::info!("  - POST /mcp/notify - Notifications");
    tracing::info!("  - GET /mcp/events - Server-Sent Events");
    tracing::info!("  - GET /health - Health check");

    let addr = format!("{}:{}", host, port);
    let transport = HttpServerTransport::new(&addr);
    http_server.start(transport).await?;

    // Keep running until interrupted
    tokio::signal::ctrl_c()
      .await
      .expect("Failed to listen for ctrl+c");
    http_server.stop().await?;

    Ok(())
  }

  /// Register all breeze tools with the MCP server
  async fn register_tools(&self, server: &mut McpServer) -> McpResult<()> {
    // Search tool
    server
      .add_tool(
        "search_code".to_string(),
        Some("Search indexed codebase for relevant files".to_string()),
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (natural language or keywords)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100
                }
            },
            "required": ["query"]
        }),
        SearchHandler {
          app: Arc::clone(&self.app),
        },
      )
      .await?;

    // Index tool
    server
      .add_tool(
        "index_directory".to_string(),
        Some("Index a directory of code files for searching".to_string()),
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to index"
                }
            },
            "required": ["path"]
        }),
        IndexHandler {
          app: Arc::clone(&self.app),
        },
      )
      .await?;

    Ok(())
  }

  /// Register tools to a guard (for HTTP server)
  async fn register_tools_to_guard(&self, server: &McpServer) -> McpResult<()> {
    // Search tool
    server
      .add_tool(
        "search_code".to_string(),
        Some("Search indexed codebase for relevant files".to_string()),
        json!({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (natural language or keywords)"
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 100
                }
            },
            "required": ["query"]
        }),
        SearchHandler {
          app: Arc::clone(&self.app),
        },
      )
      .await?;

    // Index tool
    server
      .add_tool(
        "index_directory".to_string(),
        Some("Index a directory of code files for searching".to_string()),
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the directory to index"
                }
            },
            "required": ["path"]
        }),
        IndexHandler {
          app: Arc::clone(&self.app),
        },
      )
      .await?;

    Ok(())
  }
}

/// Handler for the search_code tool
struct SearchHandler {
  app: Arc<Mutex<App>>,
}

#[async_trait]
impl ToolHandler for SearchHandler {
  async fn call(&self, arguments: HashMap<String, Value>) -> McpResult<ToolResult> {
    let query = arguments
      .get("query")
      .and_then(|v| v.as_str())
      .ok_or_else(|| McpError::Validation("Missing 'query' parameter".to_string()))?;

    let limit = arguments
      .get("limit")
      .and_then(|v| v.as_u64())
      .map(|v| v as usize)
      .unwrap_or(10);

    // Perform search
    let app = self.app.lock().await;
    let results = app
      .search(query, limit)
      .await
      .map_err(|e| McpError::Internal(e.to_string()))?;

    // Format results
    let formatted_results = serde_json::to_string(&results)?;

    Ok(ToolResult {
      content: vec![Content::text(formatted_results)],
      is_error: None,
      meta: None,
    })
  }
}

/// Handler for the index_directory tool
struct IndexHandler {
  app: Arc<Mutex<App>>,
}

#[async_trait]
impl ToolHandler for IndexHandler {
  async fn call(&self, arguments: HashMap<String, Value>) -> McpResult<ToolResult> {
    let path = arguments
      .get("path")
      .and_then(|v| v.as_str())
      .ok_or_else(|| McpError::Validation("Missing 'path' parameter".to_string()))?;

    let path = std::path::Path::new(path);
    if !path.exists() {
      return Ok(ToolResult {
        content: vec![Content::text(format!(
          "Error: Path '{}' does not exist",
          path.display()
        ))],
        is_error: Some(true),
        meta: None,
      });
    }

    if !path.is_dir() {
      return Ok(ToolResult {
        content: vec![Content::text(format!(
          "Error: Path '{}' is not a directory",
          path.display()
        ))],
        is_error: Some(true),
        meta: None,
      });
    }

    // Perform indexing
    let app = self.app.lock().await;
    match app.index(path).await {
      Ok(_) => Ok(ToolResult {
        content: vec![Content::text(format!(
          "Successfully indexed directory: {}",
          path.display()
        ))],
        is_error: None,
        meta: None,
      }),
      Err(e) => Ok(ToolResult {
        content: vec![Content::text(format!("Error indexing directory: {}", e))],
        is_error: Some(true),
        meta: None,
      }),
    }
  }
}
