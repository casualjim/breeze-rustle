use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "breeze")]
#[command(about = "High-performance semantic code indexing and search", long_about = None)]
#[command(version)]
pub struct Cli {
  /// Path to configuration file
  #[arg(short, long, global = true)]
  pub config: Option<PathBuf>,

  #[command(subcommand)]
  pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
  /// Manage projects
  #[command(subcommand)]
  Project(ProjectCommands),

  /// Manage tasks
  #[command(subcommand)]
  Task(TaskCommands),

  /// Search indexed codebase
  Search {
    /// Search query
    query: String,

    /// Number of results to return
    #[arg(short = 'n', long, default_value = "5")]
    limit: Option<usize>,

    /// Number of chunks per file
    #[arg(long, default_value = "3")]
    chunks_per_file: Option<usize>,

    /// Filter by languages (comma-separated)
    #[arg(long, value_delimiter = ',')]
    languages: Option<Vec<String>>,

    /// Search granularity (document or chunk)
    #[arg(long, value_enum)]
    granularity: Option<SearchGranularity>,

    /// Filter by node types (comma-separated, chunk mode only)
    #[arg(long, value_delimiter = ',')]
    node_types: Option<Vec<String>>,

    /// Filter by node name pattern (chunk mode only)
    #[arg(long)]
    node_name_pattern: Option<String>,

    /// Filter by parent context pattern (chunk mode only)
    #[arg(long)]
    parent_context_pattern: Option<String>,

    /// Filter by scope depth range (format: min,max)
    #[arg(long, value_parser = parse_scope_depth)]
    scope_depth: Option<(usize, usize)>,

    /// Filter by definitions (comma-separated, chunk mode only)
    #[arg(long, value_delimiter = ',')]
    has_definitions: Option<Vec<String>>,

    /// Filter by references (comma-separated, chunk mode only)
    #[arg(long, value_delimiter = ',')]
    has_references: Option<Vec<String>>,
  },

  /// Initialize configuration file
  Init {
    /// Force overwrite existing config file
    #[arg(short, long)]
    force: bool,
  },

  /// Show current configuration
  Config {
    /// Show default configuration
    #[arg(long)]
    defaults: bool,
  },

  /// Debug utilities
  #[cfg(feature = "perfprofiling")]
  Debug {
    #[command(subcommand)]
    command: DebugCommands,
  },

  /// Run the API server (uses config file for all settings)
  Serve,
}

#[derive(Subcommand)]
pub enum ProjectCommands {
  /// Create a new project
  Create {
    /// Project name
    name: String,
    /// Project directory
    directory: PathBuf,
    /// Project description
    #[arg(short, long)]
    description: Option<String>,
  },
  /// List all projects
  List,
  /// Show project details
  Show {
    /// Project ID
    id: String,
  },
  /// Update project
  Update {
    /// Project ID
    id: String,
    /// New name
    #[arg(short, long)]
    name: Option<String>,
    /// New description
    #[arg(short, long)]
    description: Option<String>,
  },
  /// Delete project
  Delete {
    /// Project ID
    id: String,
  },
  /// Index project
  Index {
    /// Project ID
    id: String,
  },
}

#[derive(Subcommand)]
pub enum TaskCommands {
  /// Show task details
  Show {
    /// Task ID
    id: String,
  },
  /// List recent tasks
  List {
    /// Maximum number of tasks to show
    #[arg(short, long, default_value = "20")]
    limit: Option<usize>,
  },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum SearchGranularity {
  Document,
  Chunk,
}

fn parse_scope_depth(s: &str) -> Result<(usize, usize), String> {
  let parts: Vec<&str> = s.split(',').collect();
  if parts.len() != 2 {
    return Err("Scope depth must be in format: min,max".to_string());
  }
  let min = parts[0].parse().map_err(|_| "Invalid min value")?;
  let max = parts[1].parse().map_err(|_| "Invalid max value")?;
  Ok((min, max))
}

#[cfg(feature = "perfprofiling")]
#[derive(Subcommand)]
pub enum DebugCommands {
  /// Chunk a directory and show statistics
  ChunkDirectory {
    /// Path to the directory to chunk
    path: PathBuf,

    /// Maximum chunk size in tokens
    #[arg(long, default_value = "2048")]
    max_chunk_size: usize,

    /// Maximum file size in bytes
    #[arg(long)]
    max_file_size: Option<u64>,

    /// Number of files to process in parallel
    #[arg(long, default_value = "16")]
    max_parallel: usize,

    /// Tokenizer to use: characters, tiktoken:model_name, or hf:org/repo
    #[arg(long, default_value = "hf:ibm-granite/granite-embedding-125m-english")]
    tokenizer: String,
  },
}

impl Cli {
  pub fn parse() -> Self {
    <Self as Parser>::parse()
  }
}
