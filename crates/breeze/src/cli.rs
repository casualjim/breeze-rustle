use clap::{Parser, Subcommand};
use std::path::PathBuf;
use crate::aiproviders::voyage::{EmbeddingModel as VoyageModel, Tier as VoyageTier};
use crate::config::EmbeddingProvider;

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
  /// Index a codebase for semantic search
  Index {
    /// Path to the codebase to index
    path: PathBuf,

    /// Database path (overrides config)
    #[arg(short, long, env = "BREEZE_DATABASE_PATH")]
    database: Option<PathBuf>,

    /// Embedding provider: local or voyage (overrides config)
    #[arg(short = 'p', long, env = "BREEZE_EMBEDDING_PROVIDER")]
    embedding_provider: Option<EmbeddingProvider>,

    /// Model to use for embeddings (overrides config)
    #[arg(short, long, env = "BREEZE_MODEL")]
    model: Option<String>,

    /// Voyage API key (overrides config/env)
    #[arg(long, env = "VOYAGE_API_KEY", alias = "breeze-voyage-api-key")]
    voyage_api_key: Option<String>,

    /// Voyage tier: free, tier1, tier2, tier3 (overrides config)
    #[arg(long, env = "BREEZE_VOYAGE_TIER")]
    voyage_tier: Option<VoyageTier>,

    /// Voyage model: voyage-3, voyage-3-lite, voyage-code-3 (overrides config)
    #[arg(long, env = "BREEZE_VOYAGE_MODEL")]
    voyage_model: Option<VoyageModel>,

    /// Maximum chunk size in tokens (overrides config)
    #[arg(long, env = "BREEZE_MAX_CHUNK_SIZE")]
    max_chunk_size: Option<usize>,

    /// Maximum file size in bytes (overrides config)
    #[arg(long, env = "BREEZE_MAX_FILE_SIZE")]
    max_file_size: Option<u64>,

    /// Number of files to process in parallel (overrides config)
    #[arg(long, env = "BREEZE_MAX_PARALLEL_FILES")]
    max_parallel_files: Option<usize>,

    /// Batch size for embedding operations (overrides config)
    #[arg(long, env = "BREEZE_BATCH_SIZE")]
    batch_size: Option<usize>,
  },

  /// Search indexed codebase
  Search {
    /// Search query
    query: String,

    /// Database path (overrides config)
    #[arg(short, long)]
    database: Option<PathBuf>,

    /// Number of results to return
    #[arg(short = 'n', long, default_value = "10")]
    limit: usize,

    /// Show full file content in results
    #[arg(short, long)]
    full: bool,
  },

  /// Show current configuration
  Config {
    /// Show default configuration
    #[arg(long)]
    defaults: bool,
  },

  /// Debug utilities
  Debug {
    #[command(subcommand)]
    command: DebugCommands,
  },
}

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
