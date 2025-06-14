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
    /// Index a codebase for semantic search
    Index {
        /// Path to the codebase to index
        path: PathBuf,

        /// Database path (overrides config)
        #[arg(short, long)]
        database: Option<PathBuf>,

        /// Model to use for embeddings (overrides config)
        #[arg(short, long)]
        model: Option<String>,

        /// Device to run on: cpu, mps, cuda (overrides config)
        #[arg(long)]
        device: Option<String>,

        /// Maximum chunk size in tokens (overrides config)
        #[arg(long)]
        max_chunk_size: Option<usize>,

        /// Maximum file size in bytes (overrides config)
        #[arg(long)]
        max_file_size: Option<u64>,

        /// Number of files to process in parallel (overrides config)
        #[arg(long)]
        max_parallel_files: Option<usize>,

        /// Batch size for embedding operations (overrides config)
        #[arg(long)]
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
