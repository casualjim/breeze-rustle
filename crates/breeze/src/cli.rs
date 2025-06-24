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
  },

  /// Search indexed codebase
  Search {
    /// Search query
    query: String,

    /// Number of results to return
    #[arg(short = 'n', long, default_value = "10")]
    limit: usize,

    /// Show full file content in results
    #[arg(short, long)]
    full: bool,
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

/// Format search results for display
pub fn format_results(results: &[breeze_indexer::SearchResult], show_full_content: bool) -> String {
  let mut output = String::new();

  for (idx, result) in results.iter().enumerate() {
    output.push_str(&format!("\n[{}] {}\n", idx + 1, result.file_path));

    // Show relevance score and chunk count
    output.push_str(&format!(
      "   Score: {:.4} | Chunks: {}\n",
      result.relevance_score, result.chunk_count
    ));

    // Show chunk previews
    if !result.chunks.is_empty() {
      output.push_str(&format!(
        "   Found {} relevant chunks:\n",
        result.chunks.len()
      ));

      for (chunk_idx, chunk) in result.chunks.iter().enumerate() {
        output.push_str(&format!(
          "\n   Chunk {} (lines {}-{}, score: {:.4}):\n",
          chunk_idx + 1,
          chunk.start_line,
          chunk.end_line,
          chunk.relevance_score
        ));

        if show_full_content {
          // Show full chunk content
          for line in chunk.content.lines() {
            output.push_str("      ");
            output.push_str(line);
            output.push('\n');
          }
        } else {
          // Show first 5 lines of chunk as preview
          let preview_lines: Vec<&str> = chunk.content.lines().take(5).collect();
          for line in preview_lines {
            output.push_str("      ");
            output.push_str(line);
            output.push('\n');
          }

          let total_lines = chunk.content.lines().count();
          if total_lines > 5 {
            output.push_str(&format!("      ... ({} more lines)\n", total_lines - 5));
          }
        }
      }
    }

    output.push_str(&format!("{}\n", "-".repeat(80)));
  }

  output
}
