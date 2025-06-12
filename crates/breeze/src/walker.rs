use futures_util::StreamExt;
use std::path::Path;
use tracing::{debug, error, info, warn};

use crate::pipeline::{BoxStream, PathWalker};
use breeze_chunkers::{ProjectFile, WalkOptions, walk_project_streaming};

/// Default implementation of PathWalker that uses breeze_chunkers::walk_project
#[derive(Debug, Clone, Default)]
pub struct ProjectWalker {
  options: WalkOptions,
}

impl ProjectWalker {
  pub fn new(options: WalkOptions) -> Self {
    Self { options }
  }
}

impl PathWalker for ProjectWalker {
  fn walk(&self, path: &Path) -> BoxStream<ProjectFile> {
    // Convert path to owned PathBuf to avoid lifetime issues
    let path_buf = path.to_path_buf();
    let options = self.options.clone();

    info!(
      path = %path.display(),
      max_chunk_size = options.max_chunk_size,
      max_parallel = options.max_parallel,
      max_file_size = ?options.max_file_size,
      "Starting project walk with streaming architecture"
    );

    // Create the stream from walk_project_streaming
    let stream = walk_project_streaming(path_buf, options);

    // Filter out errors and log them, converting to BoxStream
    let filtered_stream = stream.filter_map(|result| async move {
      match result {
        Ok(project_file) => {
          info!(
            file_path = %project_file.file_path,
            file_size = project_file.metadata.size,
            language = ?project_file.metadata.primary_language,
            line_count = project_file.metadata.line_count,
            content_hash = %project_file.metadata.content_hash,
            "Processing file"
          );
          Some(project_file)
        },
        Err(e) => {
          error!(
            error = %e,
            "Error processing file"
          );
          None
        }
      }
    });

    Box::pin(filtered_stream)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use breeze_chunkers::Tokenizer;
  use futures_util::StreamExt;
  use std::fs;
  use tempfile::TempDir;

  async fn create_test_files() -> TempDir {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();

    // Create a simple Python file
    fs::write(
      base_path.join("test.py"),
      r#"def hello():
    print("Hello, world!")

def goodbye():
    print("Goodbye!")
"#,
    )
    .unwrap();

    // Create a text file
    fs::write(
      base_path.join("readme.txt"),
      "This is a simple test file.\nIt has multiple lines.\n",
    )
    .unwrap();

    temp_dir
  }

  #[tokio::test]
  async fn test_project_walker_basic() {
    let temp_dir = create_test_files().await;
    let walker = ProjectWalker::default();

    let mut stream = walker.walk(temp_dir.path());
    let mut chunks = Vec::new();

    while let Some(chunk) = stream.next().await {
      chunks.push(chunk);
    }

    assert!(!chunks.is_empty(), "Should find some chunks");

    // Verify we got chunks from both files
    let file_paths: std::collections::HashSet<_> = chunks.iter().map(|c| &c.file_path).collect();
    assert_eq!(file_paths.len(), 2, "Should have chunks from both files");
  }

  #[tokio::test]
  async fn test_project_walker_custom_options() {
    let temp_dir = create_test_files().await;

    let options = WalkOptions {
      max_chunk_size: 50,
      tokenizer: Tokenizer::Characters,
      max_parallel: 2,
      max_file_size: Some(1024),
    };

    let walker = ProjectWalker::new(options);
    let mut stream = walker.walk(temp_dir.path());

    let mut chunk_count = 0;
    while let Some(_) = stream.next().await {
      chunk_count += 1;
    }

    // With smaller chunk size, we should get more chunks
    assert!(
      chunk_count >= 2,
      "Should get multiple chunks with small chunk size"
    );
  }
}
