use futures_util::StreamExt;
use std::path::Path;

use crate::pipeline::{BoxStream, PathWalker};
use breeze_chunkers::{ProjectChunk, WalkOptions, walk_project};

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
    fn walk(&self, path: &Path) -> BoxStream<ProjectChunk> {
        // Convert path to owned PathBuf to avoid lifetime issues
        let path_buf = path.to_path_buf();
        let options = self.options.clone();

        // Create the stream from walk_project
        let stream = walk_project(path_buf, options);

        // Filter out errors and log them, converting to BoxStream
        let filtered_stream = stream.filter_map(|result| async move {
            match result {
                Ok(chunk) => Some(chunk),
                Err(e) => {
                    eprintln!("Error processing chunk: {}", e);
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
        let file_paths: std::collections::HashSet<_> =
            chunks.iter().map(|c| &c.file_path).collect();
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
