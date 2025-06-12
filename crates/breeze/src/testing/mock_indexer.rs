use async_stream::stream;
use std::path::Path;

use crate::pipeline::*;
use breeze_chunkers::{Chunk, ChunkMetadata, ProjectChunk, SemanticChunk};

/// Mock indexer that generates example chunks for testing
pub struct MockPathWalker {
  files_per_repo: usize,
  chunks_per_file: usize,
}

impl MockPathWalker {
  pub fn new(files_per_repo: usize, chunks_per_file: usize) -> Self {
    Self {
      files_per_repo,
      chunks_per_file,
    }
  }
}

impl Default for MockPathWalker {
  fn default() -> Self {
    Self::new(3, 2)
  }
}

impl PathWalker for MockPathWalker {
  fn walk(&self, path: &Path) -> BoxStream<ProjectChunk> {
    let path_str = path.to_string_lossy().to_string();
    let files_per_repo = self.files_per_repo;
    let chunks_per_file = self.chunks_per_file;

    Box::pin(stream! {
        // Simulate finding some files
        for i in 0..files_per_repo {
            let file_path = format!("{}/file{}.rs", path_str, i);

            // Generate a few chunks per file
            for j in 0..chunks_per_file {
                let chunk = ProjectChunk {
                    file_path: file_path.clone(),
                    chunk: Chunk::Semantic(SemanticChunk {
                        text: format!("fn function_{}_{}_{}() {{\n    println!(\"Hello from function {} in file {}\");\n}}",
                            i, j, path_str.replace('/', "_"), j, i),
                        start_byte: j * 100,
                        end_byte: (j + 1) * 100,
                        start_line: j * 5 + 1,
                        end_line: (j + 1) * 5,
                        metadata: ChunkMetadata {
                            node_type: "function".to_string(),
                            node_name: Some(format!("function_{}_{}", i, j)),
                            language: "rust".to_string(),
                            parent_context: Some("module".to_string()),
                            scope_path: vec!["module".to_string(), format!("function_{}_{}", i, j)],
                            definitions: vec![format!("function_{}_{}", i, j)],
                            references: vec!["println".to_string()],
                        },
                    }),
                };
                yield chunk;
            }
        }
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use futures_util::StreamExt;

  #[tokio::test]
  async fn test_mock_indexer() {
    let walker = MockPathWalker::new(2, 3);
    let path = Path::new("/test/repo");

    let mut chunks = walker.walk(path);
    let mut count = 0;

    while let Some(chunk) = chunks.next().await {
      assert!(chunk.file_path.starts_with("/test/repo/"));
      assert!(chunk.file_path.ends_with(".rs"));
      count += 1;
    }

    assert_eq!(count, 6); // 2 files * 3 chunks
  }
}
