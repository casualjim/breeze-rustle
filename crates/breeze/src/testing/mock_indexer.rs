use async_stream::stream;
use std::path::Path;
use std::time::SystemTime;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::pipeline::*;
use breeze_chunkers::{Chunk, ChunkError, ChunkMetadata, FileMetadata, ProjectFile, SemanticChunk};

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
  fn walk(&self, path: &Path) -> BoxStream<ProjectFile> {
    let path_str = path.to_string_lossy().to_string();
    let files_per_repo = self.files_per_repo;
    let chunks_per_file = self.chunks_per_file;

    Box::pin(stream! {
        // Simulate finding some files
        for i in 0..files_per_repo {
            let file_path = format!("{}/file{}.rs", path_str, i);

            // Create a channel for chunks
            let (chunk_tx, chunk_rx) = mpsc::channel::<Result<Chunk, ChunkError>>(32);

            // Spawn task to generate chunks
            let chunks_count = chunks_per_file;
            let file_idx = i;
            let path_str_clone = path_str.clone();
            tokio::spawn(async move {
                for j in 0..chunks_count {
                    let chunk = Chunk::Semantic(SemanticChunk {
                        text: format!("fn function_{}_{}_{}() {{\n    println!(\"Hello from function {} in file {}\");\n}}",
                            file_idx, j, path_str_clone.replace('/', "_"), j, file_idx),
                        start_byte: j * 100,
                        end_byte: (j + 1) * 100,
                        start_line: j * 5 + 1,
                        end_line: (j + 1) * 5,
                        tokens: None,
                        metadata: ChunkMetadata {
                            node_type: "function".to_string(),
                            node_name: Some(format!("function_{}_{}", file_idx, j)),
                            language: "rust".to_string(),
                            parent_context: Some("module".to_string()),
                            scope_path: vec!["module".to_string(), format!("function_{}_{}", file_idx, j)],
                            definitions: vec![format!("function_{}_{}", file_idx, j)],
                            references: vec!["println".to_string()],
                        },
                    });
                    let _ = chunk_tx.send(Ok(chunk)).await;
                }
            });

            // Create file metadata
            let metadata = FileMetadata {
                primary_language: Some("Rust".to_string()),
                size: (chunks_per_file * 100) as u64,
                modified: SystemTime::now(),
                content_hash: format!("hash_file_{}", i),
                line_count: chunks_per_file * 5,
                is_binary: false,
            };

            yield ProjectFile {
                file_path,
                chunks: ReceiverStream::new(chunk_rx),
                metadata,
            };
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

    let mut files = walker.walk(path);
    let mut file_count = 0;
    let mut total_chunks = 0;

    while let Some(project_file) = files.next().await {
      assert!(project_file.file_path.starts_with("/test/repo/"));
      assert!(project_file.file_path.ends_with(".rs"));
      file_count += 1;

      // Count chunks in this file
      let mut chunk_stream = project_file.chunks;
      while let Some(chunk_result) = chunk_stream.next().await {
        assert!(chunk_result.is_ok());
        total_chunks += 1;
      }
    }

    assert_eq!(file_count, 2); // 2 files
    assert_eq!(total_chunks, 6); // 2 files * 3 chunks
  }
}
