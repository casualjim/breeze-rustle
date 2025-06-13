use crate::{
  Tokenizer,
  chunker::InnerChunker,
  languages::get_language,
  types::{Chunk, ChunkError, FileMetadata, ProjectChunk, ProjectFile},
};
use async_stream;
use blake3::Hasher;
use futures::{Stream, StreamExt};
use ignore::{DirEntry, WalkBuilder, WalkState};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{debug, error};

/// Check if an entry should be traversed (for directories) or processed (for files)
fn should_process_entry(entry: &DirEntry) -> bool {
  // Always traverse directories
  if entry.file_type().map_or(false, |ft| ft.is_dir()) {
    return true;
  }

  // For files, apply our filtering logic
  if !entry.file_type().map_or(false, |ft| ft.is_file()) {
    return false;
  }

  let path = entry.path();

  // Skip empty files
  if let Ok(metadata) = entry.metadata() {
    if metadata.len() == 0 {
      debug!("Skipping empty file: {}", path.display());
      return false;
    }
  }

  // Skip binary files
  let is_text_file = if let Ok(Some(file_type)) = infer::get_from_path(&path) {
    file_type.matcher_type() == infer::MatcherType::Text
  } else {
    // If infer can't determine, check with hyperpolyglot
    hyperpolyglot::detect(&path).is_ok()
  };

  if !is_text_file {
    debug!("Skipping binary file: {}", path.display());
    return false;
  }

  true
}

/// Options for walking a project directory
#[derive(Debug, Clone)]
pub struct WalkOptions {
  pub max_chunk_size: usize,
  pub tokenizer: Tokenizer,
  pub max_parallel: usize,
  pub max_file_size: Option<u64>,
}

impl Default for WalkOptions {
  fn default() -> Self {
    Self {
      max_chunk_size: 1000,
      tokenizer: Tokenizer::Characters,
      max_parallel: 4,
      max_file_size: Some(5 * 1024 * 1024), // 5MB default
    }
  }
}

/// Walk a project directory with options
pub fn walk_project(
  path: impl AsRef<Path>,
  options: WalkOptions,
) -> impl Stream<Item = Result<ProjectChunk, ChunkError>> {
  let path = path.as_ref().to_owned();
  let max_file_size = options.max_file_size;

  // Create a channel for streaming results with buffer matching parallelism
  let (tx, rx) = mpsc::channel::<Result<ProjectChunk, ChunkError>>(options.max_parallel);

  // Create the chunker upfront
  let chunker = match InnerChunker::new(options.max_chunk_size, options.tokenizer) {
    Ok(c) => Arc::new(c),
    Err(e) => {
      // Send error and return early
      let tx_clone = tx.clone();
      tokio::spawn(async move {
        let _ = tx_clone.send(Err(e)).await;
      });
      return ReceiverStream::new(rx);
    }
  };

  // Spawn blocking task for the walker
  tokio::task::spawn_blocking(move || {
    // Use tokio runtime handle to spawn async tasks from blocking context
    let handle = tokio::runtime::Handle::current();

    // Build the walker with parallelism
    let walker = WalkBuilder::new(&path)
      .threads(options.max_parallel)
      .max_filesize(max_file_size)
      .filter_entry(should_process_entry)
      .build_parallel();

    let tx = Arc::new(tx);

    walker.run(|| {
      let tx = tx.clone();
      let chunker = chunker.clone();
      let handle = handle.clone();

      Box::new(move |result| {
        if let Ok(entry) = result {
          let path = entry.path().to_owned();
          let tx = tx.clone();
          let chunker = chunker.clone();

          // Spawn async task on the runtime
          handle.spawn(async move {
            let mut stream = Box::pin(process_file(&path, chunker));
            while let Some(result) = stream.next().await {
              match result {
                Ok(chunk) => {
                  if tx.send(Ok(chunk)).await.is_err() {
                    return; // Receiver dropped
                  }
                }
                Err(e) => {
                  // Log based on error type
                  match &e {
                    ChunkError::IoError(io_err)
                      if io_err.kind() == std::io::ErrorKind::InvalidData =>
                    {
                      debug!("Skipping non-UTF8 file {}: {}", path.display(), e);
                    }
                    _ => {
                      error!("Error processing {}: {}", path.display(), e);
                    }
                  }
                }
              }
            }
          });
        }
        WalkState::Continue
      })
    });
  });

  ReceiverStream::new(rx)
}

/// Walk a project directory and yield ProjectFile items with chunk streams
pub fn walk_project_streaming(
  path: impl AsRef<Path>,
  options: WalkOptions,
) -> impl Stream<Item = Result<ProjectFile, ChunkError>> {
  let path = path.as_ref().to_owned();
  let max_file_size = options.max_file_size;

  // Create a channel for streaming results with buffer matching parallelism
  let (tx, rx) = mpsc::channel::<Result<ProjectFile, ChunkError>>(options.max_parallel);

  // Create the chunker upfront
  let chunker = match InnerChunker::new(options.max_chunk_size, options.tokenizer) {
    Ok(c) => Arc::new(c),
    Err(e) => {
      // Send error and return early
      let tx_clone = tx.clone();
      tokio::spawn(async move {
        let _ = tx_clone.send(Err(e)).await;
      });
      return ReceiverStream::new(rx);
    }
  };

  // Spawn blocking task for the walker
  tokio::task::spawn_blocking(move || {
    // Use tokio runtime handle to spawn async tasks from blocking context
    let handle = tokio::runtime::Handle::current();

    // Build the walker with parallelism
    let walker = WalkBuilder::new(&path)
      .threads(options.max_parallel)
      .max_filesize(max_file_size)
      .filter_entry(should_process_entry)
      .build_parallel();

    let tx = Arc::new(tx);

    walker.run(|| {
      let tx = tx.clone();
      let chunker = chunker.clone();
      let handle = handle.clone();

      Box::new(move |result| {
        if let Ok(entry) = result {
          // Only process files, not directories
          if entry.file_type().map_or(false, |ft| ft.is_file()) {
            let path = entry.path().to_owned();
            let tx = tx.clone();
            let chunker = chunker.clone();

            // Spawn async task on the runtime
            handle.spawn(async move {
              // Process file with metadata
              match process_file_with_metadata(&path, chunker).await {
                Ok(project_file) => {
                  if tx.send(Ok(project_file)).await.is_err() {
                    return; // Receiver dropped
                  }
                }
                Err(e) => {
                  // Log based on error type
                  match &e {
                    ChunkError::IoError(io_err)
                      if io_err.kind() == std::io::ErrorKind::InvalidData =>
                    {
                      debug!("Skipping non-UTF8 file {}: {}", path.display(), e);
                    }
                    _ => {
                      error!("Error processing {}: {}", path.display(), e);
                    }
                  }
                  if tx.send(Err(e)).await.is_err() {
                    return; // Receiver dropped
                  }
                }
              }
            });
          }
        }
        WalkState::Continue
      })
    });
  });

  ReceiverStream::new(rx)
}

/// Process a file and create a ProjectFile with metadata and chunk stream
async fn process_file_with_metadata<P: AsRef<Path>>(
  path: P,
  chunker: Arc<InnerChunker>,
) -> Result<ProjectFile, ChunkError> {
  let path = path.as_ref();
  let path_str = path.to_string_lossy().to_string();

  // Get file metadata
  let metadata = tokio::fs::metadata(&path)
    .await
    .map_err(ChunkError::IoError)?;

  // Read content and compute hash
  let content = tokio::fs::read_to_string(&path)
    .await
    .map_err(ChunkError::IoError)?;

  // Compute content hash
  let mut hasher = Hasher::new();
  hasher.update(content.as_bytes());
  let hash = hasher.finalize();
  let content_hash = format!("{}", hash.to_hex());

  // Count lines
  let line_count = content.matches('\n').count() + 1;

  // Detect primary language
  let primary_language = if let Ok(Some(detection)) = hyperpolyglot::detect(&path) {
    Some(detection.language().to_string())
  } else {
    None
  };

  // Get modified time
  let modified = metadata.modified().unwrap_or(std::time::SystemTime::now());

  // Create file metadata
  let file_metadata = FileMetadata {
    primary_language,
    size: metadata.len(),
    modified,
    content_hash,
    line_count,
    is_binary: false,
  };

  // Create channel for chunks with reasonable buffer for a single file
  let (chunk_tx, chunk_rx) = mpsc::channel::<Result<Chunk, ChunkError>>(32);

  // Spawn task to process chunks
  let chunker_clone = chunker.clone();
  let path_str_clone = path_str.clone();
  let content_clone = content.clone();

  tokio::spawn(async move {
    // Check if hyperpolyglot detects a supported language
    let detected_language =
      if let Ok(Some(detection)) = hyperpolyglot::detect(Path::new(&path_str_clone)) {
        let language = detection.language();
        if get_language(language).is_some() {
          Some(language.to_string())
        } else {
          None
        }
      } else {
        None
      };

    // Try semantic chunking first if we have a supported language
    if let Some(language) = detected_language {
      let mut chunk_stream = Box::pin(chunker_clone.chunk_code(
        content_clone.clone(),
        language.clone(),
        Some(path_str_clone.clone()),
      ));
      let mut had_success = false;

      while let Some(chunk_result) = chunk_stream.next().await {
        match chunk_result {
          Ok(chunk) => {
            had_success = true;
            if chunk_tx.send(Ok(chunk)).await.is_err() {
              return; // Receiver dropped
            }
          }
          Err(_) => {
            // If we haven't had any successful chunks, fall through to text chunking
            if !had_success {
              break;
            }
          }
        }
      }

      if had_success {
        return;
      }
    }

    // Fall back to text chunking
    let mut chunk_stream =
      Box::pin(chunker_clone.chunk_text(content_clone, Some(path_str_clone.clone())));
    while let Some(chunk_result) = chunk_stream.next().await {
      match chunk_result {
        Ok(chunk) => {
          if chunk_tx.send(Ok(chunk)).await.is_err() {
            return; // Receiver dropped
          }
        }
        Err(e) => {
          let _ = chunk_tx.send(Err(e)).await;
          return;
        }
      }
    }
  });

  Ok(ProjectFile {
    file_path: path_str,
    chunks: ReceiverStream::new(chunk_rx),
    metadata: file_metadata,
  })
}

/// Process a single file and yield chunks as a stream
fn process_file<P: AsRef<Path>>(
  path: P,
  chunker: Arc<InnerChunker>,
) -> impl Stream<Item = Result<ProjectChunk, ChunkError>> + Send {
  let path = path.as_ref().to_owned();

  async_stream::try_stream! {
      let path_str = path.to_string_lossy().to_string();

      // First check if it's a text file using infer (this only reads a few bytes)
      let is_text_file = if let Ok(Some(file_type)) = infer::get_from_path(&path) {
          file_type.matcher_type() == infer::MatcherType::Text
      } else {
          // If infer can't determine, check with hyperpolyglot
          hyperpolyglot::detect(&path).is_ok()
      };

      if !is_text_file {
          return; // Skip binary files
      }

      // Now we know it's a text file, read it once
      let content = tokio::fs::read_to_string(&path)
          .await
          .map_err(|e| ChunkError::IoError(e))?;

      if content.is_empty() {
          return;
      }

      // Compute content hash
      let mut hasher = Hasher::new();
      hasher.update(content.as_bytes());
      let hash = hasher.finalize();
      let mut content_hash = [0u8; 32];
      content_hash.copy_from_slice(hash.as_bytes());

      // Check if hyperpolyglot detects a supported language
      let detected_language = if let Ok(Some(detection)) = hyperpolyglot::detect(&path) {
          let language = detection.language();
          if get_language(language).is_some() {
              Some(language.to_string())
          } else {
              None
          }
      } else {
          None
      };

      // Clone content once for the EOF marker
      let content_for_eof = content.clone();

      // Try semantic chunking first if we have a supported language
      if let Some(language) = detected_language {
          let mut chunk_stream = Box::pin(chunker.chunk_code(content, language.clone(), Some(path_str.clone())));
          let mut had_success = false;

          while let Some(chunk_result) = chunk_stream.next().await {
              match chunk_result {
                  Ok(chunk) => {
                      had_success = true;
                      yield ProjectChunk {
                          file_path: path_str.clone(),
                          chunk,
                      };
                  }
                  Err(_) => {
                      // If we haven't had any successful chunks, fall through to text chunking
                      if !had_success {
                          break;
                      }
                  }
              }
          }

          // If semantic chunking succeeded, we're done
          if had_success {
              // Emit EOF marker for this file
              yield ProjectChunk {
                  file_path: path_str.clone(),
                  chunk: Chunk::EndOfFile {
                      file_path: path_str.clone(),
                      content: content_for_eof,
                      content_hash,
                  },
              };
              return;
          }
      }

      // Fall back to text chunking
      // Note: content was already moved into chunk_code if we tried semantic chunking
      let content_for_text = content_for_eof.clone();

      let mut chunk_stream = Box::pin(chunker.chunk_text(content_for_text, Some(path_str.clone())));
      while let Some(chunk_result) = chunk_stream.next().await {
          let chunk = chunk_result?;
          yield ProjectChunk {
              file_path: path_str.clone(),
              chunk,
          };
      }

      // Emit EOF marker for this file
      yield ProjectChunk {
          file_path: path_str.clone(),
          chunk: Chunk::EndOfFile {
              file_path: path_str,
              content: content_for_eof,
              content_hash,
          },
      };
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::types::Chunk;
  use std::fs;
  use tempfile::TempDir;
  use tokio_stream::StreamExt;

  async fn create_test_project() -> TempDir {
    let temp_dir = TempDir::new().unwrap();
    let base_path = temp_dir.path();

    // Create a simple project structure
    fs::create_dir_all(base_path.join("src")).unwrap();
    fs::create_dir_all(base_path.join("tests")).unwrap();
    fs::create_dir_all(base_path.join("docs")).unwrap();
    fs::create_dir_all(base_path.join("scripts")).unwrap();
    fs::create_dir_all(base_path.join(".git")).unwrap();

    // Create a Python module structure
    fs::write(
      base_path.join("src/__init__.py"),
      r#"""Main package for the test project."""

__version__ = "0.1.0"
__author__ = "Test Author"

from .calculator import Calculator
from .utils import factorial

__all__ = ["Calculator", "factorial"]
"#,
    )
    .unwrap();

    fs::write(
      base_path.join("src/calculator.py"),
      r#"""Calculator module with basic arithmetic operations."""

class Calculator:
    """A simple calculator class."""

    def __init__(self):
        self.memory = 0

    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.memory = result
        return result

    def subtract(self, a, b):
        """Subtract b from a."""
        result = a - b
        self.memory = result
        return result

    def multiply(self, a, b):
        """Multiply two numbers."""
        result = a * b
        self.memory = result
        return result

    def clear_memory(self):
        """Clear the calculator memory."""
        self.memory = 0
"#,
    )
    .unwrap();

    // Create Python file
    fs::write(
      base_path.join("scripts/test.py"),
      r#"#!/usr/bin/env python3
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

class Calculator:
    def __init__(self):
        self.value = 0

    def add(self, x):
        self.value += x
        return self.value

if __name__ == "__main__":
    print(f"5! = {factorial(5)}")
"#,
    )
    .unwrap();

    // Create another Python file
    fs::write(
      base_path.join("scripts/data_processor.py"),
      r#"#!/usr/bin/env python3
"""Data processing utilities."""

import json
import csv
from typing import List, Dict, Any

class DataProcessor:
    """Process various data formats."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data = []

    def load_json(self, filepath: str) -> List[Dict]:
        """Load data from JSON file."""
        with open(filepath, 'r') as f:
            self.data = json.load(f)
        return self.data

    def process_records(self) -> List[Dict]:
        """Process loaded records."""
        processed = []
        for record in self.data:
            if self._validate_record(record):
                processed.append(self._transform_record(record))
        return processed

    def _validate_record(self, record: Dict) -> bool:
        """Validate a single record."""
        required_fields = self.config.get('required_fields', [])
        return all(field in record for field in required_fields)

    def _transform_record(self, record: Dict) -> Dict:
        """Transform a single record."""
        # Apply transformations based on config
        return record
"#,
    )
    .unwrap();

    // Create a markdown file
    fs::write(
      base_path.join("README.md"),
      r#"# Test Project

This is a test project for the walker functionality.

## Features
- Python modules and packages
- Data processing utilities
- Calculator implementations
- Documentation

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from src import Calculator

calc = Calculator()
result = calc.add(5, 3)
print(f"Result: {result}")
```

## Testing

Run tests with pytest:

```bash
python -m pytest tests/
```
"#,
    )
    .unwrap();

    // Create a binary file (should be skipped)
    fs::write(
      base_path.join("test.bin"),
      &[0u8, 1, 2, 3, 255, 254, 253, 252],
    )
    .unwrap();

    // Create .gitignore
    fs::write(base_path.join(".gitignore"), "target/\n*.log\n").unwrap();

    // Create a file in .git (should be ignored)
    fs::write(
      base_path.join(".git/config"),
      "[core]\nrepositoryformatversion = 0\n",
    )
    .unwrap();

    // Create a requirements file
    fs::write(
      base_path.join("requirements.txt"),
      "pytest>=7.0.0\nnumpy>=1.20.0\npandas>=1.3.0\n",
    )
    .unwrap();

    temp_dir
  }

  #[tokio::test]
  async fn test_walk_project_basic() {
    let temp_dir = create_test_project().await;
    let path = temp_dir.path();

    let mut chunks = Vec::new();
    let mut stream = walk_project(
      path,
      WalkOptions {
        max_chunk_size: 500,
        tokenizer: Tokenizer::Characters,
        max_parallel: 4,
        max_file_size: None,
      },
    );

    while let Some(result) = stream.next().await {
      match result {
        Ok(chunk) => chunks.push(chunk),
        Err(e) => panic!("Unexpected error: {}", e),
      }
    }

    // Should have found and chunked multiple files
    assert!(!chunks.is_empty(), "Should have found some chunks");

    // Check we got chunks from different files
    let unique_files: std::collections::HashSet<_> = chunks.iter().map(|c| &c.file_path).collect();
    assert!(
      unique_files.len() > 1,
      "Should have chunks from multiple files"
    );

    // Check we have both semantic and text chunks
    let has_semantic = chunks.iter().any(|c| c.is_semantic());
    let has_text = chunks.iter().any(|c| c.is_text());
    assert!(has_semantic, "Should have semantic chunks");
    assert!(has_text, "Should have text chunks");

    // Debug: print all file paths
    let file_paths: Vec<_> = chunks.iter().map(|c| &c.file_path).collect();
    println!("Found files: {:?}", file_paths);

    // Verify .git files were ignored
    assert!(
      !chunks.iter().any(|c| c.file_path.contains(".git")),
      ".git files should be ignored"
    );

    // Verify binary files were skipped
    assert!(
      !chunks.iter().any(|c| c.file_path.contains("test.bin")),
      "Binary files should be skipped"
    );
  }

  #[tokio::test]
  async fn test_walk_project_languages() {
    let temp_dir = create_test_project().await;
    let path = temp_dir.path();

    let mut chunks = Vec::new();
    let mut stream = walk_project(
      path,
      WalkOptions {
        max_chunk_size: 1000,
        tokenizer: Tokenizer::Characters,
        max_parallel: 2,
        max_file_size: None,
      },
    );

    while let Some(result) = stream.next().await {
      chunks.push(result.unwrap());
    }

    // Check Python files (the only supported language currently)
    let python_chunks: Vec<_> = chunks
      .iter()
      .filter(|c| c.file_path.ends_with(".py") && c.is_semantic())
      .collect();
    assert!(
      !python_chunks.is_empty(),
      "Should have Python semantic chunks"
    );
    assert!(python_chunks.iter().all(|c| match &c.chunk {
      Chunk::Semantic(sc) => sc.metadata.language == "Python",
      _ => false,
    }));

    // Check Markdown files (should be semantic chunks)
    let md_chunks: Vec<_> = chunks
      .iter()
      .filter(|c| c.file_path.ends_with(".md") && !matches!(c.chunk, Chunk::EndOfFile { .. }))
      .collect();
    assert!(!md_chunks.is_empty(), "Should have Markdown chunks");
    assert!(md_chunks.iter().all(|c| c.is_semantic()));
  }

  #[tokio::test]
  async fn test_walk_project_empty_dir() {
    let temp_dir = TempDir::new().unwrap();
    let path = temp_dir.path();

    let mut chunks = Vec::new();
    let mut stream = walk_project(
      path,
      WalkOptions {
        max_chunk_size: 500,
        tokenizer: Tokenizer::Characters,
        max_parallel: 4,
        max_file_size: None,
      },
    );

    while let Some(result) = stream.next().await {
      chunks.push(result.unwrap());
    }

    assert!(chunks.is_empty(), "Empty directory should yield no chunks");
  }

  #[tokio::test]
  async fn test_walk_project_concurrency() {
    let temp_dir = create_test_project().await;
    let path = temp_dir.path();

    // Test with different concurrency levels
    for max_parallel in [1, 2, 8] {
      let mut chunks = Vec::new();
      let mut stream = walk_project(
        path,
        WalkOptions {
          max_chunk_size: 500,
          tokenizer: Tokenizer::Characters,
          max_parallel,
          max_file_size: None,
        },
      );

      while let Some(result) = stream.next().await {
        chunks.push(result.unwrap());
      }

      assert!(
        !chunks.is_empty(),
        "Should get chunks with concurrency level {}",
        max_parallel
      );
    }
  }

  #[tokio::test]
  async fn test_process_file_stream() {
    let temp_dir = TempDir::new().unwrap();
    let rust_file = temp_dir.path().join("test.rs");

    fs::write(
      &rust_file,
      r#"
fn main() {
    println!("Test");
}

fn helper() {
    let x = 42;
}
"#,
    )
    .unwrap();

    let chunker = Arc::new(InnerChunker::new(100, Tokenizer::Characters).unwrap());
    let mut stream = Box::pin(process_file(&rust_file, chunker));

    let mut chunks = Vec::new();
    while let Some(result) = stream.next().await {
      chunks.push(result.unwrap());
    }

    assert!(!chunks.is_empty(), "Should get chunks from Rust file");

    // Filter out EOF chunks for semantic checks
    let semantic_chunks: Vec<_> = chunks
      .iter()
      .filter(|c| !matches!(c.chunk, Chunk::EndOfFile { .. }))
      .collect();

    assert!(!semantic_chunks.is_empty(), "Should have semantic chunks");
    assert!(semantic_chunks.iter().all(|c| c.is_semantic()));
    assert!(semantic_chunks.iter().all(|c| match &c.chunk {
      Chunk::Semantic(sc) => sc.metadata.language == "Rust",
      _ => false,
    }));

    // Should have exactly one EOF chunk at the end
    assert!(matches!(
      chunks.last().unwrap().chunk,
      Chunk::EndOfFile { .. }
    ));
  }
}
