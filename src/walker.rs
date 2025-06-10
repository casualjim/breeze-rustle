use crate::{
    chunker::{InnerChunker, TokenizerType},
    types::{ChunkError, ChunkType, ProjectChunk},
    languages::get_language,
};
use futures::{Stream, StreamExt};
use async_stream;
use ignore::{WalkBuilder, WalkState};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

/// Options for walking a project directory
pub struct WalkOptions {
    pub max_chunk_size: usize,
    pub tokenizer: TokenizerType,
    pub max_parallel: usize,
    pub max_file_size: Option<u64>,
}

impl Default for WalkOptions {
    fn default() -> Self {
        Self {
            max_chunk_size: 1000,
            tokenizer: TokenizerType::Characters,
            max_parallel: 16,
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
    
    // Create a channel for streaming results
    let (tx, rx) = mpsc::channel::<Result<ProjectChunk, ChunkError>>(400_000);
    
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
            .build_parallel();
        
        let tx = Arc::new(tx);
        
        walker.run(|| {
            let tx = tx.clone();
            let chunker = chunker.clone();
            let handle = handle.clone();
            
            Box::new(move |result| {
                if let Ok(entry) = result {
                    if entry.file_type().map_or(false, |ft| ft.is_file()) {
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
                                        // Log error but continue processing other files
                                        eprintln!("Error processing {}: {}", path.display(), e);
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
        
        // Try semantic chunking first if we have a supported language
        if let Some(language) = detected_language {
            let mut chunk_stream = Box::pin(chunker.chunk_code(content.clone(), language.clone(), Some(path_str.clone())));
            let mut had_success = false;
            
            while let Some(chunk_result) = chunk_stream.next().await {
                match chunk_result {
                    Ok(chunk) => {
                        had_success = true;
                        yield ProjectChunk {
                            file_path: path_str.clone(),
                            chunk_type: ChunkType::Semantic,
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
                return;
            }
        }
        
        // Fall back to text chunking
        let mut chunk_stream = Box::pin(chunker.chunk_text(content, Some(path_str.clone())));
        while let Some(chunk_result) = chunk_stream.next().await {
            let chunk = chunk_result?;
            yield ProjectChunk {
                file_path: path_str.clone(),
                chunk_type: ChunkType::Text,
                chunk,
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_stream::StreamExt;
    use std::fs;
    use tempfile::TempDir;
    
    async fn create_test_project() -> TempDir {
        let temp_dir = TempDir::new().unwrap();
        let base_path = temp_dir.path();
        
        // Create a simple project structure
        fs::create_dir_all(base_path.join("src")).unwrap();
        fs::create_dir_all(base_path.join("tests")).unwrap();
        fs::create_dir_all(base_path.join("docs")).unwrap();
        fs::create_dir_all(base_path.join("scripts")).unwrap();
        fs::create_dir_all(base_path.join(".git")).unwrap();
        
        // Create some source files
        fs::write(
            base_path.join("src/main.rs"),
            r#"
use std::io;

fn main() {
    println!("Hello, world!");
    let result = calculate(5, 3);
    println!("Result: {}", result);
}

fn calculate(a: i32, b: i32) -> i32 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_calculate() {
        assert_eq!(calculate(2, 2), 4);
    }
}
"#
        ).unwrap();
        
        fs::write(
            base_path.join("src/lib.rs"),
            r#"
pub mod utils {
    pub fn greet(name: &str) -> String {
        format!("Hello, {}!", name)
    }
}

pub mod math {
    pub fn multiply(a: f64, b: f64) -> f64 {
        a * b
    }
}
"#
        ).unwrap();
        
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
"#
        ).unwrap();
        
        // Create JavaScript file
        fs::write(
            base_path.join("scripts/app.js"),
            r#"
const express = require('express');

function createApp() {
    const app = express();
    
    app.get('/', (req, res) => {
        res.send('Hello World!');
    });
    
    return app;
}

module.exports = { createApp };
"#
        ).unwrap();
        
        // Create a plain text file
        fs::write(
            base_path.join("README.md"),
            r#"# Test Project

This is a test project for the walker functionality.

## Features
- Rust source code
- Python scripts
- JavaScript modules
- Documentation

## Usage
Run the main program with:
```
cargo run
```
"#
        ).unwrap();
        
        // Create a binary file (should be skipped)
        fs::write(
            base_path.join("test.bin"),
            &[0u8, 1, 2, 3, 255, 254, 253, 252]
        ).unwrap();
        
        // Create .gitignore
        fs::write(
            base_path.join(".gitignore"),
            "target/\n*.log\n"
        ).unwrap();
        
        // Create a file in .git (should be ignored)
        fs::write(
            base_path.join(".git/config"),
            "[core]\nrepositoryformatversion = 0\n"
        ).unwrap();
        
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
                tokenizer: TokenizerType::Characters,
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
        let unique_files: std::collections::HashSet<_> = chunks.iter()
            .map(|c| &c.file_path)
            .collect();
        assert!(unique_files.len() > 1, "Should have chunks from multiple files");
        
        // Check we have both semantic and text chunks
        let has_semantic = chunks.iter().any(|c| c.chunk_type == ChunkType::Semantic);
        let has_text = chunks.iter().any(|c| c.chunk_type == ChunkType::Text);
        assert!(has_semantic, "Should have semantic chunks");
        assert!(has_text, "Should have text chunks");
        
        // Debug: print all file paths
        let file_paths: Vec<_> = chunks.iter().map(|c| &c.file_path).collect();
        println!("Found files: {:?}", file_paths);
        
        // Verify .git files were ignored
        assert!(!chunks.iter().any(|c| c.file_path.contains(".git")), 
               ".git files should be ignored");
        
        // Verify binary files were skipped
        assert!(!chunks.iter().any(|c| c.file_path.contains("test.bin")), 
               "Binary files should be skipped");
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
                tokenizer: TokenizerType::Characters,
                max_parallel: 2,
                max_file_size: None,
            },
        );
        
        while let Some(result) = stream.next().await {
            chunks.push(result.unwrap());
        }
        
        // Check Rust files
        let rust_chunks: Vec<_> = chunks.iter()
            .filter(|c| c.file_path.ends_with(".rs") && c.chunk_type == ChunkType::Semantic)
            .collect();
        assert!(!rust_chunks.is_empty(), "Should have Rust semantic chunks");
        assert!(rust_chunks.iter().all(|c| c.chunk.metadata.language == "Rust"));
        
        // Check Python files
        let python_chunks: Vec<_> = chunks.iter()
            .filter(|c| c.file_path.ends_with(".py") && c.chunk_type == ChunkType::Semantic)
            .collect();
        assert!(!python_chunks.is_empty(), "Should have Python semantic chunks");
        assert!(python_chunks.iter().all(|c| c.chunk.metadata.language == "Python"));
        
        // Check JavaScript files
        let js_chunks: Vec<_> = chunks.iter()
            .filter(|c| c.file_path.ends_with(".js") && c.chunk_type == ChunkType::Semantic)
            .collect();
        assert!(!js_chunks.is_empty(), "Should have JavaScript semantic chunks");
        assert!(js_chunks.iter().all(|c| c.chunk.metadata.language == "JavaScript"));
        
        // Check Markdown files (should be text chunks)
        let md_chunks: Vec<_> = chunks.iter()
            .filter(|c| c.file_path.ends_with(".md"))
            .collect();
        assert!(!md_chunks.is_empty(), "Should have Markdown chunks");
        assert!(md_chunks.iter().all(|c| c.chunk_type == ChunkType::Text));
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
                tokenizer: TokenizerType::Characters,
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
                    tokenizer: TokenizerType::Characters,
                    max_parallel,
                    max_file_size: None,
                },
            );
            
            while let Some(result) = stream.next().await {
                chunks.push(result.unwrap());
            }
            
            assert!(!chunks.is_empty(), 
                   "Should get chunks with concurrency level {}", max_parallel);
        }
    }
    
    
    #[tokio::test]
    async fn test_process_file_stream() {
        let temp_dir = TempDir::new().unwrap();
        let rust_file = temp_dir.path().join("test.rs");
        
        fs::write(&rust_file, r#"
fn main() {
    println!("Test");
}

fn helper() {
    let x = 42;
}
"#).unwrap();
        
        let chunker = Arc::new(InnerChunker::new(100, TokenizerType::Characters).unwrap());
        let mut stream = Box::pin(process_file(&rust_file, chunker));
        
        let mut chunks = Vec::new();
        while let Some(result) = stream.next().await {
            chunks.push(result.unwrap());
        }
        
        assert!(!chunks.is_empty(), "Should get chunks from Rust file");
        assert!(chunks.iter().all(|c| c.chunk_type == ChunkType::Semantic));
        assert!(chunks.iter().all(|c| c.chunk.metadata.language == "Rust"));
    }
}