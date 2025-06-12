use breeze_chunkers::{
  Chunk as RustChunk, ChunkMetadata, InnerChunker, ProjectChunk as RustProjectChunk,
  Tokenizer as RustTokenizer, WalkOptions, walk_project as rust_walk_project,
};

use futures::StreamExt;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};

#[napi]
pub enum TokenizerType {
  Characters,
  Tiktoken,
  HuggingFace,
}

#[napi]
#[derive(Clone)]
pub enum ChunkType {
  Semantic,
  Text,
}

#[napi(object)]
#[derive(Clone)]
pub struct ChunkMetadataJs {
  pub node_type: String,
  pub node_name: Option<String>,
  pub language: String,
  pub parent_context: Option<String>,
  pub scope_path: Vec<String>,
  pub definitions: Vec<String>,
  pub references: Vec<String>,
}

impl From<ChunkMetadata> for ChunkMetadataJs {
  fn from(metadata: ChunkMetadata) -> Self {
    Self {
      node_type: metadata.node_type,
      node_name: metadata.node_name,
      language: metadata.language,
      parent_context: metadata.parent_context,
      scope_path: metadata.scope_path,
      definitions: metadata.definitions,
      references: metadata.references,
    }
  }
}

#[napi(object)]
#[derive(Clone)]
pub struct SemanticChunkJs {
  pub chunk_type: ChunkType,
  pub text: String,
  pub start_byte: i32,
  pub end_byte: i32,
  pub start_line: i32,
  pub end_line: i32,
  pub metadata: ChunkMetadataJs,
}

impl From<RustChunk> for SemanticChunkJs {
  fn from(chunk: RustChunk) -> Self {
    match chunk {
      RustChunk::Semantic(sc) => Self {
        chunk_type: ChunkType::Semantic,
        text: sc.text,
        start_byte: sc.start_byte as i32,
        end_byte: sc.end_byte as i32,
        start_line: sc.start_line as i32,
        end_line: sc.end_line as i32,
        metadata: sc.metadata.into(),
      },
      RustChunk::Text(sc) => Self {
        chunk_type: ChunkType::Text,
        text: sc.text,
        start_byte: sc.start_byte as i32,
        end_byte: sc.end_byte as i32,
        start_line: sc.start_line as i32,
        end_line: sc.end_line as i32,
        metadata: sc.metadata.into(),
      },
    }
  }
}

#[napi(object)]
#[derive(Clone)]
pub struct ProjectChunkJs {
  pub file_path: String,
  pub chunk: SemanticChunkJs,
}

impl From<RustProjectChunk> for ProjectChunkJs {
  fn from(pc: RustProjectChunk) -> Self {
    Self {
      file_path: pc.file_path,
      chunk: pc.chunk.into(),
    }
  }
}

// Result type for channels
type ChunkResult<T> = std::result::Result<T, String>;

// Result type for iterator next() method - Semantic chunks
#[napi(object)]
pub struct SemanticChunkIteratorResult {
  pub done: bool,
  pub value: Option<SemanticChunkJs>,
}

// Result type for iterator next() method - Project chunks
#[napi(object)]
pub struct ProjectChunkIteratorResult {
  pub done: bool,
  pub value: Option<ProjectChunkJs>,
}

// Iterator wrapper for semantic chunks
#[napi]
pub struct SemanticChunkIterator {
  receiver: Arc<Mutex<mpsc::Receiver<ChunkResult<SemanticChunkJs>>>>,
}

#[napi]
impl SemanticChunkIterator {
  /// Get the next chunk from the iterator
  #[napi]
  pub async fn next(&self) -> Result<SemanticChunkIteratorResult> {
    let mut receiver = self.receiver.lock().await;

    match receiver.recv().await {
      Some(Ok(chunk)) => Ok(SemanticChunkIteratorResult {
        done: false,
        value: Some(chunk),
      }),
      Some(Err(e)) => Err(Error::new(Status::GenericFailure, e)),
      None => Ok(SemanticChunkIteratorResult {
        done: true,
        value: None,
      }),
    }
  }
}

// Language support functions
/// Get list of supported languages
#[napi]
pub fn supported_languages() -> Vec<String> {
  breeze_chunkers::supported_languages()
    .into_iter()
    .map(|s| s.to_string())
    .collect()
}

/// Check if a language is supported (case-insensitive)
#[napi]
pub fn is_language_supported(language: String) -> bool {
  breeze_chunkers::is_language_supported(&language)
}

// Iterator wrapper for project chunks
#[napi]
pub struct ProjectChunkIterator {
  receiver: Arc<Mutex<mpsc::Receiver<ChunkResult<ProjectChunkJs>>>>,
}

#[napi]
impl ProjectChunkIterator {
  /// Get the next chunk from the iterator
  #[napi]
  pub async fn next(&self) -> Result<ProjectChunkIteratorResult> {
    let mut receiver = self.receiver.lock().await;

    match receiver.recv().await {
      Some(Ok(chunk)) => Ok(ProjectChunkIteratorResult {
        done: false,
        value: Some(chunk),
      }),
      Some(Err(e)) => Err(Error::new(Status::GenericFailure, e)),
      None => Ok(ProjectChunkIteratorResult {
        done: true,
        value: None,
      }),
    }
  }
}

#[napi]
pub struct SemanticChunker {
  inner: Arc<InnerChunker>,
}

#[napi]
impl SemanticChunker {
  #[napi(constructor)]
  pub fn new(
    max_chunk_size: Option<i32>,
    tokenizer: Option<TokenizerType>,
    hf_model: Option<String>,
  ) -> Result<Self> {
    let max_chunk_size = max_chunk_size.unwrap_or(1500) as usize;

    // Convert JS TokenizerType to Rust Tokenizer
    let tokenizer_type = match tokenizer.unwrap_or(TokenizerType::Characters) {
      TokenizerType::Characters => RustTokenizer::Characters,
      TokenizerType::Tiktoken => RustTokenizer::Tiktoken,
      TokenizerType::HuggingFace => match hf_model {
        Some(model) => RustTokenizer::HuggingFace(model),
        None => {
          return Err(Error::new(
            Status::InvalidArg,
            "TokenizerType.HuggingFace requires hfModel parameter".to_string(),
          ));
        }
      },
    };

    let inner = InnerChunker::new(max_chunk_size, tokenizer_type).map_err(|e| {
      Error::new(
        Status::GenericFailure,
        format!("Failed to create chunker: {}", e),
      )
    })?;

    Ok(Self {
      inner: Arc::new(inner),
    })
  }

  /// Chunk code and return an async iterator
  #[napi]
  pub async fn chunk_code(
    &self,
    content: String,
    language: String,
    file_path: Option<String>,
  ) -> Result<SemanticChunkIterator> {
    let chunker = self.inner.clone();
    let stream = chunker.chunk_code(content, language, file_path);

    // Create a channel for the iterator
    let (tx, rx) = mpsc::channel(100);

    // Spawn a task to process the stream
    tokio::spawn(async move {
      let mut stream = Box::pin(stream);
      while let Some(result) = stream.next().await {
        let send_result = match result {
          Ok(chunk) => tx.send(Ok(chunk.into())).await,
          Err(e) => tx.send(Err(format!("Chunk error: {:?}", e))).await,
        };
        if send_result.is_err() {
          // Receiver dropped
          break;
        }
      }
    });

    Ok(SemanticChunkIterator {
      receiver: Arc::new(Mutex::new(rx)),
    })
  }

  /// Chunk text and return an async iterator
  #[napi]
  pub async fn chunk_text(
    &self,
    content: String,
    file_path: Option<String>,
  ) -> Result<SemanticChunkIterator> {
    let chunker = self.inner.clone();
    let stream = chunker.chunk_text(content, file_path);

    // Create a channel for the iterator
    let (tx, rx) = mpsc::channel(100);

    // Spawn a task to process the stream
    tokio::spawn(async move {
      let mut stream = Box::pin(stream);
      while let Some(result) = stream.next().await {
        let send_result = match result {
          Ok(chunk) => tx.send(Ok(chunk.into())).await,
          Err(e) => tx.send(Err(format!("Chunk error: {:?}", e))).await,
        };
        if send_result.is_err() {
          // Receiver dropped
          break;
        }
      }
    });

    Ok(SemanticChunkIterator {
      receiver: Arc::new(Mutex::new(rx)),
    })
  }
}

/// Walk a project directory and return an async iterator of chunks
#[napi]
pub async fn walk_project(
  path: String,
  max_chunk_size: Option<i32>,
  tokenizer: Option<TokenizerType>,
  hf_model: Option<String>,
  max_parallel: Option<i32>,
) -> Result<ProjectChunkIterator> {
  let max_chunk_size = max_chunk_size.unwrap_or(1500) as usize;
  let max_parallel = max_parallel.unwrap_or(8) as usize;

  // Convert JS TokenizerType to Rust Tokenizer
  let tokenizer_type = match tokenizer.unwrap_or(TokenizerType::Characters) {
    TokenizerType::Characters => RustTokenizer::Characters,
    TokenizerType::Tiktoken => RustTokenizer::Tiktoken,
    TokenizerType::HuggingFace => match hf_model {
      Some(model) => RustTokenizer::HuggingFace(model),
      None => {
        return Err(Error::new(
          Status::InvalidArg,
          "TokenizerType.HuggingFace requires hfModel parameter".to_string(),
        ));
      }
    },
  };

  // Create a channel for the iterator
  let (tx, rx) = mpsc::channel(100);

  // Spawn a task to process the stream
  tokio::spawn(async move {
    let stream = rust_walk_project(
      path,
      WalkOptions {
        max_chunk_size,
        tokenizer: tokenizer_type,
        max_parallel,
        ..Default::default()
      },
    );
    let mut stream = Box::pin(stream);

    while let Some(result) = stream.next().await {
      let send_result = match result {
        Ok(chunk) => tx.send(Ok(chunk.into())).await,
        Err(e) => tx.send(Err(format!("Chunk error: {:?}", e))).await,
      };
      if send_result.is_err() {
        // Receiver dropped
        break;
      }
    }
  });

  Ok(ProjectChunkIterator {
    receiver: Arc::new(Mutex::new(rx)),
  })
}
