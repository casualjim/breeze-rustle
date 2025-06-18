//! # breeze-chunkers
//!
//! High-performance semantic code chunking library for Rust.
//!
//! This crate provides intelligent code splitting capabilities using tree-sitter parsers
//! to break code into semantically meaningful chunks while preserving context.

mod chunker;
mod grammar_loader;
mod languages;
mod metadata_extractor;

#[cfg(feature = "perfprofiling")]
pub mod performance;
mod types;
mod walker;

use std::path::Path;
use std::sync::Arc;

use futures::stream::BoxStream;
use futures::{Stream, StreamExt};

pub use crate::chunker::InnerChunker;

// Re-export main types
pub use crate::types::{
  Chunk, ChunkError, ChunkMetadata, FileMetadata, ProjectChunk, SemanticChunk,
};
pub use crate::walker::{WalkOptions, walk_project};

/// Tokenizer type for chunk size calculation
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Tokenizer {
  /// Simple character-based tokenization
  #[serde(rename = "characters")]
  Characters,
  /// OpenAI tiktoken tokenizer with encoding name (e.g., "cl100k_base", "p50k_base")
  #[serde(rename = "tiktoken")]
  Tiktoken(#[serde(rename = "encoding")] String),
  /// Pre-loaded tiktoken tokenizer (internal use only, not exposed to bindings)
  #[doc(hidden)]
  #[serde(skip)]
  PreloadedTiktoken(std::sync::Arc<tiktoken_rs::CoreBPE>),
  /// HuggingFace tokenizer with specified model
  #[serde(rename = "huggingface")]
  HuggingFace(#[serde(rename = "model_id")] String),
  /// Pre-loaded HuggingFace tokenizer (internal use only, not exposed to bindings)
  #[doc(hidden)]
  #[serde(skip)]
  PreloadedHuggingFace(std::sync::Arc<tokenizers::Tokenizer>),
}

impl Default for Tokenizer {
  fn default() -> Self {
    Self::Characters
  }
}

impl std::fmt::Debug for Tokenizer {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Tokenizer::Characters => write!(f, "Characters"),
      Tokenizer::Tiktoken(name) => write!(f, "Tiktoken({})", name),
      Tokenizer::PreloadedTiktoken(_) => write!(f, "PreloadedTiktoken"),
      Tokenizer::HuggingFace(model) => write!(f, "HuggingFace({})", model),
      Tokenizer::PreloadedHuggingFace(_) => write!(f, "PreloadedHuggingFace"),
    }
  }
}

/// Configuration for the chunker
#[derive(Clone, Debug)]
pub struct ChunkerConfig {
  /// Maximum size of each chunk (in tokens/characters)
  pub max_chunk_size: usize,
  /// Tokenizer to use for size calculation
  pub tokenizer: Tokenizer,
}

impl Default for ChunkerConfig {
  fn default() -> Self {
    Self {
      max_chunk_size: 1500,
      tokenizer: Tokenizer::default(),
    }
  }
}

/// Main chunker for splitting code and text into semantic chunks
pub struct Chunker {
  inner: Arc<InnerChunker>,
}

impl Default for Chunker {
  /// Create a chunker with default configuration
  ///
  /// # Panics
  /// Panics if the default tokenizer (Characters) fails to initialize, which should never happen
  fn default() -> Self {
    Self::new(ChunkerConfig::default()).expect("Default chunker creation should never fail")
  }
}

impl Chunker {
  /// Create a new chunker with the given configuration
  pub fn new(config: ChunkerConfig) -> Result<Self, ChunkError> {
    let inner = InnerChunker::new(config.max_chunk_size, config.tokenizer)?;
    Ok(Self {
      inner: Arc::new(inner),
    })
  }

  /// Chunk code content with semantic understanding
  ///
  /// # Arguments
  /// * `content` - The code content to chunk
  /// * `language` - Programming language name (case-insensitive)
  /// * `file_path` - Optional file path for context
  ///
  /// # Returns
  /// A stream of semantic chunks with metadata
  pub fn chunk_code(
    &self,
    content: String,
    language: String,
    file_path: Option<String>,
  ) -> impl Stream<Item = Result<Chunk, ChunkError>> + use<> {
    self.inner.chunk_code(content, language, file_path)
  }

  /// Chunk plain text content
  ///
  /// # Arguments
  /// * `content` - The text content to chunk
  /// * `file_path` - Optional file path for context
  ///
  /// # Returns
  /// A stream of text chunks
  pub fn chunk_text(
    &self,
    content: String,
    file_path: Option<String>,
  ) -> impl Stream<Item = Result<Chunk, ChunkError>> + use<> {
    self.inner.chunk_text(content, file_path)
  }

  /// Chunk a file automatically detecting if it's code or text
  ///
  /// # Arguments
  /// * `path` - Path to the file
  ///
  /// # Returns
  /// A stream of chunks (either semantic or text based on file type)
  pub async fn chunk_file(
    &self,
    path: impl AsRef<Path>,
  ) -> Result<BoxStream<'_, Result<Chunk, ChunkError>>, ChunkError> {
    let path = path.as_ref();
    let content = tokio::fs::read_to_string(path)
      .await
      .map_err(ChunkError::IoError)?;

    let file_path = path.to_string_lossy().to_string();

    // Try to detect language
    if let Ok(Some(detection)) = hyperpolyglot::detect(path) {
      let language = detection.language();
      if is_language_supported(language) {
        return Ok(
          self
            .chunk_code(content, language.to_string(), Some(file_path))
            .boxed(),
        );
      }
    }

    // Fall back to text chunking
    Ok(self.chunk_text(content, Some(file_path)).boxed())
  }
}

/// Get list of supported programming languages
///
/// # Returns
/// A vector of language names that can be used with `chunk_code`
///
/// # Example
/// ```no_run
/// let languages = breeze_chunkers::supported_languages();
/// assert!(languages.contains(&"rust"));
/// ```
pub fn supported_languages() -> Vec<&'static str> {
  languages::supported_languages()
}

/// Check if a language is supported
///
/// # Arguments
/// * `name` - Language name to check (case-insensitive)
///
/// # Returns
/// `true` if the language is supported
///
/// # Example
/// ```no_run
/// assert!(breeze_chunkers::is_language_supported("rust"));
/// assert!(breeze_chunkers::is_language_supported("Python")); // case-insensitive
/// ```
pub fn is_language_supported(name: &str) -> bool {
  languages::is_language_supported(name)
}
