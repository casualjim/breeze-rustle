use breeze_chunkers::{
  Chunk, ChunkError, ChunkMetadata, Chunker, ChunkerConfig, ProjectChunk,
  Tokenizer as RustTokenizer, WalkOptions, is_language_supported, supported_languages,
  walk_project,
};

use futures::{Stream, StreamExt};
use pyo3::exceptions::{PyRuntimeError, PyStopAsyncIteration, PyValueError};
use pyo3::prelude::*;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;

#[pyclass(eq)]
#[derive(Clone, Debug, PartialEq)]
pub enum TokenizerType {
  #[pyo3(name = "CHARACTERS")]
  Characters,
  #[pyo3(name = "TIKTOKEN")]
  Tiktoken,
  #[pyo3(name = "HUGGINGFACE")]
  HuggingFace,
}

#[pyclass(eq)]
#[derive(Clone, Debug, PartialEq)]
pub enum ChunkType {
  #[pyo3(name = "SEMANTIC")]
  Semantic,
  #[pyo3(name = "TEXT")]
  Text,
  #[pyo3(name = "EOF")]
  EndOfFile,
  #[pyo3(name = "DELETE")]
  Delete,
}

#[pymethods]
impl TokenizerType {
  fn __repr__(&self) -> String {
    match self {
      TokenizerType::Characters => "TokenizerType.CHARACTERS",
      TokenizerType::Tiktoken => "TokenizerType.TIKTOKEN",
      TokenizerType::HuggingFace => "TokenizerType.HUGGINGFACE",
    }
    .to_string()
  }

  /// Get the string representation for use in SemanticChunker constructor
  fn value(&self) -> &'static str {
    match self {
      TokenizerType::Characters => "characters",
      TokenizerType::Tiktoken => "tiktoken",
      TokenizerType::HuggingFace => "huggingface",
    }
  }
}

#[pymethods]
impl ChunkType {
  fn __repr__(&self) -> String {
    match self {
      ChunkType::Semantic => "ChunkType.SEMANTIC",
      ChunkType::Text => "ChunkType.TEXT",
      ChunkType::EndOfFile => "ChunkType.EOF",
      ChunkType::Delete => "ChunkType.DELETE",
    }
    .to_string()
  }
}

// Python-friendly wrappers for breeze_chunkers types

#[pyclass(name = "ChunkMetadata")]
#[derive(Clone)]
pub struct PyChunkMetadata {
  #[pyo3(get)]
  pub node_type: String,
  #[pyo3(get)]
  pub node_name: Option<String>,
  #[pyo3(get)]
  pub language: String,
  #[pyo3(get)]
  pub parent_context: Option<String>,
  #[pyo3(get)]
  pub scope_path: Vec<String>,
  #[pyo3(get)]
  pub definitions: Vec<String>,
  #[pyo3(get)]
  pub references: Vec<String>,
}

impl From<ChunkMetadata> for PyChunkMetadata {
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

#[pyclass(name = "SemanticChunk")]
#[derive(Clone)]
pub struct PySemanticChunk {
  #[pyo3(get)]
  pub chunk_type: ChunkType,
  #[pyo3(get)]
  pub text: String,
  #[pyo3(get)]
  pub start_byte: usize,
  #[pyo3(get)]
  pub end_byte: usize,
  #[pyo3(get)]
  pub start_line: usize,
  #[pyo3(get)]
  pub end_line: usize,
  #[pyo3(get)]
  pub metadata: PyChunkMetadata,
  #[pyo3(get)]
  pub content: Option<String>, // Only populated for EOF chunks
  #[pyo3(get)]
  pub content_hash: Option<Vec<u8>>, // Only populated for EOF chunks
  #[pyo3(get)]
  pub expected_chunks: Option<usize>, // Only populated for EOF chunks
}

impl From<Chunk> for PySemanticChunk {
  fn from(chunk: Chunk) -> Self {
    match chunk {
      Chunk::Semantic(sc) => Self {
        chunk_type: ChunkType::Semantic,
        text: sc.text,
        start_byte: sc.start_byte,
        end_byte: sc.end_byte,
        start_line: sc.start_line,
        end_line: sc.end_line,
        metadata: sc.metadata.into(),
        content: None,
        content_hash: None,
        expected_chunks: None,
      },
      Chunk::Text(sc) => Self {
        chunk_type: ChunkType::Text,
        text: sc.text,
        start_byte: sc.start_byte,
        end_byte: sc.end_byte,
        start_line: sc.start_line,
        end_line: sc.end_line,
        metadata: sc.metadata.into(),
        content: None,
        content_hash: None,
        expected_chunks: None,
      },
      Chunk::EndOfFile {
        file_path,
        content,
        content_hash,
        expected_chunks,
      } => Self {
        chunk_type: ChunkType::EndOfFile,
        text: file_path,
        start_byte: 0,
        end_byte: 0,
        start_line: 0,
        end_line: 0,
        metadata: PyChunkMetadata {
          node_type: "eof".to_string(),
          node_name: None,
          language: "".to_string(),
          parent_context: None,
          scope_path: vec![],
          definitions: vec![],
          references: vec![],
        },
        content: Some(content),
        content_hash: Some(content_hash.to_vec()),
        expected_chunks: Some(expected_chunks),
      },
      Chunk::Delete { file_path } => Self {
        chunk_type: ChunkType::Delete,
        text: file_path,
        start_byte: 0,
        end_byte: 0,
        start_line: 0,
        end_line: 0,
        metadata: PyChunkMetadata {
          node_type: "delete".to_string(),
          node_name: None,
          language: "".to_string(),
          parent_context: None,
          scope_path: vec![],
          definitions: vec![],
          references: vec![],
        },
        content: None,
        content_hash: None,
        expected_chunks: None,
      },
    }
  }
}

#[pyclass(name = "ProjectChunk")]
#[derive(Clone)]
pub struct PyProjectChunk {
  #[pyo3(get)]
  pub file_path: String,
  #[pyo3(get)]
  pub chunk: PySemanticChunk,
}

impl From<ProjectChunk> for PyProjectChunk {
  fn from(pc: ProjectChunk) -> Self {
    Self {
      file_path: pc.file_path,
      chunk: pc.chunk.into(),
    }
  }
}

#[pyclass]
pub struct SemanticChunker {
  inner: Arc<Chunker>,
}

#[pymethods]
impl SemanticChunker {
  #[new]
  #[pyo3(signature = (max_chunk_size=None, tokenizer_type=None, tokenizer_name=None))]
  fn new(
    max_chunk_size: Option<usize>,
    tokenizer_type: Option<TokenizerType>,
    tokenizer_name: Option<String>,
  ) -> PyResult<Self> {
    let max_chunk_size = max_chunk_size.unwrap_or(1500);

    // Convert Python TokenizerType to Rust Tokenizer
    let tokenizer = match tokenizer_type.unwrap_or(TokenizerType::Characters) {
      TokenizerType::Characters => RustTokenizer::Characters,
      TokenizerType::Tiktoken => match tokenizer_name.clone() {
        Some(model) => RustTokenizer::Tiktoken(model),
        None => RustTokenizer::Tiktoken("cl100k_base".to_string()), // Default to cl100k_base
      },
      TokenizerType::HuggingFace => match tokenizer_name {
        Some(model) => RustTokenizer::HuggingFace(model),
        None => {
          return Err(PyValueError::new_err(
            "TokenizerType.HUGGINGFACE requires tokenizer_name parameter",
          ));
        }
      },
    };

    let config = ChunkerConfig {
      max_chunk_size,
      tokenizer,
    };

    let inner = Chunker::new(config)
      .map_err(|e| PyRuntimeError::new_err(format!("Failed to create chunker: {}", e)))?;

    Ok(Self {
      inner: Arc::new(inner),
    })
  }

  #[pyo3(signature = (content, language, file_path=None))]
  fn chunk_code<'py>(
    &self,
    py: Python<'py>,
    content: String,
    language: String,
    file_path: Option<String>,
  ) -> PyResult<Bound<'py, PyAny>> {
    let chunker = self.inner.clone();

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
      let stream = Box::pin(chunker.chunk_code(content, language, file_path));

      Ok(ChunkStream {
        stream: Arc::new(Mutex::new(stream)),
      })
    })
  }

  #[pyo3(signature = (content, file_path=None))]
  fn chunk_text<'py>(
    &self,
    py: Python<'py>,
    content: String,
    file_path: Option<String>,
  ) -> PyResult<Bound<'py, PyAny>> {
    let chunker = self.inner.clone();

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
      let stream = Box::pin(chunker.chunk_text(content, file_path));

      Ok(ChunkStream {
        stream: Arc::new(Mutex::new(stream)),
      })
    })
  }

  #[staticmethod]
  fn supported_languages() -> Vec<&'static str> {
    supported_languages()
  }

  #[staticmethod]
  fn is_language_supported(language: &str) -> bool {
    is_language_supported(language)
  }

  #[pyo3(signature = (path, max_chunk_size=None, tokenizer_type=None, tokenizer_name=None, max_parallel=None))]
  fn walk_project<'py>(
    &self,
    py: Python<'py>,
    path: String,
    max_chunk_size: Option<usize>,
    tokenizer_type: Option<TokenizerType>,
    tokenizer_name: Option<String>,
    max_parallel: Option<usize>,
  ) -> PyResult<Bound<'py, PyAny>> {
    let max_chunk_size = max_chunk_size.unwrap_or(1500);
    let max_parallel = max_parallel.unwrap_or(8);

    // Convert Python TokenizerType to Rust TokenizerType
    let tokenizer_type = match tokenizer_type.unwrap_or(TokenizerType::Characters) {
      TokenizerType::Characters => RustTokenizer::Characters,
      TokenizerType::Tiktoken => match tokenizer_name.clone() {
        Some(model) => RustTokenizer::Tiktoken(model),
        None => RustTokenizer::Tiktoken("cl100k_base".to_string()), // Default to cl100k_base
      },
      TokenizerType::HuggingFace => match tokenizer_name {
        Some(model) => RustTokenizer::HuggingFace(model),
        None => {
          return Err(PyValueError::new_err(
            "TokenizerType.HUGGINGFACE requires tokenizer_name parameter",
          ));
        }
      },
    };

    // Create the ProjectWalker in an async context where Tokio runtime is available
    pyo3_async_runtimes::tokio::future_into_py(py, async move {
      let stream = walk_project(
        path,
        WalkOptions {
          max_chunk_size,
          tokenizer: tokenizer_type,
          max_parallel,
          ..Default::default()
        },
      );

      Ok(ProjectWalker {
        stream: Arc::new(Mutex::new(Box::pin(stream))),
      })
    })
  }
}

/// A Python module implemented in Rust.
#[pymodule]
fn breeze(m: &Bound<'_, PyModule>) -> PyResult<()> {
  // Initialize Python logging
  pyo3_log::init();

  // Add enums
  m.add_class::<TokenizerType>()?;
  m.add_class::<ChunkType>()?;

  // Add classes
  m.add_class::<SemanticChunker>()?;
  m.add_class::<PyChunkMetadata>()?;
  m.add_class::<PySemanticChunk>()?;
  m.add_class::<PyProjectChunk>()?;
  m.add_class::<ProjectWalker>()?;
  m.add_class::<ChunkStream>()?;

  // Add version info
  m.add("__version__", env!("CARGO_PKG_VERSION"))?;

  Ok(())
}

// Type aliases to avoid clippy type complexity warnings
type ProjectChunkStream = Pin<Box<dyn Stream<Item = Result<ProjectChunk, ChunkError>> + Send>>;
type ChunkStreamType = Pin<Box<dyn Stream<Item = Result<Chunk, ChunkError>> + Send>>;

#[pyclass]
pub struct ProjectWalker {
  stream: Arc<Mutex<ProjectChunkStream>>,
}

#[pymethods]
impl ProjectWalker {
  fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
    slf
  }

  fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let stream = self.stream.clone();

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
      let mut stream_guard = stream.lock().await;
      match stream_guard.next().await {
        Some(Ok(chunk)) => Ok(PyProjectChunk::from(chunk)),
        Some(Err(e)) => Err(PyRuntimeError::new_err(format!(
          "Error processing file: {}",
          e
        ))),
        None => Err(PyStopAsyncIteration::new_err("")),
      }
    })
  }
}

#[pyclass]
pub struct ChunkStream {
  stream: Arc<Mutex<ChunkStreamType>>,
}

#[pymethods]
impl ChunkStream {
  fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
    slf
  }

  fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
    let stream = self.stream.clone();

    pyo3_async_runtimes::tokio::future_into_py(py, async move {
      let mut stream_guard = stream.lock().await;
      match stream_guard.next().await {
        Some(Ok(chunk)) => Ok(PySemanticChunk::from(chunk)),
        Some(Err(e)) => Err(match e {
          ChunkError::UnsupportedLanguage(lang) => {
            PyValueError::new_err(format!("Unsupported language: {}", lang))
          }
          ChunkError::ParseError(msg) => PyRuntimeError::new_err(format!("Parse error: {}", msg)),
          ChunkError::IoError(msg) => PyRuntimeError::new_err(format!("IO error: {}", msg)),
          ChunkError::QueryError(msg) => PyRuntimeError::new_err(format!("Query error: {}", msg)),
        }),
        None => Err(PyStopAsyncIteration::new_err("")),
      }
    })
  }
}
