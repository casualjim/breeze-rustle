mod types;
mod languages;
mod chunker;
mod metadata_extractor;
mod walker;

#[cfg(test)]
mod chunker_tests;

pub use types::{ChunkError, ChunkMetadata, SemanticChunk, ChunkType, ProjectChunk};
use chunker::{InnerChunker, TokenizerType as RustTokenizerType};
use languages::{supported_languages, is_language_supported};

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError, PyStopAsyncIteration};
use std::sync::Arc;
use std::pin::Pin;
use futures::{Stream, StreamExt};
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

#[pyclass]
pub struct SemanticChunker {
    inner: Arc<InnerChunker>,
}

#[pymethods]
impl SemanticChunker {
    #[new]
    #[pyo3(signature = (max_chunk_size=None, tokenizer=None, hf_model=None))]
    fn new(max_chunk_size: Option<usize>, tokenizer: Option<TokenizerType>, hf_model: Option<String>) -> PyResult<Self> {
        let max_chunk_size = max_chunk_size.unwrap_or(1500);
        
        // Convert Python TokenizerType to Rust TokenizerType
        let tokenizer_type = match tokenizer.unwrap_or(TokenizerType::Characters) {
            TokenizerType::Characters => RustTokenizerType::Characters,
            TokenizerType::Tiktoken => RustTokenizerType::Tiktoken,
            TokenizerType::HuggingFace => {
                match hf_model {
                    Some(model) => RustTokenizerType::HuggingFace(model),
                    None => {
                        return Err(PyValueError::new_err(
                            "TokenizerType.HUGGINGFACE requires hf_model parameter"
                        ));
                    }
                }
            }
        };
        
        let inner = InnerChunker::new(max_chunk_size, tokenizer_type)
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
            use futures::StreamExt;
            
            let mut chunks = Vec::new();
            let mut stream = Box::pin(chunker.chunk_code(&content, &language, file_path.as_deref()));
            
            while let Some(result) = stream.next().await {
                match result {
                    Ok(chunk) => chunks.push(chunk),
                    Err(e) => {
                        return Err(match e {
                            ChunkError::UnsupportedLanguage(lang) => {
                                PyValueError::new_err(format!("Unsupported language: {}", lang))
                            }
                            ChunkError::ParseError(msg) => {
                                PyRuntimeError::new_err(format!("Parse error: {}", msg))
                            }
                            ChunkError::IoError(msg) => {
                                PyRuntimeError::new_err(format!("IO error: {}", msg))
                            }
                            ChunkError::QueryError(msg) => {
                                PyRuntimeError::new_err(format!("Query error: {}", msg))
                            }
                        });
                    }
                }
            }
            
            Ok(chunks)
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
            use futures::StreamExt;
            
            let mut chunks = Vec::new();
            let mut stream = Box::pin(chunker.chunk_text(&content, file_path.as_deref()));
            
            while let Some(result) = stream.next().await {
                match result {
                    Ok(chunk) => chunks.push(chunk),
                    Err(e) => {
                        return Err(PyRuntimeError::new_err(format!("Text chunking error: {}", e)));
                    }
                }
            }
            
            Ok(chunks)
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
    
    #[pyo3(signature = (path, max_chunk_size=None, tokenizer=None, hf_model=None, max_parallel=None))]
    fn walk_project<'py>(
        &self,
        py: Python<'py>,
        path: String,
        max_chunk_size: Option<usize>,
        tokenizer: Option<TokenizerType>,
        hf_model: Option<String>,
        max_parallel: Option<usize>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let max_chunk_size = max_chunk_size.unwrap_or(1500);
        let max_parallel = max_parallel.unwrap_or(8);
        
        // Convert Python TokenizerType to Rust TokenizerType
        let tokenizer_type = match tokenizer.unwrap_or(TokenizerType::Characters) {
            TokenizerType::Characters => RustTokenizerType::Characters,
            TokenizerType::Tiktoken => RustTokenizerType::Tiktoken,
            TokenizerType::HuggingFace => {
                match hf_model {
                    Some(model) => RustTokenizerType::HuggingFace(model),
                    None => {
                        return Err(PyValueError::new_err(
                            "TokenizerType.HUGGINGFACE requires hf_model parameter"
                        ));
                    }
                }
            }
        };
        
        // Create the ProjectWalker in an async context where Tokio runtime is available
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let stream = walker::walk_project(
                path,
                walker::WalkOptions {
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
fn breeze_rustle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize Python logging
    pyo3_log::init();
    
    // Add enums
    m.add_class::<TokenizerType>()?;
    m.add_class::<ChunkType>()?;
    
    // Add classes
    m.add_class::<SemanticChunker>()?;
    m.add_class::<ChunkMetadata>()?;
    m.add_class::<SemanticChunk>()?;
    m.add_class::<ProjectChunk>()?;
    m.add_class::<ProjectWalker>()?;
    
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}

#[pyclass]
pub struct ProjectWalker {
    stream: Arc<Mutex<Pin<Box<dyn Stream<Item = Result<ProjectChunk, ChunkError>> + Send>>>>,
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
                Some(Ok(chunk)) => Ok(chunk),
                Some(Err(e)) => Err(PyRuntimeError::new_err(format!("Error processing file: {}", e))),
                None => Err(PyStopAsyncIteration::new_err("")),
            }
        })
    }
}
