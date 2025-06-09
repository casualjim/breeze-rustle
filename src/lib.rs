mod types;
mod languages;
mod chunker;
mod metadata_extractor;

#[cfg(test)]
mod chunker_tests;

pub use types::{ChunkError, ChunkMetadata, SemanticChunk};
use chunker::{InnerChunker, TokenizerType};
use languages::{supported_languages, is_language_supported};

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyRuntimeError};
use std::sync::Arc;

#[pyclass]
pub struct SemanticChunker {
    inner: Arc<InnerChunker>,
}

#[pymethods]
impl SemanticChunker {
    #[new]
    #[pyo3(signature = (max_chunk_size=None, tokenizer=None))]
    fn new(max_chunk_size: Option<usize>, tokenizer: Option<String>) -> PyResult<Self> {
        let max_chunk_size = max_chunk_size.unwrap_or(1500);
        
        // Choose tokenizer type based on parameter
        let tokenizer_type = match tokenizer.as_deref() {
            None | Some("characters") => TokenizerType::Characters,
            Some("tiktoken") => TokenizerType::Tiktoken,
            Some(model) if model.starts_with("hf:") => {
                // HuggingFace tokenizer with "hf:model_name" format
                TokenizerType::HuggingFace(model[3..].to_string())
            }
            Some(tok) => {
                return Err(PyValueError::new_err(
                    format!("Unsupported tokenizer '{}'. Supported options: 'characters', 'tiktoken', 'hf:model_name'", tok)
                ));
            }
        };
        
        let inner = InnerChunker::new(max_chunk_size, tokenizer_type)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create chunker: {}", e)))?;
        
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
    
    #[pyo3(signature = (content, language, file_path=None))]
    fn chunk_file<'py>(
        &self,
        py: Python<'py>,
        content: String,
        language: String,
        file_path: Option<String>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let chunker = self.inner.clone();
        
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            chunker
                .chunk_file(&content, &language, file_path.as_deref())
                .await
                .map_err(|e| match e {
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
            chunker
                .chunk_text(&content, file_path.as_deref())
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Text chunking error: {}", e)))
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
}

/// A Python module implemented in Rust.
#[pymodule]
fn breeze_rustle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Initialize Python logging
    pyo3_log::init();
    
    // Add classes
    m.add_class::<SemanticChunker>()?;
    m.add_class::<ChunkMetadata>()?;
    m.add_class::<SemanticChunk>()?;
    
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}
