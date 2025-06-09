use crate::types::ProjectChunk;
use futures::Stream;
use pyo3::prelude::*;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Wrapper for async iteration in Python
#[pyclass]
pub struct ProjectChunkIterator {
    stream: Pin<Box<dyn Stream<Item = Result<ProjectChunk, crate::ChunkError>> + Send>>,
}

impl ProjectChunkIterator {
    pub fn new(stream: impl Stream<Item = Result<ProjectChunk, crate::ChunkError>> + Send + 'static) -> Self {
        Self {
            stream: Box::pin(stream),
        }
    }
}

#[pymethods]
impl ProjectChunkIterator {
    fn __aiter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }
    
    fn __anext__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        use futures::StreamExt;
        use pyo3_async_runtimes::tokio::future_into_py;
        
        let fut = self.stream.next();
        
        let result = future_into_py(py, async move {
            match fut.await {
                Some(Ok(chunk)) => Ok(Some(chunk)),
                Some(Err(e)) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Error: {}", e))),
                None => Ok(None),
            }
        })?;
        
        Ok(Some(result))
    }
}