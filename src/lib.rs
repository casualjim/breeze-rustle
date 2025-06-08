mod types;
mod languages;
mod chunker;

pub use types::{ChunkError, ChunkMetadata, SemanticChunk};

use pyo3::prelude::*;

/// A Python module implemented in Rust.
#[pymodule]
fn breeze_rustle(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add classes
    m.add_class::<ChunkMetadata>()?;
    m.add_class::<SemanticChunk>()?;
    
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}
