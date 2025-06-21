// Internal modules (not exported)
pub mod aiproviders;
mod bulk_indexer;
mod converter;
mod document_builder;
mod embeddings;
mod file_watcher;
mod models;
mod pipeline;
mod project_manager;
mod reqwestx;
mod sinks;
mod task_manager;

// Public modules
mod config;
mod indexer;
mod search;

// Re-export only what's needed for the public API
pub use config::*;
pub use embeddings::EmbeddingError;
pub use indexer::{Indexer, IndexerError};
pub use models::{FileChange, FileOperation, IndexTask, Project, TaskStatus, TaskType};
pub use project_manager::ProjectManager;
pub use search::{SearchResult, hybrid_search};
pub use task_manager::TaskManager;

// Global ONNX runtime initialization to prevent multiple initialization issues
#[cfg(feature = "local-embeddings")]
use small_ctor::ctor;
#[cfg(feature = "local-embeddings")]
use std::sync::OnceLock;

#[cfg(feature = "local-embeddings")]
static ORT_INIT_RESULT: OnceLock<Result<(), String>> = OnceLock::new();

#[cfg(feature = "local-embeddings")]
#[ctor]
unsafe fn init_onnx_runtime() {
  // Set ONNX runtime to quiet mode
  // SAFETY: This is called once at program initialization before any threads are spawned
  unsafe {
    std::env::set_var("ORT_DISABLE_ALL_LOGS", "1");
  }

  // Initialize ONNX runtime with rc.10 API
  let result = ort::init()
    .with_name("breeze")
    .commit()
    .map(|_| ()) // Convert bool to ()
    .map_err(|e| format!("Failed to initialize ONNX runtime: {}", e));

  // Store the result - OnceLock ensures this only happens once
  let _ = ORT_INIT_RESULT.set(result);
}

// Ensures ONNX runtime is initialized, can be called multiple times safely
#[cfg(feature = "local-embeddings")]
pub fn ensure_ort_initialized() -> Result<(), Box<dyn std::error::Error>> {
  match ORT_INIT_RESULT.get() {
    Some(Ok(())) => Ok(()),
    Some(Err(e)) => Err(e.clone().into()),
    None => {
      // This should never happen since ctor runs before main
      Err("ONNX Runtime initialization not completed".into())
    }
  }
}

#[cfg(not(feature = "local-embeddings"))]
pub fn ensure_ort_initialized() -> Result<(), Box<dyn std::error::Error>> {
  Err("Local embeddings support not enabled. Enable the 'local-embeddings' feature to use local embedding models.".into())
}
