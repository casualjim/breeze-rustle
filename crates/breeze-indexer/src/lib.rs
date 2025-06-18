pub mod aiproviders;
pub mod config;
pub mod converter;
pub mod document_builder;
pub mod embeddings;
pub mod indexer;
pub mod models;
pub mod pipeline;
mod reqwestx;
pub mod search;
pub mod sinks;

// Re-export main types
pub use indexer::Indexer;
pub use models::CodeDocument;
pub use search::{SearchResult, hybrid_search};

// Re-export embedding types
pub use embeddings::{EmbeddingInput, EmbeddingProvider};

// Re-export config types for breeze CLI
pub use config::{
  Config, EmbeddingProvider as ConfigEmbeddingProvider, OpenAILikeConfig, VoyageConfig,
};

// Re-export voyage types for CLI
pub use aiproviders::voyage::{EmbeddingModel as VoyageModel, Tier as VoyageTier};

// Global ONNX runtime initialization to prevent multiple initialization issues
use small_ctor::ctor;
use std::sync::OnceLock;

static ORT_INIT_RESULT: OnceLock<Result<(), String>> = OnceLock::new();

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
