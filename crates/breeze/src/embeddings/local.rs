use async_trait::async_trait;
use embed_anything::embeddings::embed::{Embedder, EmbedderBuilder, EmbeddingResult};
use embed_anything::embeddings::local::text_embedding::ONNXModel;
use std::sync::Arc;

use super::{
  EmbeddingProvider,
  batching::{BatchingStrategy, LocalBatchingStrategy},
};

/// Local embedding provider using embed_anything
pub struct LocalEmbeddingProvider {
  embedder: Arc<Embedder>,
  embedding_dim: usize,
  batch_size: usize,
  #[allow(dead_code)]
  model_name: String,
}

impl LocalEmbeddingProvider {
  pub async fn new(
    model_name: String,
    batch_size: usize,
  ) -> Result<Self, Box<dyn std::error::Error>> {
    // For now, we use AllMiniLM-L6-v2 as the default
    // In the future, we can map model_name to different ONNX models
    let embedder = EmbedderBuilder::new()
      .model_architecture("bert")
      .onnx_model_id(Some(ONNXModel::AllMiniLML6V2))
      .from_pretrained_onnx()
      .map_err(|e| format!("Failed to create embedder: {}", e))?;

    // Get embedding dimension by embedding a test string
    let test_embeddings = embedder
      .embed(&["test"], None, None)
      .await
      .map_err(|e| format!("Failed to get embedding dimension: {}", e))?;

    let embedding_dim = match test_embeddings.first() {
      Some(EmbeddingResult::DenseVector(vec)) => vec.len(),
      _ => return Err("Failed to determine embedding dimension".into()),
    };

    Ok(Self {
      embedder: Arc::new(embedder),
      embedding_dim,
      batch_size,
      model_name,
    })
  }
}

#[async_trait]
impl EmbeddingProvider for LocalEmbeddingProvider {
  async fn embed(
    &self,
    inputs: &[super::EmbeddingInput<'_>],
  ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
    // Extract texts from inputs (local embedder doesn't use tokens)
    let texts: Vec<&str> = inputs.iter().map(|input| input.text).collect();

    let embeddings = self
      .embedder
      .embed(&texts, None, None)
      .await
      .map_err(|e| format!("Failed to embed: {}", e))?;

    let mut result = Vec::with_capacity(embeddings.len());
    for embedding in embeddings {
      match embedding {
        EmbeddingResult::DenseVector(vec) => result.push(vec),
        EmbeddingResult::MultiVector(_) => {
          return Err("Multi-vector embeddings not supported".into());
        }
      }
    }

    Ok(result)
  }

  fn embedding_dim(&self) -> usize {
    self.embedding_dim
  }

  fn context_length(&self) -> usize {
    // Most local models have relatively small context windows
    // This should be configurable per model in the future
    512
  }

  fn create_batching_strategy(&self) -> Box<dyn BatchingStrategy> {
    Box::new(LocalBatchingStrategy::new(self.batch_size))
  }

  fn tokenizer(&self) -> Option<Arc<tokenizers::Tokenizer>> {
    None // Local models use character-based tokenization
  }
}
