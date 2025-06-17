#![allow(unused, dead_code)]
#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::needless_return)]
#![allow(clippy::excessive_precision)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::unwrap_or_default)]
#![allow(clippy::manual_range_contains)]
#![allow(clippy::let_and_return)]
#![allow(clippy::manual_contains)]

mod bert;
mod model_info;
mod ort_bert;
mod pooling;
mod text;
mod utils;

use anyhow::anyhow;
use async_trait::async_trait;
use candle_core::{Device, Tensor};
use serde::Deserialize;
use std::sync::Arc;

use crate::embeddings::local::{bert::BertEmbed, ort_bert::OrtBertEmbedder, text::ONNXModel};

use super::{
  EmbeddingProvider,
  batching::{BatchingStrategy, LocalBatchingStrategy},
};

pub fn normalize_l2(v: &Tensor) -> candle_core::Result<Tensor> {
  v.broadcast_div(&v.sqr()?.sum_keepdim(1)?.sqrt()?)
}

pub fn select_device() -> Device {
  #[cfg(feature = "metal")]
  {
    Device::new_metal(0).unwrap_or(Device::Cpu)
  }
  #[cfg(all(not(feature = "metal"), feature = "cuda"))]
  {
    Device::cuda_if_available(0).unwrap_or(Device::Cpu)
  }
  #[cfg(not(any(feature = "metal", feature = "cuda")))]
  {
    Device::Cpu
  }
}

pub enum Dtype {
  F16,
  INT8,
  Q4,
  UINT8,
  BNB4,
  F32,
  Q4F16,
  QUANTIZED,
  BF16,
}

#[derive(Deserialize, Debug, Clone)]
pub enum EmbeddingResult {
  DenseVector(Vec<f32>),
  MultiVector(Vec<Vec<f32>>),
}

impl From<Vec<f32>> for EmbeddingResult {
  fn from(value: Vec<f32>) -> Self {
    EmbeddingResult::DenseVector(value)
  }
}

impl From<Vec<Vec<f32>>> for EmbeddingResult {
  fn from(value: Vec<Vec<f32>>) -> Self {
    EmbeddingResult::MultiVector(value)
  }
}

impl EmbeddingResult {
  pub fn to_dense(&self) -> Result<Vec<f32>, anyhow::Error> {
    match self {
      EmbeddingResult::DenseVector(x) => Ok(x.to_vec()),
      EmbeddingResult::MultiVector(_) => Err(anyhow!(
        "Multi-vector Embedding are not supported for this operation"
      )),
    }
  }

  pub fn to_multi_vector(&self) -> Result<Vec<Vec<f32>>, anyhow::Error> {
    match self {
      EmbeddingResult::MultiVector(x) => Ok(x.to_vec()),
      EmbeddingResult::DenseVector(_) => Err(anyhow!(
        "Dense Embedding are not supported for this operation"
      )),
    }
  }
}

/// Local embedding provider using embed_anything
pub struct LocalEmbeddingProvider {
  embedder: Arc<OrtBertEmbedder>,
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
    // Ensure ONNX runtime is initialized
    crate::ensure_ort_initialized()?;

    // For now, we use BGESmallENV15 as the default
    // In the future, we can map model_name to different ONNX models
    let embedder = OrtBertEmbedder::new(Some(ONNXModel::BGESmallENV15), None, None, None, None)?;

    // Get embedding dimension by embedding a test string
    let test_embeddings = embedder
      .embed(&["test"], None, None)
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
