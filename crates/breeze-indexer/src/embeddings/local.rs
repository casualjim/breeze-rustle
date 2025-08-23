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
mod ort_qwen3;
mod pooling;
mod text;
mod utils;

use anyhow::anyhow;
use async_trait::async_trait;
use candle_core::{Device, Tensor};
use serde::Deserialize;
use std::sync::Arc;

use crate::embeddings::local::{
  bert::BertEmbed, ort_bert::OrtBertEmbedder, ort_qwen3::OrtQwen3Embedder, text::ONNXModel,
};

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
  embedder: Arc<dyn BertEmbed + Send + Sync>,
  embedding_dim: usize,
  batch_size: usize,
  #[allow(dead_code)]
  model_name: String,
}

impl LocalEmbeddingProvider {
  pub async fn new(model_name: String) -> super::EmbeddingResult<Self> {
    // Ensure ONNX runtime is initialized
    crate::ensure_ort_initialized().map_err(|e| {
      super::EmbeddingError::OperationNotSupported(format!("Failed to initialize ORT: {}", e))
    })?;

    // Map model_name to different ONNX models
    let embedder: Box<dyn BertEmbed + Send + Sync> = if model_name.to_lowercase().contains("qwen3")
    {
      Box::new(OrtQwen3Embedder::new(None, None).map_err(|e| {
        super::EmbeddingError::ModelLoadFailed(format!("Failed to create Qwen3 embedder: {}", e))
      })?)
    } else {
      // Try to map model name to ONNXModel enum
      let onnx_model = match model_name.as_str() {
        "bge-small-en-v1.5" => ONNXModel::BGESmallENV15,
        "all-minilm-l6-v2" => ONNXModel::AllMiniLML6V2,
        "all-minilm-l12-v2" => ONNXModel::AllMiniLML12V2,
        _ => ONNXModel::BGESmallENV15, // Default
      };
      Box::new(
        OrtBertEmbedder::new(Some(onnx_model), None, None, None, None).map_err(|e| {
          super::EmbeddingError::ModelLoadFailed(format!("Failed to create BERT embedder: {}", e))
        })?,
      )
    };

    // Get embedding dimension by embedding a test string
    let test_embeddings = embedder.embed(&["test"], None, None).map_err(|e| {
      super::EmbeddingError::EmbeddingFailed(format!("Failed to get embedding dimension: {}", e))
    })?;

    let embedding_dim = match test_embeddings.first() {
      Some(EmbeddingResult::DenseVector(vec)) => vec.len(),
      _ => {
        return Err(super::EmbeddingError::EmbeddingFailed(
          "Failed to determine embedding dimension".to_string(),
        ));
      }
    };

    Ok(Self {
      embedder: Arc::from(embedder),
      embedding_dim,
      batch_size: 256, // Large batch size for CPU-based local models
      model_name,
    })
  }
}

#[async_trait]
impl EmbeddingProvider for LocalEmbeddingProvider {
  async fn embed(
    &self,
    inputs: &[super::EmbeddingInput<'_>],
  ) -> super::EmbeddingResult<Vec<Vec<f32>>> {
    // Extract texts from inputs (local embedder doesn't use tokens)
    let texts: Vec<&str> = inputs.iter().map(|input| input.text).collect();

    let embeddings = self
      .embedder
      .embed(&texts, None, None)
      .map_err(|e| super::EmbeddingError::EmbeddingFailed(e.to_string()))?;

    let mut result = Vec::with_capacity(embeddings.len());
    for embedding in embeddings {
      match embedding {
        EmbeddingResult::DenseVector(vec) => result.push(vec),
        EmbeddingResult::MultiVector(_) => {
          return Err(super::EmbeddingError::OperationNotSupported(
            "Multi-vector embeddings".to_string(),
          ));
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
