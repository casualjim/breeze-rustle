use breeze_chunkers::Chunk;
use candle_core::{Device, IndexOp, Tensor};
use futures_util::StreamExt;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;

use crate::embeddings::{
  EmbeddingError,
  models::{ModelType, PoolingMethod, SentenceTransformerModel},
};
use crate::pipeline::{BoxStream, Embedder, EmbeddingBatch, EmbeddingBatchItem, TextBatch};

/// Sentence transformer embedder for specific models
pub struct SentenceTransformerEmbedder {
  model: Arc<Mutex<SentenceTransformerModel>>,
  tokenizer: Arc<Tokenizer>,
  device: Device,
  model_type: ModelType,
  max_length: usize,
  pooling_method: PoolingMethod,
  embedding_dim: usize,
}

impl SentenceTransformerEmbedder {
  /// Create a new embedder with a pre-loaded model and tokenizer
  pub fn new(
    model: SentenceTransformerModel,
    tokenizer: Arc<Tokenizer>,
    device: Device,
    model_type: ModelType,
    max_length: usize,
    pooling_method: PoolingMethod,
    embedding_dim: usize,
  ) -> Self {
    Self {
      model: Arc::new(Mutex::new(model)),
      tokenizer,
      device,
      model_type,
      max_length,
      pooling_method,
      embedding_dim,
    }
  }

  /// Get the embedding dimension
  pub fn embedding_dim(&self) -> usize {
    self.embedding_dim
  }

  /// Get the tokenizer
  pub fn tokenizer(&self) -> Arc<Tokenizer> {
    self.tokenizer.clone()
  }

  /// Override the pooling method
  pub fn with_pooling_method(mut self, pooling_method: PoolingMethod) -> Self {
    self.pooling_method = pooling_method;
    self
  }

  /// Override the maximum sequence length
  pub fn with_max_length(mut self, max_length: usize) -> Self {
    self.max_length = max_length;
    self
  }

  async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, EmbeddingError> {
    if texts.is_empty() {
      return Ok(vec![]);
    }

    // Tokenize all texts
    let encodings = self
      .tokenizer
      .encode_batch(texts.to_vec(), true)
      .map_err(|e| EmbeddingError::InferenceError(format!("Tokenization failed: {}", e)))?;

    // Find the maximum length (capped by model's max)
    let max_len = encodings
      .iter()
      .map(|e| e.len())
      .max()
      .unwrap_or(0)
      .min(self.max_length);

    // Prepare input tensors
    let mut input_ids = vec![];
    let mut attention_mask = vec![];

    for encoding in &encodings {
      let ids = encoding.get_ids();
      let mask = encoding.get_attention_mask();

      // Truncate or pad
      let mut padded_ids = ids[..ids.len().min(max_len)].to_vec();
      let mut padded_mask = mask[..mask.len().min(max_len)].to_vec();

      // Pad with zeros if necessary
      padded_ids.resize(max_len, 0);
      padded_mask.resize(max_len, 0);

      input_ids.push(padded_ids);
      attention_mask.push(padded_mask);
    }

    // Flatten and create tensors
    let flat_ids: Vec<i64> = input_ids.iter().flatten().map(|&x| x as i64).collect();
    let flat_mask: Vec<f32> = attention_mask.iter().flatten().map(|&x| x as f32).collect();

    let input_ids_tensor = Tensor::new(flat_ids.as_slice(), &self.device)
      .map_err(|e| EmbeddingError::InferenceError(format!("Failed to create input tensor: {}", e)))?
      .reshape(&[texts.len(), max_len])
      .map_err(|e| {
        EmbeddingError::InferenceError(format!("Failed to reshape input tensor: {}", e))
      })?;

    let attention_mask_tensor = Tensor::new(flat_mask.as_slice(), &self.device)
      .map_err(|e| EmbeddingError::InferenceError(format!("Failed to create mask tensor: {}", e)))?
      .reshape(&[texts.len(), max_len])
      .map_err(|e| {
        EmbeddingError::InferenceError(format!("Failed to reshape mask tensor: {}", e))
      })?;

    // Run model inference
    let model = self.model.lock().await;
    let embeddings = model
      .forward(&input_ids_tensor, &attention_mask_tensor)
      .map_err(|e| EmbeddingError::InferenceError(format!("Model inference failed: {}", e)))?;

    // Apply pooling based on the selected method
    let pooled = match self.pooling_method {
      PoolingMethod::Mean => {
        // Mean pooling over the sequence dimension
        let summed = embeddings
          .sum(1)
          .map_err(|e| EmbeddingError::InferenceError(format!("Sum failed: {}", e)))?;

        // Expand attention mask to match embedding dimensions
        let mask_expanded = attention_mask_tensor
          .unsqueeze(2)
          .map_err(|e| EmbeddingError::InferenceError(format!("Unsqueeze failed: {}", e)))?
          .broadcast_as(embeddings.shape())
          .map_err(|e| EmbeddingError::InferenceError(format!("Broadcast failed: {}", e)))?;

        let mask_sum = mask_expanded
          .sum(1)
          .map_err(|e| EmbeddingError::InferenceError(format!("Mask sum failed: {}", e)))?
          .clamp(1e-9, f64::INFINITY)
          .map_err(|e| EmbeddingError::InferenceError(format!("Clamp failed: {}", e)))?;

        summed
          .broadcast_div(&mask_sum)
          .map_err(|e| EmbeddingError::InferenceError(format!("Division failed: {}", e)))?
      }
      PoolingMethod::CLS => {
        // Use the CLS token (first token) embedding
        embeddings
          .i((.., 0, ..))
          .map_err(|e| EmbeddingError::InferenceError(format!("CLS extraction failed: {}", e)))?
      }
      PoolingMethod::Max => {
        // Max pooling over the sequence dimension
        embeddings
          .max(1)
          .map_err(|e| EmbeddingError::InferenceError(format!("Max pooling failed: {}", e)))?
      }
    };

    // Normalize embeddings (L2 normalization)
    let norms = pooled
      .sqr()
      .map_err(|e| EmbeddingError::InferenceError(format!("Square failed: {}", e)))?
      .sum_keepdim(1)
      .map_err(|e| EmbeddingError::InferenceError(format!("Norm sum failed: {}", e)))?
      .sqrt()
      .map_err(|e| EmbeddingError::InferenceError(format!("Sqrt failed: {}", e)))?
      .clamp(1e-9, f64::INFINITY)
      .map_err(|e| EmbeddingError::InferenceError(format!("Norm clamp failed: {}", e)))?;

    let normalized = pooled
      .broadcast_div(&norms)
      .map_err(|e| EmbeddingError::InferenceError(format!("Normalization failed: {}", e)))?;

    // Convert to Vec<Vec<f32>>
    let result = normalized
      .to_vec2::<f32>()
      .map_err(|e| EmbeddingError::InferenceError(format!("Tensor conversion failed: {}", e)))?;

    Ok(result)
  }
}

impl Embedder for SentenceTransformerEmbedder {
  fn embed(&self, batches: BoxStream<TextBatch>) -> BoxStream<EmbeddingBatch> {
    let embedder = self.clone();

    let stream = batches
      .then(move |batch| {
        let embedder = embedder.clone();
        async move {
          // Extract texts from chunks
          let texts: Vec<String> = batch
            .iter()
            .map(|chunk| match &chunk.chunk {
              Chunk::Semantic(sc) => sc.text.clone(),
              Chunk::Text(sc) => sc.text.clone(),
            })
            .collect();

          match embedder.embed_batch(&texts).await {
            Ok(embeddings) => {
              // Create a batch item for each embedding, preserving the original chunk
              let batch_items: Vec<EmbeddingBatchItem> = embeddings
                .into_iter()
                .zip(batch.into_iter())
                .map(|(embedding, chunk)| EmbeddingBatchItem {
                  embeddings: embedding,
                  metadata: vec![chunk], // Each item has its own chunk
                })
                .collect();

              Some(batch_items)
            }
            Err(e) => {
              eprintln!("Embedding error: {}", e);
              None
            }
          }
        }
      })
      .filter_map(|x| async move { x });

    Box::pin(stream)
  }
}

impl Clone for SentenceTransformerEmbedder {
  fn clone(&self) -> Self {
    Self {
      model: self.model.clone(),
      tokenizer: self.tokenizer.clone(),
      device: self.device.clone(),
      model_type: self.model_type,
      max_length: self.max_length,
      pooling_method: self.pooling_method,
      embedding_dim: self.embedding_dim,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::path::PathBuf;

  #[tokio::test]
  #[ignore] // Requires model files
  async fn test_embedder_with_mock_batch() {
    // This test would require actual model files
    // It's here to show the expected usage pattern

    let _model_path = PathBuf::from("/tmp/test-model");
    let _device = Device::Cpu;

    // Load model and tokenizer (would need actual files)
    // let model = SentenceTransformerModel::load(ModelType::AllMiniLM, &model_path, &device).await.unwrap();
    // let tokenizer = Arc::new(Tokenizer::from_file(model_path.join("tokenizer.json")).unwrap());

    // Create embedder
    // let embedder = SentenceTransformerEmbedder::new(model, tokenizer, device, ModelType::AllMiniLM);

    // Create test batch (TextBatch is Vec<ProjectChunk>)
    // use breeze_chunkers::{ProjectChunk, Chunk, SemanticChunk, ChunkMetadata};
    // let chunk1 = ProjectChunk {
    //     file_path: "test1.txt".to_string(),
    //     chunk: Chunk::Text(SemanticChunk {
    //         text: "Hello, world!".to_string(),
    //         start_byte: 0,
    //         end_byte: 13,
    //         start_line: 1,
    //         end_line: 1,
    //         metadata: ChunkMetadata {
    //             node_type: "text".to_string(),
    //             node_name: None,
    //             language: "text".to_string(),
    //             parent_context: None,
    //             scope_path: vec![],
    //             definitions: vec![],
    //             references: vec![],
    //         },
    //     }),
    // };
    // let _batch: TextBatch = vec![chunk1];

    // Process batch
    // let stream = stream::once(async { batch }).boxed();
    // let mut results = embedder.embed(stream);

    // Check results
    // let embedding_batch = results.next().await.unwrap();
    // assert_eq!(embedding_batch.embeddings.len(), 2);
    // assert_eq!(embedding_batch.embeddings[0].len(), 384); // AllMiniLM output dimension
  }
}
