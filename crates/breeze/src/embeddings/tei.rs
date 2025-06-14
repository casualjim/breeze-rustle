use std::sync::Arc;

use candle_core::utils::metal_is_available;
use text_embeddings_backend_candle::CandleBackend;
use text_embeddings_backend_core::{Backend, Batch, Embedding, ModelType};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::info;

use breeze_chunkers::Chunk;

use crate::pipeline::EmbeddedChunk;

use super::EmbeddingError;

/// TEI-based local embedder using text-embeddings-inference backend
#[derive(Clone)]
pub struct TEIEmbedder {
  backend: Arc<Mutex<CandleBackend>>,
  tokenizer: Arc<Tokenizer>,
  model_type: ModelType,
  max_batch_size: usize,
  embedding_dim: usize,
}

impl TEIEmbedder {
  /// Create a new TEI embedder with the specified model
  pub async fn new(
    model_path: std::path::PathBuf,
    dtype: &str,
    model_type: ModelType,
  ) -> Result<Self, EmbeddingError> {
    info!("Loading TEI model from: {:?}", model_path);

    // Load embedding dimension from config
    let config_path = model_path.join("config.json");
    let embedding_dim = std::fs::File::open(&config_path)
      .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to open config.json: {}", e)))
      .and_then(|file| {
        serde_json::from_reader::<_, serde_json::Value>(file).map_err(|e| {
          EmbeddingError::ModelLoadError(format!("Failed to parse config.json: {}", e))
        })
      })
      .and_then(|config| {
        config["hidden_size"]
          .as_u64()
          .map(|size| size as usize)
          .ok_or_else(|| {
            EmbeddingError::ModelLoadError("Config missing 'hidden_size' field".to_string())
          })
      })?;

    // Load tokenizer
    let tokenizer_path = model_path.join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
      .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to load tokenizer: {}", e)))?;
    tokenizer.with_padding(None);

    let backend = CandleBackend::new(&model_path, dtype.to_string(), model_type.clone())
      .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to load TEI backend: {}", e)))?;

    // Get model configuration
    let max_batch_size = backend.max_batch_size().unwrap_or(32);

    info!(
      "TEI model loaded successfully, max batch size: {}, embedding dim: {}, has metal: {}",
      max_batch_size,
      embedding_dim,
      metal_is_available()
    );

    Ok(Self {
      backend: Arc::new(Mutex::new(backend)),
      tokenizer: Arc::new(tokenizer),
      model_type,
      max_batch_size,
      embedding_dim,
    })
  }

  /// Get the embedding dimension
  pub fn embedding_dim(&self) -> usize {
    self.embedding_dim
  }

  /// Get the tokenizer
  pub fn tokenizer(&self) -> Arc<Tokenizer> {
    self.tokenizer.clone()
  }

  /// Process a batch of chunks efficiently
  pub async fn embed_chunk_batch(
    &self,
    chunks: Vec<Chunk>,
  ) -> Result<Vec<EmbeddedChunk>, EmbeddingError> {
    if chunks.is_empty() {
      return Ok(Vec::new());
    }

    let batch = self.prepare_batch_from_owned(chunks)?;
    let chunks_for_result = batch.1; // Get the chunks back

    let backend = self.backend.lock().await;
    let mut embeddings = backend
      .embed(batch.0)
      .map_err(|e| EmbeddingError::InferenceError(format!("TEI embedding failed: {}", e)))?;

    // Build results by consuming both chunks and embeddings
    let mut results = Vec::with_capacity(chunks_for_result.len());
    for (i, chunk) in chunks_for_result.into_iter().enumerate() {
      if let Some(embedding) = embeddings.remove(&i) {
        let embedding_vec = match embedding {
          Embedding::Pooled(vec) => vec,
          Embedding::All(vecs) => {
            // For All embeddings, use mean pooling
            if vecs.is_empty() {
              return Err(EmbeddingError::InferenceError(
                "Empty embedding vector".to_string(),
              ));
            }
            let dim = vecs[0].len();
            let mut pooled = vec![0.0; dim];
            let num_vecs = vecs.len();
            for vec in vecs {
              for (j, &val) in vec.iter().enumerate() {
                pooled[j] += val;
              }
            }
            for val in &mut pooled {
              *val /= num_vecs as f32;
            }
            pooled
          }
        };

        results.push(EmbeddedChunk {
          chunk,
          embedding: embedding_vec,
        });
      } else {
        return Err(EmbeddingError::InferenceError(format!(
          "Missing embedding for chunk {}",
          i
        )));
      }
    }

    Ok(results)
  }

  /// Embed a single batch of chunks
  async fn embed_batch(&self, chunks: &[Chunk]) -> Result<Vec<EmbeddedChunk>, EmbeddingError> {
    let batch = self.prepare_batch(chunks)?;

    let backend = self.backend.lock().await;
    let embeddings = backend
      .embed(batch)
      .map_err(|e| EmbeddingError::InferenceError(format!("TEI embedding failed: {}", e)))?;

    // Convert results back to EmbeddedChunk
    let mut results = Vec::new();
    for (i, chunk) in chunks.iter().enumerate() {
      if let Some(embedding) = embeddings.get(&i) {
        let embedding_vec = match embedding {
          Embedding::Pooled(vec) => vec.clone(),
          Embedding::All(vecs) => {
            // For All embeddings, use mean pooling
            if vecs.is_empty() {
              return Err(EmbeddingError::InferenceError(
                "Empty embedding vector".to_string(),
              ));
            }
            let dim = vecs[0].len();
            let mut pooled = vec![0.0; dim];
            let num_vecs = vecs.len();
            for vec in vecs {
              for (j, &val) in vec.iter().enumerate() {
                pooled[j] += val;
              }
            }
            for val in &mut pooled {
              *val /= num_vecs as f32;
            }
            pooled
          }
        };

        results.push(EmbeddedChunk {
          chunk: chunk.clone(),
          embedding: embedding_vec,
        });
      } else {
        return Err(EmbeddingError::InferenceError(format!(
          "Missing embedding for chunk {}",
          i
        )));
      }
    }

    Ok(results)
  }

  /// Prepare a TEI batch from owned chunks, returning both batch and chunks
  fn prepare_batch_from_owned(
    &self,
    chunks: Vec<Chunk>,
  ) -> Result<(Batch, Vec<Chunk>), EmbeddingError> {
    let mut input_ids = Vec::new();
    let mut token_type_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut cumulative_seq_lengths = vec![0];
    let mut pooled_indices = Vec::new();
    let mut raw_indices = Vec::new();
    let mut max_length = 0;
    let mut chunks_out = Vec::with_capacity(chunks.len());

    for (idx, chunk) in chunks.into_iter().enumerate() {
      let semantic_chunk = match &chunk {
        Chunk::Semantic(sc) | Chunk::Text(sc) => sc,
        Chunk::EndOfFile { .. } => {
          return Err(EmbeddingError::InferenceError(
            "Cannot embed EOF markers".to_string(),
          ));
        }
      };

      // Check for pre-tokenized tokens
      let chunk_tokens = semantic_chunk.tokens.as_ref().ok_or_else(|| {
        EmbeddingError::InferenceError(
          "TEI embedder requires pre-tokenized chunks. Use HuggingFace tokenizer in chunker."
            .to_string(),
        )
      })?;

      let seq_len = chunk_tokens.len();
      max_length = max_length.max(seq_len as u32);

      // Add tokens to batch
      input_ids.extend(chunk_tokens.iter().copied());

      // For most models, token_type_ids are all zeros
      token_type_ids.extend(vec![0; seq_len]);

      // Position IDs are sequential
      position_ids.extend(0..seq_len as u32);

      // Update cumulative sequence lengths
      cumulative_seq_lengths.push(input_ids.len() as u32);

      // Track indices for pooling
      pooled_indices.push(idx as u32);
      raw_indices.push(idx as u32);

      // Keep chunk for later
      chunks_out.push(chunk);
    }

    let batch = Batch {
      input_ids,
      token_type_ids,
      position_ids,
      cumulative_seq_lengths,
      max_length,
      pooled_indices,
      raw_indices,
    };

    Ok((batch, chunks_out))
  }

  /// Prepare a TEI batch from chunks (borrowing version for compatibility)
  fn prepare_batch(&self, chunks: &[Chunk]) -> Result<Batch, EmbeddingError> {
    let mut input_ids = Vec::new();
    let mut token_type_ids = Vec::new();
    let mut position_ids = Vec::new();
    let mut cumulative_seq_lengths = vec![0];
    let mut pooled_indices = Vec::new();
    let mut raw_indices = Vec::new();
    let mut max_length = 0;

    for (idx, chunk) in chunks.iter().enumerate() {
      let semantic_chunk = match chunk {
        Chunk::Semantic(sc) | Chunk::Text(sc) => sc,
        Chunk::EndOfFile { .. } => {
          return Err(EmbeddingError::InferenceError(
            "Cannot embed EOF markers".to_string(),
          ));
        }
      };

      // Try to use pre-tokenized tokens if available
      let chunk_tokens = if let Some(ref tokens) = semantic_chunk.tokens {
        tokens.clone()
      } else {
        // Fallback: would need to tokenize here, but we don't have direct access
        // to tokenizer in this context. For now, return an error.
        return Err(EmbeddingError::InferenceError(
          "TEI embedder requires pre-tokenized chunks. Use HuggingFace tokenizer in chunker."
            .to_string(),
        ));
      };

      let seq_len = chunk_tokens.len();
      max_length = max_length.max(seq_len as u32);

      // Add tokens to batch
      input_ids.extend(chunk_tokens);

      // For most models, token_type_ids are all zeros
      token_type_ids.extend(vec![0; seq_len]);

      // Position IDs are sequential
      position_ids.extend(0..seq_len as u32);

      // Update cumulative sequence lengths
      cumulative_seq_lengths.push(input_ids.len() as u32);

      // Track indices for pooling
      pooled_indices.push(idx as u32);
      raw_indices.push(idx as u32);
    }

    Ok(Batch {
      input_ids,
      token_type_ids,
      position_ids,
      cumulative_seq_lengths,
      max_length,
      pooled_indices,
      raw_indices,
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use breeze_chunkers::{ChunkMetadata, SemanticChunk};
  use std::path::PathBuf;
  use text_embeddings_backend_core::Pool;

  #[tokio::test]
  async fn test_tei_embedder_requires_tokens() {
    // Note: This test requires a TEI model to be available
    // For CI, we might want to skip this test or use a mock
    let model_path = PathBuf::from("/tmp/test-model");
    let model_type = ModelType::Embedding(Pool::Mean);

    let embedder = match TEIEmbedder::new(model_path, "float32", model_type).await {
      Ok(e) => e,
      Err(_) => {
        eprintln!("Skipping TEI test - model not available");
        return;
      }
    };

    // Create a chunk without tokens
    let chunk = Chunk::Semantic(SemanticChunk {
      text: "Hello world".to_string(),
      tokens: None, // No tokens provided
      start_byte: 0,
      end_byte: 11,
      start_line: 1,
      end_line: 1,
      metadata: ChunkMetadata {
        node_type: "text".to_string(),
        node_name: None,
        language: "text".to_string(),
        parent_context: None,
        scope_path: vec![],
        definitions: vec![],
        references: vec![],
      },
    });

    // This should fail because no tokens were provided
    let result = embedder.embed_chunk_batch(vec![chunk]).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("pre-tokenized"));
  }

  #[tokio::test]
  async fn test_tei_batch_preparation() {
    let model_path = PathBuf::from("/tmp/test-model");
    let model_type = ModelType::Embedding(Pool::Mean);

    let embedder = match TEIEmbedder::new(model_path, "float32", model_type).await {
      Ok(e) => e,
      Err(_) => {
        eprintln!("Skipping TEI test - model not available");
        return;
      }
    };

    // Create a chunk with tokens
    let chunk = Chunk::Semantic(SemanticChunk {
      text: "Hello world".to_string(),
      tokens: Some(vec![1, 2, 3, 4, 5]), // Mock tokens
      start_byte: 0,
      end_byte: 11,
      start_line: 1,
      end_line: 1,
      metadata: ChunkMetadata {
        node_type: "text".to_string(),
        node_name: None,
        language: "text".to_string(),
        parent_context: None,
        scope_path: vec![],
        definitions: vec![],
        references: vec![],
      },
    });

    let batch = embedder.prepare_batch(&[chunk]).unwrap();

    assert_eq!(batch.input_ids, vec![1, 2, 3, 4, 5]);
    assert_eq!(batch.token_type_ids, vec![0, 0, 0, 0, 0]);
    assert_eq!(batch.position_ids, vec![0, 1, 2, 3, 4]);
    assert_eq!(batch.cumulative_seq_lengths, vec![0, 5]);
    assert_eq!(batch.max_length, 5);
  }
}
