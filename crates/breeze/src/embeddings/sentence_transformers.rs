use std::sync::Arc;

use candle_core::{backend::BackendStorage, CpuStorage, Device, Storage, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig, DTYPE};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info};

use breeze_chunkers::Chunk;

use crate::pipeline::EmbeddedChunk;

use super::EmbeddingError;

/// Sentence Transformers embedder using Candle
#[derive(Clone)]
pub struct SentenceTransformersEmbedder {
  model: Arc<Mutex<BertModel>>,
  tokenizer: Arc<Tokenizer>,
  device: Device,
  normalize: bool,
  embedding_dim: usize,
}

impl SentenceTransformersEmbedder {
  /// Create a new Sentence Transformers embedder
  pub async fn new(
    model_id: &str,
    device: Option<Device>,
    normalize: bool,
  ) -> Result<Self, EmbeddingError> {
    let device = device.unwrap_or(Device::Cpu);
    info!("Loading Sentence Transformers model: {} on {:?}", model_id, device);

    // Download model files from HuggingFace
    let (config_path, tokenizer_path, weights_path) = download_model_files(model_id).await?;

    // Load config
    let config = std::fs::read_to_string(&config_path)
      .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to read config: {}", e)))?;
    let config: BertConfig = serde_json::from_str(&config)
      .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to parse config: {}", e)))?;

    let embedding_dim = config.hidden_size;

    // Load tokenizer
    let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
      .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to load tokenizer: {}", e)))?;
    tokenizer.with_padding(None);

    // Load model weights
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DTYPE, &device) }
      .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to load weights: {}", e)))?;

    let model = BertModel::load(vb, &config)
      .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to load model: {}", e)))?;

    info!(
      "Sentence Transformers model loaded successfully, embedding dim: {}",
      embedding_dim
    );

    Ok(Self {
      model: Arc::new(Mutex::new(model)),
      tokenizer: Arc::new(tokenizer),
      device,
      normalize,
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
    self.embed_chunk_batch_with_stats(chunks, None).await
  }

  /// Process a batch of chunks efficiently with optional stats
  pub async fn embed_chunk_batch_with_stats(
    &self,
    chunks: Vec<Chunk>,
    stats: Option<(usize, usize)>, // (total_chunks, total_files)
  ) -> Result<Vec<EmbeddedChunk>, EmbeddingError> {
    if chunks.is_empty() {
      return Ok(Vec::new());
    }

    let start = std::time::Instant::now();

    // Prepare batch
    let (input_ids, attention_mask, chunk_indices) = self.prepare_batch(&chunks)?;
    let prep_time = start.elapsed();

    // Run model
    let model_start = std::time::Instant::now();
    let model = self.model.lock().await;
    let embeddings = self.compute_embeddings(&model, input_ids, attention_mask)?;
    let model_time = model_start.elapsed();

    // Build results and time the extraction (including GPU->CPU transfer)
    let extract_start = std::time::Instant::now();
    let num_chunks = chunks.len();

    // Process all embeddings on GPU first
    let processed_embeddings = self.process_embeddings_batch(&embeddings)?;

    // Log tensor info before transfer
    debug!(
      "Processed embeddings shape: {:?}, device: {:?}, dtype: {:?}",
      processed_embeddings.shape(),
      processed_embeddings.device(),
      processed_embeddings.dtype()
    );

    // Single GPU->CPU transfer
    let cpu_transfer_start = std::time::Instant::now();
    let embeddings_vec = self.batch_tensor_to_vec(&processed_embeddings)?;
    let cpu_transfer_time = cpu_transfer_start.elapsed();

    // Debug info about transfer size
    let transfer_size_mb = (embeddings_vec.len() * 4) as f64 / (1024.0 * 1024.0);
    debug!(
      "GPU->CPU transfer: {:.2} MB in {:.3}s = {:.2} MB/s",
      transfer_size_mb,
      cpu_transfer_time.as_secs_f64(),
      transfer_size_mb / cpu_transfer_time.as_secs_f64()
    );

    // Build results
    let mut results = Vec::with_capacity(num_chunks);
    for (idx, chunk) in chunks.into_iter().enumerate() {
      let start_idx = idx * self.embedding_dim;
      let end_idx = start_idx + self.embedding_dim;
      let embedding = embeddings_vec[start_idx..end_idx].to_vec();
      results.push(EmbeddedChunk { chunk, embedding });
    }
    let extract_time = extract_start.elapsed();

    if let Some((total_chunks, total_files)) = stats {
      debug!(
        "Embedding timing - prep: {:.3}s, model: {:.3}s, extract: {:.3}s (cpu_transfer: {:.3}s), total: {:.3}s for {} chunks (total processed: {} chunks, {} files)",
        prep_time.as_secs_f64(),
        model_time.as_secs_f64(),
        extract_time.as_secs_f64(),
        cpu_transfer_time.as_secs_f64(),
        start.elapsed().as_secs_f64(),
        num_chunks,
        total_chunks,
        total_files
      );
    } else {
      debug!(
        "Embedding timing - prep: {:.3}s, model: {:.3}s, extract: {:.3}s (cpu_transfer: {:.3}s), total: {:.3}s for {} chunks",
        prep_time.as_secs_f64(),
        model_time.as_secs_f64(),
        extract_time.as_secs_f64(),
        cpu_transfer_time.as_secs_f64(),
        start.elapsed().as_secs_f64(),
        num_chunks
      );
    }

    Ok(results)
  }

  fn prepare_batch(
    &self,
    chunks: &[Chunk],
  ) -> Result<(Tensor, Tensor, Vec<usize>), EmbeddingError> {
    let mut all_input_ids = Vec::new();
    let mut all_attention_mask = Vec::new();
    let mut chunk_indices = Vec::new();
    let mut max_len = 0;

    // Process each chunk
    for (idx, chunk) in chunks.iter().enumerate() {
      let (input_ids, attention_mask) = match chunk {
        Chunk::Semantic(sc) | Chunk::Text(sc) => {
          if let Some(tokens) = &sc.tokens {
            // Use pre-computed tokens
            let input_ids = tokens.clone();
            let attention_mask = vec![1u32; input_ids.len()];
            (input_ids, attention_mask)
          } else {
            // Fallback to tokenizing if tokens not available
            let encoding = self.tokenizer
              .encode(sc.text.as_str(), true)
              .map_err(|e| EmbeddingError::InferenceError(format!("Tokenization failed: {}", e)))?;

            let input_ids = encoding.get_ids().to_vec();
            let attention_mask = encoding.get_attention_mask().to_vec();
            (input_ids, attention_mask)
          }
        }
        Chunk::EndOfFile { .. } => {
          return Err(EmbeddingError::InferenceError(
            "Cannot embed EOF markers".to_string(),
          ));
        }
      };

      max_len = max_len.max(input_ids.len());
      all_input_ids.push(input_ids);
      all_attention_mask.push(attention_mask);
      chunk_indices.push(idx);
    }

    // Pad sequences to max length - more efficient without cloning
    let batch_size = chunks.len();
    let mut padded_input_ids = Vec::with_capacity(batch_size * max_len);
    let mut padded_attention_mask = Vec::with_capacity(batch_size * max_len);

    for (input_ids, attention_mask) in all_input_ids.iter().zip(all_attention_mask.iter()) {
      // Extend with existing values
      padded_input_ids.extend_from_slice(input_ids);
      padded_attention_mask.extend_from_slice(attention_mask);

      // Pad the remaining
      let pad_len = max_len - input_ids.len();
      if pad_len > 0 {
        padded_input_ids.extend(std::iter::repeat(0u32).take(pad_len));
        padded_attention_mask.extend(std::iter::repeat(0u32).take(pad_len));
      }
    }

    // Create tensors
    let input_ids = Tensor::from_vec(padded_input_ids, &[batch_size, max_len], &self.device)
      .map_err(|e| EmbeddingError::InferenceError(format!("Failed to create input tensor: {}", e)))?;

    let attention_mask = Tensor::from_vec(padded_attention_mask, &[batch_size, max_len], &self.device)
      .map_err(|e| EmbeddingError::InferenceError(format!("Failed to create mask tensor: {}", e)))?;

    Ok((input_ids, attention_mask, chunk_indices))
  }

  fn compute_embeddings(
    &self,
    model: &BertModel,
    input_ids: Tensor,
    attention_mask: Tensor,
  ) -> Result<Tensor, EmbeddingError> {
    // Create token type IDs (all zeros)
    let token_type_ids = input_ids.zeros_like()
      .map_err(|e| EmbeddingError::InferenceError(format!("Failed to create token type ids: {}", e)))?;

    // Run model
    let embeddings = model
      .forward(&input_ids, &token_type_ids, Some(&attention_mask))
      .map_err(|e| EmbeddingError::InferenceError(format!("Model forward pass failed: {}", e)))?;

    Ok(embeddings)
  }

  fn process_embeddings_batch(
    &self,
    embeddings: &Tensor,
  ) -> Result<Tensor, EmbeddingError> {
    let start = std::time::Instant::now();
    
    // Mean pooling over sequence length dimension (dim=1)
    let pool_start = std::time::Instant::now();
    let pooled = embeddings.mean(1)
      .map_err(|e| EmbeddingError::InferenceError(format!("Failed to mean pool: {}", e)))?;
    debug!("Mean pooling took: {:.6}s", pool_start.elapsed().as_secs_f64());

    // Normalize if requested
    let norm_start = std::time::Instant::now();
    let final_tensor = if self.normalize {
      // Compute L2 norm for each embedding (keepdim to broadcast)
      let norm = pooled.sqr()
        .and_then(|t| t.sum_keepdim(1))
        .and_then(|t| t.sqrt())
        .map_err(|e| EmbeddingError::InferenceError(format!("Failed to compute norm: {}", e)))?;

      // Normalize by dividing by the norm
      pooled.broadcast_div(&norm)
        .map_err(|e| EmbeddingError::InferenceError(format!("Failed to normalize: {}", e)))?
    } else {
      pooled
    };
    debug!("Normalization took: {:.6}s", norm_start.elapsed().as_secs_f64());

    // Check if tensor is already contiguous to avoid unnecessary copy
    let contig_start = std::time::Instant::now();
    let result = if final_tensor.is_contiguous() {
      debug!("Tensor is already contiguous, skipping contiguous() call");
      Ok(final_tensor)
    } else {
      debug!("Tensor is not contiguous, calling contiguous()");
      final_tensor.contiguous()
        .map_err(|e| EmbeddingError::InferenceError(format!("Failed to make tensor contiguous: {}", e)))
    };
    debug!("Contiguous check/call took: {:.6}s", contig_start.elapsed().as_secs_f64());
    debug!("Total process_embeddings_batch took: {:.6}s", start.elapsed().as_secs_f64());
    
    result
  }

  fn batch_tensor_to_vec(&self, tensor: &Tensor) -> Result<Vec<f32>, EmbeddingError> {
    let start = std::time::Instant::now();

    // First flatten the tensor
    let flatten_start = std::time::Instant::now();
    let flattened = tensor.flatten_all()
      .map_err(|e| EmbeddingError::InferenceError(format!("Failed to flatten tensor: {}", e)))?;
    debug!("Flatten took: {:.6}s", flatten_start.elapsed().as_secs_f64());

    // Now extract data directly without device transfer
    let extract_start = std::time::Instant::now();
    let result = self.tensor_to_vec(&flattened)?;
    debug!("Data extraction took: {:.6}s", extract_start.elapsed().as_secs_f64());

    debug!("Total batch_tensor_to_vec took: {:.6}s for {} elements",
           start.elapsed().as_secs_f64(), result.len());

    Ok(result)
  }

  fn tensor_to_vec(&self, tensor: &Tensor) -> Result<Vec<f32>, EmbeddingError> {
    let start = std::time::Instant::now();

    // Get storage and layout directly - no device transfer needed
    let storage_access_start = std::time::Instant::now();
    let (storage, layout) = tensor.storage_and_layout();
    debug!("  tensor_to_vec: storage_and_layout took {:.6}s", storage_access_start.elapsed().as_secs_f64());

    let data_copy_start = std::time::Instant::now();
    let result = match &*storage {
      Storage::Metal(metal_storage) => {
        debug!("  tensor_to_vec: Got Metal storage, using to_cpu_storage()");
        // For Metal storage, we need to use to_cpu() which handles the transfer
        let metal_read_start = std::time::Instant::now();

        // Log some info about the metal storage
        debug!("  tensor_to_vec: Metal storage dtype: {:?}, count: {}",
               metal_storage.dtype(), layout.shape().elem_count());

        // The to_cpu_storage method handles the device synchronization
        let to_cpu_start = std::time::Instant::now();
        let cpu_storage = metal_storage.to_cpu_storage()
          .map_err(|e| EmbeddingError::InferenceError(format!("Failed to transfer Metal buffer to CPU: {}", e)))?;

        debug!("  tensor_to_vec: Metal to_cpu_storage() call took {:.6}s", to_cpu_start.elapsed().as_secs_f64());

        // Now handle it as CPU storage
        let extract_start = std::time::Instant::now();
        let result = match &cpu_storage {
          CpuStorage::F32(data) => {
            match layout.contiguous_offsets() {
              Some((start, end)) => {
                debug!("  tensor_to_vec: Contiguous, copying range {}..{}", start, end);
                Ok(data[start..end].to_vec())
              }
              None => {
                // Non-contiguous tensor - need to gather elements
                let total_elements: usize = layout.shape().dims().iter().product();
                let mut result = Vec::with_capacity(total_elements);
                let mut index = tensor.strided_index();
                for _ in 0..total_elements {
                  if let Some(idx) = index.next() {
                    result.push(data[idx]);
                  }
                }
                Ok(result)
              }
            }
          }
          _ => Err(EmbeddingError::InferenceError(
            "Expected F32 tensor storage".to_string(),
          )),
        };
        debug!("  tensor_to_vec: Extract from CPU storage took {:.6}s", extract_start.elapsed().as_secs_f64());
        debug!("  tensor_to_vec: Total Metal->CPU->Vec took {:.6}s", metal_read_start.elapsed().as_secs_f64());
        result
      }
      Storage::Cuda(cuda_storage) => {
        debug!("  tensor_to_vec: Got CUDA storage, using to_cpu_storage()");
        let cuda_read_start = std::time::Instant::now();

        // The to_cpu_storage method handles the device synchronization
        let cpu_storage = cuda_storage.to_cpu_storage()
          .map_err(|e| EmbeddingError::InferenceError(format!("Failed to transfer CUDA buffer to CPU: {}", e)))?;

        debug!("  tensor_to_vec: CUDA to_cpu_storage took {:.6}s", cuda_read_start.elapsed().as_secs_f64());

        // Now handle it as CPU storage
        match &cpu_storage {
          CpuStorage::F32(data) => {
            match layout.contiguous_offsets() {
              Some((start, end)) => {
                debug!("  tensor_to_vec: Contiguous, copying range {}..{}", start, end);
                Ok(data[start..end].to_vec())
              }
              None => {
                // Non-contiguous tensor - need to gather elements
                let total_elements: usize = layout.shape().dims().iter().product();
                let mut result = Vec::with_capacity(total_elements);
                let mut index = tensor.strided_index();
                for _ in 0..total_elements {
                  if let Some(idx) = index.next() {
                    result.push(data[idx]);
                  }
                }
                Ok(result)
              }
            }
          }
          _ => Err(EmbeddingError::InferenceError(
            "Expected F32 tensor storage".to_string(),
          )),
        }
      }
      Storage::Cpu(cpu_storage) => {
        match cpu_storage {
          CpuStorage::F32(data) => {
            debug!("  tensor_to_vec: Got F32 data slice with len {}", data.len());

            // Handle both contiguous and non-contiguous tensors
            let check_contig_start = std::time::Instant::now();
            let contig_offsets = layout.contiguous_offsets();
            debug!("  tensor_to_vec: contiguous check took {:.6}s", check_contig_start.elapsed().as_secs_f64());

            match contig_offsets {
              Some((start, end)) => {
                debug!("  tensor_to_vec: Contiguous tensor, copying range {}..{}", start, end);
                let slice_start = std::time::Instant::now();
                let result = data[start..end].to_vec();
                debug!("  tensor_to_vec: slice.to_vec() took {:.6}s for {} elements",
                       slice_start.elapsed().as_secs_f64(), end - start);
                Ok(result)
              }
              None => {
                debug!("  tensor_to_vec: Non-contiguous tensor, need to gather elements");
                // Non-contiguous tensor - need to gather elements
                let shape = tensor.shape();
                let total_elements: usize = shape.dims().iter().product();
                debug!("  tensor_to_vec: Gathering {} elements", total_elements);

                let mut result = Vec::with_capacity(total_elements);
                let gather_start = std::time::Instant::now();
                let mut index = tensor.strided_index();
                for i in 0..total_elements {
                  if let Some(idx) = index.next() {
                    result.push(data[idx]);
                  }
                  if i % 10000 == 0 && i > 0 {
                    debug!("  tensor_to_vec: Gathered {} elements so far ({:.6}s)",
                           i, gather_start.elapsed().as_secs_f64());
                  }
                }
                debug!("  tensor_to_vec: Element gathering took {:.6}s", gather_start.elapsed().as_secs_f64());
                Ok(result)
              }
            }
          }
          _ => Err(EmbeddingError::InferenceError(
            "Expected F32 tensor storage".to_string(),
          )),
        }
      }
      _ => Err(EmbeddingError::InferenceError(
        "Unexpected storage type after CPU conversion".to_string(),
      )),
    };

    debug!("  tensor_to_vec: Data copy took {:.6}s", data_copy_start.elapsed().as_secs_f64());
    debug!("  tensor_to_vec: Total took {:.6}s", start.elapsed().as_secs_f64());

    result
  }
}

/// Download model files from HuggingFace Hub
async fn download_model_files(
  model_id: &str,
) -> Result<(std::path::PathBuf, std::path::PathBuf, std::path::PathBuf), EmbeddingError> {
  let api = Api::new()
    .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to create HF API: {}", e)))?;

  let repo = api.repo(Repo::new(
    model_id.to_string(),
    RepoType::Model,
  ));

  // Download essential files
  let config_path = repo.get("config.json").await
    .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to download config.json: {}", e)))?;

  let tokenizer_path = repo.get("tokenizer.json").await
    .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to download tokenizer.json: {}", e)))?;

  // Try safetensors first, then pytorch
  let weights_path = match repo.get("model.safetensors").await {
    Ok(path) => path,
    Err(_) => {
      info!("model.safetensors not found, trying pytorch_model.bin");
      repo.get("pytorch_model.bin").await
        .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to download model weights: {}", e)))?
    }
  };

  Ok((config_path, tokenizer_path, weights_path))
}

#[cfg(test)]
mod tests {
  use super::*;
  use breeze_chunkers::{ChunkMetadata, SemanticChunk};

  #[tokio::test]
  async fn test_sentence_transformers_basic() {
    let embedder = match SentenceTransformersEmbedder::new(
      "all-MiniLM-L6-v2",
      None,
      true,
    ).await {
      Ok(e) => e,
      Err(_) => {
        eprintln!("Skipping test - model download failed");
        return;
      }
    };

    assert_eq!(embedder.embedding_dim(), 384); // all-MiniLM-L6-v2 has 384 dims

    // Create test chunk
    let chunk = Chunk::Text(SemanticChunk {
      text: "Hello, world!".to_string(),
      tokens: None,
      start_byte: 0,
      end_byte: 13,
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

    let results = embedder.embed_chunk_batch(vec![chunk]).await.unwrap();
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].embedding.len(), 384);
  }

  #[tokio::test]
  async fn test_sentence_transformers_batch() {
    let embedder = match SentenceTransformersEmbedder::new(
      "all-MiniLM-L6-v2",
      None,
      true,
    ).await {
      Ok(e) => e,
      Err(_) => {
        eprintln!("Skipping test - model download failed");
        return;
      }
    };

    let chunks = vec![
      Chunk::Text(SemanticChunk {
        text: "First chunk".to_string(),
        tokens: None,
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
      }),
      Chunk::Text(SemanticChunk {
        text: "Second chunk with more text".to_string(),
        tokens: None,
        start_byte: 12,
        end_byte: 39,
        start_line: 2,
        end_line: 2,
        metadata: ChunkMetadata {
          node_type: "text".to_string(),
          node_name: None,
          language: "text".to_string(),
          parent_context: None,
          scope_path: vec![],
          definitions: vec![],
          references: vec![],
        },
      }),
    ];

    let results = embedder.embed_chunk_batch(chunks).await.unwrap();
    assert_eq!(results.len(), 2);
    for result in results {
      assert_eq!(result.embedding.len(), 384);
      // Check normalization
      let norm: f32 = result.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
      assert!((norm - 1.0).abs() < 0.01, "Expected normalized embedding, got norm {}", norm);
    }
  }

  #[tokio::test]
  async fn test_sentence_transformers_eof_error() {
    let embedder = match SentenceTransformersEmbedder::new(
      "all-MiniLM-L6-v2",
      None,
      true,
    ).await {
      Ok(e) => e,
      Err(_) => {
        eprintln!("Skipping test - model download failed");
        return;
      }
    };

    let chunk = Chunk::EndOfFile {
      file_path: "test.rs".to_string(),
      content: "content".to_string(),
      content_hash: [0; 32],
    };

    let result = embedder.embed_chunk_batch(vec![chunk]).await;
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("EOF markers"));
  }
}
