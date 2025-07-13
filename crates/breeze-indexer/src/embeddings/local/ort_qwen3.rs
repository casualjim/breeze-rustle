use std::sync::RwLock;

use super::bert::BertEmbed;
use super::pooling::{ModelOutput, PooledOutputType, Pooling};
use crate::embeddings::local::EmbeddingResult;
use anyhow::Error as E;
use hf_hub::Repo;
use hf_hub::api::sync::Api;
use ndarray::prelude::*;

#[cfg(feature = "cuda")]
use ort::execution_providers::CUDAExecutionProvider;
#[cfg(feature = "metal")]
use ort::execution_providers::CoreMLExecutionProvider;
#[cfg(any(feature = "cuda", feature = "metal"))]
use ort::execution_providers::ExecutionProvider;

use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use tokenizers::Tokenizer;

#[derive(Debug)]
pub struct OrtQwen3Embedder {
  pub session: RwLock<Session>,
  pub tokenizer: Tokenizer,
  pub eod_token_id: u32,
  pub max_length: usize,
}

impl OrtQwen3Embedder {
  pub fn new(model_id: Option<&str>, revision: Option<&str>) -> Result<Self, E> {
    // Default to the Qwen3 0.6B model
    let hf_model_id = model_id.unwrap_or("onnx-community/Qwen3-Embedding-0.6B-ONNX");

    let (model_filename, tokenizer_filename) = {
      let api = Api::new()?;
      let api = match revision {
        Some(rev) => api.repo(Repo::with_revision(
          hf_model_id.to_string(),
          hf_hub::RepoType::Model,
          rev.to_string(),
        )),
        None => api.repo(Repo::new(hf_model_id.to_string(), hf_hub::RepoType::Model)),
      };
      let model = api.get("onnx/model.onnx")?;
      let tokenizer = api.get("tokenizer.json")?;
      (model, tokenizer)
    };

    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
    let eod_token_id = tokenizer
      .token_to_id("<|endoftext|>")
      .ok_or_else(|| anyhow::anyhow!("EOD token not found"))?;

    // Get optimal thread count
    let threads = std::thread::available_parallelism()
      .map(|p| p.get())
      .unwrap_or(1);
    let optimal_threads = std::cmp::max(1, threads / 2);

    #[cfg(any(feature = "cuda", feature = "metal"))]
    let mut builder = Session::builder()?;
    #[cfg(not(any(feature = "cuda", feature = "metal")))]
    let builder = Session::builder()?;

    // Configure execution providers based on features
    #[cfg(any(feature = "cuda", feature = "metal"))]
    {
      let mut providers = Vec::new();

      #[cfg(feature = "cuda")]
      providers.push(CUDAExecutionProvider::default().build());

      #[cfg(feature = "metal")]
      providers.push(CoreMLExecutionProvider::default().build());

      builder = builder.with_execution_providers(providers)?;
    }

    let session = builder
      .with_optimization_level(GraphOptimizationLevel::Level3)?
      .with_intra_threads(optimal_threads)?
      .with_inter_threads(1)?
      .commit_from_file(model_filename)?;

    Ok(Self {
      session: RwLock::new(session),
      tokenizer,
      eod_token_id,
      max_length: 512,
    })
  }

  pub fn embed_batch(&self, texts: &[&str]) -> Result<Vec<EmbeddingResult>, E> {
    // Tokenize with special handling
    let mut all_input_ids = Vec::new();
    let mut all_attention_masks = Vec::new();

    for text in texts {
      let encoding = self
        .tokenizer
        .encode(*text, false)
        .map_err(|e| anyhow::anyhow!(e))?;
      let mut ids = encoding.get_ids().to_vec();
      let mut mask = encoding.get_attention_mask().to_vec();

      // Critical: Append EOD token
      ids.push(self.eod_token_id);
      mask.push(1);

      // Pad or truncate
      if ids.len() > self.max_length {
        ids.truncate(self.max_length);
        mask.truncate(self.max_length);
      } else {
        // Left padding for Qwen3
        let pad_length = self.max_length - ids.len();
        ids = vec![0; pad_length].into_iter().chain(ids).collect();
        mask = vec![0; pad_length].into_iter().chain(mask).collect();
      }

      all_input_ids.extend(ids.iter().map(|&x| x as i64));
      all_attention_masks.extend(mask.iter().map(|&x| x as i64));
    }

    let batch_size = texts.len();

    // Create tensors
    let input_ids = Array2::from_shape_vec((batch_size, self.max_length), all_input_ids)?;
    let attention_mask =
      Array2::from_shape_vec((batch_size, self.max_length), all_attention_masks)?;

    // Create position_ids
    let position_ids = Array2::from_shape_fn((batch_size, self.max_length), |(_, j)| j as i64);

    // Create empty KV cache tensors - model expects: 28 layers, 8 heads, 128 head_dim
    let num_layers = 28;
    let num_heads = 8;
    let head_dim = 128;
    let past_seq_len = 0;

    // Build inputs with KV cache
    let mut inputs: Vec<(std::borrow::Cow<str>, ort::session::SessionInputValue)> = vec![
      (
        "input_ids".into(),
        ort::value::TensorRef::from_array_view(&input_ids)?.into(),
      ),
      (
        "attention_mask".into(),
        ort::value::TensorRef::from_array_view(&attention_mask)?.into(),
      ),
      (
        "position_ids".into(),
        ort::value::TensorRef::from_array_view(&position_ids)?.into(),
      ),
    ];

    // Add empty KV cache for each layer
    let empty_kv = Array4::<f32>::zeros((batch_size, num_heads, past_seq_len, head_dim));
    for layer in 0..num_layers {
      inputs.push((
        format!("past_key_values.{}.key", layer).into(),
        ort::value::TensorRef::from_array_view(&empty_kv)?.into(),
      ));
      inputs.push((
        format!("past_key_values.{}.value", layer).into(),
        ort::value::TensorRef::from_array_view(&empty_kv)?.into(),
      ));
    }

    // Run inference
    let mut session_guard = self.session.write().unwrap();
    let outputs = session_guard.run(inputs)?;

    // Extract last hidden states
    let last_hidden_states: ndarray::ArrayViewD<f32> =
      outputs["last_hidden_state"].try_extract_array()?;
    let last_hidden_states = last_hidden_states
      .into_dimensionality::<ndarray::Ix3>()?
      .to_owned();

    // Apply last token pooling (Qwen3 uses left padding, so last token is always at position -1)
    let pooled_embeddings = last_hidden_states.slice(s![.., -1, ..]).to_owned();

    // L2 normalize
    let mut embeddings = Vec::new();
    for mut row in pooled_embeddings.axis_iter(Axis(0)) {
      let norm = row.mapv(|x| x * x).sum().sqrt();
      if norm > 0.0 {
        let normalized = row.mapv(|x| x / norm);
        embeddings.push(EmbeddingResult::DenseVector(normalized.to_vec()));
      } else {
        embeddings.push(EmbeddingResult::DenseVector(row.to_vec()));
      }
    }

    Ok(embeddings)
  }

  fn embed(
    &self,
    text_batch: &[&str],
    batch_size: Option<usize>,
  ) -> Result<Vec<EmbeddingResult>, E> {
    let batch_size = batch_size.unwrap_or(32);
    let mut all_embeddings = Vec::new();

    for mini_text_batch in text_batch.chunks(batch_size) {
      let embeddings = self.embed_batch(mini_text_batch)?;
      all_embeddings.extend(embeddings);
    }

    Ok(all_embeddings)
  }

  pub fn embed_late_chunking(
    &self,
    _text_batch: &[&str],
    _batch_size: Option<usize>,
  ) -> Result<Vec<EmbeddingResult>, E> {
    // Qwen3 doesn't support late chunking in the same way as BERT models
    // For now, we'll return an error
    Err(anyhow::anyhow!(
      "Late chunking is not supported for Qwen3 models"
    ))
  }
}

impl BertEmbed for OrtQwen3Embedder {
  fn embed(
    &self,
    text_batch: &[&str],
    batch_size: Option<usize>,
    late_chunking: Option<bool>,
  ) -> Result<Vec<EmbeddingResult>, anyhow::Error> {
    if late_chunking.unwrap_or(false) {
      self.embed_late_chunking(text_batch, batch_size)
    } else {
      self.embed(text_batch, batch_size)
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  #[test]
  fn test_ort_qwen3_embed() {
    // Skip test in CI due to incomplete ONNX model files and disk space constraints
    if std::env::var("CI").is_ok() {
      println!("Skipping Qwen3 test in CI environment due to incomplete ONNX model files");
      return;
    }

    println!("Creating Qwen3 embedder...");
    let embedder = match OrtQwen3Embedder::new(None, None) {
      Ok(e) => {
        println!("Embedder created successfully");
        e
      }
      Err(e) => {
        eprintln!("Failed to create embedder: {}", e);
        panic!("Failed to create embedder: {}", e);
      }
    };

    println!("Running embed...");
    let texts = vec![
      "Hello world",
      "Rust ONNX Runtime example",
      "Qwen3 embeddings use last token pooling",
    ];

    let embeddings = match embedder.embed(&texts, Some(32)) {
      Ok(e) => {
        println!("Embed completed, got {} results", e.len());
        e
      }
      Err(e) => {
        eprintln!("Failed to embed: {}", e);
        panic!("Failed to embed: {}", e);
      }
    };

    println!("Embeddings shape: {} embeddings", embeddings.len());

    // Print first few values of each embedding
    for (i, text) in texts.iter().enumerate() {
      println!("\nText: \"{}\"", text);
      if let EmbeddingResult::DenseVector(embedding) = &embeddings[i] {
        println!("First 5 values: {:?}", &embedding[..5.min(embedding.len())]);

        // Verify normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        println!("L2 norm: {:.6} (should be ~1.0)", norm);
      }
    }

    assert_eq!(
      embeddings.len(),
      texts.len(),
      "Should have one embedding per text"
    );
  }
}
