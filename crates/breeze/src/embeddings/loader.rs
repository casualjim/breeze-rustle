use hf_hub::{Repo, RepoType, api::tokio::Api};
use std::path::{Path, PathBuf};
use text_embeddings_backend_core::{ModelType, Pool};
use tracing::info;

use super::EmbeddingError;
use super::tei::TEIEmbedder;

/// Load a TEI embedder from a model path or HuggingFace model ID
pub async fn load_tei_embedder(
  model_id_or_path: &str,
  dtype: &str,
  pooling: Option<Pool>,
) -> Result<TEIEmbedder, EmbeddingError> {
  let model_path = resolve_model_path(model_id_or_path).await?;

  // Default to mean pooling if not specified
  let pooling = pooling.unwrap_or(Pool::Mean);
  let model_type = ModelType::Embedding(pooling);

  info!(
    "Loading TEI embedder from: {:?} with dtype: {}",
    model_path, dtype
  );
  TEIEmbedder::new(model_path, dtype, model_type).await
}

/// Resolve a model ID or path to a local directory
/// If it's a local path that exists, return it
/// Otherwise, download from HuggingFace Hub
async fn resolve_model_path(model_id_or_path: &str) -> Result<PathBuf, EmbeddingError> {
  let path = Path::new(model_id_or_path);

  if path.exists() && path.is_dir() {
    Ok(path.to_path_buf())
  } else {
    download_model_from_hub(model_id_or_path).await
  }
}

/// Download a model from HuggingFace Hub
async fn download_model_from_hub(model_id: &str) -> Result<PathBuf, EmbeddingError> {
  info!("Downloading model from HuggingFace Hub: {}", model_id);

  let api = Api::new()
    .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to create HF API: {}", e)))?;

  let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

  // Download essential files
  let config_path = repo.get("config.json").await.map_err(|e| {
    EmbeddingError::ModelLoadError(format!("Failed to download config.json: {}", e))
  })?;

  let _tokenizer_path = repo.get("tokenizer.json").await.map_err(|e| {
    EmbeddingError::ModelLoadError(format!("Failed to download tokenizer.json: {}", e))
  })?;

  // Try to download safetensors first, fallback to pytorch_model.bin
  let _model_path = match repo.get("model.safetensors").await {
    Ok(path) => path,
    Err(_) => {
      info!("model.safetensors not found, trying pytorch_model.bin");
      repo.get("pytorch_model.bin").await.map_err(|e| {
        EmbeddingError::ModelLoadError(format!("Failed to download model weights: {}", e))
      })?
    }
  };

  // Return the directory containing the downloaded files
  Ok(config_path.parent().unwrap().to_path_buf())
}

/// Get the recommended dtype for a device
pub fn dtype_for_device(device: &str) -> &'static str {
  match device {
    "cuda" => "float16", // Use fp16 for GPU
    _ => "float32",      // Use fp32 for CPU/Metal
  }
}

/// Get the embedding dimension from model config
pub async fn get_embedding_dim(model_path: &Path) -> Result<usize, EmbeddingError> {
  let config_path = model_path.join("config.json");
  let config_str = std::fs::read_to_string(config_path)
    .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to read config.json: {}", e)))?;

  // Parse config to get hidden_size
  let config: serde_json::Value = serde_json::from_str(&config_str)
    .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to parse config.json: {}", e)))?;

  let hidden_size = config["hidden_size"]
    .as_u64()
    .ok_or_else(|| EmbeddingError::ModelLoadError("No hidden_size in config".to_string()))?;

  Ok(hidden_size as usize)
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn test_dtype_for_device() {
    assert_eq!(dtype_for_device("cuda"), "float16");
    assert_eq!(dtype_for_device("cpu"), "float32");
    assert_eq!(dtype_for_device("mps"), "float32");
  }
}
