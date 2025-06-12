use std::sync::Arc;
use candle_core::{Device, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::jina_bert::{BertModel as JinaBertModel, Config as JinaConfig};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use tokenizers::Tokenizer;

use crate::embeddings::{
    EmbeddingError,
    models::{ModelType, SentenceTransformerModel},
    sentence_transformer::SentenceTransformerEmbedder,
};

/// Load a sentence transformer embedder
pub async fn load_embedder(model_type: ModelType) -> Result<SentenceTransformerEmbedder, EmbeddingError> {
    let device = select_device()?;
    load_embedder_with_device(model_type, device).await
}

/// Load a sentence transformer embedder with a specific device
pub async fn load_embedder_with_device(
    model_type: ModelType,
    device: Device,
) -> Result<SentenceTransformerEmbedder, EmbeddingError> {
    // Create API client
    let api = Api::new()
        .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to create API: {}", e)))?;
    
    let repo = api.repo(Repo::new(model_type.model_id().to_string(), RepoType::Model));
    
    // Load config
    let config_filename = repo
        .get("config.json")
        .await
        .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to download config: {}", e)))?;
    
    let config_str = std::fs::read_to_string(&config_filename)
        .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to read config: {}", e)))?;
    
    // Load model weights
    let weights_filename = repo
        .get("model.safetensors")
        .await
        .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to download weights: {}", e)))?;
    
    let weights = candle_core::safetensors::load(&weights_filename, &device)
        .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to load weights: {}", e)))?;
    let vb = VarBuilder::from_tensors(weights, DType::F32, &device);
    
    // Load tokenizer
    let tokenizer_filename = repo
        .get("tokenizer.json")
        .await
        .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to download tokenizer: {}", e)))?;
    
    let tokenizer = Tokenizer::from_file(&tokenizer_filename)
        .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to load tokenizer: {}", e)))?;
    
    // Create model based on type and extract max sequence length and hidden size
    let (model, max_seq_length, hidden_size) = match model_type {
        ModelType::Granite | ModelType::AllMiniLM => {
            let config: BertConfig = serde_json::from_str(&config_str)
                .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to parse config: {}", e)))?;
            let max_seq_length = config.max_position_embeddings;
            let hidden_size = config.hidden_size;
            let model = BertModel::load(vb, &config)
                .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to load model: {}", e)))?;
            (SentenceTransformerModel::Bert(model), max_seq_length, hidden_size)
        }
        ModelType::JinaCodeV2 => {
            let config: JinaConfig = serde_json::from_str(&config_str)
                .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to parse config: {}", e)))?;
            let max_seq_length = config.max_position_embeddings;
            let hidden_size = config.hidden_size;
            let model = JinaBertModel::new(vb, &config)
                .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to load model: {}", e)))?;
            (SentenceTransformerModel::JinaBert(model), max_seq_length, hidden_size)
        }
    };
    
    // TODO: Load pooling config from sentence_bert_config.json if available
    // For now, use mean pooling as default
    use crate::embeddings::models::PoolingMethod;
    let pooling_method = PoolingMethod::Mean;
    
    Ok(SentenceTransformerEmbedder::new(
        model,
        Arc::new(tokenizer),
        device,
        model_type,
        max_seq_length,
        pooling_method,
        hidden_size,
    ))
}

/// Select the best available device
pub fn select_device() -> Result<Device, EmbeddingError> {
    // Try CUDA first if available
    if cfg!(feature = "cuda") {
        if let Ok(device) = Device::new_cuda(0) {
            return Ok(device);
        }
    }
    
    // Try Metal Performance Shaders on macOS
    if cfg!(target_os = "macos") {
        if let Ok(device) = Device::new_metal(0) {
            return Ok(device);
        }
    }
    
    // Default to CPU
    Ok(Device::Cpu)
}

/// Get device name for display
pub fn device_name(device: &Device) -> &str {
    match device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA",
        Device::Metal(_) => "Metal (MPS)",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_device_selection() {
        let device = select_device().unwrap();
        match device {
            Device::Cpu => println!("Using CPU"),
            Device::Cuda(_) => println!("Using CUDA"),
            Device::Metal(_) => println!("Using Metal"),
        }
    }
    
    #[test]
    fn test_device_name() {
        let cpu = Device::Cpu;
        assert_eq!(device_name(&cpu), "CPU");
    }
    
    #[tokio::test]
    async fn test_load_embedder() {
        // Try loading a small model
        let embedder = load_embedder(ModelType::AllMiniLM).await.unwrap();
        
        // Test that we can embed some text
        use crate::pipeline::{Embedder, TextBatch};
        use breeze_chunkers::{ProjectChunk, Chunk, SemanticChunk, ChunkMetadata};
        use futures_util::stream;
        
        let chunk = ProjectChunk {
            file_path: "test.txt".to_string(),
            chunk: Chunk::Text(SemanticChunk {
                text: "Hello, world!".to_string(),
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
            }),
        };
        
        let batch: TextBatch = vec![chunk];
        let stream = stream::once(async { batch }).boxed();
        let mut results = embedder.embed(stream);
        
        use futures_util::StreamExt;
        let embedding_batch = results.next().await.unwrap();
        assert_eq!(embedding_batch.len(), 1);
        assert!(!embedding_batch[0].embeddings.is_empty());
        
        // Verify embedding dimension matches what the model reports
        assert_eq!(embedding_batch[0].embeddings.len(), embedder.embedding_dim());
        
        // Check normalization
        let norm: f32 = embedding_batch[0].embeddings.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized, norm={}", norm);
    }
}