use candle_core::{Device, Tensor, Module, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::jina_bert::{BertModel as JinaBertModel, Config as JinaConfig};
use hf_hub::{api::tokio::Api, Repo, RepoType};
use std::path::Path;

use crate::embeddings::EmbeddingError;

/// Supported sentence transformer models
#[derive(Debug, Clone, Copy)]
pub enum ModelType {
    Granite,
    JinaCodeV2,
    AllMiniLM,
}

impl ModelType {
    /// Get the Hugging Face model ID for this model type
    pub fn model_id(&self) -> &'static str {
        match self {
            ModelType::Granite => "ibm-granite/granite-embedding-125m-english",
            ModelType::JinaCodeV2 => "jinaai/jina-embeddings-v2-base-code", 
            ModelType::AllMiniLM => "sentence-transformers/all-MiniLM-L6-v2",
        }
    }
    
    /// Get the default pooling method for this model
    pub fn default_pooling(&self) -> PoolingMethod {
        match self {
            ModelType::Granite => PoolingMethod::Mean,
            ModelType::JinaCodeV2 => PoolingMethod::Mean,
            ModelType::AllMiniLM => PoolingMethod::Mean,
        }
    }
    
    /// Get the maximum sequence length for this model
    pub fn max_sequence_length(&self) -> usize {
        match self {
            ModelType::Granite => 512,
            ModelType::JinaCodeV2 => 8192, // Jina supports long sequences
            ModelType::AllMiniLM => 256,
        }
    }
}

/// Pooling methods for sentence transformers
#[derive(Debug, Clone, Copy)]
pub enum PoolingMethod {
    Mean,
    CLS,
    Max,
}

/// Enum containing the actual model implementations
pub enum SentenceTransformerModel {
    Bert(BertModel),
    JinaBert(JinaBertModel),
}

impl SentenceTransformerModel {
    /// Forward pass through the model
    pub fn forward(&self, input_ids: &Tensor, attention_mask: &Tensor) -> Result<Tensor, candle_core::Error> {
        match self {
            SentenceTransformerModel::Bert(model) => {
                // BERT forward returns hidden states, we need the last hidden state
                model.forward(input_ids, attention_mask, None)
            }
            SentenceTransformerModel::JinaBert(model) => {
                // Jina BERT only takes input_ids
                model.forward(input_ids)
            }
        }
    }
    
    /// Load a model from local files
    pub async fn load(
        model_type: ModelType,
        model_path: &Path,
        device: &Device,
    ) -> Result<Self, EmbeddingError> {
        // Load config
        let config_path = model_path.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to read config: {}", e)))?;
        
        // Load weights  
        let weights_path = model_path.join("model.safetensors");
        let weights = candle_core::safetensors::load(&weights_path, device)
            .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to load weights: {}", e)))?;
        let vb = VarBuilder::from_tensors(weights, DType::F32, device);
        
        match model_type {
            ModelType::Granite | ModelType::AllMiniLM => {
                // Both use standard BERT architecture
                let config: BertConfig = serde_json::from_str(&config_str)
                    .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to parse config: {}", e)))?;
                let model = BertModel::load(vb, &config)
                    .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to load model: {}", e)))?;
                Ok(SentenceTransformerModel::Bert(model))
            }
            ModelType::JinaCodeV2 => {
                // Jina uses a modified BERT architecture
                let config: JinaConfig = serde_json::from_str(&config_str)
                    .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to parse config: {}", e)))?;
                let model = JinaBertModel::new(vb, &config)
                    .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to load model: {}", e)))?;
                Ok(SentenceTransformerModel::JinaBert(model))
            }
        }
    }
    
    /// Download model files from Hugging Face Hub
    pub async fn download(model_type: ModelType, target_dir: &Path) -> Result<(), EmbeddingError> {
        let api = Api::new()
            .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to create API: {}", e)))?;
        
        let repo = api.repo(Repo::new(model_type.model_id().to_string(), RepoType::Model));
        
        // Create target directory if it doesn't exist
        std::fs::create_dir_all(target_dir)
            .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to create directory: {}", e)))?;
        
        // Download config
        let config_filename = repo
            .get("config.json")
            .await
            .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to download config: {}", e)))?;
        
        let target_config = target_dir.join("config.json");
        std::fs::copy(&config_filename, &target_config)
            .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to copy config: {}", e)))?;
        
        // Download model weights
        let weights_filename = repo
            .get("model.safetensors")
            .await
            .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to download weights: {}", e)))?;
        
        let target_weights = target_dir.join("model.safetensors");
        std::fs::copy(&weights_filename, &target_weights)
            .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to copy weights: {}", e)))?;
        
        // Download tokenizer
        let tokenizer_filename = repo
            .get("tokenizer.json")
            .await
            .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to download tokenizer: {}", e)))?;
        
        let target_tokenizer = target_dir.join("tokenizer.json");
        std::fs::copy(&tokenizer_filename, &target_tokenizer)
            .map_err(|e| EmbeddingError::ModelLoadError(format!("Failed to copy tokenizer: {}", e)))?;
        
        Ok(())
    }
}