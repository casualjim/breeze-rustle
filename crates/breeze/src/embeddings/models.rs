use candle_core::{Device, Tensor, Module, DType};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use candle_transformers::models::jina_bert::{BertModel as JinaBertModel, Config as JinaConfig};
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
                // BERT requires token_type_ids - for sentence transformers, all zeros
                let token_type_ids = input_ids.zeros_like()?;
                model.forward(input_ids, &token_type_ids, Some(attention_mask))
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
}