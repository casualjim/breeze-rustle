use serde::{Deserialize, Serialize};

/// Text embedding models from Voyage AI
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EmbeddingModel {
  /// voyage-3: 1024 dimensions, 32k context, 128 batch
  Voyage3,
  /// voyage-3.5: 1024 dimensions, 32k context, 128 batch
  Voyage35,
  /// voyage-3-lite: 1024 dimensions, 32k context, 128 batch
  Voyage3Lite,
  /// voyage-3.5-lite: 1024 dimensions, 32k context, 128 batch
  Voyage35Lite,
  /// voyage-code-3: 1024 dimensions, 32k context, 128 batch
  VoyageCode3,
  /// voyage-finance-2: 1024 dimensions, 32k context, 128 batch
  VoyageFinance2,
  /// voyage-law-2: 1024 dimensions, 16k context, 128 batch
  VoyageLaw2,
  /// voyage-multilingual-2: 1024 dimensions, 32k context, 128 batch
  VoyageMultilingual2,
  /// voyage-3-large: 1024 dimensions, 32k context, 128 batch
  Voyage3Large,
  /// 1024 dimensions, 32k context, 128 batch
  VoyageMultiModal3,
}

impl EmbeddingModel {
  /// Get the model name for API calls
  pub fn api_name(&self) -> &'static str {
    match self {
      Self::Voyage3 => "voyage-3",
      Self::Voyage35 => "voyage-3.5",
      Self::Voyage3Lite => "voyage-3-lite",
      Self::Voyage35Lite => "voyage-3.5-lite",
      Self::VoyageCode3 => "voyage-code-3",
      Self::VoyageFinance2 => "voyage-finance-2",
      Self::VoyageLaw2 => "voyage-law-2",
      Self::VoyageMultilingual2 => "voyage-multilingual-2",
      Self::Voyage3Large => "voyage-3-large",
      Self::VoyageMultiModal3 => "voyage-multimodal-3",
    }
  }

  /// Get the output dimensions for this model
  pub fn dimensions(&self) -> usize {
    match self {
      Self::Voyage3 => 1024,
      Self::Voyage35 => 1024,
      Self::Voyage3Lite => 1024,
      Self::Voyage35Lite => 1024,
      Self::VoyageCode3 => 1024,
      Self::VoyageFinance2 => 1024,
      Self::VoyageLaw2 => 1024,
      Self::VoyageMultilingual2 => 1024,
      Self::Voyage3Large => 1024,
      Self::VoyageMultiModal3 => 1024,
    }
  }

  /// Get the context length in tokens
  pub fn context_length(&self) -> usize {
    match self {
      Self::VoyageLaw2 => 16_000,
      _ => 32_000,
    }
  }

  /// Get the maximum batch size
  pub fn max_batch_size(&self) -> usize {
    128 // All models support 128
  }

  /// Get the base tokens per minute for this model (Free tier)
  pub fn base_tokens_per_minute(&self) -> u32 {
    match self {
      Self::Voyage3 | Self::Voyage35 => 8_000_000,
      Self::Voyage3Lite | Self::Voyage35Lite => 16_000_000,
      _ => 3_000_000,
    }
  }
}

/// Reranking models from Voyage AI
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RerankingModel {
  /// rerank-2: 300k context length
  Rerank2,
  /// rerank-2-lite: 16k context length
  Rerank2Lite,
}

impl RerankingModel {
  /// Get the model name for API calls
  pub fn api_name(&self) -> &'static str {
    match self {
      Self::Rerank2 => "rerank-2",
      Self::Rerank2Lite => "rerank-2-lite",
    }
  }

  /// Get the context length in tokens
  pub fn context_length(&self) -> usize {
    match self {
      Self::Rerank2Lite => 8_000,
      Self::Rerank2 => 16_000,
    }
  }

  /// Get the base tokens per minute for this model (Free tier)
  pub fn base_tokens_per_minute(&self) -> u32 {
    match self {
      Self::Rerank2 => 2_000_000,
      Self::Rerank2Lite => 4_000_000,
    }
  }
}

impl std::fmt::Display for EmbeddingModel {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "{}", self.api_name())
  }
}

impl AsRef<str> for EmbeddingModel {
  fn as_ref(&self) -> &str {
    self.api_name()
  }
}

impl std::str::FromStr for EmbeddingModel {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      "voyage-3" => Ok(Self::Voyage3),
      "voyage-3.5" => Ok(Self::Voyage35),
      "voyage-3-lite" => Ok(Self::Voyage3Lite),
      "voyage-3.5-lite" => Ok(Self::Voyage35Lite),
      "voyage-code-3" => Ok(Self::VoyageCode3),
      "voyage-finance-2" => Ok(Self::VoyageFinance2),
      "voyage-law-2" => Ok(Self::VoyageLaw2),
      "voyage-multilingual-2" => Ok(Self::VoyageMultilingual2),
      "voyage-3-large" => Ok(Self::Voyage3Large),
      "voyage-multimodal-3" => Ok(Self::VoyageMultiModal3),
      _ => Err(format!("Unknown embedding model: {}", s)),
    }
  }
}
