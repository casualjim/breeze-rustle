use super::models::EmbeddingModel;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// Voyage AI subscription tiers with their multipliers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
  Free,
  Tier1,
  Tier2,
  Tier3,
}

impl std::fmt::Display for Tier {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      Self::Free => write!(f, "Free"),
      Self::Tier1 => write!(f, "Tier 1"),
      Self::Tier2 => write!(f, "Tier 2"),
      Self::Tier3 => write!(f, "Tier 3"),
    }
  }
}
impl AsRef<str> for Tier {
  fn as_ref(&self) -> &str {
    match self {
      Self::Free => "free",
      Self::Tier1 => "tier-1",
      Self::Tier2 => "tier-2",
      Self::Tier3 => "tier-3",
    }
  }
}

impl FromStr for Tier {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s.to_lowercase().as_str() {
      "free" => Ok(Tier::Free),
      "tier-1" => Ok(Tier::Tier1),
      "tier-2" => Ok(Tier::Tier2),
      "tier-3" => Ok(Tier::Tier3),
      _ => Err(format!(
        "Invalid tier: {}. Use 'free', 'tier-1', 'tier-2', or 'tier-3'",
        s
      )),
    }
  }
}

impl Tier {
  /// Get the multiplier for this tier
  pub fn multiplier(&self) -> u32 {
    match self {
      Self::Free => 1,
      Self::Tier1 => 1,
      Self::Tier2 => 2,
      Self::Tier3 => 3,
    }
  }

  /// Get tokens per minute for a specific model
  pub fn tokens_per_minute(&self, model: EmbeddingModel) -> u32 {
    model.base_tokens_per_minute() * self.multiplier()
  }

  /// Get requests per minute for this tier
  pub fn requests_per_minute(&self) -> u32 {
    match self {
      Self::Free => 3,
      Self::Tier1 => 2000,
      Self::Tier2 => 2000 * self.multiplier(),
      Self::Tier3 => 2000 * self.multiplier(),
    }
  }

  /// Get a safety margin version of tokens per minute (90%)
  pub fn safe_tokens_per_minute(&self, model: EmbeddingModel) -> u32 {
    (self.tokens_per_minute(model) * 9) / 10
  }

  /// Get a safety margin version of requests per minute (90%)
  pub fn safe_requests_per_minute(&self) -> u32 {
    (self.requests_per_minute() * 9) / 10
  }
}

/// Input file content for embedding
#[derive(Debug, Clone)]
pub struct FileContent {
  pub content: String,
  pub path: String,
  pub language: Option<String>,
}

/// Result from embedding generation
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
  /// Embeddings for each file (aggregated from chunks)
  pub embeddings: Vec<Vec<f32>>,
  /// Indices of successfully embedded files
  pub successful_files: Vec<usize>,
  /// Indices of failed files
  pub failed_files: Vec<usize>,
  /// Total tokens used
  pub total_tokens: usize,
}
