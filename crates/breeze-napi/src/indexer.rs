use std::str::FromStr;
use std::sync::Arc;

use breeze_chunkers::Tokenizer as RustTokenizer;
use breeze_indexer::{
  ChunkResult as RustChunkResult, Config as RustIndexerConfig, Indexer as RustIndexer,
  IndexerError as RustIndexerError, Project as RustProject,
  SearchGranularity as RustSearchGranularity, SearchOptions as RustSearchOptions,
  SearchResult as RustSearchResult,
};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::chunking::TokenizerType; // reuse enum for tokenizer selection in OpenAILike config

#[napi]
#[derive(Clone, Copy)]
pub enum SearchGranularity {
  Document,
  Chunk,
}

impl From<SearchGranularity> for RustSearchGranularity {
  fn from(g: SearchGranularity) -> Self {
    match g {
      SearchGranularity::Document => RustSearchGranularity::Document,
      SearchGranularity::Chunk => RustSearchGranularity::Chunk,
    }
  }
}

#[napi(object)]
pub struct ScopeDepthJs {
  pub min: Option<u32>,
  pub max: Option<u32>,
}

#[napi(object)]
pub struct SearchOptionsJs {
  pub languages: Option<Vec<String>>,
  pub file_limit: Option<u32>,
  pub chunks_per_file: Option<u32>,
  pub granularity: Option<SearchGranularity>,

  pub node_types: Option<Vec<String>>,
  pub node_name_pattern: Option<String>,
  pub parent_context_pattern: Option<String>,
  pub scope_depth: Option<ScopeDepthJs>,
  pub has_definitions: Option<Vec<String>>,
  pub has_references: Option<Vec<String>>,
}

impl From<SearchOptionsJs> for RustSearchOptions {
  fn from(js: SearchOptionsJs) -> Self {
    let mut opts = RustSearchOptions::default();
    if let Some(langs) = js.languages {
      opts.languages = Some(langs);
    }
    if let Some(fl) = js.file_limit {
      opts.file_limit = fl as usize;
    }
    if let Some(cpf) = js.chunks_per_file {
      opts.chunks_per_file = cpf as usize;
    }
    if let Some(g) = js.granularity {
      opts.granularity = g.into();
    }
    if let Some(nt) = js.node_types {
      opts.node_types = Some(nt);
    }
    if let Some(p) = js.node_name_pattern {
      opts.node_name_pattern = Some(p);
    }
    if let Some(p) = js.parent_context_pattern {
      opts.parent_context_pattern = Some(p);
    }
    if let Some(sd) = js.scope_depth {
      let min = sd.min.unwrap_or(0) as usize;
      let max = sd.max.unwrap_or(min as u32) as usize;
      opts.scope_depth = Some((min, max));
    }
    if let Some(d) = js.has_definitions {
      opts.has_definitions = Some(d);
    }
    if let Some(r) = js.has_references {
      opts.has_references = Some(r);
    }
    opts
  }
}

#[napi(object)]
pub struct ChunkResultJs {
  pub content: String,
  pub start_line: u32,
  pub end_line: u32,
  pub start_byte: u32,
  pub end_byte: u32,
  pub relevance_score: f64,

  pub node_type: String,
  pub node_name: Option<String>,
  pub language: String,
  pub parent_context: Option<String>,
  pub scope_path: Vec<String>,
  pub definitions: Vec<String>,
  pub references: Vec<String>,
}

impl From<RustChunkResult> for ChunkResultJs {
  fn from(c: RustChunkResult) -> Self {
    Self {
      content: c.content,
      start_line: c.start_line as u32,
      end_line: c.end_line as u32,
      start_byte: c.start_byte as u32,
      end_byte: c.end_byte as u32,
      relevance_score: c.relevance_score as f64,
      node_type: c.node_type,
      node_name: c.node_name,
      language: c.language,
      parent_context: c.parent_context,
      scope_path: c.scope_path,
      definitions: c.definitions,
      references: c.references,
    }
  }
}

#[napi(object)]
pub struct SearchResultJs {
  pub id: String,
  pub file_path: String,
  pub relevance_score: f64,
  pub chunk_count: u32,
  pub chunks: Vec<ChunkResultJs>,
  pub file_size: u32,
  /// microseconds since epoch
  pub last_modified_us: i64,
  /// microseconds since epoch
  pub indexed_at_us: i64,
  pub languages: Vec<String>,
  pub primary_language: Option<String>,
}

impl From<RustSearchResult> for SearchResultJs {
  fn from(r: RustSearchResult) -> Self {
    Self {
      id: r.id,
      file_path: r.file_path,
      relevance_score: r.relevance_score as f64,
      chunk_count: r.chunk_count,
      chunks: r.chunks.into_iter().map(Into::into).collect(),
      file_size: r.file_size as u32,
      last_modified_us: r.last_modified.and_utc().timestamp_micros(),
      indexed_at_us: r.indexed_at.and_utc().timestamp_micros(),
      languages: r.languages,
      primary_language: r.primary_language,
    }
  }
}

#[napi(object)]
pub struct ProjectJs {
  pub id: String,
  pub name: String,
  pub directory: String,
  pub description: Option<String>,
  pub status: String,
  /// microseconds since epoch
  pub created_at_us: i64,
  /// microseconds since epoch
  pub updated_at_us: i64,
  /// microseconds duration
  pub rescan_interval_us: Option<i64>,
  /// microseconds since epoch
  pub last_indexed_at_us: Option<i64>,
}

impl From<RustProject> for ProjectJs {
  fn from(p: RustProject) -> Self {
    Self {
      id: p.id.to_string(),
      name: p.name,
      directory: p.directory,
      description: p.description,
      status: p.status.to_string(),
      created_at_us: p.created_at.and_utc().timestamp_micros(),
      updated_at_us: p.updated_at.and_utc().timestamp_micros(),
      rescan_interval_us: p.rescan_interval.map(|d| d.as_micros() as i64),
      last_indexed_at_us: p.last_indexed_at.map(|dt| dt.and_utc().timestamp_micros()),
    }
  }
}

// Structured, JS-friendly config to build breeze-indexer::Config
#[napi]
pub enum EmbeddingProviderJs {
  Local,
  Voyage,
  OpenAILike,
}

#[napi(object)]
pub struct VoyageConfigJs {
  pub api_key: String,
  pub tier: Option<String>,  // "free" | "tier-1" | "tier-2" | "tier-3"
  pub model: Option<String>, // voyage model id
}

#[napi(object)]
pub struct OpenAILikeConfigJs {
  pub api_base: String,
  pub api_key: Option<String>,
  pub model: String,
  pub embedding_dim: u32,
  pub context_length: u32,
  pub max_batch_size: u32,
  // tokenizer
  pub tokenizer_type: Option<TokenizerType>,
  pub tokenizer_name: Option<String>,
  // limits
  pub requests_per_minute: u32,
  pub tokens_per_minute: u32,
  pub max_concurrent_requests: Option<u32>,
  pub max_tokens_per_request: Option<u32>,
  // optional OpenAI/provider-specific output controls
  pub encoding_format: Option<String>,
  pub output_dtype: Option<String>,
  pub output_dimension: Option<u32>,
}

#[napi(object)]
pub struct OpenAIProviderEntryJs {
  pub name: String,
  pub config: OpenAILikeConfigJs,
}

#[napi(object)]
pub struct IndexerConfigJs {
  pub database_path: Option<String>,
  pub embedding_provider: EmbeddingProviderJs,
  /// For OpenAILike, name of the provider to select from openai_providers
  pub openai_provider_name: Option<String>,
  pub model: Option<String>,
  pub voyage: Option<VoyageConfigJs>,
  pub openai_providers: Option<Vec<OpenAIProviderEntryJs>>, // list of named providers
  pub max_chunk_size: Option<u32>,
  pub max_file_size: Option<f64>, // bytes; JS number -> cast to u64
  pub max_parallel_files: Option<u32>,
  pub large_file_threads: Option<u32>,
  pub embedding_workers: Option<u32>,
  pub optimize_threshold: Option<u32>,
  pub document_batch_size: Option<u32>,
}

#[napi]
pub struct Indexer {
  inner: Arc<RustIndexer>,
}

fn map_indexer_err(err: RustIndexerError) -> napi::Error {
  Error::new(Status::GenericFailure, format!("Indexer error: {}", err))
}

#[napi]
impl Indexer {
  /// Factory to create an Indexer from a structured config
  #[napi(factory)]
  pub async fn create(config: IndexerConfigJs) -> Result<Self> {
    // Start with defaults and apply overrides
    let mut cfg = RustIndexerConfig::default();

    if let Some(path) = config.database_path {
      cfg.database_path = std::path::PathBuf::from(path);
    }
    if let Some(model) = config.model {
      cfg.model = model;
    }
    if let Some(v) = config.max_chunk_size {
      cfg.max_chunk_size = v as usize;
    }
    if let Some(v) = config.max_file_size {
      cfg.max_file_size = Some(v as u64);
    }
    if let Some(v) = config.max_parallel_files {
      cfg.max_parallel_files = v as usize;
    }
    if let Some(v) = config.large_file_threads {
      cfg.large_file_threads = Some(v as usize);
    }
    if let Some(v) = config.embedding_workers {
      cfg.embedding_workers = v as usize;
    }
    if let Some(v) = config.optimize_threshold {
      cfg.optimize_threshold = v as u64;
    }
    if let Some(v) = config.document_batch_size {
      cfg.document_batch_size = v as usize;
    }

    // Embedding provider mapping
    cfg.embedding_provider = match config.embedding_provider {
      crate::indexer::EmbeddingProviderJs::Local => breeze_indexer::EmbeddingProvider::Local,
      crate::indexer::EmbeddingProviderJs::Voyage => breeze_indexer::EmbeddingProvider::Voyage,
      crate::indexer::EmbeddingProviderJs::OpenAILike => {
        let name = config.openai_provider_name.clone().ok_or_else(|| {
          Error::new(
            Status::InvalidArg,
            "openai_provider_name is required when embedding_provider is OpenAILike".to_string(),
          )
        })?;
        breeze_indexer::EmbeddingProvider::OpenAILike(name)
      }
    };

    // Voyage config
    if let Some(v) = config.voyage {
      use breeze_indexer::VoyageConfig as RVoyageConfig;
      use breeze_indexer::aiproviders::voyage::Tier as RVoyageTier;
      use breeze_indexer::aiproviders::voyage::models::EmbeddingModel as RVoyageModel;

      let tier = match v.tier.as_deref() {
        Some("tier-1") => RVoyageTier::Tier1,
        Some("tier-2") => RVoyageTier::Tier2,
        Some("tier-3") => RVoyageTier::Tier3,
        _ => RVoyageTier::Free,
      };

      // If model is provided, try parse known variants, otherwise default
      let model = match v.model {
        Some(m) => RVoyageModel::from_str(&m).unwrap_or(RVoyageModel::VoyageCode3),
        None => RVoyageModel::VoyageCode3,
      };

      cfg.voyage = Some(RVoyageConfig {
        api_key: v.api_key,
        tier,
        model,
      });
    }

    // OpenAI-like providers
    if let Some(entries) = config.openai_providers {
      let mut map = std::collections::HashMap::new();
      for entry in entries {
        use breeze_indexer::OpenAILikeConfig as ROpenAIConfig;

        let tokenizer = match entry
          .config
          .tokenizer_type
          .unwrap_or(TokenizerType::Characters)
        {
          TokenizerType::Characters => RustTokenizer::Characters,
          TokenizerType::Tiktoken => {
            let name = entry.config.tokenizer_name.clone().ok_or_else(|| {
              Error::new(
                Status::InvalidArg,
                "tokenizer_name required for Tiktoken".to_string(),
              )
            })?;
            RustTokenizer::Tiktoken(name)
          }
          TokenizerType::HuggingFace => {
            let name = entry.config.tokenizer_name.clone().ok_or_else(|| {
              Error::new(
                Status::InvalidArg,
                "tokenizer_name required for HuggingFace".to_string(),
              )
            })?;
            RustTokenizer::HuggingFace(name)
          }
        };

        let cfg_entry = ROpenAIConfig {
          api_base: entry.config.api_base,
          api_key: entry.config.api_key,
          model: entry.config.model,
          embedding_dim: entry.config.embedding_dim as usize,
          context_length: entry.config.context_length as usize,
          max_batch_size: entry.config.max_batch_size as usize,
          tokenizer,
          requests_per_minute: entry.config.requests_per_minute,
          tokens_per_minute: entry.config.tokens_per_minute,
          max_concurrent_requests: entry
            .config
            .max_concurrent_requests
            .map(|v| v as usize)
            .unwrap_or(50usize),
          max_tokens_per_request: entry.config.max_tokens_per_request.map(|v| v as usize),
          encoding_format: entry.config.encoding_format,
          output_dimension: entry.config.output_dimension.map(|v| v as usize),
          output_dtype: entry.config.output_dtype,
        };

        map.insert(entry.name, cfg_entry);
      }
      cfg.openai_providers = map;
    }

    let token = CancellationToken::new();
    let inner = RustIndexer::new(cfg, token)
      .await
      .map_err(map_indexer_err)?;

    Ok(Self {
      inner: Arc::new(inner),
    })
  }

  /// Start background workers and watchers
  #[napi]
  pub async fn start(&self) -> Result<()> {
    self.inner.start().await.map_err(map_indexer_err)
  }

  /// Stop background workers and watchers
  #[napi]
  pub fn stop(&self) {
    self.inner.stop();
  }

  /// Create a new project and automatically queue full indexing
  #[napi]
  pub async fn create_project(
    &self,
    name: String,
    directory: String,
    description: Option<String>,
    rescan_interval_secs: Option<u32>,
  ) -> Result<ProjectJs> {
    let pm = self.inner.project_manager();
    let interval = rescan_interval_secs.map(|s| std::time::Duration::from_secs(s as u64));
    let project = pm
      .create_project(name, directory, description, interval)
      .await
      .map_err(map_indexer_err)?;
    Ok(project.into())
  }

  /// List all projects
  #[napi]
  pub async fn list_projects(&self) -> Result<Vec<ProjectJs>> {
    let pm = self.inner.project_manager();
    let projects = pm
      .list_projects()
      .await
      .map_err(map_indexer_err)?
      .into_iter()
      .map(Into::into)
      .collect();
    Ok(projects)
  }

  /// Get a project by id
  #[napi]
  pub async fn get_project(&self, id: String) -> Result<Option<ProjectJs>> {
    let pm = self.inner.project_manager();
    let uuid = Uuid::parse_str(&id).map_err(|e| Error::new(Status::InvalidArg, e.to_string()))?;
    let project = pm
      .get_project(uuid)
      .await
      .map_err(map_indexer_err)?
      .map(Into::into);
    Ok(project)
  }

  /// Submit a full index task for a project
  #[napi]
  pub async fn index_project(&self, project_id: String) -> Result<String> {
    let uuid =
      Uuid::parse_str(&project_id).map_err(|e| Error::new(Status::InvalidArg, e.to_string()))?;
    let task_id = self
      .inner
      .index_project(uuid)
      .await
      .map_err(map_indexer_err)?;
    Ok(task_id.to_string())
  }

  /// Submit a partial index task for a single file in a project
  #[napi]
  pub async fn index_file(&self, project_id: String, file_path: String) -> Result<String> {
    let uuid =
      Uuid::parse_str(&project_id).map_err(|e| Error::new(Status::InvalidArg, e.to_string()))?;
    let task_id = self
      .inner
      .index_file(uuid, std::path::Path::new(&file_path))
      .await
      .map_err(map_indexer_err)?;
    Ok(task_id.to_string())
  }

  /// Perform a hybrid search across indexed code
  #[napi]
  pub async fn search(
    &self,
    query: String,
    options: Option<SearchOptionsJs>,
    project_id: Option<String>,
  ) -> Result<Vec<SearchResultJs>> {
    let project_id = if let Some(id) = project_id {
      Some(Uuid::parse_str(&id).map_err(|e| Error::new(Status::InvalidArg, e.to_string()))?)
    } else {
      None
    };
    let opts = options.map(Into::into).unwrap_or_default();
    let results = self
      .inner
      .search(&query, opts, project_id)
      .await
      .map_err(map_indexer_err)?
      .into_iter()
      .map(Into::into)
      .collect();
    Ok(results)
  }
}
