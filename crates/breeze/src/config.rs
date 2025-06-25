use breeze_indexer::aiproviders::voyage::EmbeddingModel as VoyageEmbeddingModel;
use breeze_indexer::{EmbeddingProvider, VoyageConfig as VoyageClientConfig};
use human_units::Size;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
  /// Database directory
  #[serde(default = "default_db_dir")]
  pub db_dir: PathBuf,

  /// Server configuration
  #[serde(default)]
  pub server: ServerConfig,

  /// Indexer configuration
  #[serde(default)]
  pub indexer: IndexerConfig,

  /// Embeddings configuration
  #[serde(default)]
  pub embeddings: EmbeddingsConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ServerConfig {
  /// TLS configuration
  #[serde(default)]
  pub tls: TlsConfig,

  /// Port configuration
  #[serde(default)]
  pub ports: PortConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct TlsConfig {
  /// Whether TLS is disabled (default: false)
  #[serde(default)]
  pub disabled: bool,

  /// TLS configuration using keypair files
  #[serde(skip_serializing_if = "Option::is_none")]
  pub keypair: Option<TlsKeypair>,

  /// Let's Encrypt configuration
  #[serde(skip_serializing_if = "Option::is_none")]
  pub letsencrypt: Option<LetsEncryptConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TlsKeypair {
  pub tls_cert: String,
  pub tls_key: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LetsEncryptConfig {
  pub domains: Vec<String>,
  pub emails: Vec<String>,
  #[serde(default = "default_true")]
  pub production: bool,
  #[serde(default = "default_cert_dir")]
  pub cert_dir: PathBuf,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct PortConfig {
  #[serde(default = "default_http_port")]
  pub http: u16,
  #[serde(default = "default_https_port")]
  pub https: u16,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct IndexerConfig {
  /// Chunk overlap as a ratio (0.0 to 1.0)
  #[serde(skip_serializing_if = "Option::is_none")]
  pub chunk_overlap: Option<f32>,

  /// Indexer limits
  #[serde(default)]
  pub limits: IndexerLimits,

  /// Worker configuration
  #[serde(default)]
  pub workers: IndexerWorkers,

  /// LanceDB optimization threshold - optimize when table version advances by this amount
  #[serde(default = "default_optimize_threshold")]
  pub optimize_threshold: u64,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct IndexerLimits {
  /// Maximum file size (e.g., "5mb", "100kb")
  #[serde(skip_serializing_if = "Option::is_none")]
  pub file_size: Option<Size>,

  /// Maximum chunk size in tokens
  #[serde(skip_serializing_if = "Option::is_none")]
  pub max_chunk_size: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct IndexerWorkers {
  #[serde(default = "default_small_file_workers")]
  pub small_file: usize,

  #[serde(default = "default_large_file_workers")]
  pub large_file: usize,

  #[serde(default = "default_batch_size")]
  pub batch_size: usize,
}

impl Default for IndexerWorkers {
  fn default() -> Self {
    Self {
      small_file: default_small_file_workers(),
      large_file: default_large_file_workers(),
      batch_size: default_batch_size(),
    }
  }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EmbeddingsConfig {
  /// Selected embedding provider
  #[serde(default = "default_provider")]
  pub provider: String,

  /// Number of concurrent embedding workers
  #[serde(default = "default_embedding_workers")]
  pub workers: usize,

  /// Local embeddings configuration
  #[serde(skip_serializing_if = "Option::is_none")]
  pub local: Option<LocalEmbeddingsConfig>,
  /// Voyage configuration
  #[serde(skip_serializing_if = "Option::is_none")]
  pub voyage: Option<VoyageConfig>,
  /// Additional providers (catch-all for custom providers)
  #[serde(flatten, default)]
  pub providers: HashMap<String, ProviderConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LocalEmbeddingsConfig {
  pub model: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub tokenizer: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub context_length: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub embedding_dim: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VoyageConfig {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub api_key: Option<String>,
  pub tier: String,
  pub model: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub context_length: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub embedding_dim: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub max_batch_size: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProviderConfig {
  #[serde(skip_serializing_if = "Option::is_none")]
  pub api_base: Option<String>,
  pub model: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub api_key: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub tokenizer: Option<String>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub context_length: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub embedding_dim: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub max_batch_size: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub requests_per_minute: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub tokens_per_minute: Option<u32>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub max_concurrent_requests: Option<usize>,
  #[serde(skip_serializing_if = "Option::is_none")]
  pub max_tokens_per_request: Option<usize>,
}

// Default functions
fn default_true() -> bool {
  true
}
fn default_cert_dir() -> PathBuf {
  PathBuf::from("./certs")
}
fn default_http_port() -> u16 {
  8080
}
fn default_https_port() -> u16 {
  8443
}
fn default_small_file_workers() -> usize {
  num_cpus::get()
}
fn default_large_file_workers() -> usize {
  4
}
fn default_batch_size() -> usize {
  256
}
fn default_db_dir() -> PathBuf {
  PathBuf::from("./embeddings.db")
}
fn default_provider() -> String {
  "local".to_string()
}
fn default_embedding_workers() -> usize {
  4
}

pub fn default_max_chunk_size() -> usize {
  512
}

fn default_optimize_threshold() -> u64 {
  250
}

impl Default for EmbeddingsConfig {
  fn default() -> Self {
    Self {
      provider: default_provider(),
      workers: default_embedding_workers(),
      local: None,
      voyage: None,
      providers: HashMap::new(),
    }
  }
}

impl Default for Config {
  fn default() -> Self {
    Self {
      db_dir: default_db_dir(),
      server: ServerConfig::default(),
      indexer: IndexerConfig::default(),
      embeddings: EmbeddingsConfig::default(),
    }
  }
}

impl Config {
  /// Get the default config directory path
  pub fn default_config_dir() -> anyhow::Result<PathBuf> {
    let config_dir = dirs::preference_dir()
      .ok_or_else(|| anyhow::anyhow!("Could not determine preferences directory"))?
      .join(env!("CARGO_PKG_NAME"));
    Ok(config_dir)
  }

  /// Get the default config file path
  pub fn default_config_path() -> anyhow::Result<PathBuf> {
    Ok(Self::default_config_dir()?.join("config.toml"))
  }

  /// Get the default data directory path
  pub fn default_data_dir() -> anyhow::Result<PathBuf> {
    let data_dir = dirs::data_dir()
      .ok_or_else(|| anyhow::anyhow!("Could not determine data directory"))?
      .join(env!("CARGO_PKG_NAME"));
    Ok(data_dir)
  }

  /// Load configuration using config-rs with layered sources
  pub fn load(config_path: Option<PathBuf>) -> anyhow::Result<Self> {
    let mut builder = config::Config::builder();

    // 1. Start with defaults
    builder = builder.add_source(config::Config::try_from(&Self::default())?);

    // 2. Add config file (uses provided path or default)
    let config_file = config_path.unwrap_or_else(|| {
      Self::default_config_path().unwrap_or_else(|_| PathBuf::from(".breeze.toml"))
    });
    builder = builder.add_source(config::File::from(config_file).required(false));

    // 3. Add environment variables with BREEZE_ prefix
    builder = builder.add_source(
      config::Environment::with_prefix("BREEZE")
        .separator("_")
        .ignore_empty(true),
    );

    let config: Self = builder.build()?.try_deserialize()?;
    Ok(config)
  }

  pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> anyhow::Result<()> {
    let content = toml::to_string_pretty(self)?;
    std::fs::write(path, content)?;
    Ok(())
  }

  /// Generate a default configuration with helpful comments
  pub fn generate_commented_config() -> String {
    r#"# Default configuration for Breeze

# Database configuration
db_dir = "~/.local/var/breeze/embeddings.db"

[server]

[server.ports]
http = 8528  # HTTP port
https = 8529  # HTTPS port

[server.tls]
disabled = true

# use keypair files, this does not support automatic cert rotation atm
[server.tls.keypair]
tls_cert = "cert.pem"  # Path to the TLS certificate
tls_key = "key.pem"  # Path to the TLS private key

# Or use letsencrypt over http-01 challenge
# [server.letsencrypt]
# domains = ["example.com", "www.example.com"]  # List of domains for TLS
# emails = ["someone@example.com"]  # List of contacts for TLS
# production = true  # Use production Let's Encrypt servers
# cert_dir = "./certs"  # Directory to store certificates



[indexer]
# Optional: Number of overlapping tokens between chunks (helps preserve context)
chunk_overlap = 0.1

[indexer.limits]
file_size = "5M"  # 5M - files larger than this are skipped
# Optional: Override the automatic chunk size calculation
# max_chunk_size = 512  # Maximum tokens per chunk
max_chunk_size = 4096  # Maximum tokens per chunk

[indexer.workers]
small_file = 8   # Number of files to process concurrently
large_file = 4  # Number of large files to process concurrently
batch_size = 256 # Number of chunks to collect in a batch betfore sending to the embeddings workers

[embeddings]
# The selected embedding provider, from the providers defined below
# provider = "local" | "openai" | "ollama"  | "llamacpp"| "voyage" | "cohere" | "my-provider" | ...
provider = "local"
workers = 5 # Number of concurrent chunk batchers to use for the embedding provider

[embeddings.local]
model = "BAAI/bge-small-en-v1.5"  # Local model for embeddings
tokenizer = "hf:BAAI/bge-small-en-v1.5" # Hugging Face tokenizer for BGE model
# Optional: Override if auto-detection fails
context_length = 512
embedding_dim = 384


# [embeddings.openai]
# model = "text-embedding-ada-002"
# Optional: API key can also be set via OPENAI_API_KEY env var
# api_key = "sk-..."
# Optional: Override defaults
# api_base = "https://api.openai.com/v1"  # For API-compatible providers
# context_length = 8191
# embedding_dim = 1536
# max_batch_size = 2048  # OpenAI allows large batches

# Rate limiting for the provider
# requests_per_minute = 3000
# tokens_per_minute = 1000000
# max_concurrent_requests = 50

[embeddings.ollama]
model = "nomic-embed-text"  # Ollama model for embeddings
api_base = "http://localhost:11434/v1"  # Ollama API base URL
# Optional: Override if auto-detection fails
context_length = 8192
embedding_dim = 768
max_batch_size = 64
tokenizer = "hf:nomic-ai/nomic-embed-text-v1.5" # huggingface tokenizer for Ollama model

# Rate limiting for the provider
requests_per_minute = 50
tokens_per_minute = 1_000_000
max_concurrent_requests = 10
max_tokens_per_request = 64_000

# runpodctl create pod \
#  --gpuType 'NVIDIA GeForce RTX 3090' \
#  --imageName vllm/vllm-openai:latest \
#  --volumeSize 30 \
#  --containerDiskSize 10 \
#  --name qwen3-embed \
#  --secureCloud \
#  --args '--host 0.0.0.0 --port 8000 --model Qwen/Qwen3-Embedding-0.6B --enforce-eager --trust-remote-code --max-model-len 32768 --max-num-seqs 200 --max-num-batched-tokens 192000 --gpu-memory-utilization 0.95 --disable-log-stats --api-key sk-moFgaeiq8K76FXkCNx2ydwtnxWutYotT' \
# --ports '8000/http' \
# --ports '22/tcp' \
# --volumePath /workspace \
# --env "HF_HOME=/workspace/hf_home"
[embeddings.qwen3]
model = "Qwen/Qwen3-Embedding-0.6B"
api_base = "https://wfqcjfjn5a7hiv-8000.proxy.runpod.net/v1"
context_length = 32768
embedding_dim = 1024
tokenizer = "hf:Qwen/Qwen3-Embedding-0.6B"

requests_per_minute = 10
tokens_per_minute = 1_000_000
max_concurrent_requests = 5
max_tokens_per_request = 128_000
max_batch_size = 150

[embeddings.voyage]
# api_key = "${VOYAGE_API_KEY}"  # Set your Voyage API key in the environment
tier = "tier-1"  # "free" | "tier-1" | "tier-2" | "tier-3" - affects rate limits
model = "voyage-code-3" # Best for code embeddings

# Optional overrides (Voyage has good auto-detection)
# context_length = 32768
# embedding_dim = 1024

# Voyage automatically adjusts batch sizes based on tier:
# - free: conservative batching
# - tier1-3: progressively larger batches
# max_batch_size = 128  # Override only if needed
max_concurrent_requests = 5

# [embeddings.my-provider]
# api_base = "https://api.myprovider.com/v1"
# model = "my-embedding-model"
# Optional: API key can also be set via MY_PROVIDER_API_KEY env var
# api_key = "..."
# context_length = 512
# embedding_dim = 1024

# Rate limiting for the provider
# requests_per_minute = 3000
# tokens_per_minute = 1000000
# max_concurrent_requests = 50
"#
    .to_string()
  }

  /// Convert to breeze_indexer::Config for compatibility
  pub fn to_indexer_config(&self) -> anyhow::Result<breeze_indexer::Config> {
    // Determine embedding provider
    let embedding_provider = self
      .embeddings
      .provider
      .as_str()
      .parse()
      .map_err(|e| anyhow::anyhow!("Invalid embedding provider: {e}"))?;

    // Convert voyage config if present
    let voyage = self
      .embeddings
      .voyage
      .as_ref()
      .map(|v| -> anyhow::Result<VoyageClientConfig> {
        // Get API key from config or environment variable
        let api_key = v
          .api_key
          .clone()
          .or_else(|| std::env::var("VOYAGE_API_KEY").ok())
          .ok_or_else(|| {
            anyhow::anyhow!(
              "Voyage API key not found in config or VOYAGE_API_KEY environment variable"
            )
          })?;

        Ok(VoyageClientConfig {
          api_key,
          tier: v
            .tier
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid voyage tier: {}", e))?,
          model: v
            .model
            .parse::<VoyageEmbeddingModel>()
            .map_err(|e| anyhow::anyhow!("Invalid voyage model: {}", e))?,
        })
      })
      .transpose()?;

    // Convert OpenAI-like providers
    let mut openai_providers = std::collections::HashMap::new();

    // Add providers from the flattened map
    for (name, config) in &self.embeddings.providers {
      if let Some(api_base) = &config.api_base {
        // Parse tokenizer if provided
        let tokenizer = if let Some(tok_str) = &config.tokenizer {
          tok_str
            .parse()
            .map_err(|e| anyhow::anyhow!("Invalid tokenizer: {}", e))?
        } else {
          return Err(anyhow::anyhow!(
            "tokenizer is required for provider '{}'",
            name
          ));
        };

        // Get API key from config or environment variable
        let api_key = config.api_key.clone().or_else(|| {
          let env_var = format!("{}_API_KEY", name.to_uppercase());
          std::env::var(&env_var).ok()
        });

        openai_providers.insert(
          name.clone(),
          breeze_indexer::OpenAILikeConfig {
            api_base: api_base.clone(),
            api_key,
            model: config.model.clone(),
            embedding_dim: config.embedding_dim.ok_or_else(|| {
              anyhow::anyhow!("embedding_dim is required for provider '{}'", name)
            })?,
            context_length: config.context_length.ok_or_else(|| {
              anyhow::anyhow!("context_length is required for provider '{}'", name)
            })?,
            max_batch_size: config.max_batch_size.ok_or_else(|| {
              anyhow::anyhow!("max_batch_size is required for provider '{}'", name)
            })?,
            tokenizer,
            requests_per_minute: config.requests_per_minute.ok_or_else(|| {
              anyhow::anyhow!("requests_per_minute is required for provider '{}'", name)
            })?,
            tokens_per_minute: config.tokens_per_minute.ok_or_else(|| {
              anyhow::anyhow!("tokens_per_minute is required for provider '{}'", name)
            })?,
            max_concurrent_requests: config.max_concurrent_requests.ok_or_else(|| {
              anyhow::anyhow!(
                "max_concurrent_requests is required for provider '{}'",
                name
              )
            })?,
            max_tokens_per_request: config.max_tokens_per_request,
          },
        );
      }
    }

    // Get the selected model
    let model = match &embedding_provider {
      EmbeddingProvider::Local => self
        .embeddings
        .local
        .as_ref()
        .map(|l| l.model.clone())
        .unwrap_or_else(|| "BAAI/bge-small-en-v1.5".to_string()),
      _ => "".to_string(), // Not used for non-local providers
    };

    Ok(breeze_indexer::Config {
      database_path: self.db_dir.clone(),
      embedding_provider,
      model,
      voyage,
      openai_providers,
      max_chunk_size: self.indexer.limits.max_chunk_size.unwrap_or(512),
      max_file_size: self.indexer.limits.file_size.as_deref().copied(),
      max_parallel_files: self.indexer.workers.small_file,
      large_file_threads: Some(self.indexer.workers.large_file),
      embedding_workers: self.embeddings.workers,
      optimize_threshold: self.indexer.optimize_threshold,
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::tempdir;

  #[test]
  fn test_default_config() {
    let config = Config::default();
    assert_eq!(config.embeddings.provider, "local");
    assert_eq!(config.db_dir, PathBuf::from("./embeddings.db"));
  }

  #[test]
  fn test_config_serialization() {
    let config = Config::default();
    let toml_str = toml::to_string(&config).unwrap();
    let parsed: Config = toml::from_str(&toml_str).unwrap();

    assert_eq!(config.embeddings.provider, parsed.embeddings.provider);
  }

  #[test]
  fn test_config_file_operations() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("test_config.toml");

    let config = Config::default();
    config.save_to_file(&config_path).unwrap();

    let loaded = Config::load(Some(config_path)).unwrap();
    assert_eq!(config.embeddings.provider, loaded.embeddings.provider);
  }

  #[test]
  fn test_parse_example_config() {
    let example_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
      .join("examples")
      .join("config_example.toml");

    if example_path.exists() {
      let config = Config::load(Some(example_path)).expect("Failed to parse example config");

      // Debug: print the providers map
      eprintln!(
        "Providers in map: {:?}",
        config.embeddings.providers.keys().collect::<Vec<_>>()
      );

      // Verify key fields from the example
      assert_eq!(config.db_dir, PathBuf::from("./embeddings.db"));
      assert_eq!(config.embeddings.provider, "ollama");

      // Check server ports
      assert_eq!(config.server.ports.http, 8080);
      assert_eq!(config.server.ports.https, 8443);

      // Check indexer settings
      assert_eq!(config.indexer.workers.small_file, 8);
      assert_eq!(config.indexer.workers.large_file, 4);
      assert_eq!(config.indexer.workers.batch_size, 256);

      // Check embeddings settings
      assert_eq!(config.embeddings.workers, 4);

      // Verify size parsing (5M = 5 * 1024 * 1024 bytes in human-units)
      assert_eq!(
        config.indexer.limits.file_size.as_deref().copied(),
        Some(5 * 1024 * 1024)
      );
    }
  }

  #[test]
  fn test_size_parsing() {
    use human_units::Size;

    // Test that "5M" works
    let size: Size = "5M".parse().expect("Should parse 5M");
    assert_eq!(*size, 5 * 1024 * 1024); // M is 1024-based in human-units

    // Test with serde
    #[derive(Debug, serde::Deserialize)]
    struct TestConfig {
      file_size: Option<Size>,
    }

    let toml_str = r#"file_size = "5M""#;
    let config: TestConfig = toml::from_str(toml_str).expect("Should deserialize");
    assert_eq!(config.file_size.map(|s| *s), Some(5 * 1024 * 1024));
  }

  #[test]
  fn test_config_dump_preserves_values() {
    let dir = tempdir().unwrap();
    let config_path = dir.path().join("test_config.toml");

    // Create a config with various settings
    let config = Config {
      db_dir: PathBuf::from("./test.db"),
      embeddings: EmbeddingsConfig {
        provider: "voyage".to_string(),
        ..Default::default()
      },
      indexer: IndexerConfig {
        limits: IndexerLimits {
          file_size: Some("1M".parse().expect("Invalid size")),
          ..Default::default()
        },
        workers: IndexerWorkers {
          batch_size: 512,
          ..Default::default()
        },
        ..Default::default()
      },
      ..Default::default()
    };

    // Save and reload
    config.save_to_file(&config_path).unwrap();
    let loaded = Config::load(Some(config_path)).unwrap();

    // Verify all values preserved
    assert_eq!(loaded.db_dir, PathBuf::from("./test.db"));
    assert_eq!(loaded.embeddings.provider, "voyage");
    assert_eq!(
      loaded.indexer.limits.file_size.as_deref().copied(),
      Some(1024 * 1024)
    );
    assert_eq!(loaded.indexer.workers.batch_size, 512);
  }
}
