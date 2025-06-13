# Breeze Rustle Embedding System Implementation Plan

## Overview
A refined embedding system that leverages our streaming pipeline architecture with provider-specific optimizations. The system maintains chunk-embedding pairs throughout processing, stores tokens with chunks to avoid double tokenization, and uses HuggingFace's text-embeddings-inference for robust local model support while maintaining remote provider support.

## Current Status & Next Steps

### ✅ Completed
- LanceDB sink with Arc<RwLock<Table>> design
- Basic pipeline traits and PassthroughBatcher
- Core chunking and grammar support
- DocumentBuilder trait for aggregating chunk embeddings
- Refactored to embed chunks individually, then aggregate
- EOF chunks with file content and hash for single-read optimization
- **Token Storage in Chunks** - SemanticChunk includes `tokens: Option<Vec<u32>>`
- **TEI Integration** - Full text-embeddings-inference backend with pre-tokenized support
- Comprehensive indexer testing with real models

### 🚧 In Progress
- **Remote Providers** - Implementing Voyage and OpenAI providers with rate limiting
- **CLI Application** - Building the main breeze executable
- **Configuration System** - Wiring up TOML config to provider selection

### 🎯 Next Priority Tasks

1. **Remote Provider Implementations** (High Priority)
   - Voyage API integration using async-openai
   - OpenAI API integration with tier-based rate limiting
   - Provider-specific batching strategies
   - Retry logic with exponential backoff
   
2. **CLI Application** (High Priority)
   - `breeze index` command to index a codebase
   - `breeze search` command for similarity search
   - Progress bars and status tracking
   - Configuration file support
   
3. **Configuration System Integration** (Medium Priority)
   - Load provider selection from TOML config
   - Model configuration per provider
   - Rate limiting and retry policies
   - Environment variable overrides

4. **Integration Testing** (Medium Priority)
   - End-to-end tests with real embedders
   - Performance benchmarks for different providers
   - Error recovery scenarios

## Architecture

```
                  ┌─> TEI Embedder (local)    ─┐
                  │                             │
Enhanced Walker ─>├─> Voyage Embedder (remote) ├─> DocumentBuilder → RecordBatchConverter → LanceDB
(yields ProjectFile) │                             │  (aggregation)     (Arrow format)    (Vector DB)
                  └─> OpenAI Embedder (remote) ─┘
```

**Key Points**:
- Walker yields `ProjectFile` items containing chunk streams
- Chunks include pre-tokenized data (`tokens: Option<Vec<u32>>`)
- EOF chunks contain full file content and hash - single file read optimization
- Multiple embedder implementations:
  - TEI for local models (20+ architectures)
  - Voyage for high-performance code embedding
  - OpenAI for general-purpose embedding
- DocumentBuilder aggregates chunk embeddings to file embeddings
- Token optimization: Reuse tokens from chunking when possible

## Core Types

### Enhanced SemanticChunk
```rust
pub struct SemanticChunk {
    pub text: String,
    pub tokens: Option<Vec<u32>>,  // NEW: Pre-tokenized IDs
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub metadata: ChunkMetadata,
}
```

### EOF Chunk (End of File)
```rust
pub enum Chunk {
    Semantic(SemanticChunk),
    Text(SemanticChunk),
    EndOfFile { 
        file_path: String,
        content: String,      // Full file content
        content_hash: [u8; 32],  // Blake3 hash
    },
}
```

### TEI Integration Types
```rust
pub struct EmbeddedChunk {
    pub chunk: Chunk,
    pub embedding: Vec<f32>,
}

pub struct ProjectFileWithEmbeddings {
    pub file_path: String,
    pub metadata: FileMetadata,
    pub embedded_chunks: BoxStream<Result<EmbeddedChunk, ChunkError>>,
}
```

## Implementation Plan

### Phase 1: Token-Aware Chunking

1. **Update Chunker to Store Tokens**
```rust
impl InnerChunker {
    pub fn chunk_code_with_tokens(
        &self,
        content: String,
        language: String,
        file_path: Option<String>,
    ) -> impl Stream<Item = Result<Chunk, ChunkError>> {
        // When using HuggingFace tokenizer:
        // 1. Tokenize chunk text
        // 2. Store token IDs in chunk.tokens
        // 3. Use token count for size limits
    }
}
```

2. **Tokenizer Compatibility**
```rust
pub struct TokenizerConfig {
    pub chunking_tokenizer: String,  // Must match embedding model
    pub embedding_model: String,
}
```

### Phase 2: Multiple Embedding Providers

1. **Provider Trait (shared interface)**
```rust
pub trait Embedder {
    fn embed(&self, files: BoxStream<ProjectFile>) -> BoxStream<ProjectFileWithEmbeddings>;
}
```

2. **TEI Local Provider**
```rust
pub struct TEIEmbedder {
    backend: text_embeddings_backend::Backend,
    model_type: ModelType,
    tokenizer: Option<Tokenizer>, // For pre-tokenized input
}

impl TEIEmbedder {
    pub fn embed_pretokenized(&self, chunks: &[(Vec<u32>, &str)]) -> Result<Vec<Vec<f32>>>;
    pub fn embed_text(&self, chunks: &[&str]) -> Result<Vec<Vec<f32>>>;
}
```

3. **Remote HTTP Providers**
```rust
pub struct VoyageEmbedder {
    client: async_openai::Client,
    model: String,
    rate_limiter: TokenBucketRateLimiter,
}

pub struct OpenAIEmbedder {
    client: async_openai::Client, 
    model: String,
    rate_limiter: TokenBucketRateLimiter,
}
```

### Phase 3: File-Aware Aggregation

**Note**: Aggregation is now internal to the Embedder trait implementation.

```rust
// Aggregation strategy trait for future extensibility
pub trait AggregationStrategy {
    fn aggregate(&self, chunk_embeddings: Vec<(Vec<f32>, usize)>) -> Vec<f32>;
}

pub struct WeightedAverageStrategy;

impl AggregationStrategy for WeightedAverageStrategy {
    fn aggregate(&self, chunk_embeddings: Vec<(Vec<f32>, usize)>) -> Vec<f32> {
        // chunk_embeddings: Vec of (embedding, token_count) pairs
        let total_tokens: usize = chunk_embeddings.iter()
            .map(|(_, tokens)| tokens)
            .sum();
        
        // Weighted average by token count
        let mut result = vec![0.0; chunk_embeddings[0].0.len()];
        for (embedding, token_count) in chunk_embeddings {
            let weight = *token_count as f32 / total_tokens as f32;
            for (i, val) in embedding.iter().enumerate() {
                result[i] += val * weight;
            }
        }
        result
    }
}

// Future strategies could include:
// - AttentionBasedAggregation
// - MaxPoolingAggregation  
// - HierarchicalAggregation
```

### Phase 4: LanceDB Integration ✅ COMPLETED

```rust
pub struct LanceDbSink {
    table: Arc<RwLock<lancedb::Table>>,
}

impl LanceDbSink {
    /// Create sink with pre-initialized table
    /// Database initialization is handled externally
    pub fn new(table: Arc<RwLock<lancedb::Table>>) -> Self {
        Self { table }
    }
}

impl Sink for LanceDbSink {
    fn sink(&self, batches: Pin<Box<dyn RecordBatchStream + Send>>) -> BoxStream<()> {
        // Uses merge_insert for upsert behavior on "id" column
        // Processes batches in streaming fashion
        // Includes logging and error handling
    }
}
```

**Key Design Decisions:**
- Takes `Arc<RwLock<Table>>` directly - no database initialization
- Uses merge_insert API for upsert behavior (primary key: "id")
- Fully streaming implementation
- Separation of concerns: database setup is external

### Phase 5: Rate Limiting Implementation

```rust
pub struct TokenBucketRateLimiter {
    token_bucket: Arc<Mutex<TokenBucket>>,
    request_bucket: Arc<Mutex<TokenBucket>>,
}

impl TokenBucketRateLimiter {
    pub fn new_for_voyage() -> Self {
        // 3M tokens/min, 2K requests/min
        // With 90% safety margin
        Self {
            token_bucket: TokenBucket::new(2_700_000, Duration::from_secs(60)),
            request_bucket: TokenBucket::new(1_800, Duration::from_secs(60)),
        }
    }
    
    pub fn new_for_openai(tier: OpenAITier) -> Self {
        // Tier-specific limits
        match tier {
            OpenAITier::Free => Self {
                token_bucket: TokenBucket::new(150_000, Duration::from_secs(60)),
                request_bucket: TokenBucket::new(3, Duration::from_secs(60)),
            },
            // ... other tiers
        }
    }
}
```

## Configuration

```toml
[embedding.voyage]
model = "voyage-code-2"
batch_size = 128
max_tokens_per_batch = 120000
rate_limit_tokens_per_minute = 3000000
rate_limit_requests_per_minute = 2000

[embedding.openai]
model = "text-embedding-3-small"
batch_size = 50
max_tokens_per_batch = 8000
tier = "tier-1"  # Determines rate limits

[embedding.local]
model = "BAAI/bge-small-en-v1.5"
device = "auto"  # auto-detect: mps/cuda/cpu
batch_size = 32  # Will be adjusted based on device

[chunking]
max_chunk_size = 2048
overlap_tokens = 200
tokenizer = "cl100k_base"
```

## Key Improvements

1. **Token Counting**: Every chunk knows its token count for optimal batching
2. **Maintained Pairs**: ChunkWithEmbedding preserves chunk-embedding relationship
3. **Smart Batching**: Provider-specific strategies maximize throughput
4. **Rate Limiting**: Sophisticated token + request buckets with safety margins
5. **File Aggregation**: Proper grouping and weighted averaging
6. **Overlap Support**: Configurable overlap for better RAG performance

## Error Recovery

```rust
pub struct RetryConfig {
    pub max_attempts: u32,
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub exponential_base: f64,
}

pub struct CircuitBreaker {
    pub failure_threshold: u32,
    pub success_threshold: u32,
    pub timeout: Duration,
}
```

## Success Metrics

- ✅ Process 1000+ file repositories reliably
- ✅ Maintain <100ms latency per chunk (excluding API calls)
- ✅ <5% failure rate with automatic recovery
- ✅ Memory usage bounded by streaming architecture
- ✅ Saturate API rate limits efficiently (>90% utilization)
- ✅ Support incremental updates via content hashing

## Dependencies

```toml
async-openai = "0.24"      # HTTP providers
candle-core = "0.7"        # Local inference
futures-util = "0.3"       # Stream processing
tokio = { version = "1", features = ["full"] }
lancedb = "0.10"          # Vector storage
tiktoken-rs = "0.5"       # Token counting
governor = "0.6"          # Rate limiting
```