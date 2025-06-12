# Breeze Rustle Embedding System Implementation Plan

## Overview
A refined embedding system that leverages our streaming pipeline architecture with provider-specific optimizations. The system maintains chunk-embedding pairs throughout processing and implements smart batching based on token counts.

## Current Status & Next Steps

### ✅ Completed
- LanceDB sink with Arc<RwLock<Table>> design
- Basic pipeline traits and PassthroughBatcher
- Core chunking and grammar support

### 🎯 Priority Tasks
1. **Configuration System** (Medium Priority)
   - Embedding provider configurations
   - Chunking parameters
   - Processing options
   
2. **Smart Batching** (High Priority)
   - Token-aware batching
   - Provider-specific strategies
   - Batch timeout handling

3. **Embedding Providers** (High Priority)
   - HTTP providers via async_openai
   - Local provider with sentence transformers
   - Rate limiting integration

## Architecture

```
Enhanced Walker → Smart Batcher → Embedder → File Aggregator → LanceDB
(token counting)   (provider-aware)  (rate limited)  (weighted avg)    (FTS+Vector)
```

## Core Types

### Enhanced ProjectChunk
```rust
pub struct ProjectChunk {
    pub file_path: String,
    pub chunk: Chunk,
    pub token_count: usize,  // NEW: Essential for batching
    pub overlap_with_previous: Option<String>, // NEW: For RAG
}
```

### Chunk with Embedding
```rust
pub struct ChunkWithEmbedding {
    pub chunk: ProjectChunk,
    pub embedding: Vec<f32>,
}

pub type EmbeddingBatch = Vec<ChunkWithEmbedding>;
```

## Implementation Plan

### Phase 1: Smart Batching System

1. **Smart Batcher Trait**
```rust
pub trait SmartBatcher {
    fn optimal_batch_size(&self) -> BatchConfig;
    fn batch_with_limits(&self, chunks: BoxStream<ProjectChunk>) -> BoxStream<TokenAwareBatch>;
}

pub struct BatchConfig {
    pub max_tokens_per_batch: usize,
    pub max_items_per_batch: usize,
    pub timeout_ms: u64,
}

pub struct TokenAwareBatch {
    pub chunks: Vec<ProjectChunk>,
    pub total_tokens: usize,
}
```

2. **Provider-Specific Implementations**
```rust
pub struct VoyageBatcher {
    // 128 items, 120k tokens per batch
    // Optimized for Voyage's high limits
}

pub struct OpenAIBatcher {
    // 50 items, 8k tokens per batch
    // Conservative for standard tiers
}

pub struct LocalBatcher {
    // Device-specific optimization
    // MPS: 16-32 items
    // CUDA: 64-128 items
}
```

### Phase 2: Enhanced Embedding Providers

1. **Embedding Provider Trait**
```rust
pub trait EmbeddingProvider {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
    fn rate_limiter(&self) -> &dyn RateLimiter;
}

pub trait RateLimiter {
    async fn acquire_permits(&self, tokens: usize, requests: usize);
    fn available_capacity(&self) -> (usize, usize); // (tokens, requests)
}
```

2. **HTTP Provider (async_openai)**
```rust
pub struct HttpEmbeddingProvider {
    client: async_openai::Client,
    config: HttpProviderConfig,
    rate_limiter: Box<dyn RateLimiter>,
}

pub struct HttpProviderConfig {
    pub base_url: Option<String>,
    pub model: String,
    pub api_key: String,
    pub input_type: Option<String>, // For Voyage
    pub dimensions: Option<usize>,
}
```

3. **Local Provider (candle)**
```rust
pub struct LocalEmbeddingProvider {
    model: Arc<dyn SentenceTransformer>,
    device: Device,
    batch_size: usize,
}
```

### Phase 3: File-Aware Aggregation

```rust
pub struct FileAggregator;

impl Aggregator for FileAggregator {
    fn aggregate(&self, embeddings: BoxStream<ChunkWithEmbedding>) -> BoxStream<CodeDocument> {
        embeddings
            .group_by(|chunk| chunk.chunk.file_path.clone())
            .map(|(file_path, chunks)| {
                // Weighted average by token count
                let total_tokens: usize = chunks.iter()
                    .map(|c| c.chunk.token_count)
                    .sum();
                
                let weighted_embedding = compute_weighted_average(chunks, total_tokens);
                
                CodeDocument {
                    file_path,
                    content_embedding: weighted_embedding,
                    // ... other fields
                }
            })
    }
}
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