# breeze-rustle Context & Current State

## Overview

`breeze-rustle` is a high-performance Rust library for semantic code chunking and embedding. It provides:

- Semantic code chunking using tree-sitter parsers (163 languages)
- Smart batching for different embedding providers
- Streaming pipeline architecture for processing large codebases
- Python bindings via PyO3

## Current Architecture

### Core Components

1. **breeze-chunkers**: High-performance chunking library
   - Leverages text-splitter for core chunking logic
   - Adds semantic metadata extraction
   - Supports both code and text chunking
   - Streaming API with async iterators

2. **breeze-grammars**: Tree-sitter grammar compilation
   - 163 language support via precompiled binaries
   - ~5 second build times with precompilation
   - Cross-platform support

3. **breeze**: Main application with embedding pipeline
   - Trait-based pipeline architecture
   - Provider-specific batching and rate limiting
   - LanceDB integration for storage

4. **breeze-napi**: Node.js bindings (future)

5. **breeze-py**: Python bindings via PyO3

## Pipeline Architecture

### Current Design

```text
PathWalker → Batcher → Embedder → Aggregator → Sink
```

### Refined Design (In Progress)

```text
Repository → Enhanced Walker → Smart Batcher → Embedder → File Aggregator → LanceDB
             (with tokenization)  (provider-aware)  (with rate limiting)
```

### Key Pipeline Traits

```rust
pub trait PathWalker {
    fn walk(&self, path: &Path) -> BoxStream<ProjectChunk>;
}

pub trait SmartBatcher {
    fn batch_with_limits(&self, chunks: BoxStream<ProjectChunk>) -> BoxStream<TokenAwareBatch>;
}

pub trait EmbeddingProvider {
    async fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;
    fn rate_limiter(&self) -> &dyn RateLimiter;
}

pub trait Aggregator {
    fn aggregate(&self, embeddings: BoxStream<ChunkWithEmbedding>) -> BoxStream<CodeDocument>;
}
```

## Embedding System Design

### Provider Types

1. **HTTP Providers** (via async_openai)
   - OpenAI: Standard limits, text-embedding-3-small/large
   - Voyage: High limits (3M tokens/min), voyage-code-2
   - Built-in retry logic and error handling

2. **Local Provider**
   - Sentence transformers via candle
   - Device optimization (CPU/MPS/CUDA)
   - Memory-efficient batch processing

### Smart Batching Strategy

| Provider | Batch Size | Max Tokens | Rate Limit |
|----------|------------|------------|------------|
| Voyage   | 128 items  | 120k       | 3M/min     |
| OpenAI   | 50 items   | 8k         | Varies     |
| Local    | Device-optimized | N/A   | N/A        |

### Key Improvements in Progress

1. **Token-aware batching**: Each ProjectChunk includes token count
2. **Chunk-embedding pairs**: Maintain 1:1 mapping through pipeline
3. **Provider-specific optimization**: Different batching strategies
4. **File-aware aggregation**: Group chunks by file, weighted averaging
5. **Overlap support**: Configurable chunk overlap for better RAG

## Code Document Model

```rust
pub struct CodeDocument {
    pub id: String,
    pub file_path: String,
    pub content: String,
    pub content_hash: [u8; 32],
    pub content_embedding: Vec<f32>,
    pub file_size: u64,
    pub last_modified: chrono::NaiveDateTime,
    pub indexed_at: chrono::NaiveDateTime,
}
```

## Configuration System

```toml
[repository]
path = "."
max_file_size = 5242880  # 5MB
exclude_patterns = ["*.git*", "node_modules", "target"]

[embedding]
provider = { type = "voyage", model = "voyage-code-2" }

[embedding.chunking]
max_chunk_size = 2048
chunk_overlap = 0.1
use_semantic_chunking = true

[embedding.processing]
max_concurrent_requests = 10
retry_attempts = 3
batch_timeout_seconds = 5

[storage]
database_path = "./embeddings.db"
table_name = "code_embeddings"

[processing]
max_parallel_files = 16
progress_report_interval = 100
```

## Implementation Status

### ✅ Completed

- Core chunking with breeze-chunkers
- 163 language support via breeze-grammars
- Python bindings with async support
- Project directory walker
- Basic pipeline traits
- Configuration system
- PassthroughBatcher implementation
- **LanceDB sink implementation** ✨
  - Arc<RwLock<Table>> based design
  - Merge insert for upsert behavior
  - Streaming batch processing
  - Full test coverage

### 🚧 In Progress

- Smart batching with token awareness
- HTTP embedding providers (OpenAI, Voyage)
- Local embedding provider
- File-aware aggregation

### 📋 TODO

- Configuration system for embedding providers
- Rate limiting implementation
- Chunk overlap support
- Progress tracking and resumability
- Error recovery and retry logic
- CLI application
- Integration tests

## Performance Benchmarks

Current performance (kuzu project):

- **337,327 chunks** from 4,457 files
- **26.2 seconds** total time
- **12,883 chunks/second** throughput

## Key Technical Decisions

1. **Streaming-first**: All operations use async streams
2. **Provider-specific optimization**: Different strategies for different providers
3. **Token counting**: Essential for API optimization
4. **Leverage text-splitter**: Don't reinvent chunking logic
5. **Arrow/LanceDB**: Efficient storage with vector search

## Development Commands

```bash
# Build and install Python package
maturin develop --release

# Run tests
cargo test
pytest tests/

# Build all grammars
./tools/build-grammars --all-platforms

# Run benchmarks
cargo bench
```
