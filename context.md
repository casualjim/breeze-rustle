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
                  ┌─> TEI Embedder (local)    ─┐
                  │                             │
PathWalker ────>─├─> Voyage Embedder (remote) ├─> DocumentBuilder → RecordBatchConverter → Sink
                  │                             │
                  └─> OpenAI Embedder (remote) ─┘
```

### Key Design Principles

1. **Chunking for Embedder Constraints**: Files are chunked to respect embedder token limits
2. **Semantic Boundaries**: Chunks split at function/class boundaries, preserving code structure
3. **File-Level Storage**: Despite chunking, we store one embedding per file via aggregation
4. **Token Optimization**: Store tokens in chunks to avoid double tokenization
5. **Multiple Providers**: Support local (TEI), Voyage, and OpenAI embedders
6. **Aggregation Strategy**: Weighted average by token count (pluggable for future strategies)
7. **Single File Read**: EOF chunks include file content and hash to avoid multiple file reads

### Key Pipeline Traits

```rust
pub trait PathWalker {
    fn walk(&self, path: &Path) -> BoxStream<ProjectFile>;
}

pub trait Embedder {
    fn embed(&self, files: BoxStream<ProjectFile>) -> BoxStream<ProjectFileWithEmbeddings>;
}

pub trait DocumentBuilder {
    fn build_documents(&self, files: BoxStream<ProjectFileWithEmbeddings>) -> BoxStream<CodeDocument>;
}

pub trait RecordBatchConverter {
    fn convert(&self, documents: BoxStream<CodeDocument>) -> BoxStream<RecordBatch>;
}

pub trait Sink {
    fn sink(&self, batches: BoxStream<RecordBatch>) -> BoxStream<()>;
}
```

## Embedding System Design

### Provider Types

1. **Local Provider (TEI)**
   - Text-embeddings-inference backend
   - 20+ model architectures (BERT, DistilBERT, JinaBERT, MPNet, etc.)
   - Hardware acceleration (CUDA, Metal, MKL)
   - Token optimization via pre-tokenized input

2. **Remote Providers** (via async_openai)
   - OpenAI: Standard limits, text-embedding-3-small/large
   - Voyage: High limits (3M tokens/min), voyage-code-2
   - Built-in retry logic and error handling

### Processing Strategy

| Provider | Token Handling | Optimization | Hardware |
|----------|---------------|--------------|----------|
| TEI      | Pre-tokenized | Flash attention | CUDA/Metal/CPU |
| Voyage   | Text chunks   | Large batches | Cloud API |
| OpenAI   | Text chunks   | Standard batches | Cloud API |

### Key Improvements in Progress

1. **Token storage in chunks**: Store token IDs in chunks to avoid double tokenization
2. **TEI integration**: Replace custom local embedder with text-embeddings-inference
3. **Multiple provider support**: Local (TEI), Voyage, and OpenAI embedders
4. **DocumentBuilder pattern**: Separate chunk embedding from file aggregation
5. **Tokenizer compatibility**: Ensure chunking and embedding use same tokenizer

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
- **EOF chunks with content and hash** ✨
  - Single file read optimization
  - Content and Blake3 hash in EOF chunks
  - Updated Python and Node.js bindings
  - Full test coverage
- **Comprehensive Indexer Testing** ✨
  - Mock implementations for TEIEmbedder, Table, and walk_project
  - Pipeline error propagation and cancellation
  - Channel backpressure and bounded capacity
  - Batch processing logic (accumulation, flushing, EOF handling)
  - Pipeline resilience (stage panics, receiver drops)
  - Stats tracking accuracy
  - Edge cases (empty input, EOF-only files)
  - Real embedding model integration (sentence-transformers/all-MiniLM-L6-v2)
  - Performance profiling (identified chunking as bottleneck, not parsing)
- **Token Storage in Chunks** ✨
  - Added `tokens: Option<Vec<u32>>` to SemanticChunk
  - Chunker populates tokens during chunking when using tokenizers
  - Supports character, tiktoken, and HuggingFace tokenizers
- **TEI Embedder Implementation** ✨
  - Full text-embeddings-inference backend integration
  - Supports pre-tokenized input for performance
  - Hardware acceleration (CUDA/Metal/CPU auto-detection)
  - Batch processing with configurable limits
  - Proper error handling and type safety

### 🚧 In Progress

- **Remote embedders** - Implement Voyage and OpenAI providers with rate limiting
- **CLI application** - Create the main breeze executable with commands for indexing
- **Configuration loading** - Wire up TOML config to select providers and models

### 📋 TODO

- **Documentation** - User guide, API docs, and examples
- **Python package publishing** - PyPI release workflow
- **Node.js package publishing** - npm release workflow
- **Integration tests** - End-to-end tests with real models and databases
- **Error recovery** - Retry logic and circuit breakers for remote providers

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
