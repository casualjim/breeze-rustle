# Breeze Rustle Embedding System Implementation Plan

## Overview
Implement a Rust-based embedding system that processes repository chunks through two provider types: HTTP-based (OpenAI/Voyage via async_openai) and Local (Sentence Transformers), with provider-specific batching and rate limiting.

## Simplified Architecture

```
Repository → Chunker → Provider-Specific Batcher → Embedding Provider → Result Aggregator → LanceDB
```

### Two Provider Types:
1. **HTTP Provider** (using `async_openai`): OpenAI, Voyage, and compatible APIs
2. **Local Provider**: Sentence Transformers with device optimization

## Implementation Plan

### Phase 1: Core Traits & Configuration
1. **Embedding Provider Trait**
   ```rust
   trait EmbeddingProvider {
       async fn embed_batch(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>, EmbeddingError>;
       fn optimal_batch_size(&self) -> usize;
       fn max_tokens_per_batch(&self) -> usize;
   }
   ```

2. **Configuration System** (`crates/breeze/src/config.rs`)
   - Provider configurations (OpenAI, Voyage, Local models)
   - API key management and validation
   - Device detection and model-specific optimizations

### Phase 2: HTTP Provider (async_openai based)
3. **HTTP Embedding Provider**
   - Use `async_openai::Client` with configurable base URLs
   - OpenAI: Standard configuration with `text-embedding-3-small/large`
   - Voyage: Custom base_url, `input_type="document"`, large batches (128 texts, 120k tokens)
   - Built-in retries and error handling from async_openai

4. **Provider-Specific Rate Limiting**
   - OpenAI: Standard API limits (varies by tier)
   - Voyage: High limits (3M tokens/min, 2K requests/min) with token bucket
   - Configurable via provider config

### Phase 3: Local Provider
5. **Sentence Transformers Provider**
   - Load models via HuggingFace transformers (via candle or tch)
   - Device optimization (CPU/MPS/CUDA) with automatic detection
   - Batch size optimization based on model benchmarks from Python prototype
   - Memory-efficient inference with tokio::task::spawn_blocking

### Phase 4: Stream Processing Pipeline
6. **Batching System**
   - Provider-specific batchers that consume `Stream<ProjectChunk>`
   - Voyage: Large batches optimized for token limits
   - OpenAI: Medium batches respecting API limits
   - Local: Device-optimized small batches
   - Timeout-based flushing for incomplete batches

7. **Result Aggregation**
   - Combine chunk embeddings per file using weighted averaging (by token count)
   - Handle partial failures gracefully
   - Progress tracking with resumable processing

8. **Main Application** (`crates/breeze/src/app.rs`)
   - CLI interface with provider selection
   - Configuration file support
   - Repository walking integration
   - LanceDB storage with proper error handling

### Phase 5: Dependencies & Integration
9. **Required Dependencies**
   ```toml
   async-openai = "0.x"  # HTTP provider foundation
   candle-core = "0.x"   # Local model inference (or tch)
   futures-util = "0.3"  # Stream processing
   tokio = { features = ["rt-multi-thread", "fs", "macros"] }
   ```

10. **Integration Testing**
    - End-to-end tests with real repositories
    - Mock providers for unit testing
    - Performance benchmarking against Python prototype

## Key Design Decisions

### Why async_openai?
- ✅ **Battle-tested**: Used in production OpenAI integrations
- ✅ **Built-in networking**: Retries, timeouts, proper async
- ✅ **Voyage compatibility**: Just change base_url and add input_type
- ✅ **Ecosystem**: Integrates with existing OpenAI tooling

### Batching Strategy
- **Provider-specific optimization**: Leverage known limits and characteristics
- **Stream-based**: Process chunks as they come, don't load entire repo in memory
- **Backpressure handling**: Prevent memory bloat with bounded channels

### Error Handling
- **Partial failure tolerance**: Process what we can, report failures
- **Provider fallback**: Could add fallback provider chains later
- **Resumable processing**: Track progress in LanceDB

## Success Criteria
- ✅ Process repositories with 1000+ files reliably
- ✅ Support OpenAI, Voyage, and local Sentence Transformers seamlessly
- ✅ Handle API rate limits and failures with <5% total failure rate
- ✅ Memory usage stays bounded regardless of repository size
- ✅ Resumable processing for large codebases
- ✅ Performance comparable to Python prototype
- ✅ Simple, maintainable codebase leveraging proven libraries

## Implementation Notes

### Rate Limiting Strategy
Based on the Python prototype, we need sophisticated rate limiting:

- **Voyage**: 3M tokens/min, 2K requests/min with 90% safety margin
- **OpenAI**: Varies by tier, generally much lower limits
- **Local**: No API limits, but memory/compute constraints

### Chunk Combination Strategy
From Python prototype - combine multiple chunk embeddings per file:
- Single chunk: Use as-is
- Multiple chunks: Weighted average by token count
- Ensure all embeddings have same dimensions before combining

### Device Optimization for Local Models
Based on benchmarked data from Python prototype:
- **MPS**: Smaller batches (16-64), optimized chunk sizes
- **CPU**: Larger batches possible, slower inference
- **CUDA**: Largest batches, fastest inference

This approach maximizes code reuse while providing the flexibility needed for different embedding provider characteristics.