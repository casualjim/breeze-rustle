# Embedding Providers Architecture Plan

## Overview

This plan details the implementation of a pluggable embedding provider system for breeze-rustle, starting with support for local embeddings (embed_anything) and Voyage AI as the first remote API provider.

## Goals

1. **Extensibility**: Easy to add new embedding providers (OpenAI, Cohere, etc.)
2. **Flexibility**: Each provider can have its own batching strategy
3. **Performance**: Support both sequential (local) and concurrent (API) processing
4. **Accuracy**: Use proper tokenizers for accurate token counting
5. **Configuration**: Auto-adjust chunk sizes based on model context lengths

## Architecture

### Core Abstractions

#### 1. EmbeddingProvider Trait
```rust
#[async_trait]
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a batch of texts
    async fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, Box<dyn Error + Send + Sync>>;
    
    /// Get the embedding dimension for this provider's model
    fn embedding_dim(&self) -> usize;
    
    /// Get the maximum context length in tokens
    fn context_length(&self) -> usize;
    
    /// Create the appropriate batching strategy for this provider
    fn create_batching_strategy(&self) -> Box<dyn BatchingStrategy>;
    
    /// Get the tokenizer for this provider (if applicable)
    fn tokenizer(&self) -> Option<Arc<dyn Tokenizer>>;
}
```

#### 2. BatchingStrategy Trait
```rust
#[async_trait]
pub trait BatchingStrategy: Send + Sync {
    /// Prepare chunks for batching based on provider constraints
    async fn prepare_batches(
        &self,
        chunks: Vec<ProjectChunk>,
        provider: &dyn EmbeddingProvider,
    ) -> Vec<EmbeddingBatch>;
    
    /// Maximum items per batch
    fn max_batch_size(&self) -> usize;
    
    /// Whether this strategy considers token counts
    fn is_token_aware(&self) -> bool;
    
    /// Maximum concurrent requests (for API providers)
    fn max_concurrent_requests(&self) -> Option<usize>;
}
```

### Provider Implementations

#### LocalEmbeddingProvider
- Uses `embed_anything` library
- Simple size-based batching
- Sequential processing
- 384-dimensional embeddings
- Character-based tokenization

#### VoyageEmbeddingProvider
- Uses Voyage AI API client
- Token-aware batching with HuggingFace tokenizers
- Concurrent request support
- Rate limiting based on tier
- 1024-dimensional embeddings
- Model-specific context lengths (16k-32k)

### Batching Strategies

#### LocalBatchingStrategy
```rust
pub struct LocalBatchingStrategy {
    batch_size: usize, // From config
}
```
- Simple batching by count
- No token awareness needed
- Sequential processing

#### TokenAwareBatchingStrategy
```rust
pub struct TokenAwareBatchingStrategy {
    max_tokens_per_batch: usize,
    max_items_per_batch: usize,
    max_concurrent_requests: usize,
}
```
- Batches by token count
- Respects API batch size limits
- Supports concurrent requests
- Used by Voyage and future API providers

## Configuration Updates

### Config Structure
```toml
# breeze.toml example

database_path = "./embeddings.db"
table_name = "code_files"
embedding_provider = "voyage"  # or "local"

# Voyage-specific configuration
[voyage]
api_key = "your-api-key"
tier = "tier1"  # free, tier1, tier2, tier3
model = "voyage-code-3"

# Local provider uses the existing 'model' field
model = "ibm-granite/granite-embedding-125m-english"

# Auto-adjusted based on provider/model
max_chunk_size = 30000  # tokens

# Other settings
max_file_size = 5242880
max_parallel_files = 8
batch_size = 50
```

### Environment Variables
- `BREEZE_EMBEDDING_PROVIDER`: Override embedding provider
- `BREEZE_VOYAGE_API_KEY`: Voyage API key
- `BREEZE_VOYAGE_TIER`: Voyage tier
- `BREEZE_VOYAGE_MODEL`: Voyage model

## Implementation Steps

### Phase 1: Core Infrastructure
1. ✅ Update Config struct with provider enum and voyage config
2. ✅ Create provider traits in `src/embeddings/mod.rs`
3. ✅ Create batching strategy traits in `src/embeddings/batching.rs`

### Phase 2: Provider Implementations
1. ✅ Implement LocalEmbeddingProvider in `src/embeddings/local.rs`
2. ✅ Implement VoyageEmbeddingProvider in `src/embeddings/voyage.rs`
3. ✅ Create provider factory in `src/embeddings/factory.rs`

### Phase 3: Batching Strategies
1. ✅ Implement LocalBatchingStrategy
2. ✅ Implement TokenAwareBatchingStrategy
3. ✅ Add HuggingFace tokenizer support for Voyage

### Phase 4: Integration
1. ✅ Update App to use provider factory
2. Update Indexer to use EmbeddingProvider trait
3. Modify embedder_task to use batching strategies
4. **Pass tokenizer to project walker** - The walker should handle tokenization during chunking
5. Update embedder_task to receive token counts from chunks
6. Update chunk structure to include token counts

### Phase 5: Testing & Documentation
1. Add tests for each provider
2. Add tests for batching strategies
3. ✅ Update examples
4. Update README with provider configuration

## TODO: Tokenizer Integration with Walker

The next major task is to pass the tokenizer to the project walker so it can:
1. Use the appropriate tokenizer when chunking files
2. Include accurate token counts in the chunks
3. Avoid re-tokenizing in the embedding provider

This requires modifying:
- `breeze-chunkers` to accept a tokenizer in `WalkOptions`
- The chunk structure to include token counts
- The embedder_task to use token counts from chunks instead of re-computing

## Token Counting

For Voyage models, we'll use HuggingFace tokenizers:
- Model ID: `voyageai/${model.api_name()}`
- Examples:
  - `voyageai/voyage-3`
  - `voyageai/voyage-code-3`
  - `voyageai/voyage-law-2`

## Rate Limiting & Concurrency

### Voyage Rate Limits
- **Free**: 3 req/min, 120k tokens/min
- **Tier 1**: 2000 req/min, varies by model
- **Tier 2**: 2x multiplier
- **Tier 3**: 3x multiplier

### Concurrency Strategy
- Local: Sequential processing
- Voyage: Up to 10 concurrent requests (configurable)
- Future providers: Based on their limits

## Error Handling

1. **Provider Initialization**: Validate API keys, check connectivity
2. **Embedding Failures**: Retry with exponential backoff
3. **Rate Limiting**: Use existing Voyage middleware
4. **Token Limits**: Split large texts or skip with warning

## Future Extensibility

To add a new provider:
1. Implement `EmbeddingProvider` trait
2. Create or reuse a `BatchingStrategy`
3. Add configuration section
4. Update provider factory
5. Add tests and documentation

Example future providers:
- OpenAI (text-embedding-3-*)
- Cohere (embed-v3.0)
- Anthropic (when available)
- Custom HTTP endpoints

## Migration Path

1. Existing local embeddings continue to work (default)
2. Users can opt into Voyage via config
3. No changes needed to existing code structure
4. Backward compatible with existing databases