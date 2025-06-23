# breeze-rustle Context - Embedding Pipeline Bug Investigation

## Current Issue: Missing EOF Chunks

### Problem Summary
When embedding is slow or fails, EOF chunks can be lost, causing "No EOF chunk found" errors in the document builder. This is intermittent and provider-specific.

### Root Cause Analysis

1. **Batch-Level Failures**: When embedding fails (timeout, rate limit), the entire batch fails
2. **Lost EOF Chunks**: EOF chunks in failed batches are dropped along with content chunks
3. **Silent Data Loss**: No error propagation - chunks just disappear from the pipeline

### The Pipeline Flow

```
StreamProcessor → Batcher → Embedder → DocumentBuilder → Sink
                                ↓
                          (failure = dropped chunks)
```

### Key Insights

1. **EOF chunks don't need embedding** - they're sent with empty vectors
2. **Batches mix chunks from multiple files** - one batch can contain:
   - Content chunks from files A, B
   - EOF chunks from files C, D, E (whose content was in earlier batches)
3. **Partial file failure** - if any chunk of a file fails, the entire file should be marked as failed

### Edge Cases

1. **EOF-only batches**: A batch might contain only EOF chunks for files whose content chunks were successfully processed earlier
2. **Mixed failure impact**: When a batch fails:
   - Files with content chunks in the batch → failed
   - Files with only EOF chunks in the batch → should still complete

## Proposed Solution: Proper Terminal States

Transform EmbeddedChunkWithFile into an enum that captures all states:

```rust
pub enum EmbeddedChunkWithFile {
    // Regular embedded chunk
    Embedded {
        batch_id: usize,
        file_path: String,
        chunk: PipelineChunk,  // Semantic or Text
        embedding: Vec<f32>,
    },
    // EOF marker - no embedding needed
    EndOfFile {
        batch_id: usize,
        file_path: String,
        content: String,
        content_hash: [u8; 32],
    },
    // Batch failure notification
    BatchFailure {
        batch_id: usize,
        failed_files: HashSet<String>,  // Files with content chunks in this batch
        error: String,
    }
}
```

### Benefits

1. **Type safety** - Each state is explicitly modeled, no empty embeddings hack
2. **Clear semantics** - EOF chunks are recognized as terminal markers
3. **Batch-level failure handling** - Matches how embedding actually works
4. **Selective processing** - Can handle EOF chunks for non-failed files
5. **Better observability** - Can track embedded vs EOF vs failed chunks

### Implementation Strategy

1. **In embedder**:
   - Send embedded chunks as `Embedded` variant
   - Send EOF chunks as `EndOfFile` variant immediately (no batching needed)
   - On batch failure:
     - Identify files with content chunks in the failed batch
     - Send `BatchFailure` with affected file set
     - Still send `EndOfFile` variants for unaffected files

2. **In document builder**:
   - Pattern match on the enum:
     - `Embedded`: Accumulate in file accumulator
     - `EndOfFile`: Trigger document building for that file
     - `BatchFailure`: Mark files as failed, skip their chunks
   - Only build documents for non-failed files

## Test Plan

Create a test with a mock embedding provider that:
1. Fails deterministically (e.g., every 3rd batch)
2. Process multiple files with varying chunk counts
3. Verify:
   - Files with chunks in failed batches don't produce documents
   - Files with only EOF chunks in failed batches complete successfully
   - No "No EOF chunk found" errors for partial files

## Current Code Issues

1. **No retry on embedding failure** (bulk_indexer.rs:1260-1264)
2. **EOF chunks held in pending_batches** until batch threshold met
3. **No timeout-based flushing** for small files
4. **Silent chunk dropping** on `embedded_tx.send()` failure

## Next Steps

1. Write the test to reproduce the issue
2. Implement the Failed variant
3. Update embedder error handling to create Failed chunks
4. Update document builder to handle Failed chunks
5. Add metrics/logging for failed files