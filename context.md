# breeze-rustle Context - Embedding Pipeline Bug Fix

## Problem: EOF Chunks Lost on Embedding Failure

### Root Cause

When embedding fails (timeout, rate limit, etc.), the entire batch fails including EOF chunks. Since EOF chunks don't need embedding but were being sent with the batch, they were lost when the batch failed. This caused "No EOF chunk found" errors in the document builder.

### Key Architecture Insights

1. **EOF chunks don't need embedding** - they just mark file completion
2. **Batches mix chunks from multiple files** - one batch can contain:
   - Content chunks from files A, B (need embedding)
   - EOF chunks from files C, D, E (whose content was in earlier batches)
3. **EOF chunks must maintain batch order** - they signal completion after all content chunks are processed

## Implemented Solution

Transformed EmbeddedChunkWithFile from a struct to an enum with three variants:

```rust
pub enum EmbeddedChunkWithFile {
    // Regular embedded chunk
    Embedded {
        batch_id: usize,
        file_path: String,
        chunk: PipelineChunk,
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
        failed_files: BTreeSet<String>,  // Files with content chunks in this batch
        error: String,
    }
}
```

### How It Works

1. **Embedder behavior**:
   - On success: Sends `Embedded` variants for content chunks
   - On failure: Sends `BatchFailure` with list of files that had content in the batch
   - Always: Sends `EndOfFile` variants after batch processing (success or failure)
   - EOF chunks maintain batch ordering - sent after their batch is processed

2. **Document builder behavior**:
   - Maintains a `failed_files` set
   - On `BatchFailure`: Adds files to the failed set
   - On `Embedded`: Skips if file is in failed set, otherwise accumulates
   - On `EndOfFile`: Skips if file is in failed set, otherwise builds document
   - This ensures EOF chunks for non-failed files still complete successfully

### Why This Works

- **Files with content in failed batch**: Marked as failed, all chunks ignored
- **Files with only EOF in failed batch**: Not marked as failed, EOF processed normally
- **No more EOF loss**: EOF chunks are sent regardless of embedding success/failure
- **Correct ordering**: EOF chunks still wait for their batch to complete

## What We Didn't Do (And Why)

1. **Timeout-based flushing**: Not needed - EOF chunks are sent immediately after batch processing
2. **Filtering EOF chunks at sender**: Wrong approach - document builder handles this by tracking failed files
3. **Retry logic**: Separate concern, not part of the EOF loss fix

## Test Coverage

Added comprehensive tests including:

- `test_eof_chunk_loss_on_embedding_failure`: Verifies EOF chunks aren't lost
- `test_document_builder_handles_batch_failures`: Verifies correct handling of mixed batches

## Remaining Tasks

1. **Add retry logic for embedding failures** - For transient API errors
2. **Fix silent chunk dropping** - When `embedded_tx.send()` fails
3. **Add metrics/logging** - Track failed files for observability
