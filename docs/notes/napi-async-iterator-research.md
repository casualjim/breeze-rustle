# NAPI Async Iterator Research

## Overview

This document summarizes research into implementing async iterators with NAPI-RS for the breeze-rustle project. Two main approaches were analyzed: the SMB-JS library's iterator implementation and LanceDB's stream-based approach.

## Research Sources

1. **SMB-JS**: <https://github.com/NetAppLabs/smb-js/blob/main/src/lib.rs>
2. **LanceDB**: /Users/ivan/github/lancedb/lancedb/nodejs/src/iterator.rs
3. **Current Implementation**: /Users/ivan/github/casualjim/breeze-rustle/crates/breeze-napi/src/lib.rs

## Approach 1: SMB-JS Iterator Pattern

### Key Features

- Uses `#[napi(iterator)]` macro
- Implements the `Generator` trait
- Provides full JavaScript async iterator protocol support
- Uses manual state tracking with counter

### Implementation Details

```rust
#[napi(iterator)]
pub struct JsSmbDirectoryHandleEntries {
  #[napi(js_name="[Symbol.asyncIterator]", ts_type="AsyncIterableIterator<[string, JsSmbDirectoryHandle | JsSmbFileHandle]>")]
  pub _sym: bool, // Fake member for TypeScript types
  env: SendWrapper<Env>,
  entries: Vec<JsSmbHandle>,
  count: usize
}

impl Generator for JsSmbDirectoryHandleEntries {
  type Yield = Vec<Unknown>;
  type Next = ();
  type Return = ();

  fn next(&mut self, _: Option<Self::Next>) -> Option<Self::Yield> {
    if self.entries.len() <= self.count {
      return None;
    }
    // Process current entry
    let entry = &self.entries[self.count];
    self.count += 1;
    Some(processed_entry)
  }
}
```

### Pros

- Full JavaScript async iterator protocol compliance
- Proper TypeScript type generation
- Works with `for await` loops
- Standard JavaScript iterator behavior

### Cons

- More complex implementation
- Requires pre-loading all data into memory
- Manual state management required
- More boilerplate code

## Approach 2: LanceDB Stream Pattern

### Key Features

- Simple struct wrapping Rust streams
- Manual `async fn next()` implementation
- Uses `futures::StreamExt` for stream processing
- Returns `Option<T>` directly

### Implementation Details

```rust
#[napi]
pub struct RecordBatchIterator {
    inner: SendableRecordBatchStream,
}

#[napi]
impl RecordBatchIterator {
    #[napi(catch_unwind)]
    pub async unsafe fn next(&mut self) -> napi::Result<Option<Buffer>> {
        if let Some(rst) = self.inner.next().await {
            let batch = rst.map_err(|e| {
                napi::Error::from_reason(format!("Failed to get next batch: {}", e))
            })?;
            // Process and return data
            Ok(Some(processed_data))
        } else {
            Ok(None)
        }
    }
}
```

### Pros

- Simpler implementation
- True streaming (no memory pre-loading)
- Direct integration with Rust streams
- Less boilerplate

### Cons

- Not a true JavaScript async iterator
- Requires manual iteration calls
- No `for await` loop support out of the box
- Custom iteration protocol

## Current Breeze-NAPI Implementation

### Key Features

- Callback-based approach using ThreadsafeFunctions
- Uses `on_chunk`, `on_error`, `on_complete` pattern
- Spawns async tasks with `tokio::spawn`
- Non-blocking callback execution

### Implementation Details

```rust
#[napi(ts_args_type = "content: string, language: string, filePath: string | undefined, onChunk: (chunk: SemanticChunkJs) => void, onError: (error: Error) => void, onComplete: () => void")]
pub fn chunk_code(
    &self,
    content: String,
    language: String,
    file_path: Option<String>,
    on_chunk: ThreadsafeFunction<SemanticChunkJs, ErrorStrategy::CalleeHandled>,
    on_error: ThreadsafeFunction<String, ErrorStrategy::CalleeHandled>,
    on_complete: ThreadsafeFunction<(), ErrorStrategy::CalleeHandled>,
) -> Result<()> {
    let chunker = self.inner.clone();
    
    tokio::spawn(async move {
        let mut stream = Box::pin(chunker.chunk_code(content, language, file_path));
        
        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    let js_chunk = chunk.into();
                    if on_chunk.call(Ok(js_chunk), ThreadsafeFunctionCallMode::NonBlocking).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = on_error.call(Ok(e.to_string()), ThreadsafeFunctionCallMode::NonBlocking);
                    break;
                }
            }
        }
        
        let _ = on_complete.call(Ok(()), ThreadsafeFunctionCallMode::NonBlocking);
    });
    
    Ok(())
}
```

### Pros

- Works well with current async patterns
- Non-blocking execution
- Clear error handling
- Familiar callback pattern

### Cons

- Callback hell potential
- Not as idiomatic for JavaScript iteration
- Requires multiple function parameters
- No built-in backpressure handling

## Recommendations

### Recommended Approach: LanceDB-Style with Enhancements

For breeze-rustle, I recommend implementing a hybrid approach based on the LanceDB pattern with some enhancements:

1. **Primary API**: LanceDB-style manual async iteration
2. **Secondary API**: Keep existing callback-based methods for backward compatibility
3. **Future Enhancement**: Add SMB-JS style true async iterators if needed

### Proposed Implementation

```rust
#[napi]
pub struct SemanticChunkIterator {
    inner: Pin<Box<dyn Stream<Item = Result<RustChunk, Error>> + Send>>,
}

#[napi]
impl SemanticChunkIterator {
    #[napi]
    pub async fn next(&mut self) -> napi::Result<Option<SemanticChunkJs>> {
        if let Some(result) = self.inner.next().await {
            match result {
                Ok(chunk) => Ok(Some(chunk.into())),
                Err(e) => Err(napi::Error::from_reason(format!("Chunk error: {}", e))),
            }
        } else {
            Ok(None)
        }
    }
}

// New methods on SemanticChunker
#[napi]
impl SemanticChunker {
    #[napi]
    pub fn chunk_code_iter(
        &self,
        content: String,
        language: String,
        file_path: Option<String>,
    ) -> Result<SemanticChunkIterator> {
        let stream = self.inner.chunk_code(content, language, file_path);
        Ok(SemanticChunkIterator {
            inner: Box::pin(stream),
        })
    }
}
```

### Usage Examples

```javascript
// New iterator approach
const iterator = chunker.chunkCodeIter(content, 'rust', 'main.rs');
let chunk;
while ((chunk = await iterator.next()) !== null) {
    console.log(chunk);
}

// Existing callback approach (maintained for compatibility)
chunker.chunkCode(content, 'rust', 'main.rs', 
    chunk => console.log(chunk),
    error => console.error(error),
    () => console.log('Complete')
);
```

## Implementation Priority

1. **Phase 1**: Implement LanceDB-style iterators for all chunking methods
2. **Phase 2**: Add comprehensive TypeScript definitions
3. **Phase 3**: Benchmark performance against callback approach
4. **Phase 4**: Consider SMB-JS style iterators if `for await` support is needed

## Key Considerations

1. **Memory Usage**: Stream-based approach is more memory efficient
2. **Error Handling**: Need clear error propagation from Rust to JavaScript
3. **Backpressure**: Manual iteration provides natural backpressure
4. **TypeScript Support**: Ensure proper type definitions for async iteration
5. **Backward Compatibility**: Maintain existing callback-based API

## Next Steps

1. Implement `SemanticChunkIterator`, `ProjectChunkIterator`, and `TextChunkIterator`
2. Add new iterator-returning methods to `SemanticChunker`
3. Update TypeScript definitions
4. Add comprehensive tests
5. Update documentation with usage examples

## References

- [NAPI-RS Iterator Documentation](https://napi.rs/docs/concepts/generator)
- [JavaScript Async Iteration Protocol](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Iteration_protocols#the_async_iterator_and_async_iterable_protocols)
- [Rust Futures Stream](https://docs.rs/futures/latest/futures/stream/trait.Stream.html)
