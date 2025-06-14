# Code-Splitter Analysis and Lessons Learned

## Overview

The code-splitter crate by wangxj03 provides a clean implementation for splitting code into semantic chunks using tree-sitter. This analysis documents key insights we can apply to breeze-rustle.

## Core Architecture

### 1. Trait-Based Design

```rust
pub trait Sizer {
    fn size(&self, text: &str) -> Result<usize>;
}
```

- Simple, elegant interface for measuring chunk size
- Allows pluggable implementations (characters, words, tokens)
- Zero-cost abstraction

### 2. Recursive Tree Traversal with Sibling Merging

The core algorithm in `split_node()`:

1. Check if current node fits within max_size
2. If yes, return as single chunk
3. If no, recursively split children
4. **Key innovation**: Use `try_fold` to merge adjacent chunks that fit within size limit

```rust
// Join the tail and head of neighboring chunks if possible
.try_fold(Vec::new(), |mut acc, mut next| -> Result<Vec<Chunk>> {
    if let Some(tail) = acc.pop() {
        if let Some(head) = next.first_mut() {
            let joined_size = self.joined_size(&tail, head, code)?;
            if joined_size <= self.max_size {
                // Merge chunks
                head.subtree = format!("{}\n{}", tail.subtree, head.subtree);
                head.range.start_byte = tail.range.start_byte;
                head.range.start_point = tail.range.start_point;
                head.size = joined_size;
            } else {
                acc.push(tail);
            }
        }
    }
    acc.append(&mut next);
    Ok(acc)
})
```

### 3. Simple Error Handling

- Uses `Box<dyn Error>` for maximum flexibility
- Trade-off: Less specific error types

## Key Strengths

1. **Clean API**: Builder pattern with `with_max_size()`
2. **Extensibility**: Easy to add new Sizer implementations
3. **Performance**: Works directly with `&[u8]`, converts to UTF-8 only when needed
4. **Visualization**: Nice tree formatting for debugging

## Limitations We Can Address

1. **No Parallel Processing**: Sequential only
2. **Limited Metadata**: Only size and range, no semantic info
3. **No Query Support**: Doesn't use tree-sitter queries
4. **No Overlap**: Many RAG systems benefit from chunk overlap
5. **Memory Usage**: Creates string representations for all chunks

## Integration with text-splitter

Since text-splitter already has:

- `ChunkSizer` trait (similar to code-splitter's `Sizer`)
- `CodeSplitter` with tree-sitter support
- Built-in support for tiktoken and tokenizers

We should:

1. Use text-splitter's `ChunkSizer` trait directly
2. Leverage its `CodeSplitter` implementation
3. Add our semantic enhancements on top

## Recommendations for breeze-rustle

### 1. Build on text-splitter's Foundation

- Use `text_splitter::ChunkSizer` instead of creating our own trait
- Use `text_splitter::CodeSplitter` as the base implementation
- Add our metadata extraction as a layer on top

### 2. Add Query-Based Semantic Understanding

- Use nvim-treesitter queries to identify semantic boundaries
- Extract richer metadata (function names, class context, etc.)
- Provide semantic scoring for chunks

### 3. Implement Configurable Overlap

```rust
pub struct ChunkConfig {
    pub overlap_ratio: f32,  // 0.0 to 1.0
    pub overlap_tokens: Option<usize>,  // Fixed token overlap
}
```

### 4. Add Async/Parallel Processing

- Use our existing async infrastructure
- Process multiple files concurrently
- Stream results for large codebases

### 5. Enrich Metadata

```rust
pub struct EnrichedChunk {
    pub base_chunk: text_splitter::Chunk,
    pub semantic_type: SemanticType,  // Function, Class, Method, etc.
    pub name: Option<String>,
    pub parent_context: Option<String>,
    pub imports: Vec<String>,
    pub definitions: Vec<String>,
    pub references: Vec<String>,
}
```

## Implementation Strategy

1. **Layer 1**: Use text-splitter's `CodeSplitter` for basic chunking
2. **Layer 2**: Add semantic analysis using nvim-treesitter queries
3. **Layer 3**: Implement overlap and metadata enrichment
4. **Layer 4**: Add async/parallel processing wrapper

This approach gives us:

- Battle-tested chunking from text-splitter
- Semantic understanding from nvim-treesitter
- Performance from async processing
- Rich metadata for RAG applications

## Code Example

```rust
use text_splitter::{ChunkConfig, CodeSplitter, ChunkSizer};
use crate::semantic_analyzer::SemanticAnalyzer;

pub struct BreezeRustleChunker {
    base_splitter: CodeSplitter<ChunkConfig>,
    semantic_analyzer: SemanticAnalyzer,
    overlap_config: OverlapConfig,
}

impl BreezeRustleChunker {
    pub async fn chunk_with_semantics(
        &self,
        content: &str,
        language: &str,
    ) -> Result<Vec<EnrichedChunk>> {
        // Step 1: Basic chunking with text-splitter
        let base_chunks = self.base_splitter.chunks(content);
        
        // Step 2: Enrich with semantic metadata
        let enriched = self.semantic_analyzer
            .analyze_chunks(base_chunks, language)
            .await?;
        
        // Step 3: Apply overlap if configured
        let with_overlap = self.apply_overlap(enriched)?;
        
        Ok(with_overlap)
    }
}
```

## Conclusion

The code-splitter implementation provides valuable patterns, especially the sibling merging algorithm. By building on text-splitter's foundation and adding our semantic layer, we can create a more powerful solution that combines the best of both approaches.
