# Text-Splitter Analysis for breeze-rustle

## Overview

The text-splitter crate provides a robust, production-ready implementation for semantic text chunking, including dedicated support for code via tree-sitter. It's actively maintained and used in production.

## Key Features

### 1. Core Architecture

#### ChunkSizer Trait

```rust
pub trait ChunkSizer {
    fn size(&self, chunk: &str) -> usize;
}
```

- Simple interface for measuring chunk size
- Implementations for:
  - `Characters` (default)
  - `tiktoken_rs::CoreBPE` (OpenAI tokenizers)
  - `tokenizers::Tokenizer` (HuggingFace tokenizers)

#### ChunkCapacity and ChunkConfig

- **ChunkCapacity**: Supports both fixed size and ranges (desired vs max)
- **ChunkConfig**: Complete configuration including overlap, trimming, and sizer
- Overlap support built-in (configurable via `with_overlap()`)

### 2. CodeSplitter Implementation

The `CodeSplitter` uses tree-sitter for parsing:

- Takes a tree-sitter `Language` and `ChunkConfig`
- Automatically handles invalid/unparseable code gracefully
- Uses syntax tree depth as semantic levels
- Smart merging of sibling nodes to maximize chunk size

#### Algorithm

1. Parse text into syntax tree using tree-sitter
2. Traverse tree depth-first, collecting nodes with their depths
3. Use depths as semantic levels (lower depth = higher priority)
4. Split and merge based on semantic boundaries

### 3. Key Strengths

1. **Production Ready**: Used in real applications, well-tested
2. **Flexible Sizing**: Supports characters, tokens (tiktoken/HF)
3. **Smart Chunking**: Respects semantic boundaries
4. **Overlap Support**: Built-in configurable overlap
5. **Range Support**: Can target desired size with max limit
6. **Trim Options**: Configurable whitespace handling
7. **Performance**: Memoized size calculations for efficiency

### 4. API Examples

Basic usage:

```rust
use text_splitter::{ChunkConfig, CodeSplitter};

let splitter = CodeSplitter::new(tree_sitter_rust::LANGUAGE, 1000)?;
let chunks = splitter.chunks(text).collect::<Vec<_>>();
```

With tokenizer:

```rust
use tiktoken_rs::cl100k_base;

let tokenizer = cl100k_base()?;
let config = ChunkConfig::new(1000).with_sizer(tokenizer);
let splitter = CodeSplitter::new(tree_sitter_rust::LANGUAGE, config)?;
```

With overlap:

```rust
let config = ChunkConfig::new(1000)
    .with_overlap(100)?
    .with_trim(false);
```

## What text-splitter Already Provides

1. **Tree-sitter integration** for code parsing
2. **ChunkSizer trait** for pluggable tokenization
3. **Overlap functionality** for RAG applications
4. **Smart merging** of semantic units
5. **Multiple output formats**:
   - `chunks()` - just text
   - `chunk_indices()` - with byte offsets
   - `chunk_char_indices()` - with char offsets

## What's Missing for breeze-rustle Goals

1. **Language Detection**: Need to identify language from file
2. **Multi-language Support**: Need to manage multiple parsers
3. **Semantic Metadata**: No extraction of function names, classes, etc.
4. **Query Support**: No nvim-treesitter query integration
5. **Async/Parallel**: Sequential processing only
6. **Python Bindings**: Need PyO3 wrapper

## How Far Does CodeSplitter Get Us?

**Coverage: ~60% of our requirements**

✅ Already provides:

- Semantic code chunking
- Tree-sitter integration
- Flexible sizing (tokens/chars)
- Overlap support
- Production-quality implementation

❌ Still need:

- Multi-language parser management
- Metadata extraction
- Query-based semantic analysis
- Async processing
- Python API

## Implementation Strategy for breeze-rustle

### Layer 1: Language Management

```rust
pub struct LanguageRegistry {
    parsers: HashMap<String, Language>,
}
```

### Layer 2: Enhanced Chunker

```rust
pub struct BreezeChunker {
    registry: LanguageRegistry,
    base_config: ChunkConfig<Box<dyn ChunkSizer>>,
}

impl BreezeChunker {
    pub fn chunk_with_metadata(&self, text: &str, lang: &str) 
        -> Result<Vec<EnrichedChunk>> {
        // 1. Get language parser
        let language = self.registry.get(lang)?;
        
        // 2. Create CodeSplitter
        let splitter = CodeSplitter::new(language, &self.base_config)?;
        
        // 3. Get base chunks
        let chunks = splitter.chunk_indices(text);
        
        // 4. Enrich with metadata
        self.enrich_chunks(chunks, text, language)
    }
}
```

### Layer 3: Metadata Extraction

```rust
pub struct ChunkMetadata {
    pub semantic_type: String,    // function, class, method
    pub name: Option<String>,      // extracted name
    pub parent_context: Vec<String>, // scope hierarchy
    pub imports: Vec<String>,      // for context
}
```

### Layer 4: Python API

```rust
#[pyclass]
pub struct SemanticChunker {
    inner: BreezeChunker,
}
```

## Recommendations

1. **Use text-splitter as foundation**: Don't reinvent chunking logic
2. **Build metadata layer on top**: Add our semantic enhancements
3. **Wrap for Python**: Create thin PyO3 bindings
4. **Add async later**: Can wrap sync code in async executor

## Questions to Answer

1. **Do we need syntastica initially?**
   - Not for MVP - text-splitter + individual tree-sitter parsers sufficient
   - Syntastica valuable for query support later

2. **Is text-splitter sufficient for core functionality?**
   - Yes for chunking algorithm
   - Need to add language management and metadata extraction

3. **What's the minimal path to working solution?**
   - Use text-splitter's CodeSplitter
   - Add simple language registry
   - Extract basic metadata (node types)
   - Wrap in PyO3

## Conclusion

Text-splitter provides an excellent foundation that handles the complex chunking logic. We should build our enhanced functionality on top rather than reimplementing. This approach gives us:

1. **Faster time to market**: Core chunking already solved
2. **Better reliability**: Battle-tested implementation
3. **Flexibility**: Can enhance incrementally
4. **Maintainability**: Less code to maintain

The minimal viable implementation would be ~500 lines of code on top of text-splitter, versus thousands to reimplement everything.
