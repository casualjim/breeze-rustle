# breeze-rustle Context & Current State

## Overview

`breeze-rustle` is a Rust library with Python bindings that provides semantic code chunking using text-splitter's CodeSplitter with tree-sitter parsers.

## Current Implementation Status

### ✅ Completed Tasks

1. **Removed syntastica dependencies**
   - Removed syntastica-parsers-git, syntastica-queries, syntastica-core from Cargo.toml
   - Replaced with direct tree-sitter-* crates for each language

2. **Created language registry** (`src/languages.rs`)
   - Maps language names to tree-sitter parsers using LanguageFn type
   - Supports 16 languages (PHP, Kotlin, SQL excluded due to broken crates)
   - Includes case-insensitive aliases (e.g., "python" → "Python")
   - Uses tree-sitter-language crate to handle LanguageFn → Language conversions

3. **Implemented core types** (`src/types.rs`)
   - `ChunkMetadata`: PyO3 class with node_type, node_name, language, parent_context, scope_path, definitions, references
   - `SemanticChunk`: PyO3 class with text, byte/line offsets, and metadata
   - `ChunkError`: Error enum for unsupported languages, parse errors, IO errors, query errors
   - `ChunkType`: Enum for discriminating between semantic and text chunks
   - `ProjectChunk`: Class containing file path, chunk type, and chunk data

4. **Enhanced chunker implementation** (`src/chunker.rs`)
   - `InnerChunker` wraps text-splitter's CodeSplitter
   - `chunk_code()` method creates chunks with rich metadata
   - `chunk_text()` method for plain text chunking
   - Both methods now accept owned strings and return streams
   - Integrates with metadata extractor for AST analysis

5. **Implemented Python bindings** (`src/lib.rs`)
   - `SemanticChunker` PyO3 class with async support
   - `chunk_code` and `chunk_text` methods return async iterators (`ChunkStream`)
   - `walk_project` method returns async iterator (`ProjectWalker`)
   - Static methods for language support checking
   - Proper error mapping from Rust to Python exceptions

6. **Added metadata extraction** (`src/metadata_extractor.rs`)
   - Extracts actual node_type from tree-sitter AST nodes
   - Extracts node_name from identifier nodes (functions, classes, methods, structs)
   - Builds scope_path by traversing parent nodes
   - Extracts parent_context (e.g., class name for methods)
   - Extracts definitions and references within chunks
   - Special handling for Rust structs and impl blocks

7. **Text chunking support** (`chunk_text` method)
   - Added separate method for plain text chunking
   - Works for any content regardless of language support
   - Returns chunks with minimal metadata (text_chunk type)
   - Allows users to handle unsupported languages gracefully

8. **Tokenizer support**
   - Implemented multiple tokenizers via `TokenizerType` enum:
     - `TokenizerType.CHARACTERS` - Character-based chunking (default)
     - `TokenizerType.TIKTOKEN` - OpenAI's cl100k_base tokenizer
     - `TokenizerType.HUGGINGFACE` - HuggingFace tokenizers with model specification
   - Clean API with type-safe enum values
   - Proper error handling for missing HuggingFace model names

9. **Created Python type stubs** (`python/breeze_rustle/__init__.pyi`)
   - Complete type annotations for all classes and methods
   - Includes all enums (`TokenizerType`, `ChunkType`)
   - Includes all async iterator classes (`ChunkStream`, `ProjectWalker`)
   - Comprehensive docstrings for IDE support

10. **Comprehensive test suite**
    - Tests updated for async iterator API
    - Language registry tests
    - Type system tests
    - Basic chunking tests
    - Metadata extraction tests
    - Multi-language chunking tests (Python, JavaScript, TypeScript, Rust)
    - Performance test (1MB file in ~25s - needs optimization)
    - Scope and definition extraction tests
    - Text chunking tests
    - Tokenizer enum tests
    - Async walker tests

11. **Async generators throughout**
    - All methods now return async iterators instead of collecting into lists
    - True streaming architecture for memory efficiency
    - `chunk_code` returns `ChunkStream`
    - `chunk_text` returns `ChunkStream`
    - `walk_project` returns `ProjectWalker`

### 📋 Remaining Tasks

- ~~Performance optimization (current: ~25s for 1MB file, target: <100ms)~~ ✅ Achieved! Processing 337k chunks in 26s
- ~~Investigate broken language crates (PHP, Kotlin, SQL)~~ ✅ Fixed by using correct versions
- Add more sophisticated definitions/references extraction
- ~~Create benchmark test to index entire kuzu project~~ ✅ test_async_walker.py benchmarks full project

## ✅ Project Directory Walker

### Overview

A feature to walk entire project directories and automatically process all files, yielding semantic chunks for supported languages and text chunks for other files. This feature is fully implemented with async iteration support.

### Key Features

- **Automatic file filtering**: Uses `hyperpolyglot` for language detection and `infer` for file type detection
- **Smart chunking**: Automatically uses semantic chunking for supported languages, falls back to text chunking
- **Respects .gitignore**: Uses the `ignore` crate for gitignore-aware traversal
- **Parallel processing**: Configurable parallelism for performance
- **Discriminated types**: Returns `ProjectChunk` that clearly indicates whether it's semantic or text

### New Types

```python
class ChunkType(Enum):
    SEMANTIC = "SEMANTIC"  # Properly parsed code
    TEXT = "TEXT"          # Plain text chunking

class ProjectChunk:
    file_path: str         # Path to source file
    chunk_type: ChunkType  # Type of chunk
    chunk: SemanticChunk   # The actual chunk data
    is_semantic: bool      # Helper property
    is_text: bool          # Helper property
```

### Python API

```python
# Create a chunker
chunker = SemanticChunker(
    max_chunk_size=1500,
    tokenizer=TokenizerType.CHARACTERS
)

# Walk a project - returns an async iterator
walker = await chunker.walk_project(
    path="./my_project",
    max_chunk_size=1500,  # Optional, defaults to chunker setting
    tokenizer=TokenizerType.CHARACTERS,  # Optional
    hf_model=None,  # Required if tokenizer is HUGGINGFACE
    max_parallel=8  # Number of files to process in parallel
)

# Iterate over chunks
async for chunk in walker:
    # chunk is a ProjectChunk with file_path, chunk_type, and chunk
    pass
```

### Usage Example

```python
# Process entire project
chunker = SemanticChunker()
walker = await chunker.walk_project("./my_project")

async for chunk in walker:
    if chunk.chunk_type == ChunkType.SEMANTIC:
        print(f"Code: {chunk.file_path} - {chunk.chunk.metadata.node_type}")
    else:
        print(f"Text: {chunk.file_path}")
```

## Key Technical Decisions

### Language Support

- Using LanguageFn type from tree-sitter-language crate
- Direct tree-sitter-* crate dependencies instead of syntastica
- Currently supporting: Python, JavaScript, TypeScript, TSX, Java, C++, C, C#, Go, Rust, Ruby, Swift, Scala, Shell/Bash, R
- Removed: PHP (unclear API), Kotlin (compilation issues), SQL (compilation issues)
- **Migration planned**: Moving from individual crates to compiled grammars (165+ languages)

### Architecture

- text-splitter handles the core chunking logic
- We add metadata extraction on top
- Using pyo3-async-runtimes (not pyo3-asyncio which is unmaintained)
- Async Python API for better performance

### Dependencies

Key dependencies in Cargo.toml:

- text-splitter 0.27 with code, tiktoken-rs, tokenizers features
- tree-sitter 0.25 and individual language crates
- pyo3 0.25.0 with extension-module, abi3-py39
- pyo3-async-runtimes 0.25.0 with tokio
- tree-sitter-language 0.1.5 (critical for LanguageFn support)

## Common Issues & Solutions

1. **Tree-sitter version conflicts**: Use specific versions in Cargo.toml
2. **LanguageFn vs Language**: Use tree-sitter-language crate and convert with `.into()`
3. **Language API differences**: Some use LANGUAGE constants, others use language() functions

## ✅ Grammar Integration - Phase 1 Complete

### Overview

We've successfully implemented a build-time grammar compilation system in the `breeze-grammars` crate located at `crates/breeze-grammars/`. This replaces individual tree-sitter-* dependencies with a unified approach.

### Implementation Details

1. **Architecture**:
   - `grammars.json`: Configuration file listing grammar repositories
   - `build.rs`: Downloads and compiles grammars using cc crate
   - Generated bindings follow official tree-sitter patterns using `LanguageFn::from_raw()`
   - Case-insensitive API with lowercase normalization

2. **Key Design Decisions**:
   - Use `LanguageFn::from_raw()` instead of unsafe transmutes
   - Generate both `Language` and `LanguageFn` accessors
   - Pre-compile at build time, no runtime downloads
   - Simple lowercase normalization (no complex aliasing)

3. **Current Status**:
   - ✅ Python grammar working and all tests passing
   - ✅ Proper ABI compatibility (version 14 works with tree-sitter 0.25)
   - ✅ Clean integration with text-splitter via LanguageFn

4. **Next Steps**:
   - Add Rust, JavaScript, TypeScript to test scalability
   - Create curated list of 30-50 priority languages
   - Add error handling for failed grammar compilations
   - Remove old tree-sitter-* dependencies after full migration

### Important Notes

- Grammar repositories must use format `owner/repo` (GitHub URLs are auto-constructed)
- Some grammars have source in subdirectories (use `path` field in JSON)
- Always use maturin for building Python bindings: `maturin develop`
- Bus errors were caused by improper Language/LanguageFn conversion, now fixed

See `docs/plans/tree-sitter-grammar-integration.md` for original plan.

## Build & Test Commands

```bash
# Build Python package
maturin develop --release

# Run Rust tests
cargo nextest run

# Check for issues
cargo check
cargo clippy
```

## API Usage Examples

### Basic Usage

```python
from breeze_rustle import SemanticChunker, TokenizerType, ChunkType
import asyncio

async def main():
    # Default character-based chunking
    chunker = SemanticChunker(max_chunk_size=1500)
    
    # With tiktoken tokenizer
    chunker = SemanticChunker(tokenizer=TokenizerType.TIKTOKEN)
    
    # With HuggingFace tokenizer
    chunker = SemanticChunker(
        tokenizer=TokenizerType.HUGGINGFACE,
        hf_model="bert-base-uncased"
    )
    
    # Check language support
    if SemanticChunker.is_language_supported("Python"):
        # Get an async iterator for code chunks
        chunk_stream = await chunker.chunk_code(content, "Python", "example.py")
        
        # Iterate over chunks
        async for chunk in chunk_stream:
            print(f"Lines {chunk.start_line}-{chunk.end_line}: {chunk.metadata.node_type}")
    else:
        # Use text chunking for unsupported languages
        text_stream = await chunker.chunk_text(content, "example.txt")
        
        async for chunk in text_stream:
            print(f"Text chunk: {len(chunk.text)} chars")

asyncio.run(main())
```

### Project Walking Example

```python
async def index_project(project_path: str):
    chunker = SemanticChunker(tokenizer=TokenizerType.TIKTOKEN)
    walker = await chunker.walk_project(project_path, max_parallel=16)
    
    async for project_chunk in walker:
        if project_chunk.chunk_type == ChunkType.SEMANTIC:
            # This is parsed code with full metadata
            print(f"Code: {project_chunk.chunk.metadata.node_type} in {project_chunk.file_path}")
            # Generate embeddings, store in vector DB, etc.
        else:
            # This is text content
            print(f"Text: {project_chunk.file_path}")
```

## Performance Results

With 512-character chunks on the kuzu project:

- **337,327 chunks** from 4,457 files
- **26.2 seconds** total time
- **12,883 chunks/second** throughput
- 27.7% semantic chunks, 72.3% text chunks
- Files include: .h (32.7%), .cpp (25.2%), .test (11.9%), .csv (7.8%)

## Next Steps

1. ~~Write comprehensive Python acceptance tests~~ ✅ Done
2. ~~Optimize performance~~ ✅ Achieved 12,883 chunks/s
3. **Implement tree-sitter grammar integration** 🚧 In Progress
   - Port tree-sitter-language-pack approach to Rust
   - Support 165+ languages instead of current 16
   - Compile grammars directly without tree-sitter CLI
4. Set up CI/CD with GitHub Actions
5. Package and publish to PyPI using maturin
6. Add streaming file reader for files >5MB (currently skipped)
