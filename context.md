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

4. **Enhanced chunker implementation** (`src/chunker.rs`)
   - `InnerChunker` wraps text-splitter's CodeSplitter
   - `chunk_file()` method creates chunks with rich metadata
   - Integrates with metadata extractor for AST analysis

5. **Implemented Python bindings** (`src/lib.rs`)
   - `SemanticChunker` PyO3 class with async support
   - `chunk_file` method using pyo3-async-runtimes with tokio
   - Static methods for language support checking
   - Proper error mapping from Rust to Python exceptions

6. **Added metadata extraction** (`src/metadata_extractor.rs`)
   - Extracts actual node_type from tree-sitter AST nodes
   - Extracts node_name from identifier nodes (functions, classes, methods, structs)
   - Builds scope_path by traversing parent nodes
   - Extracts parent_context (e.g., class name for methods)
   - Extracts definitions and references within chunks
   - Special handling for Rust structs and impl blocks

7. **Created Python type stubs** (`python/breeze_rustle/__init__.pyi`)
   - Complete type annotations for all classes and methods
   - Includes docstrings for IDE support

8. **Comprehensive test suite**
   - 18 passing tests including:
   - Language registry tests
   - Type system tests
   - Basic chunking tests
   - Metadata extraction tests
   - Multi-language chunking tests (Python, JavaScript, TypeScript, Rust)
   - Performance test (1MB file in <10s)
   - Scope and definition extraction tests

### 📋 Remaining Tasks
- Write Python acceptance tests
- Performance optimization (current: ~7s for 1MB file, target: <100ms)
- Set up Python packaging with maturin for PyPI distribution

## Key Technical Decisions

### Language Support
- Using LanguageFn type from tree-sitter-language crate
- Direct tree-sitter-* crate dependencies instead of syntastica
- Currently supporting: Python, JavaScript, TypeScript, TSX, Java, C++, C, C#, Go, Rust, Ruby, Swift, Scala, Shell/Bash, R
- Removed: PHP (unclear API), Kotlin (compilation issues), SQL (compilation issues)

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

## Next Steps
The immediate next step is implementing the Python bindings in `src/lib.rs` to expose the chunking functionality to Python users. This is the highest priority as it will make the library usable.