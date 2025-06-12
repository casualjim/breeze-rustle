# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

breeze-rustle is a high-performance Rust library with Python bindings that provides semantic code chunking capabilities. It uses tree-sitter parsers and nvim-treesitter queries to intelligently split code into meaningful semantic units while preserving context and extracting rich metadata.

### Chunking and Embedding Strategy

- **Chunking is for embedder constraints**: We chunk files to fit within embedding model token limits (e.g., 8k tokens)
- **Semantic boundaries**: Chunks split at function/class boundaries, not arbitrarily mid-code
- **File-level storage**: Despite chunking, we store embeddings at the file level by aggregating chunk embeddings
- **Aggregation strategy**: Currently uses weighted average by token count, but designed to be pluggable

## Coding rules

* KISS: keep it stupid simple.
* YAGNI: you aren't going to need it, only implement what was asked do not make up requirements
* Avoid allocations when you can, proactive cloning is not a good look
* Prefer streaming API's over batching API's. So return streams not vecs
* Backwards compatibility is not a goal, modify and break the existing code
* do not create a new file for every new task, work with what already exist.
* do not create limits or fallbacks for specialized libraries.
* Create the simplest possible thing that could possibly work
* Use Uuid::now_v7 not v4

## Development Commands

### Building

```bash
# Build and install Python package locally (for development)
maturin develop --release

# Build Python wheel for distribution
maturin build --release

# Build breeze-grammars crate separately (for testing)
cd crates/breeze-grammars && cargo build
```

### Testing

```bash
# Run Rust tests
cargo test

# Run Python tests
pytest tests/

# Run benchmarks
cargo bench
```

### Maintenance

```bash
# Update nvim-treesitter queries
python tools/fetch-queries

# Check Rust code
cargo check
cargo clippy
```

## Architecture

The project follows a Python → PyO3 → Rust architecture:

1. **Python API** (exposed via PyO3):
   * `SemanticChunker` class with async methods
   * `SemanticChunk` and `ChunkMetadata` data classes

2. **Rust Core**:
   * `src/`: Main library with PyO3 bindings
   * `crates/breeze-grammars/`: Grammar compilation system
   * `crates/breeze-chunkers/`: (Future) Pure Rust core without Python deps

3. **Grammar System** (`crates/breeze-grammars/`):
   * Downloads and compiles tree-sitter grammars at build time
   * Uses official `LanguageFn::from_raw()` pattern
   * Case-insensitive API with lowercase normalization

4. **Build System**:
   * `maturin` handles Rust → Python packaging
   * Grammars compiled via cc crate in build.rs
   * nvim-treesitter queries embedded at compile time

## Implementation Status

The project is in early development. The core functionality described in `docs/plans/breeze-rustle-implementation.md` is not yet implemented. Current `src/lib.rs` contains only example code.

## Key Implementation Guidelines

When implementing features:

* Follow the detailed plan in `docs/plans/breeze-rustle-implementation.md`
* Ensure no panics - use proper error handling
* Maintain zero-copy where possible for performance
* Support 100+ languages via nvim-treesitter queries
* Target <100ms parsing for 1MB files
* Never split semantic units (functions, classes) unless necessary
* Chunk files to fit embedder constraints but aggregate back to file-level embeddings
* Make aggregation strategies pluggable (start with weighted average)

## Preferred Tools

* Prefer context7 -> tavily / brave -> perplexity for documentation lookup