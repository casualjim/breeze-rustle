# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

breeze-rustle is a high-performance Rust library with Python bindings that provides semantic code chunking capabilities. It uses tree-sitter parsers and nvim-treesitter queries to intelligently split code into meaningful semantic units while preserving context and extracting rich metadata.

## Development Commands

### Building
```bash
# Build and install Python package locally (for development)
maturin develop --release

# Build Python wheel for distribution
maturin build --release
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
   - `SemanticChunker` class with async methods
   - `SemanticChunk` and `ChunkMetadata` data classes
   
2. **Rust Core** (in `src/`):
   - Tree-sitter parsing with embedded nvim-treesitter queries
   - Async/concurrent processing using smol runtime
   - Zero-copy operations where possible

3. **Build System**:
   - `maturin` handles Rust → Python packaging
   - Queries embedded at compile time via build.rs

## Implementation Status

The project is in early development. The core functionality described in `docs/plans/breeze-rustle-implementation.md` is not yet implemented. Current `src/lib.rs` contains only example code.

## Key Implementation Guidelines

When implementing features:
- Follow the detailed plan in `docs/plans/breeze-rustle-implementation.md`
- Ensure no panics - use proper error handling
- Maintain zero-copy where possible for performance
- Support 100+ languages via nvim-treesitter queries
- Target <100ms parsing for 1MB files
- Never split semantic units (functions, classes) unless necessary

## Preferred Tools

- Prefer context7 -> tavily / brave -> perplexity for documentation lookup