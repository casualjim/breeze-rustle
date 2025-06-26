# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!! NO BACKWARDS COMPATIBILITY !!!!!!!!!!!!
!!!!!!!!!!!! NO BACKWARDS COMPATIBILITY !!!!!!!!!!!!
!!!!!!!!!!!! NO BACKWARDS COMPATIBILITY !!!!!!!!!!!!
!!!!!!!!!!!! NO BACKWARDS COMPATIBILITY !!!!!!!!!!!!
!!!!!!!!!!!! NO BACKWARDS COMPATIBILITY !!!!!!!!!!!!
!!!!!!!!!!!! NO BACKWARDS COMPATIBILITY !!!!!!!!!!!!
!!!!!!!!!!!! NO BACKWARDS COMPATIBILITY !!!!!!!!!!!!
!!!!!!!!!!!! NO BACKWARDS COMPATIBILITY !!!!!!!!!!!!
!!!!!!!!!!!! NO BACKWARDS COMPATIBILITY !!!!!!!!!!!!
!!!!!!!!!!!! NO BACKWARDS COMPATIBILITY !!!!!!!!!!!!
!!!!!!!!!!!! NO BACKWARDS COMPATIBILITY !!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Do not make assumptions, the workflow is:

Read/Investigate -> Think/Research -> Propose & Discuss -> Plan -> Write down the plan in plans/docs -> Make a todo list -> Execute -> Validate -> Lint

## Project Overview

breeze-rustle is a high-performance Rust library with Python and Node.js bindings that provides semantic code chunking capabilities. It uses tree-sitter parsers and nvim-treesitter queries to intelligently split code into meaningful semantic units while preserving context and extracting rich metadata.

The project also includes:

- A CLI application for indexing and searching codebases
- An HTTP/HTTPS API server with MCP protocol support
- Vector-based semantic search using LanceDB

## Coding rules

- KISS: keep it stupid simple.
- YAGNI: you aren't going to need it, only implement what was asked do not make up requirements
- Avoid allocations when you can, proactive cloning is not a good look
- Prefer streaming API's over batching API's. So return streams not vecs
- Backwards compatibility is not a goal, modify and break the existing code
- do not create a new file for every new task, work with what already exist.
- do not create limits or fallbacks for specialized libraries.
- Create the simplest possible thing that could possibly work
- Use Uuid::now_v7 not v4
- Do NOT `#[ignore]` tests because they may download a model, it's totally allowed to access the internet. We have to install packages to run the build anyway, they come from the internet
- Use rustfmt, do NOT run `cargo fmt`
- Put imports at the top of the file for code and the top of the tests module for tests

## Development Commands

### Building

```bash
just build
```

### Testing

```bash
# Run Rust tests
just test
```

## Architecture

The project is a Rust workspace with multiple crates:

1. **Core Crates**:
   - `crates/breeze/`: CLI application and main entry point
   - `crates/breeze-chunkers/`: Core chunking library using tree-sitter
   - `crates/breeze-grammars/`: Grammar compilation system
   - `crates/breeze-indexer/`: Code indexing and embedding with LanceDB
   - `crates/breeze-server/`: HTTP/HTTPS API server and MCP server

2. **Language Bindings**:
   - `crates/breeze-py/`: Python bindings via PyO3
   - `crates/breeze-napi/`: Node.js bindings via NAPI

3. **Grammar System** (`crates/breeze-grammars/`):
   - Compiles tree-sitter grammars at build time
   - Supports 100+ languages via nvim-treesitter queries
   - Case-insensitive API with lowercase normalization

4. **Build System**:
   - Uses `just` for task running
   - `maturin` handles Python packaging
   - Grammars compiled via cc crate in build.rs

## Implementation Status

The project has a working implementation with:

- Semantic code chunking using tree-sitter
- CLI for indexing and searching codebases
- HTTP/HTTPS API server with OpenAPI docs
- Python and Node.js bindings
- Support for 100+ programming languages

## Key Implementation Guidelines

When implementing features:

- Ensure no panics - use proper error handling
- Maintain zero-copy where possible for performance
- Support 100+ languages via nvim-treesitter queries
- Never split semantic units (functions, classes) unless necessary
- Use streaming APIs for large file handling

## Preferred Tools

- Prefer breeze -> context7 -> tavily / brave for documentation lookup
