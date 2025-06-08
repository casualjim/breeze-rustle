# Syntastica Query Generation Process

## Overview

This document details how syntastica fetches, processes, and generates tree-sitter queries from nvim-treesitter.

## Query Pipeline

```
nvim-treesitter → fetch → preprocess → codegen → embedded constants
```

## Step 1: Fetching Queries

### Command

```bash
cargo xtask fetch-queries
```

### Process

1. Reads existing queries from `/queries/<language>/*.scm`
2. Parses fork source from comments:

   ```scheme
   ;; Forked from https://github.com/nvim-treesitter/nvim-treesitter/.../locals.scm
   ```

3. Downloads latest version from source URL
4. Compares with existing using S-expression parsing
5. Saves new versions with `.new` extension for review

### Key Code Location

`/xtask/src/fetch_queries.rs`

## Step 2: Query Processing

### Command

```bash
cargo xtask codegen queries
```

### Configuration

Languages defined in `syntastica-macros/languages.toml`:

```toml
[languages.rust]
group = "most"
parser.git = "https://github.com/tree-sitter/tree-sitter-rust"
parser.git.rev = "48e..."
parser.crates-io.name = "tree-sitter-rust"
parser.crates-io.version = "0.20.3"
queries.nvim_like = true
queries.injections = true
queries.locals = true
```

### Processing Steps

1. **Load Queries**: Read from `/queries/<language>/*.scm`
2. **Preprocess**: Using `syntastica-query-preprocessor`:
   - Resolve inheritance (`;; inherits: javascript`)
   - Convert Lua patterns to regex
   - Strip specified comments
   - Handle nvim-specific features
3. **Generate Versions**:
   - Standard version with all features
   - Crates.io version with git-only features removed
4. **Output**: Generate `/syntastica-queries/src/lib.rs`

### Key Code Locations

- `/xtask/src/codegen/queries.rs`
- `/syntastica-query-preprocessor/src/lib.rs`

## Step 3: Generated Output

### File Structure

```
syntastica-queries/
├── src/
│   └── lib.rs          # Generated constants
└── generated_queries/  # Processed query files
    ├── rust/
    │   ├── highlights.scm
    │   ├── injections.scm
    │   └── locals.scm
    └── ...
```

### Generated Code Format

```rust
// In syntastica-queries/src/lib.rs
pub const RUST_HIGHLIGHTS: &str = include_str!("../generated_queries/rust/highlights.scm");
pub const RUST_HIGHLIGHTS_CRATES_IO: &str = include_str!("../generated_queries/rust/highlights.crates_io.scm");
pub const RUST_INJECTIONS: &str = include_str!("../generated_queries/rust/injections.scm");
pub const RUST_LOCALS: &str = include_str!("../generated_queries/rust/locals.scm");
```

## Query Preprocessing Features

### 1. Inheritance Resolution

```scheme
;; inherits: javascript,jsx
;; This query will include all captures from javascript and jsx
```

### 2. Lua Pattern Conversion

```scheme
;; Before: ((identifier) @variable (#lua-match? @variable "^[A-Z]"))
;; After: ((identifier) @variable (#match? @variable "^[A-Z]"))
```

### 3. Comment Stripping

Removes patterns like:

- `;; Supported by ... but not by ...`
- Version-specific comments

### 4. nvim-like Processing

Special handling for Neovim-specific predicates and features.

## Comparison with breeze-rustle's Current Approach

### Current fetch-queries Script

```python
# Downloads directly via HTTP
# Filters for specific captures
# Generates runtime HashMap
# Only focuses on semantic captures
```

### Syntastica's Approach

- Maintains full query fidelity
- Supports all query types
- Compile-time embedding
- Proper preprocessing pipeline
- Version control friendly

## Integration Options for breeze-rustle

### Option 1: Direct Dependency

```toml
[dependencies]
syntastica-queries = "0.6.0"
```

```rust
use syntastica_queries::RUST_LOCALS;

let query = tree_sitter::Query::new(rust_language, RUST_LOCALS)?;
```

### Option 2: Custom Build Process

```rust
// build.rs
fn main() {
    // Fetch only locals.scm
    // Process with syntastica-query-preprocessor
    // Generate constants
}
```

### Option 3: Runtime Loading

```rust
// Use syntastica's preprocessing at runtime
use syntastica_query_preprocessor::preprocess_query;

let processed = preprocess_query(raw_query, &PreprocessorConfig {
    ignore_predicates: vec!["nvim-ts-rainbow".to_string()],
    ...
})?;
```

## Key Takeaways

1. **Query Fidelity**: Syntastica preserves the original nvim-treesitter queries with proper attribution
2. **Preprocessing**: Handles edge cases like inheritance and Lua patterns
3. **Multiple Versions**: Supports both full-featured and crates.io-compatible versions
4. **Build Integration**: Uses xtask pattern for maintainable code generation
5. **Compile-time Embedding**: Zero runtime overhead for query loading

## Recommendations

For breeze-rustle, we should:

1. **Start with syntastica-queries dependency** for immediate functionality
2. **Use their preprocessing logic** to handle query edge cases correctly
3. **Consider a custom build process** if we need to optimize for size
4. **Learn from their xtask pattern** for maintainable tooling

The syntastica approach is more robust than our current script and handles many edge cases we haven't considered.
