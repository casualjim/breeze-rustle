# Syntastica Research Notes

## Overview

Syntastica is a modern syntax highlighting library using tree-sitter, designed as a replacement for syntect. It provides a comprehensive solution for tree-sitter-based syntax highlighting with support for 50+ languages.

**Repository**: <https://github.com/RubixDev/syntastica>  
**License**: MPL-2.0  
**Version**: 0.6.0 (as of research date)

## Architecture

### Workspace Structure

```
syntastica/
├── syntastica/              # Main library interface
├── syntastica-core/         # Shared types and traits
├── syntastica-highlight/    # Highlighting engine (fork of tree-sitter-highlight)
├── syntastica-queries/      # Pre-compiled query collection
├── syntastica-parsers*/     # Various parser collections
├── syntastica-themes/       # Theme collection
├── syntastica-query-preprocessor/  # Query processing
└── supporting crates...
```

### Key Components for breeze-rustle

1. **syntastica-queries**
   - Contains pre-compiled tree-sitter queries for all languages
   - Queries are embedded at compile time using `include_str!()`
   - Three query types per language:
     - `highlights.scm` - Syntax highlighting
     - `injections.scm` - Language injection points
     - `locals.scm` - Scope tracking (most relevant for chunking!)

2. **syntastica-query-preprocessor**
   - Processes nvim-treesitter format queries
   - Converts Lua patterns to regex
   - Handles query inheritance
   - Could be used directly or as reference implementation

## Query System

### Query Sources

- Forked from nvim-treesitter project
- Located in `/queries/<language>/*.scm`
- Generated into `/syntastica-queries/generated_queries/`
- Updated via `cargo xtask codegen queries`

### locals.scm Format

The `locals.scm` queries define semantic information perfect for code chunking:

```scheme
; Example locals.scm patterns
@local.scope       ; Defines scope boundaries (functions, classes, blocks)
@local.definition  ; Variable/function definitions
@local.reference   ; Variable/function references
```

### Supported Languages

Over 50 languages including:

- Major languages: Rust, Python, JavaScript, TypeScript, Go, Java, C/C++
- Web: HTML, CSS, JSX/TSX
- Config: TOML, YAML, JSON
- Scripting: Bash, Fish, Lua
- And many more...

## Integration Options for breeze-rustle

### Option 1: Direct Dependency

```toml
[dependencies]
syntastica-queries = "0.6.0"
```

**Pros:**

- All queries pre-compiled and ready
- No maintenance burden
- Consistent with upstream nvim-treesitter

**Cons:**

- Includes all query types (we only need locals.scm)
- Extra dependency weight

### Option 2: Fork Query Processing

Adapt syntastica's approach:

1. Use their fetch-queries script pattern
2. Process only locals.scm files
3. Embed at compile time

**Pros:**

- Minimal dependency footprint
- Customized for our needs
- Learn from their implementation

**Cons:**

- Need to maintain query updates
- More implementation work

### Option 3: Use syntastica-query-preprocessor

```toml
[dependencies]
syntastica-query-preprocessor = "0.6.0"
```

**Pros:**

- Handles nvim-treesitter format conversion
- Lua pattern support
- Query inheritance

**Cons:**

- May include features we don't need

## Key Implementation Insights

### 1. Query Processing Pipeline

```
nvim-treesitter queries → preprocessor → regex conversion → embedded strings
```

### 2. Language Detection

- Uses file extensions and content-based detection
- Maps to tree-sitter language names

### 3. Parser Management

- Caches parsers using DashMap
- Supports multiple parser sources (crates.io, git, dynamic)

### 4. Build Process

- `cargo xtask` pattern for code generation
- Fetches queries from nvim-treesitter
- Generates Rust constants at build time

## Recommendations for breeze-rustle

1. **Start with syntastica-queries dependency**
   - Quick to implement
   - Battle-tested queries
   - Can optimize later if needed

2. **Focus on locals.scm queries**
   - These define semantic boundaries
   - Perfect for chunk detection
   - Already maintained by community

3. **Consider query preprocessor**
   - If we need custom query modifications
   - Handles edge cases and compatibility

4. **Architecture patterns to adopt:**
   - Parser caching strategy
   - Language detection approach
   - Query compilation patterns

## Code Examples

### Using syntastica-queries

```rust
use syntastica_queries::{RUST_LOCALS, PYTHON_LOCALS, JAVASCRIPT_LOCALS};

// Access pre-compiled queries
let rust_locals_query = RUST_LOCALS;
```

### Query structure they use

```rust
// In syntastica-queries/src/lib.rs
pub const RUST_HIGHLIGHTS: &str = include_str!("../generated_queries/rust/highlights.scm");
pub const RUST_INJECTIONS: &str = include_str!("../generated_queries/rust/injections.scm");
pub const RUST_LOCALS: &str = include_str!("../generated_queries/rust/locals.scm");
```

## Next Steps

1. Evaluate dependency size of syntastica-queries
2. Test locals.scm queries for chunk boundary detection
3. Prototype integration with our chunking algorithm
4. Decide on direct dependency vs adapted approach

## References

- [Syntastica Documentation](https://rubixdev.github.io/syntastica/)
- [nvim-treesitter queries](https://github.com/nvim-treesitter/nvim-treesitter/tree/master/queries)
- [Tree-sitter Documentation](https://tree-sitter.github.io/)
