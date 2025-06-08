# Syntastica Features for breeze-rustle

## Executive Summary

We'll use `syntastica-queries` as a dependency to get pre-compiled, community-maintained tree-sitter queries for 50+ languages. This provides immediate access to semantic information needed for code chunking.

## Key Features We'll Use

### 1. Pre-compiled locals.scm Queries

**What they provide:**

- `@local.scope` - Marks scope boundaries (functions, classes, blocks)
- `@local.definition` - Variable/function definitions
- `@local.reference` - Variable/function references

**Example access:**

```rust
use syntastica_queries::{RUST_LOCALS, PYTHON_LOCALS, JAVASCRIPT_LOCALS};

// Direct access to query strings
let rust_query = tree_sitter::Query::new(rust_language, RUST_LOCALS)?;
```

### 2. Language Coverage

Syntastica provides queries for all major languages we need:

- **Systems**: Rust, C, C++, Go, Zig
- **Web**: JavaScript, TypeScript, JSX, TSX, HTML, CSS
- **Scripting**: Python, Ruby, Lua, Bash
- **JVM**: Java, Kotlin, Scala
- **Data**: JSON, YAML, TOML, XML
- **And 40+ more languages**

### 3. Query Processing Features

The queries are pre-processed to handle:

- **Inheritance**: Queries that inherit from other languages (e.g., TSX inherits from TypeScript)
- **Lua pattern conversion**: nvim-treesitter Lua patterns converted to regex
- **Cross-platform compatibility**: Works on all platforms without Neovim

## Implementation Strategy

### Step 1: Add Dependency

```toml
[dependencies]
syntastica-queries = "0.6.0"
tree-sitter = "0.25.6"
# Add specific language parsers as needed
tree-sitter-rust = "0.24.0"
tree-sitter-python = "0.24.0"
# etc...
```

### Step 2: Create Language Registry

```rust
use syntastica_queries::*;
use std::collections::HashMap;

pub struct LanguageRegistry {
    parsers: HashMap<&'static str, tree_sitter::Language>,
    queries: HashMap<&'static str, &'static str>,
}

impl LanguageRegistry {
    pub fn new() -> Self {
        let mut queries = HashMap::new();
        
        // Map language names to their locals queries
        queries.insert("rust", RUST_LOCALS);
        queries.insert("python", PYTHON_LOCALS);
        queries.insert("javascript", JAVASCRIPT_LOCALS);
        queries.insert("typescript", TYPESCRIPT_LOCALS);
        // ... add more as needed
        
        Self {
            parsers: HashMap::new(),
            queries,
        }
    }
}
```

### Step 3: Use for Semantic Chunking

```rust
pub fn find_semantic_boundaries(
    source: &str,
    language: &str,
) -> Result<Vec<SemanticChunk>> {
    let parser = self.get_parser(language)?;
    let query_str = self.queries.get(language)
        .ok_or("Unsupported language")?;
    
    let query = tree_sitter::Query::new(
        parser.language(),
        query_str
    )?;
    
    // Find @local.scope captures for chunk boundaries
    let scope_index = query.capture_index_for_name("local.scope")
        .ok_or("No scope captures")?;
    
    // Parse and find chunks...
}
```

## Advantages Over Custom Implementation

1. **Maintained by Community**: nvim-treesitter queries are actively maintained
2. **Edge Cases Handled**: Inheritance, special predicates, cross-language injections
3. **Immediate Availability**: No need to implement fetch/update mechanisms
4. **Tested**: Used by thousands of Neovim users daily
5. **Comprehensive**: Covers edge cases we might miss

## What We Still Need to Implement

1. **Chunk Boundary Detection**: Logic to use `@local.scope` for determining chunks
2. **Size-based Splitting**: When semantic units exceed size limits
3. **Context Preservation**: Including relevant context with chunks
4. **Metadata Extraction**: Getting function names, types, etc. from captures

## Migration Path

1. **Phase 1**: Remove `fetch-queries` script
2. **Phase 2**: Add syntastica-queries dependency
3. **Phase 3**: Update `SemanticChunker` to use syntastica queries
4. **Phase 4**: Add language detection and parser management

## Example: Using LOCALS Queries

Here's what a locals.scm query looks like (Rust example):

```scheme
; Scopes
(function_item) @local.scope
(impl_item) @local.scope
(block) @local.scope

; Definitions
(parameter (identifier) @local.definition)
(let_declaration (identifier) @local.definition)

; References
(identifier) @local.reference
```

We can use these captures to:

- Find function/class boundaries (`@local.scope`)
- Track variable definitions
- Understand code structure

## Performance Considerations

- Queries are embedded at compile time (zero runtime overhead)
- No file I/O needed for query loading
- Parser initialization can be cached
- Query compilation can be cached per language

## Next Steps

1. Add syntastica-queries to Cargo.toml
2. Create a simple proof-of-concept using Rust locals
3. Design the chunk boundary detection algorithm
4. Implement the full SemanticChunker using syntastica queries
