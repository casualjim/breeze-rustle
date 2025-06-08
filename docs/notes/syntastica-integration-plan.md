# Syntastica Integration Plan for breeze-rustle

## Overview

This document outlines the concrete steps to integrate syntastica-queries into breeze-rustle, replacing our current approach with a more robust solution.

## Integration Approach

We'll use syntastica-parsers and syntastica-queries as dependencies to get both tree-sitter parsers and pre-compiled queries for semantic code analysis.

## Implementation Steps

### Phase 1: Setup Dependencies

**Update Cargo.toml:**

```toml
[dependencies]
# Parser and query management
syntastica-parsers = { version = "0.6.0", features = ["all"] }
syntastica-queries = "0.6.0"

# Async runtime (already have)
smol = "2.0.2"
flume = "0.11.1"

# Python bindings (already have)
pyo3 = "0.25.0"

# Tree-sitter core
tree-sitter = "0.25.6"

# Optional: for query preprocessing if needed later
# syntastica-query-preprocessor = "0.6.0"
```

Note: Using `"all"` provides parsers for 50+ languages including less common ones like bibtex, dockerfile, fish, gleam, julia, latex, ocaml, and more. This aligns with the original goal of supporting 100+ languages.

### Phase 2: Create Core Types and Errors

**src/types.rs:**

```rust
use pyo3::prelude::*;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ChunkError {
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),
    
    #[error("Failed to parse content: {0}")]
    ParseError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Query error: {0}")]
    QueryError(String),
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    #[pyo3(get)]
    pub node_type: String,        // "function", "class", "method"
    #[pyo3(get)]
    pub node_name: Option<String>, // "parse_document", "MyClass"
    #[pyo3(get)]
    pub language: String,
    #[pyo3(get)]
    pub parent_context: Option<String>, // "class MyClass" for methods
    #[pyo3(get)]
    pub scope_path: Vec<String>,   // ["module", "MyClass", "parse_document"]
    #[pyo3(get)]
    pub definitions: Vec<String>,  // Variable/function names defined
    #[pyo3(get)]
    pub references: Vec<String>,   // Variable/function names referenced
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct SemanticChunk {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub start_byte: usize,
    #[pyo3(get)]
    pub end_byte: usize,
    #[pyo3(get)]
    pub start_line: usize,
    #[pyo3(get)]
    pub end_line: usize,
    #[pyo3(get)]
    pub metadata: ChunkMetadata,
}

impl SemanticChunk {
    /// Calculate line numbers from byte offsets
    pub fn from_node(node: &Node, source: &str, metadata: ChunkMetadata) -> Self {
        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        let text = source[start_byte..end_byte].to_string();
        
        // Calculate line numbers
        let start_line = source[..start_byte].matches('\n').count() + 1;
        let end_line = source[..end_byte].matches('\n').count() + 1;
        
        Self {
            text,
            start_byte,
            end_byte,
            start_line,
            end_line,
            metadata,
        }
    }
}
```

### Phase 3: Language Registry

**src/languages.rs:**

```rust
use syntastica_queries::*;
use syntastica_parsers::{Lang, LANGUAGES};
use tree_sitter::Language;
use std::collections::HashMap;
use once_cell::sync::Lazy;

pub struct LanguageConfig {
    pub language: Language,
    pub locals_query: &'static str,
    pub highlights_query: &'static str,
    pub injections_query: &'static str,
}

// Build a complete language registry for all supported languages
static LANGUAGE_CONFIGS: Lazy<HashMap<Lang, LanguageConfig>> = Lazy::new(|| {
    use Lang::*;
    
    let mut configs = HashMap::new();
    
    // Syntastica provides all these with the "all" feature
    configs.insert(Rust, LanguageConfig {
        language: syntastica_parsers::rust(),
        locals_query: RUST_LOCALS,
        highlights_query: RUST_HIGHLIGHTS,
        injections_query: RUST_INJECTIONS,
    });
    
    configs.insert(Python, LanguageConfig {
        language: syntastica_parsers::python(),
        locals_query: PYTHON_LOCALS,
        highlights_query: PYTHON_HIGHLIGHTS,
        injections_query: PYTHON_INJECTIONS,
    });
    
    configs.insert(Javascript, LanguageConfig {
        language: syntastica_parsers::javascript(),
        locals_query: JAVASCRIPT_LOCALS,
        highlights_query: JAVASCRIPT_HIGHLIGHTS,
        injections_query: JAVASCRIPT_INJECTIONS,
    });
    
    // ... Continue for all languages in LANGUAGES array
    // This can be generated with a macro if needed
    
    configs
});

pub fn get_language_config(lang: Lang) -> Option<&'static LanguageConfig> {
    LANGUAGE_CONFIGS.get(&lang)
}

pub fn supported_languages() -> Vec<&'static str> {
    LANGUAGES.iter()
        .map(|lang| lang.name())
        .collect()
}

pub fn detect_language(path: &str) -> Option<Lang> {
    // Syntastica provides file type detection
    let file_type = syntastica_core::language_set::FileType::from_path(path)?;
    Lang::from_file_type(file_type)
}
```

### Phase 4: Query Processing

**src/query_utils.rs:**

```rust
use tree_sitter::{Query, QueryCursor, Node, Tree};
use crate::types::*;

pub struct QueryProcessor {
    query: Query,
    scope_capture_index: Option<u32>,
    definition_capture_index: Option<u32>,
    reference_capture_index: Option<u32>,
}

impl QueryProcessor {
    pub fn new(language: Language, query_str: &str) -> Result<Self> {
        let query = Query::new(&language, query_str)?;
        
        Ok(Self {
            scope_capture_index: query.capture_index_for_name("local.scope"),
            definition_capture_index: query.capture_index_for_name("local.definition"),
            reference_capture_index: query.capture_index_for_name("local.reference"),
            query,
        })
    }
    
    pub fn find_scopes<'a>(&self, tree: &'a Tree, source: &'a [u8]) -> Vec<Node<'a>> {
        let mut cursor = QueryCursor::new();
        let mut scopes = Vec::new();
        
        if let Some(scope_idx) = self.scope_capture_index {
            for match_ in cursor.matches(&self.query, tree.root_node(), source) {
                for capture in match_.captures {
                    if capture.index == scope_idx {
                        scopes.push(capture.node);
                    }
                }
            }
        }
        
        scopes
    }
}
```

### Phase 5: Chunking Algorithm

**src/chunker.rs:**

```rust
use crate::{languages::*, query_utils::*, types::*};
use dashmap::DashMap;
use tree_sitter::{Parser, Tree, Node};
use std::sync::Arc;

// Internal implementation that works with language strings
pub struct InnerChunker {
    parser_cache: Arc<DashMap<String, Parser>>,
    query_cache: Arc<DashMap<String, QueryProcessor>>,
    max_chunk_size: usize,
}

impl InnerChunker {
    pub fn new(max_chunk_size: usize) -> Self {
        Self {
            parser_cache: Arc::new(DashMap::new()),
            query_cache: Arc::new(DashMap::new()),
            max_chunk_size,
        }
    }

    pub async fn chunk_file(
        &self, 
        content: &str, 
        language: &str, 
        file_path: Option<&str>
    ) -> Result<Vec<SemanticChunk>, ChunkError> {
        // Map language string to syntastica Lang enum internally
        let lang = language_from_string(language)
            .ok_or_else(|| ChunkError::UnsupportedLanguage(language.to_string()))?;
        
        let config = get_language_config(lang)
            .ok_or_else(|| ChunkError::UnsupportedLanguage(language.to_string()))?;
        
        // Get or create parser (thread-safe with DashMap)
        let mut parser = self.parser_cache
            .entry(language.to_string())
            .or_insert_with(|| {
                let mut parser = Parser::new();
                parser.set_language(&config.language)
                    .expect("Failed to set language");
                parser
            })
            .clone();
        
        // Parse the source
        let tree = parser.parse(content, None)
            .ok_or("Failed to parse content")?;
        
        // Get or create query processor
        let processor = self.query_cache
            .entry(language.to_string())
            .or_insert_with(|| {
                QueryProcessor::new(config.language.clone(), config.locals_query)
                    .expect("Failed to create query processor")
            })
            .clone();
        
        // Find semantic boundaries
        let scopes = processor.find_scopes(&tree, content.as_bytes());
        
        // Convert scopes to chunks
        self.create_chunks(content, scopes, language, file_path)
    }
    
    fn create_chunks(
        &self, 
        source: &str, 
        scopes: Vec<Node>, 
        language: &str,
        file_path: Option<&str>
    ) -> Result<Vec<SemanticChunk>, Box<dyn std::error::Error + Send + Sync>> {
        let mut chunks = Vec::new();
        
        for scope in scopes {
            let metadata = self.extract_metadata(&scope, source, language)?;
            let chunk = SemanticChunk::from_node(&scope, source, metadata);
            
            // Check if chunk needs splitting
            if chunk.text.len() > self.max_chunk_size {
                // TODO: Implement intelligent splitting
                chunks.push(chunk);
            } else {
                chunks.push(chunk);
            }
        }
        
        Ok(chunks)
    }
    
    fn extract_metadata(
        &self,
        node: &Node,
        source: &str,
        language: &str
    ) -> Result<ChunkMetadata, Box<dyn std::error::Error + Send + Sync>> {
        // Extract node name
        let node_name = self.extract_node_name(node, source);
        
        // Extract parent context
        let parent_context = self.extract_parent_context(node, source);
        
        // Build scope path
        let scope_path = self.build_scope_path(node, source);
        
        Ok(ChunkMetadata {
            node_type: node.kind().to_string(),
            node_name,
            language: language.to_string(),
            parent_context,
            scope_path,
            definitions: vec![], // TODO: Extract from query captures
            references: vec![],  // TODO: Extract from query captures
        })
    }
    
    fn extract_node_name(&self, node: &Node, source: &str) -> Option<String> {
        // Look for identifier child nodes
        for child in node.children(&mut node.walk()) {
            if child.kind() == "identifier" || child.kind() == "name" {
                return Some(source[child.byte_range()].to_string());
            }
        }
        None
    }
    
    fn extract_parent_context(&self, node: &Node, source: &str) -> Option<String> {
        let parent = node.parent()?;
        match parent.kind() {
            "class_definition" | "class_declaration" => {
                let name = self.extract_node_name(&parent, source)?;
                Some(format!("class {}", name))
            }
            "function_definition" | "function_declaration" | "method_definition" => {
                let name = self.extract_node_name(&parent, source)?;
                Some(format!("def {}", name))
            }
            "impl_item" => {
                Some("impl".to_string())
            }
            _ => None
        }
    }
    
    fn build_scope_path(&self, node: &Node, source: &str) -> Vec<String> {
        let mut path = vec!["module".to_string()];
        let mut current = Some(*node);
        
        while let Some(n) = current {
            if let Some(name) = self.extract_node_name(&n, source) {
                path.push(name);
            }
            current = n.parent();
        }
        
        path.reverse();
        path
    }
}

// Helper function to map language strings to syntastica Lang enum
fn language_from_string(language: &str) -> Option<syntastica_parsers::Lang> {
    use syntastica_parsers::Lang;
    
    // This would be generated or use a match statement for all languages
    match language.to_lowercase().as_str() {
        "rust" => Some(Lang::Rust),
        "python" => Some(Lang::Python),
        "javascript" | "js" => Some(Lang::Javascript),
        "typescript" | "ts" => Some(Lang::Typescript),
        // ... add all supported languages
        _ => None
    }
}
```

### Phase 6: Python Bindings

**src/lib.rs:**

```rust
use pyo3::prelude::*;
use pyo3_asyncio;
use std::sync::Arc;

#[pyclass]
pub struct SemanticChunker {
    inner: Arc<InnerChunker>,
    executor: Arc<smol::Executor<'static>>,
}

#[pymethods]
impl SemanticChunker {
    #[new]
    fn new(max_chunk_size: Option<usize>) -> PyResult<Self> {
        let max_chunk_size = max_chunk_size.unwrap_or(16384);
        Ok(Self {
            inner: Arc::new(InnerChunker::new(max_chunk_size)),
            executor: Arc::new(smol::Executor::new()),
        })
    }
    
    /// Chunk a single file into semantic units
    #[pyo3(signature = (content, language, file_path=None))]
    fn chunk_file<'p>(
        &self,
        py: Python<'p>,
        content: String,
        language: String,
        file_path: Option<String>,
    ) -> PyResult<&'p PyAny> {
        let chunker = self.inner.clone();
        pyo3_asyncio::smol::future_into_py(py, async move {
            chunker.chunk_file(&content, &language, file_path.as_deref())
                .await
                .map_err(|e| match e {
                    ChunkError::UnsupportedLanguage(lang) => {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Unsupported language: {}", lang)
                        )
                    }
                    other => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(other.to_string())
                })
        })
    }
    
    /// Chunk multiple files concurrently
    #[pyo3(signature = (files))]
    fn chunk_files<'p>(
        &self,
        py: Python<'p>,
        files: Vec<(String, String, String)>, // (content, language, path)
    ) -> PyResult<&'p PyAny> {
        let chunker = self.inner.clone();
        let executor = self.executor.clone();
        
        pyo3_asyncio::smol::future_into_py(py, async move {
            use futures_lite::StreamExt;
            
            // Create channel for work distribution
            let (tx, rx) = flume::bounded::<(usize, String, String, String)>(files.len());
            
            // Send work items
            for (idx, (content, lang, path)) in files.into_iter().enumerate() {
                tx.send_async((idx, content, lang, path)).await
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            }
            drop(tx);
            
            // Process concurrently with controlled parallelism
            let mut results = vec![Vec::new(); rx.len()];
            let stream = rx.into_stream();
            
            stream
                .for_each_concurrent(num_cpus::get(), |(idx, content, lang, path)| {
                    let chunker = chunker.clone();
                    async move {
                        match chunker.chunk_file(&content, &lang, Some(&path)).await {
                            Ok(chunks) => {
                                results[idx] = chunks;
                            }
                            Err(e) => {
                                eprintln!("Failed to chunk {}: {}", path, e);
                            }
                        }
                    }
                })
                .await;
            
            Ok(results)
        })
    }
    
    /// Get supported languages
    #[staticmethod]
    fn supported_languages() -> Vec<String> {
        supported_languages().into_iter()
            .map(|s| s.to_string())
            .collect()
    }
    
    /// Check if a language is supported
    #[staticmethod]
    fn is_language_supported(language: &str) -> bool {
        supported_languages().contains(&language)
    }
}

#[pymodule]
fn breeze_rustle(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    // Initialize logging bridge
    pyo3_log::init();
    
    m.add_class::<SemanticChunker>()?;
    m.add_class::<SemanticChunk>()?;
    m.add_class::<ChunkMetadata>()?;
    
    // Add version info
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}
```

## Benefits of This Approach

1. **Comprehensive Language Support**: 50+ languages included by default
2. **Community Maintained**: Queries updated by nvim-treesitter community
3. **Robust Query Processing**: Handles inheritance, Lua patterns, etc.
4. **Zero Runtime Overhead**: Queries and parsers compiled into binary
5. **Type Safety**: Strongly typed throughout
6. **Simple Dependencies**: Single `syntastica-parsers` crate manages everything
7. **Future Proof**: New languages automatically available when syntastica updates

## Accepted Trade-offs

1. **Binary Size**: ~10-20MB for all parsers (acceptable for comprehensive support)
2. **Compile Time**: One-time cost during CI/CD builds
3. **Memory**: Negligible overhead from parser availability

## Distribution Strategy

Since we're publishing as a Python wheel:
- Users download pre-built binaries, never compile
- We build once in CI/CD with all languages included
- No need for feature flags or variants
- Single wheel supports all 50+ languages
- "Just works" experience for end users

## Python Type Stubs

**python/breeze_rustle/__init__.pyi:**

```python
from typing import List, Optional, Tuple

class ChunkMetadata:
    node_type: str
    node_name: Optional[str]
    language: str
    parent_context: Optional[str]
    scope_path: List[str]
    definitions: List[str]
    references: List[str]

class SemanticChunk:
    text: str
    start_byte: int
    end_byte: int
    start_line: int
    end_line: int
    metadata: ChunkMetadata

class SemanticChunker:
    def __init__(self, max_chunk_size: int = 16384) -> None: ...
    
    async def chunk_file(
        self,
        content: str,
        language: str,
        file_path: Optional[str] = None
    ) -> List[SemanticChunk]: ...
    
    async def chunk_files(
        self,
        files: List[Tuple[str, str, str]]  # [(content, language, path), ...]
    ) -> List[List[SemanticChunk]]: ...
    
    @staticmethod
    def supported_languages() -> List[str]: ...
    
    @staticmethod
    def is_language_supported(language: str) -> bool: ...

__version__: str
```

## Migration Checklist

- [ ] Remove tools/fetch-queries script
- [ ] Add syntastica-parsers with `"all"` feature
- [ ] Add syntastica-queries dependency
- [ ] Remove individual tree-sitter-* dependencies
- [ ] Implement core types with proper PyO3 annotations
- [ ] Create language registry using syntastica APIs internally
- [ ] Implement query processing for all query types (locals, highlights, textobjects)
- [ ] Build chunking algorithm with proper error handling
- [ ] Add Python bindings matching original API
- [ ] Port comprehensive test suite from original plan
- [ ] Create Python type stubs
- [ ] Add build.rs for validation
- [ ] Update documentation

## Testing Strategy

1. **Unit Tests**: Test each component in isolation
2. **Integration Tests**: Test full chunking pipeline
3. **Language Tests**: Ensure each language chunks correctly
4. **Performance Tests**: Benchmark against requirements
5. **Python Tests**: Test PyO3 bindings

## Performance Optimization

1. **Parser Caching**: Reuse parsers across files
2. **Query Caching**: Compile queries once per language
3. **Concurrent Processing**: Use smol for parallel file processing
4. **Zero-Copy**: Use string slices where possible

## Error Handling Strategy

The implementation uses specific error types to enable proper fallback behavior:

1. **UnsupportedLanguage**: Returns `PyValueError` - allows Python to catch and fallback to character-based chunking
2. **ParseError**: Returns `PyRuntimeError` - for malformed code in supported languages
3. **Other Errors**: Returns `PyRuntimeError` - for IO errors, query errors, etc.

This allows the Python server to implement a clean fallback pattern:

```python
from breeze_rustle import SemanticChunker

async def chunk_with_fallback(content: str, language: str, path: str):
    chunker = SemanticChunker()
    try:
        # Try semantic chunking first
        chunks = await chunker.chunk_file(content, language, path)
        return chunks
    except ValueError as e:
        if "Unsupported language" in str(e):
            # Fall back to character-based chunking
            return character_based_chunking(content)
        raise
```

## Future Enhancements

1. **Custom Queries**: Allow users to provide custom locals.scm
2. **Query Composition**: Combine multiple query types
3. **Streaming**: Support chunking large files incrementally
4. **Language Detection**: Improve detection beyond file extensions

## Conclusion

This integration plan leverages syntastica's robust parser and query infrastructure while maintaining the original breeze-rustle API design. The error handling strategy ensures graceful degradation for unsupported languages, allowing the Python server to seamlessly fall back to character-based chunking when semantic chunking is not available.
