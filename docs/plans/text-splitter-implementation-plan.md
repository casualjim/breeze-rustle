# breeze-rustle Text-Splitter Implementation Plan

## Overview

`breeze-rustle` is a Rust library that provides high-performance semantic code chunking by leveraging the `text-splitter` crate with tree-sitter parsers. It adds rich metadata extraction and Python bindings for use in RAG applications.

## Project Goals

1. **Fast semantic chunking** - Use text-splitter's proven CodeSplitter implementation
2. **Multi-language support** - Support 17+ languages via direct tree-sitter crates
3. **Rich metadata** - Extract node types, names, scopes, and context from code
4. **Python bindings** - Clean async API for Python consumers
5. **Hyperpolyglot compatibility** - Work seamlessly with hyperpolyglot language detection

## Architecture

```
breeze-rustle/
├── Cargo.toml          # Rust dependencies including text-splitter
├── pyproject.toml      # Python packaging with maturin
├── src/
│   ├── lib.rs          # PyO3 Python bindings and module definition
│   ├── types.rs        # Core types (SemanticChunk, ChunkMetadata)
│   ├── languages.rs    # Language registry mapping names to parsers
│   ├── chunker.rs      # Enhanced chunker wrapping text-splitter
│   └── metadata.rs     # Metadata extraction from AST nodes
├── python/
│   └── breeze_rustle/
│       ├── __init__.py
│       └── __init__.pyi  # Type stubs
└── tests/
    ├── test_chunking.rs
    └── test_python.py
```

## Core Components

### 1. Language Registry

Maps hyperpolyglot language names to tree-sitter parsers:

```rust
// src/languages.rs
use std::collections::HashMap;
use tree_sitter::Language;
use once_cell::sync::Lazy;

pub static LANGUAGE_REGISTRY: Lazy<HashMap<&'static str, Language>> = Lazy::new(|| {
    let mut registry = HashMap::new();
    
    // Map standard names to parsers
    registry.insert("Python", tree_sitter_python::language());
    registry.insert("JavaScript", tree_sitter_javascript::language());
    registry.insert("TypeScript", tree_sitter_typescript::language_typescript());
    registry.insert("TSX", tree_sitter_typescript::language_tsx());
    registry.insert("Java", tree_sitter_java::language());
    registry.insert("C++", tree_sitter_cpp::language());
    registry.insert("C", tree_sitter_c::language());
    registry.insert("C#", tree_sitter_c_sharp::language());
    registry.insert("Go", tree_sitter_go::language());
    registry.insert("Rust", tree_sitter_rust::language());
    registry.insert("Ruby", tree_sitter_ruby::language());
    registry.insert("PHP", tree_sitter_php::language());
    registry.insert("Swift", tree_sitter_swift::language());
    registry.insert("Kotlin", tree_sitter_kotlin::language());
    registry.insert("Scala", tree_sitter_scala::language());
    registry.insert("SQL", tree_sitter_sql::language());
    registry.insert("Shell", tree_sitter_bash::language());
    registry.insert("R", tree_sitter_r::language());
    
    registry
});

pub fn get_language(name: &str) -> Option<Language> {
    LANGUAGE_REGISTRY.get(name).copied()
}

pub fn supported_languages() -> Vec<&'static str> {
    LANGUAGE_REGISTRY.keys().copied().collect()
}
```

### 2. Core Types

```rust
// src/types.rs
use pyo3::prelude::*;

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
```

### 3. Enhanced Chunker

Wraps text-splitter's CodeSplitter and adds metadata extraction:

```rust
// src/chunker.rs
use text_splitter::{ChunkConfig, CodeSplitter, ChunkSizer};
use crate::{languages::*, types::*, metadata::*};

pub struct BreezeChunker {
    chunk_config: ChunkConfig<Box<dyn ChunkSizer>>,
}

impl BreezeChunker {
    pub fn new(max_chunk_size: usize, sizer: Box<dyn ChunkSizer>) -> Self {
        let chunk_config = ChunkConfig::new(max_chunk_size)
            .with_sizer(sizer)
            .with_trim(false);
        
        Self { chunk_config }
    }
    
    pub async fn chunk_file(
        &self,
        content: &str,
        language: &str,
        file_path: Option<&str>,
    ) -> Result<Vec<SemanticChunk>, ChunkError> {
        // Get tree-sitter language
        let ts_language = get_language(language)
            .ok_or_else(|| ChunkError::UnsupportedLanguage(language.to_string()))?;
        
        // Create CodeSplitter
        let splitter = CodeSplitter::new(ts_language, self.chunk_config.clone())?;
        
        // Get base chunks with indices
        let chunks: Vec<_> = splitter.chunk_indices(content).collect();
        
        // Parse once for metadata extraction
        let mut parser = Parser::new();
        parser.set_language(&ts_language)?;
        let tree = parser.parse(content, None)
            .ok_or_else(|| ChunkError::ParseError("Failed to parse".into()))?;
        
        // Enrich chunks with metadata
        let mut enriched_chunks = Vec::new();
        for (offset, chunk_text) in chunks {
            let metadata = extract_metadata_for_chunk(
                &tree,
                content,
                offset,
                chunk_text.len(),
                language,
            )?;
            
            let chunk = SemanticChunk {
                text: chunk_text.to_string(),
                start_byte: offset,
                end_byte: offset + chunk_text.len(),
                start_line: content[..offset].matches('\n').count() + 1,
                end_line: content[..offset + chunk_text.len()].matches('\n').count() + 1,
                metadata,
            };
            
            enriched_chunks.push(chunk);
        }
        
        Ok(enriched_chunks)
    }
}
```

### 4. Python API

```rust
// src/lib.rs
use pyo3::prelude::*;
use pyo3_asyncio;

#[pyclass]
pub struct SemanticChunker {
    inner: Arc<BreezeChunker>,
}

#[pymethods]
impl SemanticChunker {
    #[new]
    #[pyo3(signature = (max_chunk_size=None, tokenizer=None))]
    fn new(max_chunk_size: Option<usize>, tokenizer: Option<String>) -> PyResult<Self> {
        let max_chunk_size = max_chunk_size.unwrap_or(1500);
        
        // Choose sizer based on tokenizer parameter
        let sizer: Box<dyn ChunkSizer> = match tokenizer.as_deref() {
            Some("tiktoken") => {
                Box::new(tiktoken_rs::cl100k_base()?)
            }
            Some("characters") | None => {
                Box::new(text_splitter::Characters)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "Unsupported tokenizer"
                ));
            }
        };
        
        Ok(Self {
            inner: Arc::new(BreezeChunker::new(max_chunk_size, sizer)),
        })
    }
    
    #[pyo3(signature = (content, language, file_path=None))]
    fn chunk_file<'p>(
        &self,
        py: Python<'p>,
        content: String,
        language: String,
        file_path: Option<String>,
    ) -> PyResult<&'p PyAny> {
        let chunker = self.inner.clone();
        pyo3_asyncio::tokio::future_into_py(py, async move {
            chunker.chunk_file(&content, &language, file_path.as_deref())
                .await
                .map_err(|e| match e {
                    ChunkError::UnsupportedLanguage(lang) => {
                        PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            format!("Unsupported language: {}", lang)
                        )
                    }
                    other => PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                        other.to_string()
                    )
                })
        })
    }
    
    #[staticmethod]
    fn supported_languages() -> Vec<&'static str> {
        supported_languages()
    }
    
    #[staticmethod]
    fn is_language_supported(language: &str) -> bool {
        get_language(language).is_some()
    }
}

#[pymodule]
fn breeze_rustle(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    pyo3_log::init();
    
    m.add_class::<SemanticChunker>()?;
    m.add_class::<SemanticChunk>()?;
    m.add_class::<ChunkMetadata>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    Ok(())
}
```

## Dependencies

```toml
[dependencies]
# Text splitting
text-splitter = { version = "0.27", features = ["code", "tiktoken-rs", "tokenizers"] }

# Tokenization
tokenizers = { version = "0.21", features = ["http"] }
tiktoken-rs = "0.7"

# Tree-sitter parsers
tree-sitter = "0.25"
tree-sitter-python = "0.23"
tree-sitter-javascript = "0.23"
tree-sitter-typescript = "0.23"
tree-sitter-java = "0.23"
tree-sitter-cpp = "0.24"
tree-sitter-c = "0.23"
tree-sitter-c-sharp = "0.22"
tree-sitter-go = "0.23"
tree-sitter-rust = "0.24"
tree-sitter-ruby = "0.23"
tree-sitter-php = "0.23"
tree-sitter-swift = "0.23"
tree-sitter-kotlin = "0.23"
tree-sitter-scala = "0.23"
tree-sitter-sql = "0.3"
tree-sitter-bash = "0.23"
tree-sitter-r = "0.23"

# Python bindings
pyo3 = { version = "0.23", features = ["extension-module"] }
pyo3-asyncio = { version = "0.23", features = ["tokio-runtime"] }
pyo3-log = "0.12"

# Utils
once_cell = "1.20"
thiserror = "2.0"
log = "0.4"

[dev-dependencies]
tokio = { version = "1", features = ["full"] }
pytest-runner = "6.0"
```

## Acceptance Criteria

### 1. Core Functionality
- [x] Uses text-splitter's CodeSplitter for chunking
- [x] Supports 17+ languages via direct tree-sitter crates
- [x] Maps hyperpolyglot language names correctly
- [x] Extracts rich metadata from chunks

### 2. Metadata Extraction
- [ ] Extract node_type from tree-sitter AST
- [ ] Extract node_name from identifier nodes
- [ ] Build scope_path by traversing parent nodes
- [ ] Extract parent_context (e.g., class name for methods)
- [ ] Basic definitions/references extraction

### 3. Performance
- [ ] Parse 1MB file in <100ms
- [ ] Minimal memory allocations
- [ ] Efficient chunk size calculation

### 4. Error Handling
- [ ] Return empty Vec for unsupported languages
- [ ] Never panic on malformed code
- [ ] Clear error types for Python to handle

### 5. Python Integration
- [ ] Published as wheel on PyPI
- [ ] No Rust toolchain required for users
- [ ] Type stubs for IDE support
- [ ] Works on Linux, macOS, Windows

## Key Differences from Original Plans

1. **No Query Integration**: text-splitter handles semantic boundaries
2. **No build.rs**: No need to fetch queries at compile time
3. **Direct Dependencies**: Individual tree-sitter-* crates instead of syntastica
4. **Simpler Architecture**: Leverage text-splitter's proven implementation
5. **Flexible Tokenization**: Support characters, tiktoken, or HF tokenizers

## Implementation Phases

### Phase 1: Core Implementation (MVP)
- Set up text-splitter integration
- Create language registry for top 10 languages
- Basic metadata extraction (node type and name)
- Python bindings with async support

### Phase 2: Enhanced Metadata
- Scope path extraction
- Parent context tracking
- Basic definitions/references (if feasible without queries)

### Phase 3: Full Language Support
- Add remaining tree-sitter language crates
- Comprehensive testing across languages
- Performance optimization

### Phase 4: Production Ready
- Python packaging and distribution
- Documentation and examples
- Integration tests with hyperpolyglot

## Testing Strategy

```python
# tests/test_acceptance.py
import pytest
from breeze_rustle import SemanticChunker, SemanticChunk

class TestBasicFunctionality:
    def test_supported_languages(self):
        """Should support at least 17 languages"""
        languages = SemanticChunker.supported_languages()
        assert len(languages) >= 17
        assert "Python" in languages
        assert "Rust" in languages
        assert "JavaScript" in languages
    
    async def test_simple_python_file(self):
        """Should chunk a simple Python file correctly"""
        content = '''
def hello(name):
    message = f"Hello {name}"
    return message

class Greeter:
    def __init__(self, lang):
        self.lang = lang
    
    def greet(self, name):
        return hello(name)
'''
        chunker = SemanticChunker()
        chunks = await chunker.chunk_file(content, "Python")
        
        assert len(chunks) > 0
        
        # Check metadata
        for chunk in chunks:
            assert chunk.metadata.language == "Python"
            assert chunk.metadata.node_type is not None
    
    async def test_hyperpolyglot_names(self):
        """Should work with hyperpolyglot language names"""
        test_cases = [
            ("Python", "def test(): pass"),
            ("C++", "int main() { return 0; }"),
            ("C#", "class Program { }"),
        ]
        
        chunker = SemanticChunker()
        for lang, code in test_cases:
            chunks = await chunker.chunk_file(code, lang)
            assert len(chunks) > 0
```

## Success Metrics

1. All acceptance tests pass
2. <100ms parsing for 1MB files
3. Zero panics or segfaults
4. Works with hyperpolyglot detection
5. Clean Python API with type hints

## Migration Notes

This plan supersedes:
- `docs/plans/breeze-rustle-implementation.md`
- `docs/notes/syntastica-integration-plan.md`

The text-splitter approach provides a simpler, more maintainable solution while meeting all core requirements.