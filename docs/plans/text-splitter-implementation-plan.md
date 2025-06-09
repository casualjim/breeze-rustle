# breeze-rustle Text-Splitter Implementation Plan

**Note: This plan has been largely implemented. See context.md for current status.**

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
use pyo3_async_runtimes;

// Tokenizer enum for type safety
#[pyclass(eq)]
#[derive(Clone, Debug, PartialEq)]
pub enum TokenizerType {
    #[pyo3(name = "CHARACTERS")]
    Characters,
    #[pyo3(name = "TIKTOKEN")]
    Tiktoken,
    #[pyo3(name = "HUGGINGFACE")]
    HuggingFace,
}

#[pyclass]
pub struct SemanticChunker {
    inner: Arc<InnerChunker>,
}

#[pymethods]
impl SemanticChunker {
    #[new]
    #[pyo3(signature = (max_chunk_size=None, tokenizer=None, hf_model=None))]
    fn new(max_chunk_size: Option<usize>, tokenizer: Option<TokenizerType>, hf_model: Option<String>) -> PyResult<Self> {
        let max_chunk_size = max_chunk_size.unwrap_or(1500);
        
        // Convert Python TokenizerType to Rust implementation
        let tokenizer_type = match tokenizer.unwrap_or(TokenizerType::Characters) {
            TokenizerType::Characters => RustTokenizerType::Characters,
            TokenizerType::Tiktoken => RustTokenizerType::Tiktoken,
            TokenizerType::HuggingFace => {
                match hf_model {
                    Some(model) => RustTokenizerType::HuggingFace(model),
                    None => {
                        return Err(PyValueError::new_err(
                            "TokenizerType.HUGGINGFACE requires hf_model parameter"
                        ));
                    }
                }
            }
        };
        
        let inner = InnerChunker::new(max_chunk_size, tokenizer_type)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create chunker: {}", e)))?;
        
        Ok(Self {
            inner: Arc::new(inner),
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
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
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
    
    #[pyo3(signature = (content, file_path=None))]
    fn chunk_text<'p>(
        &self,
        py: Python<'p>,
        content: String,
        file_path: Option<String>,
    ) -> PyResult<&'p PyAny> {
        let chunker = self.inner.clone();
        
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            chunker
                .chunk_text(&content, file_path.as_deref())
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Text chunking error: {}", e)))
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
    
    // Add enum
    m.add_class::<TokenizerType>()?;
    
    // Add classes
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
pyo3 = { version = "0.25", features = ["extension-module", "abi3-py39"] }
pyo3-async-runtimes = { version = "0.25", features = ["tokio-runtime"] }
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

- [x] Extract node_type from tree-sitter AST
- [x] Extract node_name from identifier nodes
- [x] Build scope_path by traversing parent nodes
- [x] Extract parent_context (e.g., class name for methods)
- [x] Basic definitions/references extraction

### 3. Text Chunking Support

- [x] Separate `chunk_text` method for plain text
- [x] Works for any content regardless of language support
- [x] Returns chunks with minimal metadata
- [x] Handles unsupported languages gracefully

### 4. Tokenizer Support

- [x] Multiple tokenizer implementations (Characters, Tiktoken, HuggingFace)
- [x] Type-safe `TokenizerType` enum
- [x] Support for custom HuggingFace models
- [x] Clean error handling for missing model names

### 5. Performance

- [ ] Parse 1MB file in <100ms
- [ ] Minimal memory allocations
- [ ] Efficient chunk size calculation

### 6. Error Handling

- [x] Return error for unsupported languages
- [x] Never panic on malformed code
- [x] Clear error types for Python to handle

### 7. Python Integration

- [ ] Published as wheel on PyPI
- [ ] No Rust toolchain required for users
- [x] Type stubs for IDE support with TokenizerType enum
- [ ] Works on Linux, macOS, Windows

## Key Differences from Original Plans

1. **No Query Integration**: text-splitter handles semantic boundaries
2. **No build.rs**: No need to fetch queries at compile time
3. **Direct Dependencies**: Individual tree-sitter-* crates instead of syntastica
4. **Simpler Architecture**: Leverage text-splitter's proven implementation
5. **Flexible Tokenization**: Support characters, tiktoken, or HF tokenizers

## Implementation Phases

### Phase 1: Core Implementation (MVP) ✅

- ✅ Set up text-splitter integration
- ✅ Create language registry for top 10 languages
- ✅ Basic metadata extraction (node type and name)
- ✅ Python bindings with async support

### Phase 2: Enhanced Metadata ✅

- ✅ Scope path extraction
- ✅ Parent context tracking
- ✅ Basic definitions/references (if feasible without queries)

### Phase 3: Full Language Support ✅

- ✅ Add remaining tree-sitter language crates (16 total)
- ✅ Comprehensive testing across languages
- ⏳ Performance optimization (partially done)

### Phase 4: Production Ready ✅

- ✅ Python packaging and distribution (CI/CD setup)
- ✅ Documentation and examples (README & DOCUMENTATION.md)
- ✅ Integration tests with hyperpolyglot

### Phase 5: Project Directory Walker 🚧

- 🚧 Add `walk_project` function for processing entire directories
- 🚧 Integrate hyperpolyglot and infer for file filtering
- 🚧 Implement discriminated union types (ProjectChunk)
- 🚧 Support async iteration over chunks

## Testing Strategy

```python
# tests/test_acceptance.py
import pytest
from breeze_rustle import SemanticChunker, SemanticChunk, TokenizerType

class TestBasicFunctionality:
    def test_supported_languages(self):
        """Should support at least 16 languages"""
        languages = SemanticChunker.supported_languages()
        assert len(languages) >= 16
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
    
    async def test_tokenizer_types(self):
        """Should support different tokenizer types"""
        content = "def test(): pass"
        
        # Test each tokenizer type
        for tokenizer in [TokenizerType.CHARACTERS, TokenizerType.TIKTOKEN]:
            chunker = SemanticChunker(tokenizer=tokenizer)
            chunks = await chunker.chunk_file(content, "Python")
            assert len(chunks) > 0
    
    async def test_text_chunking(self):
        """Should handle plain text chunking for unsupported languages"""
        content = "This is plain text content."
        chunker = SemanticChunker()
        
        # Test with unsupported language
        if not SemanticChunker.is_language_supported("COBOL"):
            chunks = await chunker.chunk_text(content)
            assert len(chunks) > 0
            assert chunks[0].metadata.language == "text"
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

## Project Directory Walker Feature

### Overview
A new feature to walk entire project directories and automatically chunk all processable files.

### Design Goals
1. **Zero configuration**: Just point at a directory and get chunks
2. **Smart filtering**: Skip binary files, respect .gitignore
3. **Automatic language detection**: Use hyperpolyglot for accurate detection
4. **Graceful fallback**: Use text chunking for unsupported languages
5. **Type safety**: Clear discrimination between semantic and text chunks

### Dependencies to Add
```toml
ignore = "0.4"              # Gitignore-aware traversal
futures = "0.3"             # Async streams
hyperpolyglot = "0.1.7"     # Language detection
infer = "0.19.0"            # File type detection
tokio-stream = "0.1"        # Async streaming
```

### New Types
```rust
#[pyclass]
pub enum ChunkType {
    #[pyo3(name = "SEMANTIC")]
    Semantic,
    #[pyo3(name = "TEXT")]
    Text,
}

#[pyclass]
pub struct ProjectChunk {
    #[pyo3(get)]
    pub file_path: String,
    #[pyo3(get)]
    pub chunk_type: ChunkType,
    #[pyo3(get)]
    pub chunk: SemanticChunk,
}
```

### File Processing Strategy
1. Walk directory using `ignore` crate (respects .gitignore)
2. Use `infer` to skip binary files (images, videos, etc.)
3. Use `hyperpolyglot` to detect programming language
4. If language is supported → semantic chunking
5. If language is unsupported or unknown → text chunking
6. Skip files that can't be processed

### Supported Text Formats
When language detection fails, these extensions trigger text chunking:
- Documentation: .txt, .md, .rst
- Config: .yaml, .yml, .toml, .json, .ini, .cfg, .conf
- Web: .xml, .html, .htm
- Scripts: .sh, .bat, .ps1
- Data: .csv, .tsv
- Other: Dockerfile, Makefile, README, .gitignore, .env

### Expected Usage
```python
# Simple usage - process entire project
async for chunk in walk_project("./my_project"):
    if chunk.is_semantic:
        # This is parsed code with full metadata
        print(f"{chunk.chunk.metadata.node_type} in {chunk.file_path}")
    else:
        # This is text content
        print(f"Text chunk from {chunk.file_path}")

# For building search index
async for chunk in walk_project("./src", tokenizer=TokenizerType.TIKTOKEN):
    embedding = await generate_embedding(chunk.chunk.text)
    await store_chunk(chunk, embedding)
```
