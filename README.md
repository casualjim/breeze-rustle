# breeze-rustle

High-performance semantic code chunking for [Breeze](https://github.com/casualjim/breeze), powered by tree-sitter and nvim-treesitter queries.

## Overview

`breeze-rustle` is a Rust library with Python bindings that provides intelligent code chunking using tree-sitter parsers and nvim-treesitter's battle-tested query files. It splits code into semantic units (functions, classes, methods) while preserving context and extracting rich metadata.

**Key insight**: While we chunk code to fit within embedding model constraints, we store embeddings at the file level. The chunking ensures we respect semantic boundaries (never splitting functions/classes arbitrarily) while staying within token limits. File embeddings are created by aggregating chunk embeddings.

### Key Features

- **🚀 Fast**: Written in Rust with async/concurrent processing
- **🎯 Semantic**: Uses nvim-treesitter queries for accurate code understanding
- **📦 Zero Dependencies**: Distributed as a wheel - no Rust toolchain needed
- **🌍 163 Languages**: Comprehensive language support via tree-sitter grammars
- **🔍 Rich Metadata**: Extracts scopes, definitions, references, and more
- **⚡ Async First**: Native Python async/await support

## Installation

```bash
pip install breeze-rustle
```

## Quick Start

```python
import asyncio
from breeze_rustle import SemanticChunker

async def main():
    chunker = SemanticChunker(max_chunk_size=16384)

    # Chunk a single file
    code = """
def process_data(items):
    results = []
    for item in items:
        if item > 0:
            results.append(item * 2)
    return results
"""

    chunks = await chunker.chunk_file(code, "python")

    for chunk in chunks:
        print(f"Chunk: {chunk.metadata.node_type} - {chunk.metadata.node_name}")
        print(f"  Definitions: {chunk.metadata.definitions}")
        print(f"  References: {chunk.metadata.references}")
        print(f"  Lines: {chunk.start_line}-{chunk.end_line}")

asyncio.run(main())
```

## Features

### Semantic Chunking

breeze-rustle chunks code to fit within embedding model constraints while respecting code structure:

- **Purpose**: Fit within embedder token limits (e.g., 8k tokens)
- **Boundaries**: Only splits at semantic boundaries (functions, classes)
- **Largest units**: Creates the largest possible chunks that fit constraints
- **File handling**: Small files = one chunk, large files = multiple chunks
- **Aggregation**: Multiple chunks from one file are aggregated back to a single file embedding

### Rich Metadata

Each chunk includes:

```python
chunk.metadata.node_type      # "function", "class", "method", etc.
chunk.metadata.node_name      # "process_data", "MyClass", etc.
chunk.metadata.scope_path     # ["module", "MyClass", "my_method"]
chunk.metadata.definitions    # ["results", "item"]
chunk.metadata.references     # ["items", "append"]
chunk.metadata.parent_context # "class MyClass"
```

### Concurrent Processing

Process multiple files efficiently:

```python
files = [
    (content1, "python", "file1.py"),
    (content2, "javascript", "file2.js"),
    (content3, "rust", "file3.rs"),
]

results = await chunker.chunk_files(files)
```

### Language Support

breeze-rustle supports **163 programming languages** through compiled tree-sitter grammars, making it one of the most language-comprehensive code analysis tools available.

**Supported Language Categories:**

- **Major Languages**: Python, JavaScript, TypeScript, Java, C/C++, C#, Go, Rust, Swift, Kotlin, Ruby, etc.
- **Web Technologies**: HTML, CSS, Vue, Svelte, Astro, TSX, JSX, SCSS, etc.
- **Systems Languages**: Zig, V, D, Assembly, CUDA, Verilog, VHDL, etc.
- **Functional Languages**: Haskell, OCaml, Elm, Clojure, Erlang, Elixir, Scheme, etc.
- **Domain-Specific**: SQL, GraphQL, Dockerfile, Terraform, Prisma, Protobuf, etc.
- **Configuration**: YAML, TOML, JSON, XML, HCL, Nix, etc.
- **And many more specialized languages**

```python
# Check supported languages
languages = SemanticChunker.supported_languages()
print(f"Supports {len(languages)} languages")  # 163 languages!

# Check specific language (case-insensitive)
if SemanticChunker.is_language_supported("rust"):
    chunks = await chunker.chunk_file(rust_code, "rust")
```

## How It Works

1. **Tree-sitter Parsing**: Uses tree-sitter to build a concrete syntax tree
2. **nvim-treesitter Queries**: Applies community-maintained queries to identify:
   - Scopes (`@local.scope`)
   - Definitions (`@local.definition`)
   - References (`@local.reference`)
3. **Intelligent Chunking**: Groups semantic units respecting size limits
4. **Metadata Extraction**: Enriches chunks with structural information

## Performance

- Parses 1MB files in <100ms
- Concurrent processing with `smol` async runtime
- Zero-copy operations where possible
- Efficient caching of parsers and queries

## Development

### Building from Source

```bash
# Clone the repository
git clone https://github.com/casualjim/breeze-rustle
cd breeze-rustle

# Install Rust toolchain
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install maturin
pip install maturin

# Build and install locally
maturin develop --release
```

### Running Tests

```bash
# Python tests
pytest tests/

# Rust tests
cargo test

# Benchmarks
cargo bench
```

### Updating Queries

The nvim-treesitter queries are embedded at build time:

```bash
# Update queries from nvim-treesitter
python tools/fetch-queries

# This regenerates src/queries.rs
# Commit the changes
```

## Architecture

```
┌─────────────────────────┐
│   Python (breeze)       │
├─────────────────────────┤
│   PyO3 Bindings         │
│   - Async/await support │
│   - Type conversions    │
├─────────────────────────┤
│   Rust Core             │
│   - Tree-sitter parsing │
│   - Query execution     │
│   - Chunking logic      │
├─────────────────────────┤
│   Embedded Queries      │
│   - nvim-treesitter     │
│   - 163 languages       │
└─────────────────────────┘
```

## API Reference

### SemanticChunker

```python
class SemanticChunker:
    def __init__(self, max_chunk_size: int = 16384) -> None:
        """
        Initialize a semantic chunker.

        Args:
            max_chunk_size: Maximum tokens per chunk (default: 16384)
        """

    async def chunk_file(
        self,
        content: str,
        language: str,
        file_path: Optional[str] = None
    ) -> List[SemanticChunk]:
        """
        Chunk a single file into semantic units.

        Args:
            content: File content to chunk
            language: Programming language (e.g., "python", "rust")
            file_path: Optional file path for better error messages

        Returns:
            List of semantic chunks with metadata
        """

    async def chunk_files(
        self,
        files: List[Tuple[str, str, str]]
    ) -> List[List[SemanticChunk]]:
        """
        Chunk multiple files concurrently.

        Args:
            files: List of (content, language, path) tuples

        Returns:
            List of chunk lists, one per input file
        """
```

### SemanticChunk

```python
@dataclass
class SemanticChunk:
    text: str                # The chunk content
    start_byte: int         # Start position in original file
    end_byte: int           # End position in original file
    start_line: int         # Start line number (1-based)
    end_line: int           # End line number (1-based)
    metadata: ChunkMetadata # Rich metadata about the chunk
```

### ChunkMetadata

```python
@dataclass
class ChunkMetadata:
    node_type: str                    # AST node type
    node_name: Optional[str]          # Symbol name if available
    language: str                     # Programming language
    parent_context: Optional[str]     # Parent scope description
    scope_path: List[str]            # Full scope hierarchy
    definitions: List[str]           # Symbols defined in this chunk
    references: List[str]            # Symbols referenced in this chunk
```

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Acknowledgments

- [tree-sitter](https://tree-sitter.github.io/) for the parsing framework
- [nvim-treesitter](https://github.com/nvim-treesitter/nvim-treesitter) for the amazing query files
- [tree-sitter-language-pack](https://github.com/Goldziher/tree-sitter-language-pack) for the comprehensive grammar curation
- [PyO3](https://pyo3.rs/) for Rust-Python bindings
- [smol](https://github.com/smol-rs/smol) for the async runtime
