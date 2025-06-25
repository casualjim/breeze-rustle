# breeze-rustle

A high-performance Rust library for semantic code chunking and search, with bindings for Python and Node.js. Uses tree-sitter parsers to intelligently split code into meaningful semantic units while preserving context and extracting rich metadata.

## Features

- **Semantic Code Chunking**: Split code at function/class boundaries using tree-sitter
- **100+ Languages**: Support via nvim-treesitter queries
- **Code Search**: Semantic search using vector embeddings
- **Multiple Bindings**: Rust CLI, Python (PyO3), and Node.js (NAPI)
- **Streaming APIs**: Handle large files efficiently
- **API Server**: HTTP/HTTPS with OpenAPI and MCP protocol support

## Installation

At the moment there are no binaries produced yet from the CI build. But you can install the project from source.

### Build from source

The easiest way to get the required tools is to use: [mise](https://mise.jdx.dev/). But for the build to work we need:

- Rust 1.84 or above
- Working LLVM installation (tree-sitter parsers are in C or C++)
- Node 20 (we use [volta](https://volta.sh/))

```bash
git clone https://github.com/casualjim/breeze-rustle 
cd breeze-rustle

# Install dependencies and build
just build
```

### Install the CLI

```bash
cargo install --path crates/breeze
```

Note: This will take a long time because we have many tree-sitter parsers to link.

## Usage

### CLI Commands

```bash
# Initialize configuration
breeze init

# Index a codebase
breeze index <path>

# Search indexed code
breeze search <query>

# Start the API server
breeze serve

# Debug chunking
breeze debug chunk-directory <path>
```

### Configuration

Create a `.breeze.toml` file or use `breeze init`. See the example configuration in the repository.

## Development

```bash
# Run tests
just test

# Build the project
just build

# Clean build artifacts
just clean
```

## Architecture

The project is organized as a Rust workspace with multiple crates:

- `breeze` - CLI application
- `breeze-chunkers` - Core chunking library using tree-sitter
- `breeze-grammars` - Grammar compilation system
- `breeze-indexer` - Code indexing and embedding
- `breeze-server` - HTTP/HTTPS API server
- `breeze-py` - Python bindings
- `breeze-napi` - Node.js bindings

## License

This project is licensed under the MIT License - see the LICENSE file for details.
