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

```bash
git clone https://github.com/casualjim/breeze-rustle
cd breeze-rustle

# Install dependencies and build
mise build
```

### Install the CLI

```bash
# CPU based local embeddings
cargo install --path crates/breeze
# Cuda based local embeddings
cargo install --path crates/breeze --features cuda
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

Create a config file or use `breeze init` to generate one. The configuration file is searched in the following locations (in order):

1. `.breeze.toml` (current directory)
2. `~/Library/Application Support/com.github.casualjim.breeze.server/config.toml` (macOS only)
3. `~/.config/breeze/config.toml` (user config)
4. `/etc/breeze/config.toml` (system-wide)
5. `/usr/share/breeze/config.toml` (system-wide)

You can also specify a custom config path using the `--config` flag. See the [default configuration](crates/breeze/examples/config_default.toml) in the repository.

#### Embeddings providers

This application can use local embeddings, via the onnxruntime. Technically we also support candle transformers but they were really slow because candle waits for all GPU tasks to be freed before copying the embedding numbers back to CPU space and that causes big delays.

We provide a local embedding engine so that you can get started right away, but you'll probably want to use an API. You'll get higher quality embeddings and much faster indexing.

The remote embeddings providers are assumed to be OpenAI API compatible, which works for most providers like: vllm, infinity, ollama, llamacpp as well as voyage, cohere, ...

## Development

```bash
# Run tests
mise test

# Build the project
mise build

# Clean build artifacts
mise clean
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

## Performance

For me, on my macbook M4 Pro, compiled with `cargo install --path crates/breeze --features metal`.

See [docs/performance.md](docs/performance.md)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
