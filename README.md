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
just build
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

Create a `.breeze.toml` file or use `breeze init`. See the [default configuration](crates/breeze/examples/config_default.toml) in the repository.

#### Embeddings providers

This application can use local embeddings, via the onnxruntime. Technically we also support candle transformers but they were really slow because candle waits for all GPU tasks to be freed before
copying the embedding numbers back to CPU space and that causes big delays.

We provide a local embedding engine so that you can get started right away, but you'll probably want to use an API. You'll get higher quality embeddings and much faster indexing.

The remote embeddings providers are assumed to be OpenAI API compatible, which works for most providers like: vllm, infinity, ollama, llamacpp as well as voyage, cohere, ...

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

## Performance

For me, on my macbook M4 Pro, compiled with `cargo install --path crates/breeze --features metal`.

### Lance (storage engine)

Repository: [lance](https://github.com/lancedb/lance)

#### Project stats (lance)

```shellsession
$ tokei ~/github/lancedb/lance --exclude target --exclude node_modules --exclude .venv

===============================================================================
 Language            Files        Lines         Code     Comments       Blanks
===============================================================================
 Batch                   1           35           26            1            8
 C                       1           87           47           28           12
 CSS                     1            4            4            0            0
 Java                   44         7031         4435         1785          811
 JSON                    3           15           15            0            0
 Makefile                2           75           53            7           15
 Protocol Buffers        8         1804          620          962          222
 Python                142        31448        25314         1400         4734
 ReStructuredText       28         4197         3133            0         1064
 Shell                   3          318          206           79           33
 Plain Text             14           77            0           76            1
 TOML                   21         1650         1312          198          140
 XML                     3          642          585           31           26
 YAML                   10           80           77            2            1
-------------------------------------------------------------------------------
 Jupyter Notebooks       4            0            0            0            0
 |- Markdown             3           85            0           70           15
 |- Python               4          408          339           19           50
 (Total)                            493          339           89           65
-------------------------------------------------------------------------------
 Markdown               28         1206            0          805          401
 |- BASH                 3            7            7            0            0
 |- Java                 1          123          118            2            3
 |- Python               4          115           86            9           20
 |- Rust                 2           30           26            1            3
 |- Shell                5           66           66            0            0
 (Total)                           1547          303          817          427
-------------------------------------------------------------------------------
 Rust                  392       186664       158916         8404        19344
 |- Markdown           299         8659          164         6901         1594
 (Total)                         195323       159080        15305        20938
===============================================================================
 Total                 705       235333       194743        13778        26812
===============================================================================
```

#### Local embeddings

Using the local embeddings provider (baai/bge-small-en-1.5):

Indexing time: 147s

#### Ollama

Using the ollama embeddings provider with nomic-embed-text (should be better quality than local)

Indexing time: 165s

#### Voyage

Using the voyage-code-3 model.

Indexing time: 48s

### LanceDB

Repository: [lancedb](https://github.com/lancedb/lancedb)

#### Project stats (lancedb)

```shellsession
===============================================================================
 Language            Files        Lines         Code     Comments       Blanks
===============================================================================
 CSS                     2          119          102            3           14
 Dockerfile              2           54           27           14           13
 HTML                    2          181          137           31           13
 Java                    2          243          138           77           28
 JavaScript             10          461          296          100           65
 JSON                   30        20851        20850            0            1
 Makefile                1           40           31            0            9
 PowerShell              2           84           58           14           12
 Python                 97        29127        24199         1401         3527
 Shell                   9          396          209          100           87
 SVG                     5           26           26            0            0
 Plain Text              8           37            0           37            0
 TOML                    8          480          434           15           31
 TypeScript             56        17042        11702         3786         1554
 XML                     2          433          415            1           17
 YAML                    3          920          897            3           20
-------------------------------------------------------------------------------
 Jupyter Notebooks      12            0            0            0            0
 |- Markdown            12          684           21          500          163
 |- Python              12         1203          976           48          179
 (Total)                           1887          997          548          342
-------------------------------------------------------------------------------
 Markdown              247        20381            0        12820         7561
 |- BASH                15           80           77            2            1
 |- JavaScript          13          209          181           19            9
 |- JSON                 3           77           77            0            0
 |- Markdown             1          124            0          124            0
 |- Python              71         2683         2150          148          385
 |- Rust                 6           65           63            0            2
 |- Shell               13           77           71            6            0
 |- SQL                  2            6            6            0            0
 |- TOML                 4           17           16            0            1
 |- TypeScript          10          130          122            0            8
 (Total)                          23849         2763        13119         7967
-------------------------------------------------------------------------------
 Rust                   84        25353        21693          632         3028
 |- Markdown            52         2471           94         1924          453
 (Total)                          27824        21787         2556         3481
===============================================================================
 Total                 582       116228        81214        19034        15980
===============================================================================
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
