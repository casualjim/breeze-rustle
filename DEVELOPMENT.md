# Development Guide

This guide covers building breeze-rustle and optimizing the build process.

## Quick Start

```bash
# First time setup - build all grammars for all platforms
./tools/build-grammars --all-platforms

# Build and install Python package locally
maturin develop --release

# Run tests
cargo test
pytest tests/
```

## Understanding the Build Process

The most time-consuming part of building breeze-rustle is compiling the 40+ tree-sitter grammars. Each grammar is a C/C++ parser that needs to be compiled into a static library.

### Build Strategies

1. **Precompiled Binaries (Recommended)**
   - Run `./tools/build-grammars` once to compile all grammars
   - Subsequent builds will use the precompiled binaries from `crates/breeze-grammars/precompiled/`
   - Build time: ~5 seconds (vs ~10+ minutes without precompilation)

2. **Source Compilation**
   - If no precompiled binaries exist, the build will compile from source
   - This happens automatically but is very slow

## Speeding Up Development Builds

### 1. Use Precompiled Grammars

```bash
# Build grammars for your current platform only (fastest)
./tools/build-grammars

# Build for all platforms (needed for distribution)
./tools/build-grammars --all-platforms

# The binaries are stored in:
# crates/breeze-grammars/precompiled/<platform>/
```

### 2. Install sccache

sccache dramatically speeds up recompilation:

```bash
# Install sccache
cargo install sccache

# It will be automatically detected and used by the build process
```

### 3. Partial Grammar Sets

For development, you can temporarily reduce the grammar set:

```bash
# Use a smaller grammar set for testing
cp crates/breeze-grammars/grammars-minimal.json crates/breeze-grammars/grammars.json

# Don't forget to restore the full set before committing!
cp crates/breeze-grammars/grammars-full.json crates/breeze-grammars/grammars.json
```

## Build Tools

### build-grammars

The `tools/build-grammars` script handles grammar compilation:

```bash
# Options:
./tools/build-grammars --help

# Fetch grammar sources only (no compilation)
./tools/build-grammars --fetch-only

# Compile only (assumes sources are fetched)
./tools/build-grammars --compile-only

# Build for specific platform
./tools/build-grammars --platform linux-x86_64-glibc

# Control parallelism
./tools/build-grammars -j 4
```

### Supported Platforms

- `linux-x86_64-glibc` - Standard Linux (Ubuntu, Debian, etc.)
- `linux-x86_64-musl` - Alpine Linux
- `linux-aarch64-glibc` - ARM64 Linux
- `linux-aarch64-musl` - ARM64 Alpine
- `windows-x86_64` - Windows 64-bit
- `windows-aarch64` - Windows ARM64
- `macos-x86_64` - Intel Macs
- `macos-aarch64` - Apple Silicon Macs

## CI/CD Optimization

The GitHub Actions workflow caches precompiled grammars:

- Cache key: `grammars-${{ hashFiles('crates/breeze-grammars/grammars.json') }}-v1`
- Grammars are built once and shared across all job matrices
- To bust the cache, change the version suffix (e.g., `-v1` to `-v2`)

## Troubleshooting

### Slow First Build

The first build will be slow as it needs to:
1. Clone all grammar repositories
2. Compile each grammar from C/C++ source

This is normal and only happens once. Use `./tools/build-grammars` to precompile.

### Grammar Compilation Errors

Some grammars may fail to compile due to:
- Missing `parser.c` (needs generation from `grammar.js`)
- Platform-specific issues
- Network issues during cloning

The build will continue with available grammars and report failures.

### Cross-Compilation

Cross-compilation requires [Zig](https://ziglang.org/):

```bash
# Install Zig
# macOS: brew install zig
# Linux: snap install zig --classic --beta
# Or download from https://ziglang.org/download/

# Build for all platforms
./tools/build-grammars --all-platforms
```

## Adding New Grammars

To add a new grammar:

1. Add entry to `crates/breeze-grammars/grammars.json`:
   ```json
   {
     "name": "language-name",
     "repo": "https://github.com/org/tree-sitter-language",
     "rev": "commit-hash",
     "path": "optional/subdirectory",
     "symbol_name": "optional_symbol_override"
   }
   ```

2. Rebuild grammars:
   ```bash
   ./tools/build-grammars
   ```

### Special Cases

- **symbol_name**: Use when the exported symbol doesn't match the grammar name (e.g., C# exports `tree_sitter_c_sharp` but is named `csharp`)
- **path**: Use when the grammar source is in a subdirectory
- **rev**: Pin to specific commit for reproducibility

## Architecture Notes

The build system has several layers:

1. **Python tool** (`tools/build-grammars`): Fetches and compiles grammars with Zig
2. **Rust build.rs**: Detects and uses precompiled binaries or falls back to source compilation
3. **Grammar bindings**: Auto-generated Rust code to load grammars

This separation allows for:
- Fast incremental builds (precompiled binaries)
- Cross-platform compilation (Zig)
- Fallback to source compilation when needed
- Easy grammar updates
