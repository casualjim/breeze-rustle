# Justfile for breeze-rustle polyglot project
# https://github.com/casey/just

# Default recipe shows available commands
default:
    @just --list

# Install all development dependencies
install:
    cargo binstall flamegraph
    cargo binstall cargo-zigbuild
    cargo binstall cargo-bloat
    cargo binstall cargo-tree

    uv venv --python 3.12
    uv sync --extra dev

    cd crates/breeze-napi && npm install

# Build all components
build: build-rust build-python build-node

# Build only Rust crates (excluding Python and Node.js bindings)
build-rust:
    cargo zigbuild --release --workspace --exclude breeze-py --exclude breeze-napi

# Build Python bindings
build-python:
    cd crates/breeze-py && maturin develop --release

# Build Node.js bindings
build-node:
    cd crates/breeze-napi && npm run build

# Build grammars (if not using precompiled)
build-grammars:
    cd crates/breeze-grammars && cargo build --release

# Run all tests
test: test-rust test-python test-node

# Run Rust tests with nextest
test-rust:
    cargo nextest run --workspace --features local-embeddings

# Run Python tests
test-python:
    cd crates/breeze-py && pytest -v

# Run Node.js tests
test-node:
    cd crates/breeze-napi && npm run test:all

# Run tests for a specific crate
test-crate crate:
    cargo nextest run -p {{crate}}

# Watch and run tests on file changes
watch-test:
    cargo watch -x "nextest run"

# Run benchmarks
bench:
    cargo bench

# Check code (lint, format, clippy)
check:
    cargo check --workspace
    cargo fmt --all -- --check
    cargo clippy --workspace --all-targets -- -D warnings

# Format code
fmt:
    rustfmt

# Fix common issues
fix:
    cargo fix --workspace --allow-dirty --allow-staged
    cargo clippy --workspace --all-targets --fix -- -D warnings
    cargo fmt --all

# Clean build artifacts
clean:
    cargo clean
    rm -rf crates/breeze-py/target
    rm -rf crates/breeze-napi/target
    rm -rf crates/breeze-napi/*.node
    rm -rf crates/breeze-napi/index.native.js
    rm -rf crates/breeze-napi/index.native.d.ts
    find . -type d -name "__pycache__" -exec rm -rf {} +
    find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Development workflow commands

# Run Python development build and test
dev-python:
    cd crates/breeze-py && maturin develop && pytest -v

# Run Node.js development build and test
dev-node:
    cd crates/breeze-napi && npm run build:debug && npm run test:all

# Interactive Python REPL with breeze imported
python-repl:
    cd crates/breeze-py && maturin develop && python -c "import breeze_python; import asyncio; print('breeze_python module loaded')" -i

# Build and serve documentation
docs:
    cargo doc --workspace --no-deps --open

# CI-related commands

# Run CI checks locally
ci-check: check test

# Build wheels for Python (requires maturin)
build-wheels:
    cd crates/breeze-py && maturin build --release --zig

# Build for specific platform
build-platform target:
    cd crates/breeze-py && maturin build --release --zig --target {{target}}

# Utility commands

# Update dependencies
update:
    cargo update
    cd crates/breeze-napi && npm update

# Show project structure
tree:
    tree -I 'target|__pycache__|node_modules|.git|.pytest_cache' -L 3

# Count lines of code
loc:
    tokei . --exclude target --exclude node_modules --exclude .venv

# List all todos in the codebase
todos:
    rg -i "todo|fixme|hack|xxx" --type rust --type python --type javascript

# Grammar-specific commands

# Fetch latest nvim-treesitter queries
fetch-queries:
    python tools/fetch-queries

# Build grammars for all platforms (requires zig)
build-all-grammars:
    ./tools/build-grammars --all-platforms

# Language-specific test commands

# Test specific language support
test-language lang:
    cargo test -p breeze-grammars {{lang}}
    cd crates/breeze-py && pytest -v -k {{lang}}

# Test all supported languages
test-all-languages:
    cd crates/breeze-py && pytest -v tests/test_all_languages_chunking.py

# Performance testing

# Run performance benchmarks
perf:
    cargo bench --workspace
    cd crates/breeze-py && pytest benchmarks/ -v

# Profile with flamegraph (requires cargo-flamegraph)
profile:
    cargo flamegraph --bin breeze-example

# Release commands

# Prepare for release (run all checks)
pre-release: fmt check test
    @echo "All checks passed! Ready for release."

# Tag a new version
tag version:
    git tag -a v{{version}} -m "Release version {{version}}"
    git push origin v{{version}}

# Publish to crates.io (Rust crates)
publish-crates:
    cargo publish -p breeze-grammars
    sleep 10
    cargo publish -p breeze-chunkers
    sleep 10
    cargo publish -p breeze

# Publish to PyPI (Python package)
publish-python:
    cd crates/breeze-py && maturin publish

# Publish to npm (Node.js package)
publish-npm:
    cd crates/breeze-napi && npm publish

# Docker commands (if needed in future)

# Build Docker image
docker-build:
    docker build -t breeze-rustle .

# Run in Docker
docker-run:
    docker run --rm -it breeze-rustle

# Environment setup

# Setup development environment
setup: install build
    @echo "Development environment ready!"

# Verify environment
verify:
    @echo "Checking environment..."
    @command -v rustc >/dev/null || (echo "Rust not found" && exit 1)
    @command -v cargo >/dev/null || (echo "Cargo not found" && exit 1)
    @command -v python >/dev/null || (echo "Python not found" && exit 1)
    @command -v node >/dev/null || (echo "Node.js not found" && exit 1)
    @command -v uv >/dev/null || (echo "uv not found" && exit 1)
    @echo "Rust version: $(rustc --version)"
    @echo "Python version: $(python --version)"
    @echo "Node version: $(node --version)"
    @echo "Environment OK!"
