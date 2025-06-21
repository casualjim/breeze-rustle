# Justfile for breeze-rustle polyglot project
# https://github.com/casey/just

# Default recipe shows available commands
default:
    @just --list

# Install all development dependencies
install:
    cargo install cargo-nextest
    cargo install flamegraph
    cargo install cargo-zigbuild

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
    cargo nextest run --workspace

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

# Clean build artifacts
clean:
    cargo clean
    rm -rf crates/breeze-py/target
    rm -rf crates/breeze-napi/target
    rm -rf crates/breeze-napi/*.node
    rm -rf crates/breeze-napi/index.js
    rm -rf crates/breeze-napi/index.d.ts
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

# Performance testing

# Run performance benchmarks
perf:
    cargo bench --workspace
    cd crates/breeze-py && pytest benchmarks/ -v

# Profile with flamegraph (requires cargo-flamegraph)
profile:
    cargo flamegraph --bin breeze-example

# Release commands


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
