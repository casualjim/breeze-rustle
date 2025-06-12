# Grammar Integration Implementation Notes

## Key Learnings

### 1. Proper LanguageFn Creation

The official tree-sitter language bindings use this pattern:

```rust
extern "C" {
    fn tree_sitter_python() -> *const ();
}

pub const LANGUAGE: LanguageFn = unsafe { LanguageFn::from_raw(tree_sitter_python) };
```

**Important**: Never use `std::mem::transmute` to convert between `Language` and `LanguageFn`. This causes bus errors and crashes.

### 2. Build Script Pattern

When generating bindings in build.rs:

```rust
// Correct: Generate extern declarations returning *const ()
bindings.push_str(&format!(
    "extern \"C\" {{ fn tree_sitter_{}() -> *const (); }}\n",
    fn_name
));

// Then create LanguageFn constants
bindings.push_str(&format!(
    "pub const {}_LANGUAGE: LanguageFn = unsafe {{ LanguageFn::from_raw(tree_sitter_{}) }};\n",
    const_name, fn_name
));
```

### 3. Case Normalization

We chose lowercase normalization for simplicity:
- All grammar names in grammars.json should be lowercase
- API normalizes input to lowercase: `name.to_lowercase()`
- No complex aliasing system needed

### 4. ABI Compatibility

Tree-sitter has a sliding window of ABI compatibility:
- tree-sitter 0.25 expects ABI version 15
- Minimum compatible version is 13
- Python grammar (ABI 14) works fine
- Check with: `lang.abi_version()`

### 5. Workspace Structure

The grammar compilation crate should be separate:
- Located at `crates/breeze-grammars/`
- Has its own Cargo.toml with minimal dependencies
- Generates bindings at build time
- Main crate depends on it via `path = "crates/breeze-grammars"`

### 6. Grammar Repository Structure

Different grammars have different layouts:
- Most have source in `src/` directory
- Some (like TypeScript) have multiple grammars: `typescript/src/` and `tsx/src/`
- PHP has source in `php/src/`
- Use the `path` field in grammars.json when needed

### 7. Build Dependencies

Essential build dependencies for breeze-grammars:
```toml
[dependencies]
tree-sitter = "0.25"
tree-sitter-language = "0.1.5"

[build-dependencies]
cc = "1.0"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

### 8. Error Handling

Common issues and solutions:
- **Bus error**: Usually means improper Language/LanguageFn conversion
- **Missing parser.c**: Grammar might use different source layout
- **Compilation fails**: Check if grammar needs C++ compilation (scanner.cc)
- **Clone fails**: Ensure proper GitHub URL construction

## Testing Pattern

Always test new grammars with:
1. Build the breeze-grammars crate standalone
2. Run its tests to verify grammar loads
3. Build main crate with maturin
4. Run Python tests to verify integration

## Future Considerations

1. **Caching**: Grammar downloads are currently not cached between clean builds
2. **Binary size**: Each grammar adds ~1-2MB to the binary
3. **Build time**: Compiling many grammars is slow (~3-5s per grammar)
4. **Feature flags**: Consider adding features to select grammar groups