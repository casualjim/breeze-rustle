# Language Parser API Inconsistencies

## Overview

During the implementation of breeze-rustle, we discovered that tree-sitter language parsers have inconsistent APIs, preventing us from adding support for Kotlin and SQL dialects. This document captures the investigation findings and potential solutions.

## The Problem

The breeze-rustle language registry expects all parsers to provide a `LanguageFn` type (from the `tree-sitter-language` crate). However, different tree-sitter language crates expose their languages through different APIs:

### Working Parsers (Using Constants)

Most parsers provide `LANGUAGE` or similar constants that are already `LanguageFn`:

```rust
tree_sitter_python::LANGUAGE       // LanguageFn
tree_sitter_rust::LANGUAGE          // LanguageFn
tree_sitter_javascript::LANGUAGE    // LanguageFn
tree_sitter_php::LANGUAGE_PHP       // LanguageFn (different name)
```

### Problematic Parsers (Using Functions)

Some parsers only provide functions that return `tree_sitter::Language`:

```rust
tree_sitter_kotlin::language()      // fn() -> Language
tree_sitter_sql::language()         // fn() -> Language
```

## Why This Matters

Our language registry is defined as:

```rust
pub static LANGUAGE_REGISTRY: LazyLock<HashMap<&'static str, LanguageFn>> = LazyLock::new(|| {
    let mut registry = HashMap::new();
    registry.insert("Python", tree_sitter_python::LANGUAGE);  // Works
    registry.insert("Kotlin", tree_sitter_kotlin::language);   // ERROR: Type mismatch
    // ...
});
```

We cannot cast `fn() -> Language` to `LanguageFn` because:
1. Rust doesn't allow non-primitive casts with `as`
2. `LanguageFn` and `fn() -> Language` are fundamentally different types
3. The internal representation may differ

## Investigation Results

### Test Code

```rust
#[test]
fn test_missing_language_apis() {
    // PHP works fine
    let php_lang: LanguageFn = tree_sitter_php::LANGUAGE_PHP;
    
    // Kotlin returns Language, not LanguageFn
    let kotlin_lang = tree_sitter_kotlin::language();
    
    // SQL also returns Language, not LanguageFn
    let sql_lang = tree_sitter_sql::language();
    
    // PHP can convert to Language
    let php_as_lang: tree_sitter::Language = php_lang.into();
}
```

### Compilation Errors

```
error[E0605]: non-primitive cast: `fn() -> tree_sitter::Language {tree_sitter_kotlin::language}` as `tree_sitter_language::LanguageFn`
```

## Current Impact

- **PHP**: Successfully added (uses `LANGUAGE_PHP` constant)
- **Kotlin**: Cannot be added due to API mismatch
- **SQL**: Cannot be added due to API mismatch
- **Total Languages**: 17 supported (would be 19 with Kotlin and SQL)

## Potential Solutions

### 1. Wrapper Constants (Quick Fix)

Create static wrappers that convert at initialization:

```rust
// This approach requires unsafe code or API changes
static KOTLIN_LANG: LazyLock<LanguageFn> = LazyLock::new(|| {
    // Need to find a way to convert Language to LanguageFn
});
```

**Pros**: Maintains current architecture
**Cons**: May require unsafe code, hacky solution

### 2. Registry Redesign

Change the registry to handle both API patterns:

```rust
enum LanguageProvider {
    Fn(LanguageFn),
    Getter(fn() -> tree_sitter::Language),
}

pub static LANGUAGE_REGISTRY: LazyLock<HashMap<&'static str, LanguageProvider>> = ...
```

**Pros**: Clean, supports both patterns
**Cons**: Requires refactoring all code that uses the registry

### 3. Direct Language Storage

Store the actual `Language` instances instead of functions:

```rust
pub static LANGUAGE_REGISTRY: LazyLock<HashMap<&'static str, tree_sitter::Language>> = LazyLock::new(|| {
    let mut registry = HashMap::new();
    registry.insert("Python", tree_sitter_python::LANGUAGE.into());
    registry.insert("Kotlin", tree_sitter_kotlin::language());  // Now works
    // ...
});
```

**Pros**: Simpler, works with both patterns
**Cons**: Changes the registry type, may affect performance

### 4. Use Alternative Crates

- **tree-sitter-kotlin-sg**: Alternative Kotlin parser that might have different API
- **syntastica**: Provides consistent API for all languages
- **Custom builds**: Generate our own bindings with consistent APIs

**Pros**: Consistent API across all languages
**Cons**: Additional dependencies, migration effort

### 5. Skip Problematic Languages

Simply document that Kotlin and SQL are not supported due to upstream API issues.

**Pros**: No code changes needed
**Cons**: Reduced language support

## Recommendation

For now, we've chosen **Option 5** (skip problematic languages) because:

1. We already support 17 languages, which covers most use cases
2. The API inconsistency is an upstream issue
3. PHP (which was also problematic) now works, showing progress
4. Future versions of these crates may fix the API inconsistency

If comprehensive language support becomes critical, we should consider:
1. **Short term**: Implement Option 3 (direct Language storage)
2. **Long term**: Migrate to syntastica for consistent API and better maintenance

## Related Issues

- Context: The crates are already in `Cargo.toml` but commented out in code
- Previous attempts to add these languages failed with compilation errors
- This is why comments like "// Skip PHP for now - API unclear" exist in the code

## Future Work

1. Monitor tree-sitter-kotlin and tree-sitter-sql for API updates
2. Consider creating a compatibility layer if many more languages have this issue
3. Investigate syntastica as a comprehensive solution
4. File issues upstream to standardize the API across all tree-sitter language crates