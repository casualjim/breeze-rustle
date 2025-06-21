# breeze-grammars

[![Crates.io](https://img.shields.io/crates/v/breeze-grammars)](https://crates.io/crates/breeze-grammars)
[![License](https://img.shields.io/crates/l/breeze-grammars)](https://github.com/casualjim/breeze-rustle/blob/main/LICENSE)

Tree-sitter grammar compilation and management for breeze-rustle. This crate bundles a comprehensive collection of [tree-sitter](https://tree-sitter.github.io/tree-sitter/) grammars compiled at build time.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
breeze-grammars = "0.1"
```

## Important Notes

- The grammar configurations in this crate are based on the excellent curation work done by the [tree-sitter-language-pack](https://github.com/Goldziher/tree-sitter-language-pack) project. They have done the hard work of identifying stable versions, proper repository structures, and compilation requirements for each grammar.
- All grammars are compiled at build time using the `cc` crate - no runtime downloads required
- All languages bundled by this crate are licensed under permissive open-source licenses (MIT, Apache 2.0 etc.) only - no GPL licensed languages are included

## Features

- 165+ Languages: Support for all major programming languages and many domain-specific languages
- Compile-time Grammar Compilation: All grammars are compiled into your binary at build time
- Type-Safe: Full Rust type safety with `tree_sitter::Language` and `tree_sitter_language::LanguageFn`
- Zero GPL Dependencies: All bundled languages use permissive licenses (MIT, Apache 2.0, etc.)
- Stable Versions: Uses specific commit hashes for reproducible builds

## Usage

This library provides a simple API for loading tree-sitter grammars:

```rust
use breeze_grammars::{get_language, get_language_fn, is_language_supported, supported_languages};

// Check if a language is supported (case-insensitive)
if is_language_supported("rust") {
    // Get the language (returns Option<tree_sitter::Language>)
    let language = get_language("Rust").unwrap();

    // Use with tree-sitter parser
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&language).unwrap();
}

// Get a LanguageFn for lower-level usage
let language_fn = get_language_fn("python").unwrap();

// List all supported languages
for lang in supported_languages() {
    println!("Supported: {}", lang);
}
```

See the list of available languages below to get the name of the language you want to use.

## Available Languages

Each language below is identified by the key used to retrieve it from the `get_language` and `get_language_fn` functions.

- [actionscript](https://github.com/Rileran/tree-sitter-actionscript) - MIT License
- [ada](https://github.com/briot/tree-sitter-ada) - MIT License
- [agda](https://github.com/tree-sitter/tree-sitter-agda) - MIT License
- [apex](https://github.com/aheber/tree-sitter-sfapex) - MIT License
- [arduino](https://github.com/tree-sitter-grammars/tree-sitter-arduino) - MIT License
- [asm](https://github.com/rush-rs/tree-sitter-asm) - MIT License
- [astro](https://github.com/virchau13/tree-sitter-astro) - MIT License
- [bash](https://github.com/tree-sitter/tree-sitter-bash) - MIT License
- [beancount](https://github.com/polarmutex/tree-sitter-beancount) - MIT License
- [bibtex](https://github.com/latex-lsp/tree-sitter-bibtex) - MIT License
- [bicep](https://github.com/tree-sitter-grammars/tree-sitter-bicep) - MIT License
- [bitbake](https://github.com/tree-sitter-grammars/tree-sitter-bitbake) - MIT License
- [c](https://github.com/tree-sitter/tree-sitter-c) - MIT License
- [cairo](https://github.com/tree-sitter-grammars/tree-sitter-cairo) - MIT License
- [capnp](https://github.com/tree-sitter-grammars/tree-sitter-capnp) - MIT License
- [chatito](https://github.com/tree-sitter-grammars/tree-sitter-chatito) - MIT License
- [clarity](https://github.com/xlittlerag/tree-sitter-clarity) - MIT License
- [clojure](https://github.com/sogaiu/tree-sitter-clojure) - CC0 1.0 Universal License
- [cmake](https://github.com/uyha/tree-sitter-cmake) - MIT License
- [comment](https://github.com/stsewd/tree-sitter-comment) - MIT License
- [commonlisp](https://github.com/tree-sitter-grammars/tree-sitter-commonlisp) - MIT License
- [cpon](https://github.com/tree-sitter-grammars/tree-sitter-cpon) - MIT License
- [cpp](https://github.com/tree-sitter/tree-sitter-cpp) - MIT License
- [csharp](https://github.com/tree-sitter/tree-sitter-c-sharp) - MIT License
- [css](https://github.com/tree-sitter/tree-sitter-css) - MIT License
- [csv](https://github.com/tree-sitter-grammars/tree-sitter-csv) - MIT License
- [cuda](https://github.com/tree-sitter-grammars/tree-sitter-cuda) - MIT License
- [d](https://github.com/gdamore/tree-sitter-d) - MIT License
- [dart](https://github.com/UserNobody14/tree-sitter-dart) - MIT License
- [dockerfile](https://github.com/camdencheek/tree-sitter-dockerfile) - MIT License
- [doxygen](https://github.com/tree-sitter-grammars/tree-sitter-doxygen) - MIT License
- [dtd](https://github.com/tree-sitter-grammars/tree-sitter-xml) - MIT License
- [elisp](https://github.com/Wilfred/tree-sitter-elisp) - MIT License
- [elixir](https://github.com/elixir-lang/tree-sitter-elixir) - MIT License
- [elm](https://github.com/elm-tooling/tree-sitter-elm) - MIT License
- [embeddedtemplate](https://github.com/tree-sitter/tree-sitter-embedded-template) - MIT License
- [erlang](https://github.com/WhatsApp/tree-sitter-erlang) - MIT License
- [fennel](https://github.com/TravonteD/tree-sitter-fennel) - MIT License
- [firrtl](https://github.com/tree-sitter-grammars/tree-sitter-firrtl) - Apache License 2.0
- [fish](https://github.com/ram02z/tree-sitter-fish) - Unlicense license
- [fortran](https://github.com/stadelmanma/tree-sitter-fortran) - MIT License
- [func](https://github.com/tree-sitter-grammars/tree-sitter-func) - MIT License
- [gdscript](https://github.com/PrestonKnopp/tree-sitter-gdscript) - MIT License
- [gitattributes](https://github.com/tree-sitter-grammars/tree-sitter-gitattributes) - MIT License
- [gitcommit](https://github.com/gbprod/tree-sitter-gitcommit) - WTFPL License
- [gitignore](https://github.com/shunsambongi/tree-sitter-gitignore) - MIT License
- [gleam](https://github.com/gleam-lang/tree-sitter-gleam) - Apache-2.0 license
- [glsl](https://github.com/tree-sitter-grammars/tree-sitter-glsl) - MIT License
- [gn](https://github.com/tree-sitter-grammars/tree-sitter-gn) - MIT License
- [go](https://github.com/tree-sitter/tree-sitter-go) - MIT License
- [gomod](https://github.com/camdencheek/tree-sitter-go-mod) - MIT License
- [gosum](https://github.com/tree-sitter-grammars/tree-sitter-go-sum) - MIT License
- [graphql](https://github.com/bkegley/tree-sitter-graphql) - MIT License
- [groovy](https://github.com/Decodetalkers/tree-sitter-groovy) - MIT License
- [gstlaunch](https://github.com/tree-sitter-grammars/tree-sitter-gstlaunch) - MIT License
- [hack](https://github.com/slackhq/tree-sitter-hack) - MIT License
- [hare](https://github.com/tree-sitter-grammars/tree-sitter-hare) - MIT License
- [haskell](https://github.com/tree-sitter/tree-sitter-haskell) - MIT License
- [haxe](https://github.com/vantreeseba/tree-sitter-haxe) - MIT License
- [hcl](https://github.com/tree-sitter-grammars/tree-sitter-hcl) - Apache License 2.0
- [heex](https://github.com/phoenixframework/tree-sitter-heex) - MIT License
- [hlsl](https://github.com/tree-sitter-grammars/tree-sitter-hlsl) - MIT License
- [html](https://github.com/tree-sitter/tree-sitter-html) - MIT License
- [hyprlang](https://github.com/tree-sitter-grammars/tree-sitter-hyprlang) - MIT License
- [ispc](https://github.com/tree-sitter-grammars/tree-sitter-ispc) - MIT License
- [janet](https://github.com/GrayJack/tree-sitter-janet) - BSD-3-Clause license
- [java](https://github.com/tree-sitter/tree-sitter-java) - MIT License
- [javascript](https://github.com/tree-sitter/tree-sitter-javascript) - MIT License
- [jsdoc](https://github.com/tree-sitter/tree-sitter-jsdoc) - MIT License
- [json](https://github.com/tree-sitter/tree-sitter-json) - MIT License
- [jsonnet](https://github.com/sourcegraph/tree-sitter-jsonnet) - MIT License
- [julia](https://github.com/tree-sitter/tree-sitter-julia) - MIT License
- [kconfig](https://github.com/tree-sitter-grammars/tree-sitter-kconfig) - MIT License
- [kdl](https://github.com/tree-sitter-grammars/tree-sitter-kdl) - MIT License
- [kotlin](https://github.com/fwcd/tree-sitter-kotlin) - MIT License
- [latex](https://github.com/latex-lsp/tree-sitter-latex) - MIT License
- [linkerscript](https://github.com/tree-sitter-grammars/tree-sitter-linkerscript) - MIT License
- [llvm](https://github.com/benwilliamgraham/tree-sitter-llvm) - MIT License
- [lua](https://github.com/tree-sitter-grammars/tree-sitter-lua) - MIT License
- [luadoc](https://github.com/tree-sitter-grammars/tree-sitter-luadoc) - MIT License
- [luap](https://github.com/tree-sitter-grammars/tree-sitter-luap) - MIT License
- [luau](https://github.com/tree-sitter-grammars/tree-sitter-luau) - MIT License
- [magik](https://github.com/krn-robin/tree-sitter-magik) - MIT License
- [make](https://github.com/tree-sitter-grammars/tree-sitter-make) - MIT License
- [markdown](https://github.com/tree-sitter-grammars/tree-sitter-markdown) - MIT License
- [markdown_inline](https://github.com/tree-sitter-grammars/tree-sitter-markdown) - MIT License
- [matlab](https://github.com/acristoffers/tree-sitter-matlab) - MIT License
- [mermaid](https://github.com/monaqa/tree-sitter-mermaid) - MIT License
- [meson](https://github.com/tree-sitter-grammars/tree-sitter-meson) - MIT License
- [netlinx](https://github.com/Norgate-AV/tree-sitter-netlinx) - MIT License
- [ninja](https://github.com/alemuller/tree-sitter-ninja) - MIT License
- [nix](https://github.com/nix-community/tree-sitter-nix) - MIT License
- [nqc](https://github.com/tree-sitter-grammars/tree-sitter-nqc) - MIT License
- [objc](https://github.com/tree-sitter-grammars/tree-sitter-objc) - MIT License
- [ocaml/ocaml_interface](https://github.com/tree-sitter/tree-sitter-ocaml) - MIT License
- [odin](https://github.com/tree-sitter-grammars/tree-sitter-odin) - MIT License
- [org](https://github.com/milisims/tree-sitter-org) - MIT License
- [pascal](https://github.com/Isopod/tree-sitter-pascal) - MIT License
- [pem](https://github.com/tree-sitter-grammars/tree-sitter-pem) - MIT License
- [perl](https://github.com/tree-sitter-perl/tree-sitter-perl) - Artistic License 2.0
- [pgn](https://github.com/rolandwalker/tree-sitter-pgn) - BSD-2-Clause license
- [php](https://github.com/tree-sitter/tree-sitter-php) - MIT License
- [po](https://github.com/tree-sitter-grammars/tree-sitter-po) - MIT License
- [pony](https://github.com/tree-sitter-grammars/tree-sitter-pony) - MIT License
- [powershell](https://github.com/airbus-cert/tree-sitter-powershell) - MIT License
- [printf](https://github.com/tree-sitter-grammars/tree-sitter-printf) - ISC License
- [prisma](https://github.com/LumaKernel/tree-sitter-prisma) - MIT License
- [properties](https://github.com/tree-sitter-grammars/tree-sitter-properties) - MIT License
- [proto](https://github.com/coder3101/tree-sitter-proto) - MIT License
- [psv](https://github.com/amaanq/tree-sitter-csv) - MIT License
- [puppet](https://github.com/tree-sitter-grammars/tree-sitter-puppet) - MIT License
- [purescript](https://github.com/postsolar/tree-sitter-purescript) - MIT License
- [pymanifest](https://github.com/tree-sitter-grammars/tree-sitter-pymanifest) - MIT License
- [python](https://github.com/tree-sitter/tree-sitter-python) - MIT License
- [qmldir](https://github.com/tree-sitter-grammars/tree-sitter-qmldir) - MIT License
- [qmljs](https://github.com/yuja/tree-sitter-qmljs) - MIT License
- [query](https://github.com/tree-sitter-grammars/tree-sitter-query) - Apache License 2.0
- [r](https://github.com/r-lib/tree-sitter-r) - MIT License
- [racket](https://github.com/6cdh/tree-sitter-racket) - MIT License
- [rbs](https://github.com/joker1007/tree-sitter-rbs) - MIT License
- [re2c](https://github.com/tree-sitter-grammars/tree-sitter-re2c) - MIT License
- [readline](https://github.com/tree-sitter-grammars/tree-sitter-readline) - MIT License
- [rego](https://github.com/FallenAngel97/tree-sitter-rego) - MIT License
- [requirements](https://github.com/tree-sitter-grammars/tree-sitter-requirements) - MIT License
- [ron](https://github.com/tree-sitter-grammars/tree-sitter-ron) - Apache License 2.0
- [rst](https://github.com/stsewd/tree-sitter-rst) - MIT License
- [ruby](https://github.com/tree-sitter/tree-sitter-ruby) - MIT License
- [rust](https://github.com/tree-sitter/tree-sitter-rust) - MIT License
- [scala](https://github.com/tree-sitter/tree-sitter-scala) - MIT License
- [scheme](https://github.com/6cdh/tree-sitter-scheme) - MIT License
- [scss](https://github.com/tree-sitter-grammars/tree-sitter-scss) - MIT License
- [slang](https://github.com/tree-sitter-grammars/tree-sitter-slang) - MIT License
- [smali](https://github.com/tree-sitter-grammars/tree-sitter-smali) - MIT License
- [smithy](https://github.com/indoorvivants/tree-sitter-smithy) - MIT License
- [solidity](https://github.com/JoranHonig/tree-sitter-solidity) - MIT License
- [sparql](https://github.com/GordianDziwis/tree-sitter-sparql) - MIT License
- [sql](https://github.com/derekstride/tree-sitter-sql) - MIT License
- [squirrel](https://github.com/tree-sitter-grammars/tree-sitter-squirrel) - MIT License
- [starlark](https://github.com/tree-sitter-grammars/tree-sitter-starlark) - MIT License
- [svelte](https://github.com/tree-sitter-grammars/tree-sitter-svelte) - MIT License
- [swift](https://github.com/alex-pinkus/tree-sitter-swift) - MIT License
- [tablegen](https://github.com/tree-sitter-grammars/tree-sitter-tablegen) - MIT License
- [tcl](https://github.com/tree-sitter-grammars/tree-sitter-tcl) - MIT License
- [test](https://github.com/tree-sitter-grammars/tree-sitter-test) - MIT License
- [thrift](https://github.com/tree-sitter-grammars/tree-sitter-thrift) - MIT License
- [toml](https://github.com/tree-sitter-grammars/tree-sitter-toml) - MIT License
- [tsv](https://github.com/amaanq/tree-sitter-csv) - MIT License
- [tsx](https://github.com/tree-sitter/tree-sitter-typescript) - MIT License
- [twig](https://github.com/gbprod/tree-sitter-twig) - WTFPL License
- [typescript](https://github.com/tree-sitter/tree-sitter-typescript) - MIT License
- [typst](https://github.com/uben0/tree-sitter-typst) - MIT License
- [udev](https://github.com/tree-sitter-grammars/tree-sitter-udev) - MIT License
- [ungrammar](https://github.com/tree-sitter-grammars/tree-sitter-ungrammar) - MIT License
- [uxntal](https://github.com/tree-sitter-grammars/tree-sitter-uxntal) - MIT License
- [v](https://github.com/nedpals/tree-sitter-v) - MIT License
- [verilog](https://github.com/tree-sitter/tree-sitter-verilog) - MIT License
- [vhdl](https://github.com/alemuller/tree-sitter-vhdl) - MIT License
- [vim](https://github.com/tree-sitter-grammars/tree-sitter-vim) - MIT License
- [vue](https://github.com/tree-sitter-grammars/tree-sitter-vue) - MIT License
- [wgsl](https://github.com/szebniok/tree-sitter-wgsl) - MIT License
- [xcompose](https://github.com/tree-sitter-grammars/tree-sitter-xcompose) - MIT License
- [xml](https://github.com/tree-sitter-grammars/tree-sitter-xml) - MIT License
- [yaml](https://github.com/tree-sitter-grammars/tree-sitter-yaml) - MIT License
- [yuck](https://github.com/tree-sitter-grammars/tree-sitter-yuck) - MIT License
- [zig](https://github.com/maxxnino/tree-sitter-zig) - MIT License

## Adding New Languages

To add support for a new language:

1. Add an entry to `grammars.json` with:
   - `name`: The language identifier (lowercase)
   - `repo`: GitHub repository URL
   - `rev`: Git commit hash for stability
   - `path`: (optional) Subdirectory containing the grammar

2. Run `cargo build` to compile the new grammar

3. Update tests to verify the new language works

## Technical Details

The build process:
1. Reads `grammars.json` configuration
2. Clones specified grammar repositories at build time
3. Compiles each grammar using the `cc` crate
4. Generates Rust bindings using `tree_sitter_language::LanguageFn`
5. Provides a unified API for language access

Grammar repositories are cached in `target/debug/build/breeze-grammars-*/out/grammar_cache/` to speed up rebuilds.

## License

This crate is part of the breeze-rustle project and is licensed under the MIT License. See the [LICENSE](../../LICENSE) file for details.

## Acknowledgments

- [tree-sitter-language-pack](https://github.com/Goldziher/tree-sitter-language-pack) for the comprehensive grammar curation and testing
- [tree-sitter](https://tree-sitter.github.io/) for the parsing framework
- All the individual grammar authors listed above
