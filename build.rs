use std::env;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    // For now, let's use a simpler approach - we'll use the pre-compiled parsers
    // but wrap them in a way that works with tree-sitter 0.25
    
    let out_dir = env::var("OUT_DIR").unwrap();
    std::fs::write(
        std::path::Path::new(&out_dir).join("language_bindings.rs"),
        r#"
// Temporary stub until we implement proper grammar compilation
use tree_sitter::LanguageFn;

pub fn get_language_function(_name: &str) -> Option<LanguageFn> {
    // TODO: Implement proper grammar loading
    None
}
"#,
    ).unwrap();
}