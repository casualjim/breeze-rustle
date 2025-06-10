use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct Grammar {
    name: String,
    repo: String,
    branch: String,
    #[serde(default)]
    path: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct GrammarsConfig {
    grammars: Vec<Grammar>,
}

fn main() {
    println!("cargo:rerun-if-changed=grammars.json");
    println!("cargo:rerun-if-changed=build.rs");
    
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);
    let cache_dir = out_path.join("grammar_cache");
    fs::create_dir_all(&cache_dir).unwrap();
    
    // Read grammars configuration
    let config_str = fs::read_to_string("grammars.json")
        .expect("Failed to read grammars.json");
    let config: GrammarsConfig = serde_json::from_str(&config_str)
        .expect("Failed to parse grammars.json");
    
    let mut compiled_grammars = Vec::new();
    
    for grammar in &config.grammars {
        println!("cargo:warning=Building grammar: {}", grammar.name);
        
        let grammar_dir = cache_dir.join(&grammar.name);
        
        // Clone or update the repository
        if !grammar_dir.exists() {
            let output = Command::new("git")
                .args(&["clone", "--depth", "1", "-b", &grammar.branch, &grammar.repo, grammar_dir.to_str().unwrap()])
                .output()
                .expect("Failed to clone grammar repository");
                
            if !output.status.success() {
                panic!("Failed to clone {}: {}", grammar.name, String::from_utf8_lossy(&output.stderr));
            }
        }
        
        // Determine source directory
        let src_dir = if let Some(path) = &grammar.path {
            grammar_dir.join(path)
        } else {
            grammar_dir.join("src")
        };
        
        if !src_dir.exists() {
            eprintln!("Warning: No source directory found for {}", grammar.name);
            continue;
        }
        
        // Compile the grammar
        let mut build = cc::Build::new();
        build.include(&src_dir);
        build.include(&grammar_dir);
        
        // Add parser.c
        let parser_c = src_dir.join("parser.c");
        if parser_c.exists() {
            build.file(parser_c);
        } else {
            eprintln!("Warning: No parser.c found for {}", grammar.name);
            continue;
        }
        
        // Add scanner.c or scanner.cc if present
        let scanner_c = src_dir.join("scanner.c");
        let scanner_cc = src_dir.join("scanner.cc");
        
        if scanner_c.exists() {
            build.file(scanner_c);
        } else if scanner_cc.exists() {
            build.file(scanner_cc);
            build.cpp(true);
        }
        
        // Set C standard
        build.flag_if_supported("-std=c11");
        
        // Compile the grammar
        build.compile(&format!("tree-sitter-{}", grammar.name));
        
        compiled_grammars.push(grammar.name.clone());
    }
    
    // Generate bindings file
    let bindings_path = out_path.join("grammars.rs");
    let mut bindings = String::new();
    
    bindings.push_str("// Auto-generated grammar bindings\n\n");
    bindings.push_str("use std::collections::HashMap;\n");
    bindings.push_str("use tree_sitter::Language;\n\n");
    
    // Generate extern declarations
    for name in &compiled_grammars {
        let fn_name = name.replace("-", "_");
        bindings.push_str(&format!(
            "extern \"C\" {{ fn tree_sitter_{}() -> Language; }}\n",
            fn_name
        ));
    }
    
    bindings.push_str("\n");
    bindings.push_str("pub fn load_grammar(name: &str) -> Option<Language> {\n");
    bindings.push_str("    match name {\n");
    
    // Generate match arms
    for name in &compiled_grammars {
        let fn_name = name.replace("-", "_");
        bindings.push_str(&format!(
            "        \"{}\" => Some(unsafe {{ tree_sitter_{}() }}),\n",
            name, fn_name
        ));
    }
    
    bindings.push_str("        _ => None,\n");
    bindings.push_str("    }\n");
    bindings.push_str("}\n\n");
    
    bindings.push_str("pub fn available_grammars() -> Vec<&'static str> {\n");
    bindings.push_str("    vec![\n");
    for name in &compiled_grammars {
        bindings.push_str(&format!("        \"{}\",\n", name));
    }
    bindings.push_str("    ]\n");
    bindings.push_str("}\n");
    
    fs::write(bindings_path, bindings).unwrap();
}