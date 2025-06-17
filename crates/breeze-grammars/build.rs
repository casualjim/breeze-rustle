use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Grammar {
  name: String,
  #[serde(skip_serializing_if = "Option::is_none")]
  symbol_name: Option<String>,
}

fn get_target_parser_path() -> Result<PathBuf, String> {
  // First check if PARSER_LIB environment variable is set
  if let Ok(parser_lib) = env::var("PARSER_LIB") {
    eprintln!("Using PARSER_LIB from environment: {}", parser_lib);
    return Ok(PathBuf::from(parser_lib));
  }

  // Get the target triple
  let target = env::var("TARGET").unwrap_or_else(|_| {
    // If TARGET is not set, fall back to host
    env::var("HOST").unwrap_or_else(|_| "unknown".to_string())
  });

  eprintln!("Building for target: {}", target);

  // Map Rust target triple to npm package name
  let npm_package = match target.as_str() {
    "aarch64-apple-darwin" => "@kumos/tree-sitter-parsers-darwin-arm64",
    "x86_64-apple-darwin" => "@kumos/tree-sitter-parsers-darwin-x64",
    "aarch64-unknown-linux-gnu" => "@kumos/tree-sitter-parsers-linux-arm64",
    "x86_64-unknown-linux-gnu" => "@kumos/tree-sitter-parsers-linux-x64",
    "aarch64-unknown-linux-musl" => "@kumos/tree-sitter-parsers-linux-arm64-musl",
    "x86_64-unknown-linux-musl" => "@kumos/tree-sitter-parsers-linux-x64-musl",
    "aarch64-pc-windows-msvc" => "@kumos/tree-sitter-parsers-win32-arm64",
    "x86_64-pc-windows-msvc" | "x86_64-pc-windows-gnu" => "@kumos/tree-sitter-parsers-win32-x64",
    _ => {
      // For unknown targets, try to use npx which will auto-install if needed
      eprintln!("Unknown target '{}', trying npx fallback", target);
      return get_npm_parser_path();
    }
  };

  // Try to find the platform-specific package in node_modules
  let node_modules = find_node_modules()?;
  let package_dir = node_modules.join(npm_package);

  if package_dir.exists() {
    eprintln!("Found platform package at: {}", package_dir.display());

    // Construct expected filename based on target
    let expected_filename = match target.as_str() {
      "aarch64-apple-darwin" => "libtree-sitter-parsers-all-macos-aarch64.a",
      "x86_64-apple-darwin" => "libtree-sitter-parsers-all-macos-x86_64.a",
      "aarch64-unknown-linux-gnu" => "libtree-sitter-parsers-all-linux-aarch64.a",
      "x86_64-unknown-linux-gnu" => "libtree-sitter-parsers-all-linux-x86_64.a",
      "aarch64-unknown-linux-musl" => "libtree-sitter-parsers-all-linux-aarch64-musl.a",
      "x86_64-unknown-linux-musl" => "libtree-sitter-parsers-all-linux-x86_64-musl.a",
      "aarch64-pc-windows-msvc" => "libtree-sitter-parsers-all-windows-aarch64.a",
      "x86_64-pc-windows-msvc" | "x86_64-pc-windows-gnu" => {
        "libtree-sitter-parsers-all-windows-x86_64.a"
      }
      _ => {
        return Err(format!(
          "No parser library found in package: {}",
          npm_package
        ));
      }
    };

    let lib_path = package_dir.join(expected_filename);
    if lib_path.exists() {
      eprintln!("Found parser library: {}", lib_path.display());
      return Ok(lib_path);
    } else {
      return Err(format!(
        "Expected library {} not found in package: {}",
        expected_filename, npm_package
      ));
    }
  }

  // Fall back to npx
  eprintln!(
    "Platform package {} not found, falling back to npx",
    npm_package
  );
  get_npm_parser_path()
}

fn find_node_modules() -> Result<PathBuf, String> {
  // Start from the current directory and walk up looking for node_modules
  let mut current = env::current_dir().map_err(|e| format!("Failed to get current dir: {}", e))?;

  loop {
    let node_modules = current.join("node_modules");
    if node_modules.exists() && node_modules.is_dir() {
      return Ok(node_modules);
    }

    if !current.pop() {
      return Err("Could not find node_modules directory".to_string());
    }
  }
}

fn get_npm_parser_path() -> Result<PathBuf, String> {
  // Use npx which will auto-install if needed
  eprintln!("Getting tree-sitter parsers path via npx (will auto-install if needed)...");
  let output = Command::new("npx")
    .arg("--yes") // Automatically install if not present
    .arg("@kumos/tree-sitter-parsers")
    .output()
    .map_err(|e| {
      format!(
        "Failed to run npx: {}. Make sure Node.js and npm are installed.",
        e
      )
    })?;

  if !output.status.success() {
    return Err(format!(
      "Failed to get parser path: {}",
      String::from_utf8_lossy(&output.stderr)
    ));
  }

  let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();

  if path_str.is_empty() {
    return Err("Got empty path from tree-sitter-parsers-path".to_string());
  }

  Ok(PathBuf::from(path_str))
}

fn main() {
  println!("cargo:rerun-if-changed=build.rs");

  let out_dir = env::var("OUT_DIR").unwrap();
  let out_path = Path::new(&out_dir);

  // Get target-specific parsers
  let parser_lib_path = get_target_parser_path()
    .expect("Failed to get npm parsers. Make sure Node.js and npm are installed.");

  eprintln!(
    "Using tree-sitter parsers from {}",
    parser_lib_path.display()
  );

  use_npm_parsers(&parser_lib_path, out_path);
}

fn use_npm_parsers(parser_lib_path: &Path, out_path: &Path) {
  // Link the npm parser library
  let lib_name = parser_lib_path.file_stem().unwrap().to_str().unwrap();
  let lib_name = lib_name.strip_prefix("lib").unwrap_or(lib_name);

  println!("cargo:rustc-link-lib=static={}", lib_name);
  println!(
    "cargo:rustc-link-search=native={}",
    parser_lib_path.parent().unwrap().display()
  );

  // Always link C++ standard library as some grammars use C++ scanners
  if cfg!(target_os = "macos") {
    println!("cargo:rustc-link-lib=c++");
  } else if cfg!(target_os = "linux") {
    println!("cargo:rustc-link-lib=stdc++");
  } else if cfg!(target_os = "windows") {
    // Windows doesn't need explicit C++ runtime linking with MSVC
  }

  // Check if there's a metadata file alongside the library
  let metadata_path = parser_lib_path.parent().unwrap().join(
    parser_lib_path
      .file_name()
      .unwrap()
      .to_str()
      .unwrap()
      .replace(".a", ".json")
      .replace("libtree-sitter-parsers-all-", "grammars-"),
  );

  if metadata_path.exists() {
    eprintln!("Found grammar metadata at: {}", metadata_path.display());
    let metadata_str = fs::read_to_string(&metadata_path).expect("Failed to read grammar metadata");
    let grammars: Vec<Grammar> =
      serde_json::from_str(&metadata_str).expect("Failed to parse grammar metadata");
    generate_bindings(out_path, &grammars);
  } else {
    panic!("No grammar metadata found. The npm package must provide a grammars-*.json file.");
  }
}

fn generate_bindings(out_path: &Path, compiled_grammars: &[Grammar]) {
  let bindings_path = out_path.join("grammars.rs");
  let mut bindings = String::new();

  bindings.push_str("// Auto-generated grammar bindings\n\n");
  bindings.push_str("use tree_sitter::Language;\n");
  bindings.push_str("use tree_sitter_language::LanguageFn;\n\n");

  // Generate extern declarations
  for grammar in compiled_grammars {
    let fn_name = if let Some(symbol) = &grammar.symbol_name {
      symbol.clone()
    } else if grammar.name == "csharp" {
      "c_sharp".to_string()
    } else {
      grammar.name.replace("-", "_")
    };
    bindings.push_str(&format!(
      "extern \"C\" {{ fn tree_sitter_{}() -> *const (); }}\n",
      fn_name
    ));
  }

  bindings.push('\n');

  // Generate LanguageFn constants
  for grammar in compiled_grammars {
    let fn_name = if let Some(symbol) = &grammar.symbol_name {
      symbol.clone()
    } else if grammar.name == "csharp" {
      "c_sharp".to_string()
    } else {
      grammar.name.replace("-", "_")
    };
    let const_name = grammar.name.to_uppercase();
    bindings.push_str(&format!(
      "pub const {}_LANGUAGE: LanguageFn = unsafe {{ LanguageFn::from_raw(tree_sitter_{}) }};\n",
      const_name, fn_name
    ));
  }

  bindings.push('\n');
  bindings.push_str("pub fn load_grammar(name: &str) -> Option<Language> {\n");
  bindings.push_str("    match name {\n");

  // Generate match arms
  for grammar in compiled_grammars {
    let const_name = grammar.name.to_uppercase();
    bindings.push_str(&format!(
      "        \"{}\" => Some({}_LANGUAGE.into()),\n",
      grammar.name, const_name
    ));
  }

  bindings.push_str("        _ => None,\n");
  bindings.push_str("    }\n");
  bindings.push_str("}\n\n");

  bindings.push_str("pub fn load_grammar_fn(name: &str) -> Option<LanguageFn> {\n");
  bindings.push_str("    match name {\n");

  // Generate match arms for LanguageFn
  for grammar in compiled_grammars {
    let const_name = grammar.name.to_uppercase();
    bindings.push_str(&format!(
      "        \"{}\" => Some({}_LANGUAGE),\n",
      grammar.name, const_name
    ));
  }

  bindings.push_str("        _ => None,\n");
  bindings.push_str("    }\n");
  bindings.push_str("}\n\n");

  bindings.push_str("pub fn available_grammars() -> &'static [&'static str] {\n");
  bindings.push_str("    &[\n");
  for grammar in compiled_grammars {
    bindings.push_str(&format!("        \"{}\",\n", grammar.name));
  }
  bindings.push_str("    ]\n");
  bindings.push_str("}\n");

  fs::write(bindings_path, bindings).unwrap();
}
