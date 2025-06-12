use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Grammar {
    name: String,
    repo: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    branch: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rev: Option<String>,
    #[serde(default)]
    path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    symbol_name: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct GrammarsConfig {
    grammars: Vec<Grammar>,
}

fn clone_repo(grammar: &Grammar, cache_dir: &Path) -> Result<(), String> {
    let grammar_dir = cache_dir.join(&grammar.name);
    
    // Skip if already cloned
    if grammar_dir.exists() {
        return Ok(());
    }
    
    let repo_url = if grammar.repo.starts_with("http") {
        grammar.repo.clone()
    } else {
        format!("https://github.com/{}", grammar.repo)
    };
    
    let output = if let Some(rev) = &grammar.rev {
        // Clone and checkout specific commit
        let clone_output = Command::new("git")
            .args(&["clone", &repo_url, grammar_dir.to_str().unwrap()])
            .output()
            .expect("Failed to clone grammar repository");
        
        if !clone_output.status.success() {
            return Err(format!("Failed to clone {}: {}", grammar.name, String::from_utf8_lossy(&clone_output.stderr)));
        }
        
        // Checkout specific revision
        Command::new("git")
            .current_dir(&grammar_dir)
            .args(&["checkout", rev])
            .output()
            .expect("Failed to checkout revision")
    } else if let Some(branch) = &grammar.branch {
        // Clone specific branch with shallow clone
        Command::new("git")
            .args(&["clone", "--depth", "1", "-b", branch, &repo_url, grammar_dir.to_str().unwrap()])
            .output()
            .expect("Failed to clone grammar repository")
    } else {
        // Clone default branch with shallow clone
        Command::new("git")
            .args(&["clone", "--depth", "1", &repo_url, grammar_dir.to_str().unwrap()])
            .output()
            .expect("Failed to clone grammar repository")
    };
        
    if !output.status.success() {
        return Err(format!("Failed to setup {}: {}", grammar.name, String::from_utf8_lossy(&output.stderr)));
    }
    
    Ok(())
}

fn main() {
    println!("cargo:rerun-if-changed=grammars.json");
    println!("cargo:rerun-if-changed=build.rs");
    
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);
    
    // Check for precompiled binaries first
    let target = env::var("TARGET").unwrap();
    let arch = if target.contains("x86_64") {
        "x86_64"
    } else if target.contains("aarch64") {
        "aarch64"
    } else {
        // Fall back to building from source for unsupported architectures
        compile_from_source(out_path);
        return;
    };
    
    let os = if target.contains("windows") {
        "windows"
    } else if target.contains("darwin") {
        "macos"
    } else if target.contains("linux") {
        "linux"
    } else {
        // Fall back to building from source for unsupported OS
        compile_from_source(out_path);
        return;
    };
    
    // Map to the new platform directory structure
    let platform_suffix = if target.contains("musl") {
        "musl"
    } else if target.contains("linux") {
        "glibc" 
    } else {
        ""
    };
    
    let platform_dir = if platform_suffix.is_empty() {
        format!("{}-{}", os, arch)
    } else {
        format!("{}-{}-{}", os, arch, platform_suffix)
    };
    
    let precompiled_dir = Path::new("precompiled").join(&platform_dir);
    
    // If precompiled binaries exist, use them
    if precompiled_dir.exists() && precompiled_dir.join("grammars.json").exists() {
        println!("cargo:warning=Using precompiled grammars from {}", precompiled_dir.display());
        use_precompiled_binaries(&precompiled_dir, out_path);
        return;
    }
    
    // Otherwise, compile from source
    println!("cargo:warning=No precompiled grammars found, building from source");
    compile_from_source(out_path);
}

fn use_precompiled_binaries(precompiled_dir: &Path, out_path: &Path) {
    // Copy all library files
    let lib_pattern = if cfg!(windows) { "lib" } else { "a" };
    
    for entry in fs::read_dir(precompiled_dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();
        
        // Copy library files
        if path.extension().map_or(false, |ext| ext == lib_pattern) {
            let filename = path.file_name().unwrap();
            let dest = out_path.join(filename);
            fs::copy(&path, &dest).unwrap();
            
            // Tell Cargo where to find the library
            let lib_name = path.file_stem().unwrap().to_str().unwrap();
            if lib_name.starts_with("lib") {
                // Unix-style library
                println!("cargo:rustc-link-lib=static={}", &lib_name[3..]);
            } else {
                // Windows-style library
                println!("cargo:rustc-link-lib=static={}", lib_name);
            }
        }
    }
    
    println!("cargo:rustc-link-search=native={}", out_path.display());
    
    // Check if any grammars use C++ scanners by looking for marker files
    let has_cpp_grammars = precompiled_dir.read_dir()
        .map(|entries| {
            entries.filter_map(|e| e.ok())
                .any(|entry| {
                    entry.path().extension()
                        .map_or(false, |ext| ext == "cpp")
                })
        })
        .unwrap_or(false);
    
    // Link C++ standard library if needed
    if has_cpp_grammars {
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-lib=c++");
        } else if cfg!(target_os = "linux") {
            println!("cargo:rustc-link-lib=stdc++");
        }
    }
    
    // Copy the grammars.rs bindings file
    if let Ok(bindings) = fs::read_to_string(precompiled_dir.join("grammars.rs")) {
        fs::write(out_path.join("grammars.rs"), bindings).unwrap();
    } else {
        // Generate bindings from the grammars.json metadata
        let grammars: Vec<Grammar> = serde_json::from_str(
            &fs::read_to_string(precompiled_dir.join("grammars.json"))
                .expect("Missing grammars.json in precompiled directory")
        ).expect("Invalid grammars.json");
        
        generate_bindings(out_path, &grammars);
    }
}

fn compile_from_source(out_path: &Path) {
    // Set up sccache if available
    if which::which("sccache").is_ok() {
        println!("cargo:warning=sccache detected, enabling compilation caching");
        env::set_var("CC", "sccache cc");
        env::set_var("CXX", "sccache c++");
        env::set_var("RUSTC_WRAPPER", "sccache");
    }
    
    // Check for external cache directory (for CI)
    let cache_dir = if let Ok(external_cache) = env::var("BREEZE_GRAMMAR_CACHE") {
        println!("cargo:warning=Using external grammar cache: {}", external_cache);
        Path::new(&external_cache).to_path_buf()
    } else {
        out_path.join("grammar_cache")
    };
    
    fs::create_dir_all(&cache_dir).unwrap();
    
    // Read grammars configuration
    let config_str = fs::read_to_string("grammars.json")
        .expect("Failed to read grammars.json");
    let config: GrammarsConfig = serde_json::from_str(&config_str)
        .expect("Failed to parse grammars.json");
    
    println!("cargo:warning=Processing {} grammars", config.grammars.len());
    
    // Set parallelism for cc crate
    let num_cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(num_cpus::get());
    env::set_var("CARGO_BUILD_JOBS", num_cpus.to_string());
    
    // Phase 1: Clone all repositories in parallel
    println!("cargo:warning=Phase 1: Cloning repositories in parallel");
    let repos_to_clone: Vec<_> = config.grammars.iter()
        .filter(|g| !cache_dir.join(&g.name).exists())
        .collect();
    
    if !repos_to_clone.is_empty() {
        println!("cargo:warning=Cloning {} new repositories", repos_to_clone.len());
        
        // Create thread pool for cloning
        let mut handles = vec![];
        let failed_clones = Arc::new(Mutex::new(Vec::new()));
        
        for grammar in repos_to_clone {
            let grammar = grammar.clone();
            let cache_dir = cache_dir.clone();
            let failed_clones = failed_clones.clone();
            
            let handle = thread::spawn(move || {
                println!("cargo:warning=Cloning {}", grammar.name);
                if let Err(e) = clone_repo(&grammar, &cache_dir) {
                    eprintln!("Failed to clone {}: {}", grammar.name, e);
                    failed_clones.lock().unwrap().push(grammar.name);
                }
            });
            
            handles.push(handle);
        }
        
        // Wait for all clones to complete
        for handle in handles {
            handle.join().expect("Thread panicked");
        }
        
        let failed = Arc::try_unwrap(failed_clones)
            .expect("Failed to unwrap Arc")
            .into_inner()
            .expect("Failed to extract from Mutex");
        
        if !failed.is_empty() {
            panic!("Failed to clone repositories: {:?}", failed);
        }
    }
    
    // Phase 2: Compile all grammars in a single cc::Build invocation
    println!("cargo:warning=Phase 2: Compiling all grammars with parallel cc");
    
    let mut compiled_grammars: Vec<Grammar> = Vec::new();
    let mut failed_grammars = Vec::new();
    
    // Process each grammar and collect all source files
    let mut all_builds = Vec::new();
    
    for grammar in &config.grammars {
        let grammar_dir = cache_dir.join(&grammar.name);
        
        // Determine source directory
        let src_dir = if let Some(path) = &grammar.path {
            grammar_dir.join(path).join("src")
        } else {
            grammar_dir.join("src")
        };
        
        if !src_dir.exists() {
            eprintln!("No source directory found for {} at {:?}", grammar.name, src_dir);
            failed_grammars.push(grammar.name.clone());
            continue;
        }
        
        // Check for required files
        let parser_c = src_dir.join("parser.c");
        if !parser_c.exists() {
            eprintln!("No parser.c found for {}", grammar.name);
            failed_grammars.push(grammar.name.clone());
            continue;
        }
        
        let scanner_c = src_dir.join("scanner.c");
        let scanner_cc = src_dir.join("scanner.cc");
        let has_scanner = scanner_c.exists() || scanner_cc.exists();
        let _is_cpp = scanner_cc.exists();
        
        // Store build info for this grammar
        all_builds.push((
            grammar.clone(),
            parser_c,
            scanner_c,
            scanner_cc,
            has_scanner,
            _is_cpp,
            src_dir,
            grammar_dir,
        ));
    }
    
    // Compile each grammar individually but let cc use parallelism internally
    for (grammar, parser_c, scanner_c, scanner_cc, has_scanner, is_cpp, src_dir, grammar_dir) in &all_builds {
        println!("cargo:warning=Compiling grammar: {}", grammar.name);
        
        let mut build = cc::Build::new();
        build.include(src_dir);
        build.include(grammar_dir);
        
        // Set C standard
        build.flag_if_supported("-std=c11");
        
        // Add parser.c
        build.file(parser_c);
        
        // Add scanner if present
        if *has_scanner {
            if scanner_c.exists() {
                build.file(scanner_c);
            } else if scanner_cc.exists() {
                build.file(scanner_cc);
                build.cpp(true);
                build.flag_if_supported("-std=c++14");
            }
        }
        
        // Enable optimizations
        build.opt_level(3);
        build.flag_if_supported("-fno-exceptions");
        
        // Additional optimization flags for maximum performance
        build.flag_if_supported("-funroll-loops");
        build.flag_if_supported("-fomit-frame-pointer");
        build.flag_if_supported("-ffast-math");  // Safe for parser code
        build.flag_if_supported("-finline-functions");
        
        // Size optimizations that often improve performance too
        build.flag_if_supported("-ffunction-sections");
        build.flag_if_supported("-fdata-sections");
        
        // For static libraries, enable interprocedural optimizations
        build.flag_if_supported("-fipa-pta");
        build.flag_if_supported("-fdevirtualize-at-ltrans");
        
        // If using sccache, it handles LTO caching well
        if which::which("sccache").is_ok() {
            build.flag_if_supported("-flto");
        }
        
        // Compile the grammar
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            build.compile(&format!("tree-sitter-{}", grammar.name));
        })) {
            Ok(_) => {
                compiled_grammars.push(grammar.clone());
            }
            Err(_) => {
                eprintln!("Failed to compile {}", grammar.name);
                failed_grammars.push(grammar.name.clone());
            }
        }
    }
    
    if !failed_grammars.is_empty() {
        eprintln!("Warning: Failed to compile {} grammars: {:?}", failed_grammars.len(), failed_grammars);
    }
    
    // Sort for consistent output
    compiled_grammars.sort_by(|a, b| a.name.cmp(&b.name));
    
    println!("cargo:warning=Successfully compiled {} grammars", compiled_grammars.len());
    
    // Generate bindings
    generate_bindings(out_path, &compiled_grammars);
    
    // Track which grammars use C++ scanners
    let mut has_cpp_scanner = false;
    
    for (grammar, _, _, _, _, is_cpp, _, _) in &all_builds {
        if compiled_grammars.iter().any(|g| g.name == grammar.name) && *is_cpp {
            has_cpp_scanner = true;
            // Create marker file for future reference
            let marker_path = out_path.join(format!("{}.cpp", grammar.name));
            fs::write(marker_path, "").unwrap();
        }
    }
    
    // Link C++ standard library if any grammar uses C++ scanner
    println!("cargo:rustc-link-search=native={}", out_path.display());
    if has_cpp_scanner {
        if cfg!(target_os = "macos") {
            println!("cargo:rustc-link-lib=c++");
        } else if cfg!(target_os = "linux") {
            println!("cargo:rustc-link-lib=stdc++");
        }
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
    
    bindings.push_str("\n");
    
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
    
    bindings.push_str("\n");
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