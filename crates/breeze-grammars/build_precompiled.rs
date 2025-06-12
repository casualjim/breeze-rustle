use serde::{Deserialize, Serialize};
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;

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
}

#[derive(Debug, Deserialize, Serialize)]
struct GrammarsConfig {
  grammars: Vec<Grammar>,
}

#[derive(Clone)]
struct BuildTarget {
  rust_target: &'static str,
  zig_target: &'static str,
  platform_dir: &'static str,
}

const BUILD_TARGETS: &[BuildTarget] = &[
  // Linux targets
  BuildTarget {
    rust_target: "x86_64-unknown-linux-gnu",
    zig_target: "x86_64-linux-gnu",
    platform_dir: "linux-x86_64-glibc",
  },
  BuildTarget {
    rust_target: "x86_64-unknown-linux-musl",
    zig_target: "x86_64-linux-musl",
    platform_dir: "linux-x86_64-musl",
  },
  BuildTarget {
    rust_target: "aarch64-unknown-linux-gnu",
    zig_target: "aarch64-linux-gnu",
    platform_dir: "linux-aarch64-glibc",
  },
  BuildTarget {
    rust_target: "aarch64-unknown-linux-musl",
    zig_target: "aarch64-linux-musl",
    platform_dir: "linux-aarch64-musl",
  },
  // Windows targets
  BuildTarget {
    rust_target: "x86_64-pc-windows-gnu",
    zig_target: "x86_64-windows-gnu",
    platform_dir: "windows-x86_64",
  },
  BuildTarget {
    rust_target: "aarch64-pc-windows-gnu",
    zig_target: "aarch64-windows-gnu",
    platform_dir: "windows-aarch64",
  },
  // macOS targets (can't cross-compile to macOS with zig, but include for native builds)
  BuildTarget {
    rust_target: "x86_64-apple-darwin",
    zig_target: "x86_64-macos-none",
    platform_dir: "macos-x86_64",
  },
  BuildTarget {
    rust_target: "aarch64-apple-darwin",
    zig_target: "aarch64-macos-none",
    platform_dir: "macos-aarch64",
  },
];

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
      return Err(format!(
        "Failed to clone {}: {}",
        grammar.name,
        String::from_utf8_lossy(&clone_output.stderr)
      ));
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
      .args(&[
        "clone",
        "--depth",
        "1",
        "-b",
        branch,
        &repo_url,
        grammar_dir.to_str().unwrap(),
      ])
      .output()
      .expect("Failed to clone grammar repository")
  } else {
    // Clone default branch with shallow clone
    Command::new("git")
      .args(&[
        "clone",
        "--depth",
        "1",
        &repo_url,
        grammar_dir.to_str().unwrap(),
      ])
      .output()
      .expect("Failed to clone grammar repository")
  };

  if !output.status.success() {
    return Err(format!(
      "Failed to setup {}: {}",
      grammar.name,
      String::from_utf8_lossy(&output.stderr)
    ));
  }

  Ok(())
}

fn main() {
  println!("cargo:rerun-if-changed=grammars.json");
  println!("cargo:rerun-if-changed=build_precompiled.rs");

  // Check if zig is available
  let use_zig = which::which("zig").is_ok();
  if !use_zig {
    println!("cargo:warning=Zig not found, building only for current platform");
    println!(
      "cargo:warning=Install Zig to enable cross-compilation: https://ziglang.org/download/"
    );
  }

  // Determine which targets to build
  let targets_to_build: Vec<BuildTarget> = if use_zig {
    // Cross-compile to all targets if --all-targets is set
    if env::var("BREEZE_BUILD_ALL_TARGETS").is_ok() {
      println!("cargo:warning=Building for all targets with Zig cross-compilation");
      BUILD_TARGETS.to_vec()
    } else {
      // Otherwise just build for current target
      let target = env::var("TARGET").unwrap();
      BUILD_TARGETS
        .iter()
        .find(|t| t.rust_target == target)
        .cloned()
        .map(|t| vec![t])
        .unwrap_or_else(|| {
          println!(
            "cargo:warning=Unknown target {}, building for host only",
            target
          );
          vec![]
        })
    }
  } else {
    // Without zig, only build for current platform
    vec![]
  };

  // If no specific targets, build for host
  let targets_to_build = if targets_to_build.is_empty() {
    let target = env::var("TARGET").unwrap();
    let platform_dir = determine_platform_dir(&target);
    vec![BuildTarget {
      rust_target: Box::leak(target.into_boxed_str()),
      zig_target: "",
      platform_dir: Box::leak(platform_dir.into_boxed_str()),
    }]
  } else {
    targets_to_build
  };

  // Use a temporary directory for building
  let temp_dir = env::temp_dir().join("breeze-grammars-build");
  fs::create_dir_all(&temp_dir).expect("Failed to create temp directory");

  // Check for external cache directory (for CI)
  let cache_dir = if let Ok(external_cache) = env::var("BREEZE_GRAMMAR_CACHE") {
    println!(
      "cargo:warning=Using external grammar cache: {}",
      external_cache
    );
    Path::new(&external_cache).to_path_buf()
  } else {
    temp_dir.join("grammar_cache")
  };

  fs::create_dir_all(&cache_dir).unwrap();

  // Read grammars configuration
  let config_str = fs::read_to_string("grammars.json").expect("Failed to read grammars.json");
  let config: GrammarsConfig =
    serde_json::from_str(&config_str).expect("Failed to parse grammars.json");

  println!(
    "cargo:warning=Processing {} grammars for precompilation",
    config.grammars.len()
  );

  // Phase 1: Clone all repositories in parallel
  println!("cargo:warning=Phase 1: Cloning repositories in parallel");
  let repos_to_clone: Vec<_> = config
    .grammars
    .iter()
    .filter(|g| !cache_dir.join(&g.name).exists())
    .collect();

  if !repos_to_clone.is_empty() {
    println!(
      "cargo:warning=Cloning {} new repositories",
      repos_to_clone.len()
    );

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

  // Phase 2: Build for each target
  for target in targets_to_build {
    println!("cargo:warning=Building for target: {}", target.platform_dir);
    build_for_target(&config, &cache_dir, &target, use_zig);
  }
}

fn build_for_target(
  config: &GrammarsConfig,
  cache_dir: &Path,
  target: &BuildTarget,
  use_zig: bool,
) {
  let precompiled_base = Path::new("precompiled");
  let platform_dir = precompiled_base.join(target.platform_dir);
  fs::create_dir_all(&platform_dir).expect("Failed to create precompiled directory");

  let mut compiled_grammars = Vec::new();
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
      eprintln!(
        "No source directory found for {} at {:?}",
        grammar.name, src_dir
      );
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
      grammar.name.clone(),
      parser_c,
      scanner_c,
      scanner_cc,
      has_scanner,
      _is_cpp,
      src_dir,
      grammar_dir,
    ));
  }

  // Set parallelism for cc crate
  let num_cpus = std::thread::available_parallelism()
    .map(|n| n.get())
    .unwrap_or(num_cpus::get());
  env::set_var("CARGO_BUILD_JOBS", num_cpus.to_string());

  // Compile each grammar to the precompiled directory
  for (name, parser_c, scanner_c, scanner_cc, has_scanner, _is_cpp, src_dir, grammar_dir) in
    all_builds
  {
    println!(
      "cargo:warning=Compiling grammar: {} for {}",
      name, target.platform_dir
    );

    let mut build = cc::Build::new();

    // Configure for cross-compilation with zig
    if use_zig && !target.zig_target.is_empty() {
      build.compiler("zig");
      build.target(target.rust_target);
      // Add zig-specific flags
      build.flag("-target");
      build.flag(target.zig_target);
      // Use zig's cross-compilation capabilities
      env::set_var("CC", "zig cc");
      env::set_var("CXX", "zig c++");
      env::set_var("AR", "zig ar");
    }

    build.include(&src_dir);
    build.include(&grammar_dir);

    // Set C standard
    build.flag_if_supported("-std=c11");

    // Add parser.c
    build.file(&parser_c);

    // Add scanner if present
    if has_scanner {
      if scanner_c.exists() {
        build.file(&scanner_c);
      } else if scanner_cc.exists() {
        build.file(&scanner_cc);
        build.cpp(true);
        build.flag_if_supported("-std=c++14");
      }
    }

    // Enable optimizations
    build.opt_level(3);
    build.flag_if_supported("-fno-exceptions");

    // Additional optimization flags
    if use_zig && !target.zig_target.is_empty() {
      // Zig-specific optimizations
      build.flag_if_supported("-O3"); // Redundant with opt_level but ensures it's passed
      build.flag_if_supported("-flto"); // Link-time optimization
      build.flag_if_supported("-march=native"); // Won't work for cross-compile, but zig ignores it properly
    }

    // General optimization flags that work with both zig and native compilers
    build.flag_if_supported("-funroll-loops");
    build.flag_if_supported("-fomit-frame-pointer");
    build.flag_if_supported("-ffast-math"); // Safe for parser code
    build.flag_if_supported("-finline-functions");

    // Size optimizations that often improve performance too
    build.flag_if_supported("-ffunction-sections");
    build.flag_if_supported("-fdata-sections");

    // For static libraries, enable interprocedural optimizations
    build.flag_if_supported("-fipa-pta");
    build.flag_if_supported("-fdevirtualize-at-ltrans");

    // Set output directory to precompiled platform directory
    build.out_dir(&platform_dir);

    // Compile the grammar
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
      build.compile(&format!("tree-sitter-{}", name));
    })) {
      Ok(_) => {
        compiled_grammars.push(name);
      }
      Err(_) => {
        eprintln!("Failed to compile {}", name);
        failed_grammars.push(name);
      }
    }
  }

  if !failed_grammars.is_empty() {
    eprintln!(
      "Warning: Failed to compile {} grammars: {:?}",
      failed_grammars.len(),
      failed_grammars
    );
  }

  // Sort for consistent output
  compiled_grammars.sort();

  println!(
    "cargo:warning=Successfully compiled {} grammars to {}",
    compiled_grammars.len(),
    platform_dir.display()
  );

  // Save metadata about compiled grammars
  let metadata =
    serde_json::to_string_pretty(&compiled_grammars).expect("Failed to serialize grammar list");
  fs::write(platform_dir.join("grammars.json"), metadata).expect("Failed to write grammars.json");

  // Generate bindings file
  generate_bindings(&platform_dir, &compiled_grammars);
}

fn determine_platform_dir(target: &str) -> String {
  let arch = if target.contains("x86_64") {
    "x86_64"
  } else if target.contains("aarch64") {
    "aarch64"
  } else {
    "unknown"
  };

  let os = if target.contains("windows") {
    "windows"
  } else if target.contains("darwin") {
    "macos"
  } else if target.contains("linux") {
    if target.contains("musl") {
      "linux-musl"
    } else {
      "linux-glibc"
    }
  } else {
    "unknown"
  };

  format!("{}-{}", os, arch)
}

fn generate_bindings(platform_dir: &Path, compiled_grammars: &[String]) {
  let bindings_path = platform_dir.join("grammars.rs");
  let mut bindings = String::new();

  bindings.push_str("// Auto-generated grammar bindings\n\n");
  bindings.push_str("use tree_sitter::Language;\n");
  bindings.push_str("use tree_sitter_language::LanguageFn;\n\n");

  // Generate extern declarations
  for name in compiled_grammars {
    let fn_name = if name == "csharp" {
      "c_sharp".to_string()
    } else {
      name.replace("-", "_")
    };
    bindings.push_str(&format!(
      "extern \"C\" {{ fn tree_sitter_{}() -> *const (); }}\n",
      fn_name
    ));
  }

  bindings.push_str("\n");

  // Generate LanguageFn constants
  for name in compiled_grammars {
    let fn_name = if name == "csharp" {
      "c_sharp".to_string()
    } else {
      name.replace("-", "_")
    };
    let const_name = name.to_uppercase();
    bindings.push_str(&format!(
      "pub const {}_LANGUAGE: LanguageFn = unsafe {{ LanguageFn::from_raw(tree_sitter_{}) }};\n",
      const_name, fn_name
    ));
  }

  bindings.push_str("\n");
  bindings.push_str("pub fn load_grammar(name: &str) -> Option<Language> {\n");
  bindings.push_str("    match name {\n");

  // Generate match arms
  for name in compiled_grammars {
    let const_name = name.to_uppercase();
    bindings.push_str(&format!(
      "        \"{}\" => Some({}_LANGUAGE.into()),\n",
      name, const_name
    ));
  }

  bindings.push_str("        _ => None,\n");
  bindings.push_str("    }\n");
  bindings.push_str("}\n\n");

  bindings.push_str("pub fn load_grammar_fn(name: &str) -> Option<LanguageFn> {\n");
  bindings.push_str("    match name {\n");

  // Generate match arms for LanguageFn
  for name in compiled_grammars {
    let const_name = name.to_uppercase();
    bindings.push_str(&format!(
      "        \"{}\" => Some({}_LANGUAGE),\n",
      name, const_name
    ));
  }

  bindings.push_str("        _ => None,\n");
  bindings.push_str("    }\n");
  bindings.push_str("}\n\n");

  bindings.push_str("pub fn available_grammars() -> &'static [&'static str] {\n");
  bindings.push_str("    &[\n");
  for name in compiled_grammars {
    bindings.push_str(&format!("        \"{}\",\n", name));
  }
  bindings.push_str("    ]\n");
  bindings.push_str("}\n");

  fs::write(bindings_path, bindings).unwrap();
}
