use breeze_grammars::{get_language, get_language_fn, supported_languages};

fn main() {
    let languages = supported_languages();
    println!("Testing {} languages...\n", languages.len());
    
    let mut successes = 0;
    let mut failures = Vec::new();
    
    for lang_name in &languages {
        print!("Testing {:<20} ", format!("{}...", lang_name));
        
        // Test get_language
        match get_language(lang_name) {
            Some(language) => {
                // Basic sanity check - language should have a valid node kind count
                let node_kind_count = language.node_kind_count();
                if node_kind_count > 0 {
                    // Test get_language_fn as well
                    match get_language_fn(lang_name) {
                        Some(_fn) => {
                            println!("✓ (node kinds: {})", node_kind_count);
                            successes += 1;
                        }
                        None => {
                            println!("✗ (get_language_fn failed)");
                            failures.push((lang_name, "get_language_fn returned None"));
                        }
                    }
                } else {
                    println!("✗ (invalid language: 0 node kinds)");
                    failures.push((lang_name, "0 node kinds"));
                }
            }
            None => {
                println!("✗ (get_language failed)");
                failures.push((lang_name, "get_language returned None"));
            }
        }
    }
    
    println!("\n=== Summary ===");
    println!("Total languages: {}", languages.len());
    println!("Successful: {}", successes);
    println!("Failed: {}", failures.len());
    
    if !failures.is_empty() {
        println!("\nFailed languages:");
        for (lang, reason) in &failures {
            println!("  {} - {}", lang, reason);
        }
        std::process::exit(1);
    }
    
    println!("\nAll languages loaded successfully! 🎉");
}