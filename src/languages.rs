use std::collections::HashMap;
use std::sync::LazyLock;
use tree_sitter_language::LanguageFn;

// Build language registry mapping language names to tree-sitter parsers
pub static LANGUAGE_REGISTRY: LazyLock<HashMap<&'static str, LanguageFn>> = LazyLock::new(|| {
    let mut registry = HashMap::new();
    
    // Map standard names to parsers using LANGUAGE constants and function calls
    registry.insert("Python", tree_sitter_python::LANGUAGE);
    registry.insert("JavaScript", tree_sitter_javascript::LANGUAGE);
    registry.insert("TypeScript", tree_sitter_typescript::LANGUAGE_TYPESCRIPT);
    registry.insert("TSX", tree_sitter_typescript::LANGUAGE_TSX);
    registry.insert("Java", tree_sitter_java::LANGUAGE);
    registry.insert("C++", tree_sitter_cpp::LANGUAGE);
    registry.insert("C", tree_sitter_c::LANGUAGE);
    registry.insert("C#", tree_sitter_c_sharp::LANGUAGE);
    registry.insert("Go", tree_sitter_go::LANGUAGE);
    registry.insert("Rust", tree_sitter_rust::LANGUAGE);
    registry.insert("Ruby", tree_sitter_ruby::LANGUAGE);
    // Skip PHP for now - API unclear
    registry.insert("Swift", tree_sitter_swift::LANGUAGE);
    registry.insert("Scala", tree_sitter_scala::LANGUAGE);
    registry.insert("Shell", tree_sitter_bash::LANGUAGE);
    registry.insert("Bash", tree_sitter_bash::LANGUAGE);
    registry.insert("R", tree_sitter_r::LANGUAGE);
    
    // Add common aliases
    registry.insert("python", tree_sitter_python::LANGUAGE);
    registry.insert("javascript", tree_sitter_javascript::LANGUAGE);
    registry.insert("typescript", tree_sitter_typescript::LANGUAGE_TYPESCRIPT);
    registry.insert("tsx", tree_sitter_typescript::LANGUAGE_TSX);
    registry.insert("java", tree_sitter_java::LANGUAGE);
    registry.insert("cpp", tree_sitter_cpp::LANGUAGE);
    registry.insert("c++", tree_sitter_cpp::LANGUAGE);
    registry.insert("c", tree_sitter_c::LANGUAGE);
    registry.insert("csharp", tree_sitter_c_sharp::LANGUAGE);
    registry.insert("c#", tree_sitter_c_sharp::LANGUAGE);
    registry.insert("go", tree_sitter_go::LANGUAGE);
    registry.insert("rust", tree_sitter_rust::LANGUAGE);
    registry.insert("ruby", tree_sitter_ruby::LANGUAGE);
    // Skip php for now
    registry.insert("swift", tree_sitter_swift::LANGUAGE);
    registry.insert("scala", tree_sitter_scala::LANGUAGE);
    registry.insert("shell", tree_sitter_bash::LANGUAGE);
    registry.insert("bash", tree_sitter_bash::LANGUAGE);
    registry.insert("sh", tree_sitter_bash::LANGUAGE);
    registry.insert("r", tree_sitter_r::LANGUAGE);
    
    registry
});

pub fn get_language(name: &str) -> Option<LanguageFn> {
    LANGUAGE_REGISTRY.get(name).cloned()
}

pub fn supported_languages() -> Vec<&'static str> {
    let mut languages: Vec<_> = LANGUAGE_REGISTRY.keys()
        .copied()
        .filter(|&name| {
            // Only return the canonical names (capitalized)
            name.chars().next().map_or(false, |c| c.is_uppercase())
        })
        .collect();
    languages.sort();
    languages
}

pub fn is_language_supported(name: &str) -> bool {
    LANGUAGE_REGISTRY.contains_key(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_language() {
        // Test exact matches
        assert!(get_language("Python").is_some());
        assert!(get_language("Rust").is_some());
        assert!(get_language("JavaScript").is_some());
        
        // Test lowercase aliases
        assert!(get_language("python").is_some());
        assert!(get_language("rust").is_some());
        assert!(get_language("javascript").is_some());
        
        // Test special cases
        assert!(get_language("C++").is_some());
        assert!(get_language("cpp").is_some());
        assert!(get_language("C#").is_some());
        assert!(get_language("csharp").is_some());
        
        // Test unknown language
        assert!(get_language("COBOL").is_none());
    }

    #[test]
    fn test_supported_languages() {
        let languages = supported_languages();
        assert!(!languages.is_empty());
        
        // Should have at least the languages we added (16 after removing PHP)
        assert!(languages.len() >= 16);
        
        // Check for some expected languages
        assert!(languages.contains(&"Python"));
        assert!(languages.contains(&"Rust"));
        assert!(languages.contains(&"JavaScript"));
        assert!(languages.contains(&"C++"));
        assert!(languages.contains(&"C#"));
        
        // Ensure all are capitalized (canonical names)
        for lang in &languages {
            assert!(lang.chars().next().unwrap().is_uppercase());
        }
    }

    #[test]
    fn test_is_language_supported() {
        assert!(is_language_supported("Python"));
        assert!(is_language_supported("python"));
        assert!(is_language_supported("C++"));
        assert!(is_language_supported("cpp"));
        assert!(!is_language_supported("COBOL"));
    }
}