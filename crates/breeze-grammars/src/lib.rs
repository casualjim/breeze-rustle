// Include the auto-generated bindings
include!(concat!(env!("OUT_DIR"), "/grammars.rs"));

/// Get a language by name (case-insensitive, normalized to lowercase)
pub fn get_language(name: &str) -> Option<tree_sitter::Language> {
    load_grammar(&name.to_lowercase())
}

/// Get a LanguageFn by name (case-insensitive, normalized to lowercase)
pub fn get_language_fn(name: &str) -> Option<tree_sitter_language::LanguageFn> {
    load_grammar_fn(&name.to_lowercase())
}

/// Get all supported language names
pub fn supported_languages() -> Vec<&'static str> {
    available_grammars().to_vec()
}

/// Check if a language is supported (case-insensitive)
pub fn is_language_supported(name: &str) -> bool {
    load_grammar(&name.to_lowercase()).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_load_grammars() {
        let languages = supported_languages();
        assert!(!languages.is_empty(), "Should have loaded at least one grammar");
        
        // Test Python
        if languages.contains(&"python") {
            let lang = get_language("python");
            assert!(lang.is_some(), "python should be loaded");
            
            let lang_fn = get_language_fn("python");
            assert!(lang_fn.is_some(), "python LanguageFn should be loaded");
        }
    }
    
    #[test]
    fn test_case_insensitive() {
        // Test case variations
        if is_language_supported("python") {
            assert!(is_language_supported("Python"));
            assert!(is_language_supported("PYTHON"));
        }
    }
}