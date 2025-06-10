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
    
    #[test]
    fn test_new_languages() {
        // Test that all our new languages are available
        let languages = ["python", "rust", "javascript", "typescript", "go"];
        
        for lang in &languages {
            assert!(
                is_language_supported(lang),
                "{} should be supported",
                lang
            );
            
            let language = get_language(lang);
            assert!(
                language.is_some(),
                "{} language should be loaded",
                lang
            );
            
            let language_fn = get_language_fn(lang);
            assert!(
                language_fn.is_some(),
                "{} LanguageFn should be loaded",
                lang
            );
        }
        
        // Verify we have at least these 5 languages
        let all_languages = supported_languages();
        assert!(
            all_languages.len() >= 5,
            "Should have at least 5 languages, got {}",
            all_languages.len()
        );
    }
    
    #[test]
    fn test_parse_simple_code() {
        // Test parsing simple code with each grammar
        let test_cases = vec![
            ("python", "def hello():\n    pass"),
            ("rust", "fn main() {}"),
            ("javascript", "function hello() {}"),
            ("typescript", "function hello(): void {}"),
            ("go", "func main() {}"),
        ];
        
        for (lang_name, code) in test_cases {
            if let Some(language) = get_language(lang_name) {
                let mut parser = tree_sitter::Parser::new();
                parser.set_language(&language).unwrap();
                
                let tree = parser.parse(code, None);
                assert!(
                    tree.is_some(),
                    "{} should parse simple code",
                    lang_name
                );
                
                let tree = tree.unwrap();
                assert!(
                    !tree.root_node().has_error(),
                    "{} parse tree should not have errors",
                    lang_name
                );
            }
        }
    }
}