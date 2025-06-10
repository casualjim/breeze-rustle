use tree_sitter_language::LanguageFn;
use crate::grammar_loader;

pub fn get_language(name: &str) -> Option<LanguageFn> {
    grammar_loader::get_language_fn(name)
}

pub fn supported_languages() -> Vec<&'static str> {
    grammar_loader::supported_languages()
}

pub fn is_language_supported(name: &str) -> bool {
    grammar_loader::is_language_supported(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_language() {
        // Test all available languages
        assert!(get_language("python").is_some());
        assert!(get_language("Python").is_some()); // Case insensitive
        assert!(get_language("PYTHON").is_some());
        
        assert!(get_language("rust").is_some());
        assert!(get_language("Rust").is_some());
        
        assert!(get_language("javascript").is_some());
        assert!(get_language("JavaScript").is_some());
        
        assert!(get_language("typescript").is_some());
        assert!(get_language("TypeScript").is_some());
        
        assert!(get_language("go").is_some());
        assert!(get_language("Go").is_some());
        
        // Test unknown languages
        assert!(get_language("COBOL").is_none());
        assert!(get_language("fortran").is_none());
    }

    #[test]
    fn test_supported_languages() {
        let languages = supported_languages();
        assert!(!languages.is_empty());
        
        // We now have 5 languages
        assert_eq!(languages.len(), 5);
        assert!(languages.contains(&"python"));
        assert!(languages.contains(&"rust"));
        assert!(languages.contains(&"javascript"));
        assert!(languages.contains(&"typescript"));
        assert!(languages.contains(&"go"));
    }

    #[test]
    fn test_is_language_supported() {
        // Test all supported languages
        assert!(is_language_supported("Python"));
        assert!(is_language_supported("python"));
        assert!(is_language_supported("PYTHON"));
        
        assert!(is_language_supported("rust"));
        assert!(is_language_supported("Rust"));
        assert!(is_language_supported("RUST"));
        
        assert!(is_language_supported("javascript"));
        assert!(is_language_supported("JavaScript"));
        
        assert!(is_language_supported("typescript"));
        assert!(is_language_supported("TypeScript"));
        
        assert!(is_language_supported("go"));
        assert!(is_language_supported("Go"));
        
        // These are not supported
        assert!(!is_language_supported("C++"));
        assert!(!is_language_supported("COBOL"));
        assert!(!is_language_supported("Fortran"));
    }
}