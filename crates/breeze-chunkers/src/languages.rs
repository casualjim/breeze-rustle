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
        // Currently only Python is available in breeze-grammars
        assert!(get_language("python").is_some());
        assert!(get_language("Python").is_some()); // Case insensitive
        assert!(get_language("PYTHON").is_some());
        
        // Test unknown languages (will be added later)
        assert!(get_language("rust").is_none());
        assert!(get_language("javascript").is_none());
        assert!(get_language("COBOL").is_none());
    }

    #[test]
    fn test_supported_languages() {
        let languages = supported_languages();
        assert!(!languages.is_empty());
        
        // Currently only Python is available
        assert_eq!(languages.len(), 1);
        assert!(languages.contains(&"python"));
    }

    #[test]
    fn test_is_language_supported() {
        // Currently only Python is available
        assert!(is_language_supported("Python"));
        assert!(is_language_supported("python"));
        assert!(is_language_supported("PYTHON"));
        
        // These will be supported later
        assert!(!is_language_supported("rust"));
        assert!(!is_language_supported("C++"));
        assert!(!is_language_supported("COBOL"));
    }
}