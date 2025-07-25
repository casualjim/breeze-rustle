use crate::grammar_loader;
use tree_sitter_language::LanguageFn;

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
    // Fortran is now supported in breeze-grammars
    assert!(get_language("fortran").is_some());
  }

  #[test]
  fn test_supported_languages() {
    let languages = supported_languages();
    assert!(!languages.is_empty());

    // We now have 163 languages from breeze-grammars
    assert_eq!(languages.len(), 163);

    // Verify core languages are present
    assert!(languages.contains(&"python"));
    assert!(languages.contains(&"rust"));
    assert!(languages.contains(&"javascript"));
    assert!(languages.contains(&"typescript"));
    assert!(languages.contains(&"go"));

    // Verify additional languages
    assert!(languages.contains(&"java"));
    assert!(languages.contains(&"cpp"));
    assert!(languages.contains(&"c"));
    assert!(languages.contains(&"csharp"));
    assert!(languages.contains(&"ruby"));
    assert!(languages.contains(&"html"));
    assert!(languages.contains(&"css"));
    assert!(languages.contains(&"sql"));
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
    // Fortran is now supported in breeze-grammars
    assert!(is_language_supported("Fortran"));
  }
}
