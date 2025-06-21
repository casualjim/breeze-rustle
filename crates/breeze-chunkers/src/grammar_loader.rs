use tree_sitter_language::LanguageFn;

/// Get a language as LanguageFn
pub fn get_language_fn(name: &str) -> Option<LanguageFn> {
  breeze_grammars::get_language_fn(name)
}

/// Check if a language is supported
pub fn is_language_supported(name: &str) -> bool {
  breeze_grammars::is_language_supported(name)
}

/// Get all supported language names
pub fn supported_languages() -> Vec<&'static str> {
  breeze_grammars::supported_languages()
}
