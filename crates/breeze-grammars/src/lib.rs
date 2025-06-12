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
    assert!(
      !languages.is_empty(),
      "Should have loaded at least one grammar"
    );

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
  fn test_expanded_language_support() {
    // Test that we have all 44 languages
    let all_languages = supported_languages();
    assert!(
      all_languages.len() >= 44,
      "Should have at least 44 languages, got {}",
      all_languages.len()
    );

    // Test core languages (top 10)
    let core_languages = [
      "python",
      "javascript",
      "java",
      "cpp",
      "c",
      "csharp",
      "typescript",
      "sql",
      "php",
      "go",
    ];

    for lang in &core_languages {
      assert!(is_language_supported(lang), "{} should be supported", lang);
    }

    // Test web languages including the ones that were missing
    let web_languages = ["html", "css", "tsx"];
    for lang in &web_languages {
      assert!(is_language_supported(lang), "{} should be supported", lang);
    }

    // Test popular languages (11-20)
    let popular_languages = [
      "rust",
      "swift",
      "kotlin",
      "ruby",
      "r",
      "bash",
      "scala",
      "dart",
      "powershell",
    ];

    for lang in &popular_languages {
      assert!(is_language_supported(lang), "{} should be supported", lang);
    }

    // Test that all languages can be loaded
    for lang_name in all_languages {
      let language = get_language(lang_name);
      assert!(
        language.is_some(),
        "{} language should be loaded",
        lang_name
      );

      let language_fn = get_language_fn(lang_name);
      assert!(
        language_fn.is_some(),
        "{} LanguageFn should be loaded",
        lang_name
      );
    }
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
      ("c", "int main() { return 0; }"),
      ("cpp", "int main() { return 0; }"),
      (
        "java",
        "public class Main { public static void main(String[] args) {} }",
      ),
      ("ruby", "def hello\n  puts 'hello'\nend"),
      ("html", "<html><body>Hello</body></html>"),
    ];

    for (lang_name, code) in test_cases {
      if let Some(language) = get_language(lang_name) {
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&language).unwrap();

        let tree = parser.parse(code, None);
        assert!(tree.is_some(), "{} should parse simple code", lang_name);

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
