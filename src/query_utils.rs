use tree_sitter::{Query, QueryCursor, Node, Tree, Language};
use crate::types::ChunkError;

pub struct QueryProcessor {
    query: Query,
    scope_capture_index: Option<u32>,
    definition_capture_index: Option<u32>,
    reference_capture_index: Option<u32>,
}

impl QueryProcessor {
    pub fn new(language: Language, query_str: &str) -> Result<Self, ChunkError> {
        let query = Query::new(&language, query_str)
            .map_err(|e| ChunkError::QueryError(format!("Failed to create query: {}", e)))?;
        
        Ok(Self {
            scope_capture_index: query.capture_index_for_name("local.scope"),
            definition_capture_index: query.capture_index_for_name("local.definition"),
            reference_capture_index: query.capture_index_for_name("local.reference"),
            query,
        })
    }
    
    pub fn find_scopes<'a>(&self, tree: &'a Tree, source: &'a [u8]) -> Vec<Node<'a>> {
        let mut cursor = QueryCursor::new();
        let mut scopes = Vec::new();
        
        if let Some(scope_idx) = self.scope_capture_index {
            for match_ in cursor.matches(&self.query, tree.root_node(), source) {
                for capture in match_.captures.iter() {
                    if capture.index == scope_idx {
                        scopes.push(capture.node);
                    }
                }
            }
        }
        
        scopes
    }
    
    pub fn find_definitions<'a>(&self, tree: &'a Tree, source: &'a [u8]) -> Vec<(Node<'a>, String)> {
        let mut cursor = QueryCursor::new();
        let mut definitions = Vec::new();
        
        if let Some(def_idx) = self.definition_capture_index {
            let matches: Vec<_> = cursor.matches(&self.query, tree.root_node(), source).collect();
            for match_ in matches {
                for capture in match_.captures.iter() {
                    if capture.index == def_idx {
                        let text = std::str::from_utf8(&source[capture.node.byte_range()])
                            .unwrap_or_default()
                            .to_string();
                        definitions.push((capture.node, text));
                    }
                }
            }
        }
        
        definitions
    }
    
    pub fn find_references<'a>(&self, tree: &'a Tree, source: &'a [u8]) -> Vec<(Node<'a>, String)> {
        let mut cursor = QueryCursor::new();
        let mut references = Vec::new();
        
        if let Some(ref_idx) = self.reference_capture_index {
            let matches: Vec<_> = cursor.matches(&self.query, tree.root_node(), source).collect();
            for match_ in matches {
                for capture in match_.captures.iter() {
                    if capture.index == ref_idx {
                        let text = std::str::from_utf8(&source[capture.node.byte_range()])
                            .unwrap_or_default()
                            .to_string();
                        references.push((capture.node, text));
                    }
                }
            }
        }
        
        references
    }
    
    pub fn find_all_captures<'a>(&self, tree: &'a Tree, source: &'a [u8]) -> Vec<(String, Node<'a>)> {
        let mut cursor = QueryCursor::new();
        let mut captures = Vec::new();
        
        let matches = cursor.matches(&self.query, tree.root_node(), source);
        for match_ in matches {
            for capture in match_.captures.iter() {
                let capture_name = self.query.capture_names()[capture.index as usize].clone();
                captures.push((capture_name, capture.node));
            }
        }
        
        captures
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter::Parser;
    
    #[test]
    fn test_query_processor_creation() {
        let language = tree_sitter_rust::language();
        
        // Valid query
        let query_str = r#"
            (function_item) @local.scope
            (identifier) @local.definition
        "#;
        
        let processor = QueryProcessor::new(language, query_str);
        assert!(processor.is_ok());
        
        let processor = processor.unwrap();
        assert!(processor.scope_capture_index.is_some());
        assert!(processor.definition_capture_index.is_some());
        assert!(processor.reference_capture_index.is_none()); // Not in query
    }
    
    #[test]
    fn test_query_processor_invalid_query() {
        let language = tree_sitter_rust::language();
        
        // Invalid query syntax
        let query_str = "((invalid query";
        
        let processor = QueryProcessor::new(language, query_str);
        assert!(processor.is_err());
        
        match processor {
            Err(ChunkError::QueryError(msg)) => {
                assert!(msg.contains("Failed to create query"));
            }
            _ => panic!("Expected QueryError"),
        }
    }
    
    #[test]
    fn test_find_scopes() {
        let language = tree_sitter_rust::LANGUAGE.into();
        let mut parser = Parser::new();
        parser.set_language(language).unwrap();
        
        let source = r#"
fn main() {
    println!("Hello");
}

fn helper() {
    let x = 42;
}
"#;
        
        let tree = parser.parse(source, None).unwrap();
        
        let query_str = "(function_item) @local.scope";
        let processor = QueryProcessor::new(language, query_str).unwrap();
        
        let scopes = processor.find_scopes(&tree, source.as_bytes());
        assert_eq!(scopes.len(), 2); // Two functions
        
        // Check that we found function nodes
        for scope in &scopes {
            assert_eq!(scope.kind(), "function_item");
        }
    }
    
    #[test]
    fn test_find_definitions() {
        let language = tree_sitter_rust::LANGUAGE.into();
        let mut parser = Parser::new();
        parser.set_language(&language).unwrap();
        
        let source = r#"
fn main() {
    let x = 42;
    let y = "hello";
}
"#;
        
        let tree = parser.parse(source, None).unwrap();
        
        // Query for let binding identifiers
        let query_str = r#"
            (let_declaration
                pattern: (identifier) @local.definition)
        "#;
        
        let processor = QueryProcessor::new(language, query_str).unwrap();
        let definitions = processor.find_definitions(&tree, source.as_bytes());
        
        assert_eq!(definitions.len(), 2);
        assert_eq!(definitions[0].1, "x");
        assert_eq!(definitions[1].1, "y");
    }
    
    #[test]
    fn test_find_all_captures() {
        let language = tree_sitter_rust::LANGUAGE.into();
        let mut parser = Parser::new();
        parser.set_language(&language).unwrap();
        
        let source = "fn main() {}";
        let tree = parser.parse(source, None).unwrap();
        
        let query_str = r#"
            (function_item) @function
            (identifier) @name
        "#;
        
        let processor = QueryProcessor::new(language, query_str).unwrap();
        let captures = processor.find_all_captures(&tree, source.as_bytes());
        
        assert!(!captures.is_empty());
        
        // Should find both the function and its name
        let function_captures: Vec<_> = captures.iter()
            .filter(|(name, _)| name == "function")
            .collect();
        let name_captures: Vec<_> = captures.iter()
            .filter(|(name, _)| name == "name")
            .collect();
        
        assert_eq!(function_captures.len(), 1);
        assert_eq!(name_captures.len(), 1);
        assert_eq!(name_captures[0].1.kind(), "identifier");
    }
}