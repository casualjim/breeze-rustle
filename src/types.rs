use pyo3::prelude::*;
use thiserror::Error;
use tree_sitter::Node;

#[derive(Error, Debug)]
pub enum ChunkError {
    #[error("Unsupported language: {0}")]
    UnsupportedLanguage(String),
    
    #[error("Failed to parse content: {0}")]
    ParseError(String),
    
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
    
    #[error("Query error: {0}")]
    QueryError(String),
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct ChunkMetadata {
    #[pyo3(get)]
    pub node_type: String,        // "function", "class", "method"
    #[pyo3(get)]
    pub node_name: Option<String>, // "parse_document", "MyClass"
    #[pyo3(get)]
    pub language: String,
    #[pyo3(get)]
    pub parent_context: Option<String>, // "class MyClass" for methods
    #[pyo3(get)]
    pub scope_path: Vec<String>,   // ["module", "MyClass", "parse_document"]
    #[pyo3(get)]
    pub definitions: Vec<String>,  // Variable/function names defined
    #[pyo3(get)]
    pub references: Vec<String>,   // Variable/function names referenced
}

#[pymethods]
impl ChunkMetadata {
    fn __repr__(&self) -> String {
        format!(
            "ChunkMetadata(node_type='{}', node_name={:?}, language='{}', parent_context={:?})",
            self.node_type, self.node_name, self.language, self.parent_context
        )
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct SemanticChunk {
    #[pyo3(get)]
    pub text: String,
    #[pyo3(get)]
    pub start_byte: usize,
    #[pyo3(get)]
    pub end_byte: usize,
    #[pyo3(get)]
    pub start_line: usize,
    #[pyo3(get)]
    pub end_line: usize,
    #[pyo3(get)]
    pub metadata: ChunkMetadata,
}

#[pymethods]
impl SemanticChunk {
    fn __repr__(&self) -> String {
        format!(
            "SemanticChunk(lines={}-{}, type='{}', name={:?})",
            self.start_line, self.end_line, self.metadata.node_type, self.metadata.node_name
        )
    }
    
    fn __len__(&self) -> usize {
        self.text.len()
    }
}

impl SemanticChunk {
    /// Calculate line numbers from byte offsets
    pub fn from_node(node: &Node, source: &str, metadata: ChunkMetadata) -> Self {
        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        let text = source[start_byte..end_byte].to_string();
        
        // Calculate line numbers
        let start_line = source[..start_byte].matches('\n').count() + 1;
        let end_line = source[..end_byte].matches('\n').count() + 1;
        
        Self {
            text,
            start_byte,
            end_byte,
            start_line,
            end_line,
            metadata,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tree_sitter::{Language, Parser};
    
    #[test]
    fn test_chunk_metadata_creation() {
        let metadata = ChunkMetadata {
            node_type: "function".to_string(),
            node_name: Some("test_func".to_string()),
            language: "rust".to_string(),
            parent_context: Some("impl MyStruct".to_string()),
            scope_path: vec!["module".to_string(), "MyStruct".to_string(), "test_func".to_string()],
            definitions: vec!["x".to_string(), "y".to_string()],
            references: vec!["println".to_string()],
        };
        
        assert_eq!(metadata.node_type, "function");
        assert_eq!(metadata.node_name, Some("test_func".to_string()));
        assert_eq!(metadata.language, "rust");
        assert_eq!(metadata.parent_context, Some("impl MyStruct".to_string()));
        assert_eq!(metadata.scope_path.len(), 3);
        assert_eq!(metadata.definitions.len(), 2);
        assert_eq!(metadata.references.len(), 1);
    }
    
    #[test]
    fn test_semantic_chunk_from_node() {
        // Create a simple source code
        let source = "fn main() {\n    println!(\"Hello\");\n}";
        
        // Parse with tree-sitter (using rust parser as example)
        let mut parser = Parser::new();
        let lang: Language = tree_sitter_rust::LANGUAGE.into();
        parser.set_language(&lang).unwrap();

        let tree = parser.parse(source, None).unwrap();
        let root = tree.root_node();
        
        // Find the function node
        let function_node = root.child(0).unwrap();
        
        let metadata = ChunkMetadata {
            node_type: "function".to_string(),
            node_name: Some("main".to_string()),
            language: "rust".to_string(),
            parent_context: None,
            scope_path: vec!["module".to_string(), "main".to_string()],
            definitions: vec![],
            references: vec!["println".to_string()],
        };
        
        let chunk = SemanticChunk::from_node(&function_node, source, metadata);
        
        assert_eq!(chunk.text, source);
        assert_eq!(chunk.start_byte, 0);
        assert_eq!(chunk.end_byte, source.len());
        assert_eq!(chunk.start_line, 1);
        assert_eq!(chunk.end_line, 3);
    }
    
    #[test]
    fn test_chunk_error_display() {
        let err = ChunkError::UnsupportedLanguage("cobol".to_string());
        assert_eq!(err.to_string(), "Unsupported language: cobol");
        
        let err = ChunkError::ParseError("syntax error at line 5".to_string());
        assert_eq!(err.to_string(), "Failed to parse content: syntax error at line 5");
        
        let err = ChunkError::QueryError("invalid capture name".to_string());
        assert_eq!(err.to_string(), "Query error: invalid capture name");
    }
    
    #[test]
    fn test_line_number_calculation() {
        let source = "line1\nline2\nline3\nline4\nline5";
        
        // Test various byte positions
        assert_eq!(source[..0].matches('\n').count() + 1, 1); // Start of file
        assert_eq!(source[..6].matches('\n').count() + 1, 2); // After first newline
        assert_eq!(source[..12].matches('\n').count() + 1, 3); // After second newline
        assert_eq!(source[..source.len()].matches('\n').count() + 1, 5); // End of file
    }
}