use crate::{languages::*, types::*};
use text_splitter::{ChunkConfig, CodeSplitter};

// Simple chunker that creates a new splitter for each request
pub struct InnerChunker {
    max_chunk_size: usize,
}

impl InnerChunker {
    pub fn new(max_chunk_size: usize) -> Self {
        Self { max_chunk_size }
    }

    pub async fn chunk_file(
        &self, 
        content: &str, 
        language: &str, 
        file_path: Option<&str>
    ) -> Result<Vec<SemanticChunk>, ChunkError> {
        // Get tree-sitter language
        let language_fn = get_language(language)
            .ok_or_else(|| ChunkError::UnsupportedLanguage(language.to_string()))?;
        let ts_language: tree_sitter::Language = language_fn.into();
        
        // Create config and splitter
        let config = ChunkConfig::new(self.max_chunk_size);
        let splitter = CodeSplitter::new(ts_language, config)
            .map_err(|e| ChunkError::ParseError(format!("Failed to create splitter: {}", e)))?;
        
        // Get base chunks with indices
        let chunks: Vec<_> = splitter.chunk_indices(content).collect();
        
        // Convert to our SemanticChunk format
        let mut semantic_chunks = Vec::new();
        
        for (idx, (offset, chunk_text)) in chunks.into_iter().enumerate() {
            // Calculate line numbers
            let start_line = content[..offset].matches('\n').count() + 1;
            let end_line = content[..offset + chunk_text.len()].matches('\n').count() + 1;
            
            let metadata = ChunkMetadata {
                node_type: "code_chunk".to_string(),
                node_name: file_path.map(|_p| format!("chunk_{}", idx + 1)),
                language: language.to_string(),
                parent_context: file_path.map(|p| p.to_string()),
                scope_path: vec![],
                definitions: vec![],
                references: vec![],
            };
            
            semantic_chunks.push(SemanticChunk {
                text: chunk_text.to_string(),
                start_byte: offset,
                end_byte: offset + chunk_text.len(),
                start_line,
                end_line,
                metadata,
            });
        }
        
        Ok(semantic_chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_inner_chunker_creation() {
        let _chunker = InnerChunker::new(1000);
        // Just verify it creates successfully
    }
    
    #[tokio::test]
    async fn test_chunk_simple_rust_code() {
        let chunker = InnerChunker::new(100);
        
        let code = r#"
fn main() {
    println!("Hello, world!");
}

fn helper() {
    let x = 42;
}
"#;
        
        let result = chunker.chunk_file(code, "Rust", None).await;
        assert!(result.is_ok());
        
        let chunks = result.unwrap();
        assert!(!chunks.is_empty());
        
        // Check that chunks have proper metadata
        for chunk in &chunks {
            assert_eq!(chunk.metadata.language, "Rust");
            assert_eq!(chunk.metadata.node_type, "code_chunk");
        }
    }
    
    #[tokio::test]
    async fn test_unsupported_language() {
        let chunker = InnerChunker::new(1000);
        
        let result = chunker.chunk_file("code", "COBOL", None).await;
        assert!(result.is_err());
        
        match result {
            Err(ChunkError::UnsupportedLanguage(lang)) => {
                assert_eq!(lang, "COBOL");
            }
            _ => panic!("Expected UnsupportedLanguage error"),
        }
    }
    
    #[tokio::test] 
    async fn test_language_case_insensitive() {
        let chunker = InnerChunker::new(1000);
        
        let code = "def main(): pass";
        
        // These should all work
        assert!(chunker.chunk_file(code, "python", None).await.is_ok());
        assert!(chunker.chunk_file(code, "Python", None).await.is_ok());
    }
}