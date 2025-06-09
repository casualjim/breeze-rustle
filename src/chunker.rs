use crate::{languages::*, types::*, metadata_extractor::extract_metadata};
use text_splitter::{ChunkConfig, CodeSplitter, TextSplitter, ChunkSizer};

pub enum TokenizerType {
    Characters,
    Tiktoken,
    HuggingFace(String), // model name
}

// Concrete chunk sizer enum to avoid trait object issues
pub enum ConcreteSizer {
    Characters(text_splitter::Characters),
    Tiktoken(tiktoken_rs::CoreBPE),
    HuggingFace(tokenizers::tokenizer::Tokenizer),
}

impl ChunkSizer for ConcreteSizer {
    fn size(&self, chunk: &str) -> usize {
        match self {
            ConcreteSizer::Characters(sizer) => sizer.size(chunk),
            ConcreteSizer::Tiktoken(sizer) => sizer.size(chunk),
            ConcreteSizer::HuggingFace(sizer) => sizer.size(chunk),
        }
    }
}

// Simple chunker that creates a new splitter for each request
pub struct InnerChunker {
    max_chunk_size: usize,
    chunk_sizer: ConcreteSizer,
}

impl InnerChunker {
    pub fn new(max_chunk_size: usize, tokenizer_type: TokenizerType) -> Result<Self, ChunkError> {
        let chunk_sizer = match tokenizer_type {
            TokenizerType::Characters => ConcreteSizer::Characters(text_splitter::Characters),
            TokenizerType::Tiktoken => {
                let tiktoken = tiktoken_rs::cl100k_base()
                    .map_err(|e| ChunkError::ParseError(format!("Failed to create tiktoken: {}", e)))?;
                ConcreteSizer::Tiktoken(tiktoken)
            }
            TokenizerType::HuggingFace(model) => {
                let tokenizer = tokenizers::tokenizer::Tokenizer::from_pretrained(&model, None)
                    .map_err(|e| ChunkError::ParseError(format!("Failed to load HF tokenizer: {}", e)))?;
                ConcreteSizer::HuggingFace(tokenizer)
            }
        };
        
        Ok(Self { 
            max_chunk_size,
            chunk_sizer,
        })
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
        
        // Create config and splitter with our chunk sizer
        let config = ChunkConfig::new(self.max_chunk_size)
            .with_sizer(&self.chunk_sizer);
        let splitter = CodeSplitter::new(ts_language.clone(), config)
            .map_err(|e| ChunkError::ParseError(format!("Failed to create splitter: {}", e)))?;
        
        // Get base chunks with indices
        let chunks: Vec<_> = splitter.chunk_indices(content).collect();
        
        // Convert to our SemanticChunk format
        let mut semantic_chunks = Vec::new();
        
        for (idx, (offset, chunk_text)) in chunks.into_iter().enumerate() {
            // Calculate line numbers
            let start_line = content[..offset].matches('\n').count() + 1;
            let end_line = content[..offset + chunk_text.len()].matches('\n').count() + 1;
            
            // Extract metadata from AST
            let metadata = match extract_metadata(
                content,
                offset,
                offset + chunk_text.len(),
                ts_language.clone(),
                language,
            ) {
                Ok(mut meta) => {
                    // If no node name was extracted, use a default
                    if meta.node_name.is_none() {
                        meta.node_name = file_path.map(|_p| format!("chunk_{}", idx + 1));
                    }
                    // Add file path as parent context if not already set
                    if meta.parent_context.is_none() && file_path.is_some() {
                        meta.parent_context = file_path.map(|p| p.to_string());
                    }
                    meta
                }
                Err(_) => {
                    // Fallback metadata if extraction fails
                    ChunkMetadata {
                        node_type: "code_chunk".to_string(),
                        node_name: file_path.map(|_p| format!("chunk_{}", idx + 1)),
                        language: language.to_string(),
                        parent_context: file_path.map(|p| p.to_string()),
                        scope_path: vec![],
                        definitions: vec![],
                        references: vec![],
                    }
                }
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
    
    pub async fn chunk_text(
        &self,
        content: &str,
        file_path: Option<&str>
    ) -> Result<Vec<SemanticChunk>, ChunkError> {
        // Create config and text splitter with our chunk sizer
        let config = ChunkConfig::new(self.max_chunk_size)
            .with_sizer(&self.chunk_sizer)
            .with_trim(false);
        let splitter = TextSplitter::new(config);
        
        // Get base chunks with indices
        let chunks: Vec<_> = splitter.chunk_indices(content).collect();
        
        // Convert to our SemanticChunk format with minimal metadata
        let mut text_chunks = Vec::new();
        
        for (idx, (offset, chunk_text)) in chunks.into_iter().enumerate() {
            // Calculate line numbers
            let start_line = content[..offset].matches('\n').count() + 1;
            let end_line = content[..offset + chunk_text.len()].matches('\n').count() + 1;
            
            // Create minimal metadata for text chunks
            let metadata = ChunkMetadata {
                node_type: "text_chunk".to_string(),
                node_name: Some(format!("text_chunk_{}", idx + 1)),
                language: "text".to_string(),
                parent_context: file_path.map(|p| p.to_string()),
                scope_path: vec![],
                definitions: vec![],
                references: vec![],
            };
            
            text_chunks.push(SemanticChunk {
                text: chunk_text.to_string(),
                start_byte: offset,
                end_byte: offset + chunk_text.len(),
                start_line,
                end_line,
                metadata,
            });
        }
        
        Ok(text_chunks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_inner_chunker_creation() {
        let _chunker = InnerChunker::new(1000, TokenizerType::Characters).unwrap();
        // Just verify it creates successfully
    }
    
    #[tokio::test]
    async fn test_chunk_simple_rust_code() {
        let chunker = InnerChunker::new(100, TokenizerType::Characters).unwrap();
        
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
            // Now we extract actual node types from AST
            assert!(!chunk.metadata.node_type.is_empty());
        }
    }
    
    #[tokio::test]
    async fn test_unsupported_language() {
        let chunker = InnerChunker::new(1000, TokenizerType::Characters).unwrap();
        
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
        let chunker = InnerChunker::new(1000, TokenizerType::Characters).unwrap();
        
        let code = "def main(): pass";
        
        // These should all work
        assert!(chunker.chunk_file(code, "python", None).await.is_ok());
        assert!(chunker.chunk_file(code, "Python", None).await.is_ok());
    }
}