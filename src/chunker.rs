use crate::{languages::*, types::*, metadata_extractor::extract_metadata_from_tree};
use text_splitter::{ChunkConfig, CodeSplitter, TextSplitter, ChunkSizer};

pub enum TokenizerType {
    Characters,
    Tiktoken,
    HuggingFace(String), // model name
}

// Concrete chunk sizer enum to avoid trait object issues
#[derive(Clone)]
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
#[derive(Clone)]
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

    pub fn chunk_code(
        &self, 
        content: String, 
        language: String, 
        file_path: Option<String>
    ) -> impl futures::Stream<Item = Result<SemanticChunk, ChunkError>> {
        let chunker = self.clone();
        
        async_stream::try_stream! {
            let (tree, chunks, line_offsets) = chunker.setup_code_chunking(&content, &language)?;
            
            // Only do the actual chunk yielding inside the stream
            for (idx, (offset, chunk_text)) in chunks.into_iter().enumerate() {
                // Binary search for line numbers (convert to 1-based)
                let start_line = line_offsets.binary_search(&offset).unwrap_or_else(|i| i) + 1;
                let end_offset = offset + chunk_text.len();
                let end_line = line_offsets.binary_search(&end_offset).unwrap_or_else(|i| i) + 1;
                
                // Extract metadata from pre-parsed AST
                let metadata = match extract_metadata_from_tree(
                    &tree,
                    &content,
                    offset,
                    end_offset,
                    &language,
                ) {
                    Ok(mut meta) => {
                        // If no node name was extracted, use a default
                        if meta.node_name.is_none() {
                            meta.node_name = file_path.as_ref().map(|_p| format!("chunk_{}", idx + 1));
                        }
                        // Add file path as parent context if not already set
                        if meta.parent_context.is_none() && file_path.is_some() {
                            meta.parent_context = file_path.clone();
                        }
                        meta
                    }
                    Err(_) => {
                        // Fallback metadata if extraction fails
                        ChunkMetadata {
                            node_type: "code_chunk".to_string(),
                            node_name: file_path.as_ref().map(|_p| format!("chunk_{}", idx + 1)),
                            language: language.clone(),
                            parent_context: file_path.clone(),
                            scope_path: vec![],
                            definitions: vec![],
                            references: vec![],
                        }
                    }
                };
                
                yield SemanticChunk {
                    text: chunk_text.to_string(),
                    start_byte: offset,
                    end_byte: end_offset,
                    start_line,
                    end_line,
                    metadata,
                };
            }
        }
    }
    
    fn setup_code_chunking<'a>(
        &self,
        content: &'a str,
        language: &str,
    ) -> Result<(tree_sitter::Tree, Vec<(usize, &'a str)>, Vec<usize>), ChunkError> {
        // Get tree-sitter language
        let language_fn = get_language(language)
            .ok_or_else(|| ChunkError::UnsupportedLanguage(language.to_string()))?;
        let ts_language: tree_sitter::Language = language_fn.into();
        
        // Parse the content once upfront
        let mut parser = tree_sitter::Parser::new();
        parser.set_language(&ts_language)
            .map_err(|e| ChunkError::ParseError(format!("Failed to set language: {:?}", e)))?;
        
        let tree = parser.parse(content, None)
            .ok_or_else(|| ChunkError::ParseError("Failed to parse content".to_string()))?;
        
        // Create config and splitter with our chunk sizer
        let config = ChunkConfig::new(self.max_chunk_size)
            .with_sizer(&self.chunk_sizer);
        let splitter = CodeSplitter::new(ts_language.clone(), config)
            .map_err(|e| ChunkError::ParseError(format!("Failed to create splitter: {}", e)))?;
        
        // Get base chunks with indices
        let chunks: Vec<_> = splitter.chunk_indices(content).collect();
        
        // Pre-calculate line offsets for efficient line number computation
        let line_offsets: Vec<usize> = std::iter::once(0)
            .chain(content.match_indices('\n').map(|(i, _)| i + 1))
            .collect();
        
        Ok((tree, chunks, line_offsets))
    }
    
    pub fn chunk_text(
        &self,
        content: String,
        file_path: Option<String>
    ) -> impl futures::Stream<Item = Result<SemanticChunk, ChunkError>> {
        let chunker = self.clone();
        
        async_stream::try_stream! {
            let (chunks, line_offsets) = chunker.setup_text_chunking(&content)?;
            
            // Only do the actual chunk yielding inside the stream
            for (idx, (offset, chunk_text)) in chunks.into_iter().enumerate() {
                // Binary search for line numbers (convert to 1-based)
                let start_line = line_offsets.binary_search(&offset).unwrap_or_else(|i| i) + 1;
                let end_offset = offset + chunk_text.len();
                let end_line = line_offsets.binary_search(&end_offset).unwrap_or_else(|i| i) + 1;
                
                // Create minimal metadata for text chunks
                let metadata = ChunkMetadata {
                    node_type: "text_chunk".to_string(),
                    node_name: Some(format!("text_chunk_{}", idx + 1)),
                    language: "text".to_string(),
                    parent_context: file_path.clone(),
                    scope_path: vec![],
                    definitions: vec![],
                    references: vec![],
                };
                
                yield SemanticChunk {
                    text: chunk_text.to_string(),
                    start_byte: offset,
                    end_byte: offset + chunk_text.len(),
                    start_line,
                    end_line,
                    metadata,
                };
            }
        }
    }
    
    fn setup_text_chunking<'a>(
        &self,
        content: &'a str,
    ) -> Result<(Vec<(usize, &'a str)>, Vec<usize>), ChunkError> {
        // Create config and text splitter with our chunk sizer
        let config = ChunkConfig::new(self.max_chunk_size)
            .with_sizer(&self.chunk_sizer)
            .with_trim(false);
        let splitter = TextSplitter::new(config);
        
        // Get base chunks with indices
        let chunks: Vec<_> = splitter.chunk_indices(content).collect();
        
        // Pre-calculate line offsets for efficient line number computation
        let line_offsets: Vec<usize> = std::iter::once(0)
            .chain(content.match_indices('\n').map(|(i, _)| i + 1))
            .collect();
        
        Ok((chunks, line_offsets))
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
        use futures::StreamExt;
        
        let chunker = InnerChunker::new(100, TokenizerType::Characters).unwrap();
        
        let code = r#"
fn main() {
    println!("Hello, world!");
}

fn helper() {
    let x = 42;
}
"#;
        
        let mut chunks = Vec::new();
        let mut stream = Box::pin(chunker.chunk_code(code.to_string(), "Rust".to_string(), None));
        
        while let Some(result) = stream.next().await {
            assert!(result.is_ok());
            chunks.push(result.unwrap());
        }
        
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
        use futures::StreamExt;
        
        let chunker = InnerChunker::new(1000, TokenizerType::Characters).unwrap();
        
        let mut stream = Box::pin(chunker.chunk_code("code".to_string(), "COBOL".to_string(), None));
        
        // The first item should be an error
        let result = stream.next().await.unwrap();
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
        use futures::StreamExt;
        
        let chunker = InnerChunker::new(1000, TokenizerType::Characters).unwrap();
        
        let code = "def main(): pass";
        
        // These should all work
        let mut stream1 = Box::pin(chunker.chunk_code(code.to_string(), "python".to_string(), None));
        assert!(stream1.next().await.unwrap().is_ok());
        
        let mut stream2 = Box::pin(chunker.chunk_code(code.to_string(), "Python".to_string(), None));
        assert!(stream2.next().await.unwrap().is_ok());
    }
}