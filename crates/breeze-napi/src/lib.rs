use breeze_chunkers::{
    ChunkMetadata, Chunk as RustChunk, ProjectChunk as RustProjectChunk,
    InnerChunker, Tokenizer as RustTokenizer,
    supported_languages, is_language_supported,
    walk_project as rust_walk_project, WalkOptions,
};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::sync::Arc;
use futures::StreamExt;
use napi::threadsafe_function::{ErrorStrategy, ThreadsafeFunction, ThreadsafeFunctionCallMode};

#[napi]
pub enum TokenizerType {
    Characters,
    Tiktoken,
    HuggingFace,
}

#[napi]
pub enum ChunkType {
    Semantic,
    Text,
}

#[napi(object)]
#[derive(Clone)]
pub struct ChunkMetadataJs {
    pub node_type: String,
    pub node_name: Option<String>,
    pub language: String,
    pub parent_context: Option<String>,
    pub scope_path: Vec<String>,
    pub definitions: Vec<String>,
    pub references: Vec<String>,
}

impl From<ChunkMetadata> for ChunkMetadataJs {
    fn from(metadata: ChunkMetadata) -> Self {
        Self {
            node_type: metadata.node_type,
            node_name: metadata.node_name,
            language: metadata.language,
            parent_context: metadata.parent_context,
            scope_path: metadata.scope_path,
            definitions: metadata.definitions,
            references: metadata.references,
        }
    }
}

#[napi(object)]
#[derive(Clone)]
pub struct SemanticChunkJs {
    pub chunk_type: ChunkType,
    pub text: String,
    pub start_byte: i32,
    pub end_byte: i32,
    pub start_line: i32,
    pub end_line: i32,
    pub metadata: ChunkMetadataJs,
}

impl From<RustChunk> for SemanticChunkJs {
    fn from(chunk: RustChunk) -> Self {
        match chunk {
            RustChunk::Semantic(sc) => Self {
                chunk_type: ChunkType::Semantic,
                text: sc.text,
                start_byte: sc.start_byte as i32,
                end_byte: sc.end_byte as i32,
                start_line: sc.start_line as i32,
                end_line: sc.end_line as i32,
                metadata: sc.metadata.into(),
            },
            RustChunk::Text(sc) => Self {
                chunk_type: ChunkType::Text,
                text: sc.text,
                start_byte: sc.start_byte as i32,
                end_byte: sc.end_byte as i32,
                start_line: sc.start_line as i32,
                end_line: sc.end_line as i32,
                metadata: sc.metadata.into(),
            },
        }
    }
}

#[napi(object)]
#[derive(Clone)]
pub struct ProjectChunkJs {
    pub file_path: String,
    pub chunk: SemanticChunkJs,
}

impl From<RustProjectChunk> for ProjectChunkJs {
    fn from(pc: RustProjectChunk) -> Self {
        Self {
            file_path: pc.file_path,
            chunk: pc.chunk.into(),
        }
    }
}

#[napi]
pub struct SemanticChunker {
    inner: Arc<InnerChunker>,
}

#[napi]
impl SemanticChunker {
    #[napi(constructor)]
    pub fn new(
        max_chunk_size: Option<i32>,
        tokenizer: Option<TokenizerType>,
        hf_model: Option<String>,
    ) -> Result<Self> {
        let max_chunk_size = max_chunk_size.unwrap_or(1500) as usize;
        
        // Convert JS TokenizerType to Rust Tokenizer
        let tokenizer_type = match tokenizer.unwrap_or(TokenizerType::Characters) {
            TokenizerType::Characters => RustTokenizer::Characters,
            TokenizerType::Tiktoken => RustTokenizer::Tiktoken,
            TokenizerType::HuggingFace => {
                match hf_model {
                    Some(model) => RustTokenizer::HuggingFace(model),
                    None => {
                        return Err(Error::new(
                            Status::InvalidArg,
                            "TokenizerType.HuggingFace requires hfModel parameter".to_string(),
                        ));
                    }
                }
            }
        };
        
        let inner = InnerChunker::new(max_chunk_size, tokenizer_type)
            .map_err(|e| Error::new(Status::GenericFailure, format!("Failed to create chunker: {}", e)))?;
        
        Ok(Self {
            inner: Arc::new(inner),
        })
    }
    
    #[napi(ts_args_type = "content: string, language: string, filePath: string | undefined, onChunk: (chunk: SemanticChunkJs) => void, onError: (error: Error) => void, onComplete: () => void")]
    pub fn chunk_code(
        &self,
        content: String,
        language: String,
        file_path: Option<String>,
        on_chunk: ThreadsafeFunction<SemanticChunkJs, ErrorStrategy::CalleeHandled>,
        on_error: ThreadsafeFunction<String, ErrorStrategy::CalleeHandled>,
        on_complete: ThreadsafeFunction<(), ErrorStrategy::CalleeHandled>,
    ) -> Result<()> {
        let chunker = self.inner.clone();
        
        tokio::spawn(async move {
            let mut stream = Box::pin(chunker.chunk_code(content, language, file_path));
            
            while let Some(result) = stream.next().await {
                match result {
                    Ok(chunk) => {
                        let js_chunk = chunk.into();
                        if on_chunk.call(Ok(js_chunk), ThreadsafeFunctionCallMode::NonBlocking).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = on_error.call(Ok(e.to_string()), ThreadsafeFunctionCallMode::NonBlocking);
                        break;
                    }
                }
            }
            
            let _ = on_complete.call(Ok(()), ThreadsafeFunctionCallMode::NonBlocking);
        });
        
        Ok(())
    }
    
    #[napi(ts_args_type = "content: string, filePath: string | undefined, onChunk: (chunk: SemanticChunkJs) => void, onError: (error: Error) => void, onComplete: () => void")]
    pub fn chunk_text(
        &self,
        content: String,
        file_path: Option<String>,
        on_chunk: ThreadsafeFunction<SemanticChunkJs, ErrorStrategy::CalleeHandled>,
        on_error: ThreadsafeFunction<String, ErrorStrategy::CalleeHandled>,
        on_complete: ThreadsafeFunction<(), ErrorStrategy::CalleeHandled>,
    ) -> Result<()> {
        let chunker = self.inner.clone();
        
        tokio::spawn(async move {
            let mut stream = Box::pin(chunker.chunk_text(content, file_path));
            
            while let Some(result) = stream.next().await {
                match result {
                    Ok(chunk) => {
                        let js_chunk = chunk.into();
                        if on_chunk.call(Ok(js_chunk), ThreadsafeFunctionCallMode::NonBlocking).is_err() {
                            break;
                        }
                    }
                    Err(e) => {
                        let _ = on_error.call(Ok(e.to_string()), ThreadsafeFunctionCallMode::NonBlocking);
                        break;
                    }
                }
            }
            
            let _ = on_complete.call(Ok(()), ThreadsafeFunctionCallMode::NonBlocking);
        });
        
        Ok(())
    }
    
    #[napi]
    pub fn supported_languages() -> Vec<String> {
        supported_languages().iter().map(|&s| s.to_string()).collect()
    }
    
    #[napi]
    pub fn is_language_supported(language: String) -> bool {
        is_language_supported(&language)
    }
}

#[napi(ts_args_type = "path: string, maxChunkSize: number | undefined, tokenizer: TokenizerType | undefined, hfModel: string | undefined, maxParallel: number | undefined, onChunk: (chunk: ProjectChunkJs) => void, onError: (error: Error) => void, onComplete: () => void")]
pub fn walk_project(
    path: String,
    max_chunk_size: Option<i32>,
    tokenizer: Option<TokenizerType>,
    hf_model: Option<String>,
    max_parallel: Option<i32>,
    on_chunk: ThreadsafeFunction<ProjectChunkJs, ErrorStrategy::CalleeHandled>,
    on_error: ThreadsafeFunction<String, ErrorStrategy::CalleeHandled>,
    on_complete: ThreadsafeFunction<(), ErrorStrategy::CalleeHandled>,
) -> Result<()> {
    let max_chunk_size = max_chunk_size.unwrap_or(1500) as usize;
    let max_parallel = max_parallel.unwrap_or(8) as usize;
    
    // Convert JS TokenizerType to Rust Tokenizer
    let tokenizer_type = match tokenizer.unwrap_or(TokenizerType::Characters) {
        TokenizerType::Characters => RustTokenizer::Characters,
        TokenizerType::Tiktoken => RustTokenizer::Tiktoken,
        TokenizerType::HuggingFace => {
            match hf_model {
                Some(model) => RustTokenizer::HuggingFace(model),
                None => {
                    return Err(Error::new(
                        Status::InvalidArg,
                        "TokenizerType.HuggingFace requires hfModel parameter".to_string(),
                    ));
                }
            }
        }
    };
    
    tokio::spawn(async move {
        let mut stream = rust_walk_project(
            path,
            WalkOptions {
                max_chunk_size,
                tokenizer: tokenizer_type,
                max_parallel,
                ..Default::default()
            },
        );
        
        while let Some(result) = stream.next().await {
            match result {
                Ok(chunk) => {
                    let js_chunk = chunk.into();
                    if on_chunk.call(Ok(js_chunk), ThreadsafeFunctionCallMode::NonBlocking).is_err() {
                        break;
                    }
                }
                Err(e) => {
                    let _ = on_error.call(Ok(e.to_string()), ThreadsafeFunctionCallMode::NonBlocking);
                    break;
                }
            }
        }
        
        let _ = on_complete.call(Ok(()), ThreadsafeFunctionCallMode::NonBlocking);
    });
    
    Ok(())
}

