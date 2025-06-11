use async_stream::stream;
use futures_util::StreamExt;

use crate::pipeline::*;
use breeze_chunkers::{ProjectChunk, Chunk};

/// Mock batcher that groups chunks into configurable batch sizes
pub struct MockBatcher {
    batch_size: usize,
    max_tokens: Option<usize>,
}

impl MockBatcher {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            max_tokens: None,
        }
    }
    
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

impl Default for MockBatcher {
    fn default() -> Self {
        Self::new(2)
    }
}

impl Batcher for MockBatcher {
    fn batch(&self, chunks: BoxStream<ProjectChunk>) -> BoxStream<TextBatch> {
        let batch_size = self.batch_size;
        let max_tokens = self.max_tokens;
        
        Box::pin(stream! {
            let mut buffer = Vec::new();
            let mut metadata_buffer = Vec::new();
            let mut current_tokens = 0;
            
            let mut chunks = chunks;
            while let Some(chunk) = chunks.next().await {
                let text = match &chunk.chunk {
                    Chunk::Semantic(sc) => &sc.text,
                    Chunk::Text(sc) => &sc.text,
                };
                
                let token_count = text.len() / 4; // Rough approximation
                
                // Check if adding this chunk would exceed token limit
                if let Some(max) = max_tokens {
                    if current_tokens + token_count > max && !buffer.is_empty() {
                        // Yield current batch before adding new chunk
                        yield TextBatch {
                            texts: buffer.drain(..).collect(),
                            metadata: metadata_buffer.drain(..).collect(),
                        };
                        current_tokens = 0;
                    }
                }
                
                buffer.push(text.clone());
                metadata_buffer.push(BatchMetadata {
                    file_path: chunk.file_path.clone(),
                    chunk_index: metadata_buffer.len(),
                    token_count,
                });
                current_tokens += token_count;
                
                if buffer.len() >= batch_size {
                    yield TextBatch {
                        texts: buffer.drain(..).collect(),
                        metadata: metadata_buffer.drain(..).collect(),
                    };
                    current_tokens = 0;
                }
            }
            
            // Yield remaining items
            if !buffer.is_empty() {
                yield TextBatch {
                    texts: buffer,
                    metadata: metadata_buffer,
                };
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    use crate::testing::MockPathWalker;
    use crate::config::Config;
    use std::path::Path;
    
    #[tokio::test]
    async fn test_mock_batcher_by_count() {
        let walker = MockPathWalker::new(2, 5);
        let batcher = MockBatcher::new(3);
        
        let chunks = walker.walk(Path::new("/test"));
        let mut batches = batcher.batch(chunks);
        
        let mut batch_count = 0;
        let mut total_texts = 0;
        
        while let Some(batch) = batches.next().await {
            assert!(batch.texts.len() <= 3);
            assert_eq!(batch.texts.len(), batch.metadata.len());
            batch_count += 1;
            total_texts += batch.texts.len();
        }
        
        assert_eq!(batch_count, 4); // 10 chunks / 3 per batch = 4 batches
        assert_eq!(total_texts, 10); // 2 files * 5 chunks
    }
    
    #[tokio::test]
    async fn test_mock_batcher_with_tokens() {
        let walker = MockPathWalker::new(1, 3);
        let batcher = MockBatcher::new(10).with_max_tokens(50);
        
        let chunks = walker.walk(Path::new("/test"));
        let mut batches = batcher.batch(chunks);
        
        while let Some(batch) = batches.next().await {
            let total_tokens: usize = batch.metadata.iter()
                .map(|m| m.token_count)
                .sum();
            assert!(total_tokens <= 50 || batch.texts.len() == 1);
        }
    }
}