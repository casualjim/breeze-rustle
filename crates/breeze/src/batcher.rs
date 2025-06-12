use std::num::NonZeroUsize;
use futures_util::StreamExt;

use crate::pipeline::{Batcher, BoxStream, TextBatch};
use breeze_chunkers::ProjectChunk;

/// A pass-through batcher that creates batches of configurable size
pub struct PassthroughBatcher {
    batch_size: NonZeroUsize,
}

impl PassthroughBatcher {
    /// Create a new batcher with the specified batch size
    pub fn new(batch_size: NonZeroUsize) -> Self {
        Self { batch_size }
    }

    /// Create a single-item batcher (batch size = 1)
    pub fn single() -> Self {
        Self::new(NonZeroUsize::new(1).unwrap())
    }
}

impl Default for PassthroughBatcher {
    fn default() -> Self {
        Self::new(NonZeroUsize::new(32).unwrap()) // Default batch size of 32
    }
}

impl Batcher for PassthroughBatcher {
    fn batch(&self, chunks: BoxStream<ProjectChunk>) -> BoxStream<TextBatch> {
        let batch_size = self.batch_size.get();
        
        let stream = chunks
            .ready_chunks(batch_size)
            .map(move |chunks| chunks);

        Box::pin(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;

    fn create_test_chunk(index: usize) -> ProjectChunk {
        use breeze_chunkers::{SemanticChunk, ChunkMetadata, Chunk};
        
        let text = format!("Test chunk {}", index);
        let metadata = ChunkMetadata {
            node_type: "test".to_string(),
            node_name: Some(format!("test_{}", index)),
            language: "python".to_string(),
            parent_context: None,
            scope_path: vec!["module".to_string()],
            definitions: vec![],
            references: vec![],
        };
        
        let semantic_chunk = SemanticChunk {
            text,
            start_byte: index * 100,
            end_byte: (index + 1) * 100,
            start_line: index * 10,
            end_line: (index + 1) * 10,
            metadata,
        };
        
        ProjectChunk {
            file_path: format!("test{}.py", index),
            chunk: Chunk::Semantic(semantic_chunk),
        }
    }

    #[tokio::test]
    async fn test_passthrough_single() {
        let batcher = PassthroughBatcher::single();
        let chunks = vec![create_test_chunk(0), create_test_chunk(1)];
        let stream = stream::iter(chunks).boxed();
        
        let batches: Vec<TextBatch> = batcher.batch(stream).collect().await;
        
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 1);
        assert_eq!(batches[0][0].file_path, "test0.py");
        assert_eq!(batches[1].len(), 1);
        assert_eq!(batches[1][0].file_path, "test1.py");
    }

    #[tokio::test]
    async fn test_passthrough_batched() {
        let batcher = PassthroughBatcher::new(NonZeroUsize::new(2).unwrap());
        let chunks = vec![
            create_test_chunk(0), 
            create_test_chunk(1),
            create_test_chunk(2),
        ];
        let stream = stream::iter(chunks).boxed();
        
        let batches: Vec<TextBatch> = batcher.batch(stream).collect().await;
        
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 2);
        assert_eq!(batches[0][0].file_path, "test0.py");
        assert_eq!(batches[0][1].file_path, "test1.py");
        assert_eq!(batches[1].len(), 1);
        assert_eq!(batches[1][0].file_path, "test2.py");
    }
    
    #[tokio::test]
    async fn test_default_batch_size() {
        let batcher = PassthroughBatcher::default();
        assert_eq!(batcher.batch_size.get(), 32);
    }
    
    #[tokio::test]
    async fn test_exact_batch_size() {
        let batcher = PassthroughBatcher::new(NonZeroUsize::new(3).unwrap());
        let chunks: Vec<_> = (0..6).map(create_test_chunk).collect();
        let stream = stream::iter(chunks).boxed();
        
        let batches: Vec<TextBatch> = batcher.batch(stream).collect().await;
        
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 3);
    }
    
    #[tokio::test]
    async fn test_empty_stream() {
        let batcher = PassthroughBatcher::new(NonZeroUsize::new(5).unwrap());
        let stream = stream::empty().boxed();
        
        let batches: Vec<TextBatch> = batcher.batch(stream).collect().await;
        
        assert_eq!(batches.len(), 0);
    }
    
    #[tokio::test]
    async fn test_preserves_chunk_order() {
        let batcher = PassthroughBatcher::new(NonZeroUsize::new(2).unwrap());
        let chunks: Vec<_> = (0..5).map(create_test_chunk).collect();
        let stream = stream::iter(chunks).boxed();
        
        let batches: Vec<TextBatch> = batcher.batch(stream).collect().await;
        
        assert_eq!(batches.len(), 3); // [0,1], [2,3], [4]
        assert_eq!(batches[0][0].file_path, "test0.py");
        assert_eq!(batches[0][1].file_path, "test1.py");
        assert_eq!(batches[1][0].file_path, "test2.py");
        assert_eq!(batches[1][1].file_path, "test3.py");
        assert_eq!(batches[2][0].file_path, "test4.py");
    }
}