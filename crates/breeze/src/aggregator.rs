use std::collections::HashMap;
use futures_util::{StreamExt, FutureExt};

use crate::pipeline::{Aggregator, BoxStream, EmbeddingBatch, EmbeddingBatchItem};
use crate::models::CodeDocument;
use breeze_chunkers::{ProjectChunk, Chunk};

/// File-aware aggregator that groups chunks by file and computes weighted average embeddings
pub struct FileAggregator;

impl FileAggregator {
    /// Create a new file aggregator
    pub fn new() -> Self {
        Self
    }
}

impl Default for FileAggregator {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper struct to accumulate chunks and embeddings for a file
#[derive(Default)]
struct FileAccumulator {
    chunks: Vec<ProjectChunk>,
    embeddings: Vec<Vec<f32>>,
    total_length: usize,
}

impl FileAccumulator {
    fn add(&mut self, chunk: ProjectChunk, embedding: Vec<f32>) {
        let chunk_length = match &chunk.chunk {
            Chunk::Semantic(sc) => sc.text.len(),
            Chunk::Text(sc) => sc.text.len(),
        };
        self.total_length += chunk_length;
        self.chunks.push(chunk);
        self.embeddings.push(embedding);
    }
    
    /// Compute weighted average embedding based on chunk text lengths
    fn compute_weighted_embedding(&self) -> Vec<f32> {
        if self.embeddings.is_empty() {
            return vec![];
        }
        
        let embedding_dim = self.embeddings[0].len();
        let mut weighted_sum = vec![0.0f32; embedding_dim];
        
        for (chunk, embedding) in self.chunks.iter().zip(&self.embeddings) {
            let chunk_length = match &chunk.chunk {
                Chunk::Semantic(sc) => sc.text.len(),
                Chunk::Text(sc) => sc.text.len(),
            } as f32;
            
            let weight = chunk_length / self.total_length as f32;
            
            for (i, &value) in embedding.iter().enumerate() {
                weighted_sum[i] += value * weight;
            }
        }
        
        weighted_sum
    }
}

impl Aggregator for FileAggregator {
    fn aggregate(&self, embeddings: BoxStream<EmbeddingBatch>) -> BoxStream<CodeDocument> {
        // First collect all embeddings by file
        let collected = embeddings
            .fold(HashMap::<String, FileAccumulator>::new(), |mut acc, batch| async move {
                for item in batch {
                    if let Some(chunk) = item.metadata.into_iter().next() {
                        let file_path = chunk.file_path.clone();
                        acc.entry(file_path)
                            .or_default()
                            .add(chunk, item.embeddings);
                    }
                }
                acc
            });
        
        // Then process each file
        let stream = collected
            .into_stream()
            .flat_map(|accumulators| {
                futures_util::stream::iter(accumulators.into_iter())
                    .then(|(file_path, accumulator)| async move {
                        let weighted_embedding = accumulator.compute_weighted_embedding();
                        
                        match CodeDocument::from_file(&file_path).await {
                            Ok(mut document) => {
                                document.update_embedding(weighted_embedding);
                                Some(document)
                            }
                            Err(e) => {
                                tracing::error!("Failed to read file {}: {}", file_path, e);
                                None
                            }
                        }
                    })
                    .filter_map(|x| async move { x })
            });
        
        Box::pin(stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;
    use breeze_chunkers::{SemanticChunk, ChunkMetadata};
    use std::fs;
    use tempfile::TempDir;
    
    fn create_test_chunk(file_path: &str, text: &str, start: usize) -> ProjectChunk {
        let metadata = ChunkMetadata {
            node_type: "test".to_string(),
            node_name: None,
            language: "python".to_string(),
            parent_context: None,
            scope_path: vec![],
            definitions: vec![],
            references: vec![],
        };
        
        let semantic_chunk = SemanticChunk {
            text: text.to_string(),
            start_byte: start,
            end_byte: start + text.len(),
            start_line: 1,
            end_line: 1,
            metadata,
        };
        
        ProjectChunk {
            file_path: file_path.to_string(),
            chunk: Chunk::Semantic(semantic_chunk),
        }
    }
    
    fn create_test_embedding(dim: usize, value: f32) -> Vec<f32> {
        vec![value; dim]
    }
    
    #[tokio::test]
    async fn test_file_aggregation() {
        // Create temporary test files
        let temp_dir = TempDir::new().unwrap();
        let file1_path = temp_dir.path().join("file1.py");
        let file2_path = temp_dir.path().join("file2.py");
        
        fs::write(&file1_path, "Hello World").unwrap();
        fs::write(&file2_path, "World").unwrap();
        
        let aggregator = FileAggregator::new();
        
        // Create test data
        let batch1 = vec![
            EmbeddingBatchItem {
                embeddings: create_test_embedding(3, 1.0),
                metadata: vec![create_test_chunk(&file1_path.to_string_lossy(), "Hello ", 0)],
            },
            EmbeddingBatchItem {
                embeddings: create_test_embedding(3, 2.0),
                metadata: vec![create_test_chunk(&file2_path.to_string_lossy(), "World", 0)],
            },
        ];
        
        let batch2 = vec![
            EmbeddingBatchItem {
                embeddings: create_test_embedding(3, 3.0),
                metadata: vec![create_test_chunk(&file1_path.to_string_lossy(), "World", 6)],
            },
        ];
        
        let stream = stream::iter(vec![batch1, batch2]).boxed();
        let mut results: Vec<CodeDocument> = aggregator.aggregate(stream).collect().await;
        
        // Sort by file path for consistent testing
        results.sort_by(|a, b| a.file_path.cmp(&b.file_path));
        
        assert_eq!(results.len(), 2);
        
        // Check file1.py
        assert_eq!(results[0].file_path, file1_path.to_string_lossy());
        assert_eq!(results[0].content, "Hello World");
        // Weighted average: (1.0 * 6 + 3.0 * 5) / 11 = 21/11 ≈ 1.909
        let expected_embedding = 21.0 / 11.0;
        assert!((results[0].content_embedding[0] - expected_embedding).abs() < 0.001);
        
        // Check file2.py
        assert_eq!(results[1].file_path, file2_path.to_string_lossy());
        assert_eq!(results[1].content, "World");
        assert_eq!(results[1].content_embedding, vec![2.0, 2.0, 2.0]);
    }
    
    #[tokio::test]
    async fn test_empty_stream() {
        let aggregator = FileAggregator::new();
        let stream = stream::empty().boxed();
        let results: Vec<CodeDocument> = aggregator.aggregate(stream).collect().await;
        assert_eq!(results.len(), 0);
    }
    
    #[tokio::test]
    async fn test_single_file_single_chunk() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("single.py");
        fs::write(&file_path, "def hello(): pass").unwrap();
        
        let aggregator = FileAggregator::new();
        
        let batch = vec![
            EmbeddingBatchItem {
                embeddings: vec![1.0, 2.0, 3.0],
                metadata: vec![create_test_chunk(&file_path.to_string_lossy(), "def hello(): pass", 0)],
            },
        ];
        
        let stream = stream::once(async { batch }).boxed();
        let results: Vec<CodeDocument> = aggregator.aggregate(stream).collect().await;
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, file_path.to_string_lossy());
        assert_eq!(results[0].content, "def hello(): pass");
        assert_eq!(results[0].content_embedding, vec![1.0, 2.0, 3.0]);
    }
    
    #[tokio::test]
    async fn test_multiple_files_in_single_batch() {
        let temp_dir = TempDir::new().unwrap();
        let file1 = temp_dir.path().join("file1.py");
        let file2 = temp_dir.path().join("file2.py");
        let file3 = temp_dir.path().join("file3.py");
        
        fs::write(&file1, "content1").unwrap();
        fs::write(&file2, "content2").unwrap();
        fs::write(&file3, "content3").unwrap();
        
        let aggregator = FileAggregator::new();
        
        let batch = vec![
            EmbeddingBatchItem {
                embeddings: vec![1.0],
                metadata: vec![create_test_chunk(&file1.to_string_lossy(), "content1", 0)],
            },
            EmbeddingBatchItem {
                embeddings: vec![2.0],
                metadata: vec![create_test_chunk(&file2.to_string_lossy(), "content2", 0)],
            },
            EmbeddingBatchItem {
                embeddings: vec![3.0],
                metadata: vec![create_test_chunk(&file3.to_string_lossy(), "content3", 0)],
            },
        ];
        
        let stream = stream::once(async { batch }).boxed();
        let mut results: Vec<CodeDocument> = aggregator.aggregate(stream).collect().await;
        results.sort_by(|a, b| a.file_path.cmp(&b.file_path));
        
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].content_embedding, vec![1.0]);
        assert_eq!(results[1].content_embedding, vec![2.0]);
        assert_eq!(results[2].content_embedding, vec![3.0]);
    }
    
    #[tokio::test]
    async fn test_weighted_average_calculation() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("weighted.py");
        
        // Content with different chunk sizes for testing weighted average
        fs::write(&file_path, "short\nlonger chunk here").unwrap();
        
        let aggregator = FileAggregator::new();
        
        // First chunk: "short" (5 chars), embedding [1.0, 1.0]
        // Second chunk: "longer chunk here" (17 chars), embedding [3.0, 3.0]
        // Expected weighted average: (1.0 * 5 + 3.0 * 17) / 22 = 56/22 ≈ 2.545
        let batch = vec![
            EmbeddingBatchItem {
                embeddings: vec![1.0, 1.0],
                metadata: vec![create_test_chunk(&file_path.to_string_lossy(), "short", 0)],
            },
            EmbeddingBatchItem {
                embeddings: vec![3.0, 3.0],
                metadata: vec![create_test_chunk(&file_path.to_string_lossy(), "longer chunk here", 6)],
            },
        ];
        
        let stream = stream::once(async { batch }).boxed();
        let results: Vec<CodeDocument> = aggregator.aggregate(stream).collect().await;
        
        assert_eq!(results.len(), 1);
        let expected_value = (1.0 * 5.0 + 3.0 * 17.0) / 22.0;
        assert!((results[0].content_embedding[0] - expected_value).abs() < 0.001);
        assert!((results[0].content_embedding[1] - expected_value).abs() < 0.001);
    }
    
    #[tokio::test]
    async fn test_multiple_batches_same_file() {
        let temp_dir = TempDir::new().unwrap();
        let file_path = temp_dir.path().join("multi_batch.py");
        fs::write(&file_path, "chunk1\nchunk2\nchunk3").unwrap();
        
        let aggregator = FileAggregator::new();
        
        let batch1 = vec![
            EmbeddingBatchItem {
                embeddings: vec![1.0],
                metadata: vec![create_test_chunk(&file_path.to_string_lossy(), "chunk1", 0)],
            },
        ];
        
        let batch2 = vec![
            EmbeddingBatchItem {
                embeddings: vec![2.0],
                metadata: vec![create_test_chunk(&file_path.to_string_lossy(), "chunk2", 7)],
            },
            EmbeddingBatchItem {
                embeddings: vec![3.0],
                metadata: vec![create_test_chunk(&file_path.to_string_lossy(), "chunk3", 14)],
            },
        ];
        
        let stream = stream::iter(vec![batch1, batch2]).boxed();
        let results: Vec<CodeDocument> = aggregator.aggregate(stream).collect().await;
        
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].file_path, file_path.to_string_lossy());
        assert_eq!(results[0].content, "chunk1\nchunk2\nchunk3");
        
        // Weighted average: (1*6 + 2*6 + 3*6) / 18 = 36/18 = 2.0
        assert_eq!(results[0].content_embedding[0], 2.0);
    }
}