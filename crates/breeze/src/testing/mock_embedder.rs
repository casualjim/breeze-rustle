use async_stream::stream;
use futures_util::StreamExt;
use crate::pipeline::*;

/// Mock embedder that generates fake embeddings for testing
pub struct MockEmbedder {
    dimension: usize,
    delay_ms: Option<u64>,
}

impl MockEmbedder {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            delay_ms: None,
        }
    }
    
    pub fn with_delay_ms(mut self, delay_ms: u64) -> Self {
        self.delay_ms = Some(delay_ms);
        self
    }
    
    /// Generate a deterministic embedding based on text content
    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        let mut embedding = vec![0.0; self.dimension];
        let text_hash = text.bytes().fold(0u32, |acc, b| acc.wrapping_add(b as u32));
        
        for i in 0..self.dimension {
            // Generate deterministic values based on text hash
            embedding[i] = ((text_hash.wrapping_mul(i as u32 + 1) % 1000) as f32) / 1000.0;
        }
        
        // Normalize
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for val in &mut embedding {
                *val /= norm;
            }
        }
        
        embedding
    }
}

impl Default for MockEmbedder {
    fn default() -> Self {
        Self::new(384)
    }
}

impl Embedder for MockEmbedder {
    fn embed(&self, batches: BoxStream<TextBatch>) -> BoxStream<EmbeddingBatch> {
        let delay_ms = self.delay_ms;
        let embedder = self.clone();
        
        Box::pin(stream! {
            let mut batches = batches;
            while let Some(batch) = batches.next().await {
                // Simulate processing delay
                if let Some(delay) = delay_ms {
                    tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
                }
                
                let embeddings = batch.texts.iter()
                    .map(|text| embedder.generate_embedding(text))
                    .collect();
                
                yield EmbeddingBatch {
                    embeddings,
                    metadata: batch.metadata,
                };
            }
        })
    }
}

impl Clone for MockEmbedder {
    fn clone(&self) -> Self {
        Self {
            dimension: self.dimension,
            delay_ms: self.delay_ms,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::StreamExt;
    
    #[tokio::test]
    async fn test_mock_embedder() {
        let embedder = MockEmbedder::new(128);
        
        let batch = TextBatch {
            texts: vec!["hello world".to_string(), "test text".to_string()],
            metadata: vec![
                BatchMetadata {
                    file_path: "test.rs".to_string(),
                    chunk_index: 0,
                    token_count: 2,
                },
                BatchMetadata {
                    file_path: "test.rs".to_string(),
                    chunk_index: 1,
                    token_count: 2,
                },
            ],
        };
        
        let batches = Box::pin(futures_util::stream::once(async { batch }));
        let mut embeddings = embedder.embed(batches);
        
        let result = embeddings.next().await.unwrap();
        assert_eq!(result.embeddings.len(), 2);
        assert_eq!(result.embeddings[0].len(), 128);
        assert_eq!(result.embeddings[1].len(), 128);
        
        // Check normalization
        for embedding in &result.embeddings {
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.001);
        }
        
        // Check deterministic
        let embedder2 = MockEmbedder::new(128);
        let batch2 = TextBatch {
            texts: vec!["hello world".to_string()],
            metadata: vec![BatchMetadata {
                file_path: "test.rs".to_string(),
                chunk_index: 0,
                token_count: 2,
            }],
        };
        
        let batches2 = Box::pin(futures_util::stream::once(async { batch2 }));
        let mut embeddings2 = embedder2.embed(batches2);
        let result2 = embeddings2.next().await.unwrap();
        
        assert_eq!(result.embeddings[0], result2.embeddings[0]);
    }
}