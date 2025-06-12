use crate::pipeline::*;
use async_stream::stream;
use breeze_chunkers::{Chunk, ChunkMetadata, ProjectChunk, SemanticChunk};
use futures_util::StreamExt;

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

            // Create embeddings for each chunk
            let batch_items: Vec<EmbeddingBatchItem> = batch.into_iter()
                .map(|chunk| {
                    let text = match &chunk.chunk {
                        Chunk::Semantic(sc) => &sc.text,
                        Chunk::Text(sc) => &sc.text,
                    };

                    let embedding = embedder.generate_embedding(text);

                    EmbeddingBatchItem {
                        embeddings: embedding,
                        metadata: vec![chunk],
                    }
                })
                .collect();

            yield batch_items;
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

    let chunk1 = ProjectChunk {
      file_path: "test1.rs".to_string(),
      chunk: Chunk::Text(SemanticChunk {
        text: "hello world".to_string(),
        start_byte: 0,
        end_byte: 11,
        start_line: 1,
        end_line: 1,
        metadata: ChunkMetadata {
          node_type: "text".to_string(),
          node_name: None,
          language: "rust".to_string(),
          parent_context: None,
          scope_path: vec![],
          definitions: vec![],
          references: vec![],
        },
      }),
    };

    let chunk2 = ProjectChunk {
      file_path: "test2.rs".to_string(),
      chunk: Chunk::Text(SemanticChunk {
        text: "test text".to_string(),
        start_byte: 0,
        end_byte: 9,
        start_line: 1,
        end_line: 1,
        metadata: ChunkMetadata {
          node_type: "text".to_string(),
          node_name: None,
          language: "rust".to_string(),
          parent_context: None,
          scope_path: vec![],
          definitions: vec![],
          references: vec![],
        },
      }),
    };

    let batch: TextBatch = vec![chunk1, chunk2];

    let batches = Box::pin(futures_util::stream::once(async { batch }));
    let mut embeddings = embedder.embed(batches);

    let result = embeddings.next().await.unwrap();
    assert_eq!(result.len(), 2); // Two batch items
    assert_eq!(result[0].embeddings.len(), 128);
    assert_eq!(result[1].embeddings.len(), 128);

    // Check normalization
    for batch_item in &result {
      let norm: f32 = batch_item
        .embeddings
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
      assert!((norm - 1.0).abs() < 0.001);
    }

    // Check deterministic
    let embedder2 = MockEmbedder::new(128);
    let chunk = ProjectChunk {
      file_path: "test.rs".to_string(),
      chunk: Chunk::Text(SemanticChunk {
        text: "hello world".to_string(),
        start_byte: 0,
        end_byte: 11,
        start_line: 1,
        end_line: 1,
        metadata: ChunkMetadata {
          node_type: "text".to_string(),
          node_name: None,
          language: "rust".to_string(),
          parent_context: None,
          scope_path: vec![],
          definitions: vec![],
          references: vec![],
        },
      }),
    };
    let batch2: TextBatch = vec![chunk];

    let batches2 = Box::pin(futures_util::stream::once(async { batch2 }));
    let mut embeddings2 = embedder2.embed(batches2);
    let result2 = embeddings2.next().await.unwrap();

    assert_eq!(result[0].embeddings, result2[0].embeddings);
  }
}
