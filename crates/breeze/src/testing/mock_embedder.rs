use crate::pipeline::*;
use async_stream::stream;
use breeze_chunkers::{Chunk, ChunkError, ProjectFile};
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
  fn embed(&self, files: BoxStream<ProjectFile>) -> BoxStream<ProjectFileWithEmbeddings> {
    let embedder = self.clone();

    Box::pin(files.map(move |project_file| {
      let embedder = embedder.clone();
      let delay_ms = embedder.delay_ms;
      let file_path = project_file.file_path.clone();
      let metadata = project_file.metadata.clone();

      // Create a stream that embeds each chunk individually
      let embedded_chunks = Box::pin(stream! {
        let mut chunks = project_file.chunks;

        while let Some(chunk_result) = chunks.next().await {
          match chunk_result {
            Ok(chunk) => {
              // Simulate processing delay
              if let Some(delay) = delay_ms {
                tokio::time::sleep(tokio::time::Duration::from_millis(delay)).await;
              }

              let text = match &chunk {
                Chunk::Semantic(sc) => &sc.text,
                Chunk::Text(sc) => &sc.text,
              };

              // Generate embedding for this chunk
              let embedding = embedder.generate_embedding(text);

              yield Ok(EmbeddedChunk { chunk, embedding });
            }
            Err(e) => yield Err(e),
          }
        }
      });

      ProjectFileWithEmbeddings {
        file_path,
        metadata,
        embedded_chunks,
      }
    }))
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
    use tokio::sync::mpsc;
    use tokio_stream::wrappers::ReceiverStream;
    use breeze_chunkers::{ChunkMetadata, FileMetadata, SemanticChunk};
    use std::time::SystemTime;

    let embedder = MockEmbedder::new(128);

    // Create a test ProjectFile with chunks
    let (chunk_tx, chunk_rx) = mpsc::channel::<Result<Chunk, ChunkError>>(32);

    // Send some chunks
    tokio::spawn(async move {
      let chunk1 = Chunk::Text(SemanticChunk {
        text: "hello world".to_string(),
        start_byte: 0,
        end_byte: 11,
        start_line: 1,
        end_line: 1,
        tokens: None,
        metadata: ChunkMetadata {
          node_type: "text".to_string(),
          node_name: None,
          language: "rust".to_string(),
          parent_context: None,
          scope_path: vec![],
          definitions: vec![],
          references: vec![],
        },
      });

      let chunk2 = Chunk::Text(SemanticChunk {
        text: "test text".to_string(),
        start_byte: 12,
        end_byte: 21,
        start_line: 2,
        end_line: 2,
        tokens: None,
        metadata: ChunkMetadata {
          node_type: "text".to_string(),
          node_name: None,
          language: "rust".to_string(),
          parent_context: None,
          scope_path: vec![],
          definitions: vec![],
          references: vec![],
        },
      });

      let _ = chunk_tx.send(Ok(chunk1)).await;
      let _ = chunk_tx.send(Ok(chunk2)).await;
    });

    let project_file = ProjectFile {
      file_path: "test.rs".to_string(),
      chunks: ReceiverStream::new(chunk_rx),
      metadata: FileMetadata {
        primary_language: Some("Rust".to_string()),
        size: 100,
        modified: SystemTime::now(),
        content_hash: "hash123".to_string(),
        line_count: 10,
        is_binary: false,
      },
    };

    let files = Box::pin(futures_util::stream::once(async { project_file }));
    let mut files_with_embeddings = embedder.embed(files);

    // Get the first file with embeddings
    let file_with_embeddings = files_with_embeddings.next().await.unwrap();
    assert_eq!(file_with_embeddings.file_path, "test.rs");

    // Collect embedded chunks
    let embedded_chunks: Vec<EmbeddedChunk> = file_with_embeddings.embedded_chunks
      .filter_map(|result| async move { result.ok() })
      .collect()
      .await;

    assert_eq!(embedded_chunks.len(), 2); // Two chunks
    assert_eq!(embedded_chunks[0].embedding.len(), 128);
    assert_eq!(embedded_chunks[1].embedding.len(), 128);

    // Check that embeddings are normalized
    for embedded_chunk in &embedded_chunks {
      let norm: f32 = embedded_chunk.embedding
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        .sqrt();
      assert!((norm - 1.0).abs() < 0.001);
    }

    // Check that embeddings are different for different chunks
    assert_ne!(embedded_chunks[0].embedding, embedded_chunks[1].embedding);
  }
}
