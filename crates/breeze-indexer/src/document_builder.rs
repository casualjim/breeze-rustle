use tracing::{debug, error};

use crate::models::CodeDocument;
use crate::pipeline::FileAccumulator;
use breeze_chunkers::Chunk;

/// Build a document from accumulated file chunks using weighted average
pub(crate) async fn build_document_from_accumulator(
  accumulator: FileAccumulator,
  embedding_dim: usize,
) -> Option<CodeDocument> {
  if accumulator.embedded_chunks.is_empty() {
    return None;
  }

  let file_path = accumulator.file_path;
  let embedded_chunks = accumulator.embedded_chunks;

  // Calculate weights based on token counts
  let mut weights = Vec::new();
  let mut total_weight = 0.0;

  for embedded_chunk in &embedded_chunks {
    match &embedded_chunk.chunk {
      Chunk::Semantic(sc) | Chunk::Text(sc) => {
        // Use actual token count if available, otherwise estimate
        let token_count = sc
          .tokens
          .as_ref()
          .map(|tokens| tokens.len() as f32)
          .unwrap_or_else(|| (sc.text.len() as f32 / 4.0).max(1.0));
        weights.push(token_count);
        total_weight += token_count;
      }
      Chunk::EndOfFile { .. } => continue, // Skip EOF markers
    }
  }

  // Normalize weights
  for weight in &mut weights {
    *weight /= total_weight;
  }

  // Compute weighted average embedding
  let mut aggregated_embedding = vec![0.0; embedding_dim];

  for (i, embedded_chunk) in embedded_chunks.iter().enumerate() {
    if matches!(embedded_chunk.chunk, Chunk::EndOfFile { .. }) {
      continue;
    }
    let weight = weights[i];
    for (j, &value) in embedded_chunk.embedding.iter().enumerate() {
      if j < embedding_dim {
        aggregated_embedding[j] += value * weight;
      }
    }
  }

  debug!(
      file_path = %file_path,
      num_chunks = embedded_chunks.len(),
      total_tokens_approx = total_weight as u64,
      "Aggregated embeddings for file"
  );

  // Check if we have an EOF chunk with content
  let (content, content_hash) = if let Some(eof_chunk) = embedded_chunks
    .iter()
    .find(|ec| matches!(ec.chunk, Chunk::EndOfFile { .. }))
  {
    if let Chunk::EndOfFile {
      ref content,
      ref content_hash,
      ..
    } = eof_chunk.chunk
    {
      (content.clone(), *content_hash)
    } else {
      // This shouldn't happen but handle gracefully
      error!("EOF chunk found but missing content for {}", file_path);
      return None;
    }
  } else {
    // No EOF chunk found - this shouldn't happen in normal flow
    error!("No EOF chunk found for {}", file_path);
    return None;
  };

  // Create document with content from EOF chunk
  let mut doc = CodeDocument::new(file_path, content);
  doc.update_embedding(aggregated_embedding);
  doc.update_content_hash(content_hash);
  Some(doc)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::pipeline::EmbeddedChunk;
  use breeze_chunkers::{Chunk, ChunkMetadata, SemanticChunk};
  use std::io::Write;
  use tempfile::NamedTempFile;

  fn create_test_chunk(text: &str, embedding: Vec<f32>) -> EmbeddedChunk {
    EmbeddedChunk {
      chunk: Chunk::Text(SemanticChunk {
        text: text.to_string(),
        start_byte: 0,
        end_byte: text.len(),
        start_line: 1,
        end_line: 1,
        tokens: None,
        metadata: ChunkMetadata {
          node_type: "text".to_string(),
          node_name: None,
          language: "text".to_string(),
          parent_context: None,
          scope_path: vec![],
          definitions: vec![],
          references: vec![],
        },
      }),
      embedding,
    }
  }

  #[tokio::test]
  async fn test_weighted_average_builder() {
    // Create a temporary file
    let mut temp_file = NamedTempFile::new().unwrap();
    let content = "short\nthis is a longer chunk\nmedium chunk";
    write!(temp_file, "{}", content).unwrap();
    let file_path = temp_file.path().to_string_lossy().to_string();

    // Create test accumulator
    let mut accumulator = FileAccumulator::new(file_path.clone());

    // Add chunks
    accumulator.add_chunk(create_test_chunk("short", vec![1.0, 0.0, 0.0])); // weight ~1.25
    accumulator.add_chunk(create_test_chunk(
      "this is a longer chunk",
      vec![0.0, 1.0, 0.0],
    )); // weight ~5.5
    accumulator.add_chunk(create_test_chunk("medium chunk", vec![0.0, 0.0, 1.0])); // weight ~3.0

    // Add EOF chunk with content
    let hash = blake3::hash(content.as_bytes());
    let mut content_hash = [0u8; 32];
    content_hash.copy_from_slice(hash.as_bytes());
    accumulator.add_chunk(EmbeddedChunk {
      chunk: Chunk::EndOfFile {
        file_path: file_path.clone(),
        content: content.to_string(),
        content_hash,
      },
      embedding: vec![],
    });

    let doc = build_document_from_accumulator(accumulator, 3)
      .await
      .unwrap();

    // Verify document properties
    assert_eq!(doc.file_path, file_path);
    assert_eq!(doc.content, content);
    assert_eq!(doc.content_embedding.len(), 3);

    // The weighted average should favor the longer chunk
    assert!(doc.content_embedding[0] < 0.2); // Should be small
    assert!(doc.content_embedding[1] > 0.5); // Should be largest
    assert!(doc.content_embedding[2] > 0.2 && doc.content_embedding[2] < 0.4); // Should be medium

    // Sum should be approximately 1.0
    let sum: f32 = doc.content_embedding.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
  }

  #[tokio::test]
  async fn test_empty_chunks() {
    let accumulator = FileAccumulator::new("empty.txt".to_string());
    let result = build_document_from_accumulator(accumulator, 3).await;
    assert!(result.is_none());
  }
}
