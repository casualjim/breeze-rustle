use futures_util::StreamExt;
use tracing::{debug, warn};

use crate::models::CodeDocument;
use crate::pipeline::{BoxStream, DocumentBuilder, EmbeddedChunk, ProjectFileWithEmbeddings};
use breeze_chunkers::Chunk;

/// A document builder that uses weighted average of chunk embeddings
/// Weights are based on the token count of each chunk
pub struct WeightedAverageDocumentBuilder {
    embedding_dim: usize,
}

impl WeightedAverageDocumentBuilder {
    pub fn new(embedding_dim: usize) -> Self {
        Self { embedding_dim }
    }
}

impl DocumentBuilder for WeightedAverageDocumentBuilder {
    fn build_documents(&self, files: BoxStream<ProjectFileWithEmbeddings>) -> BoxStream<CodeDocument> {
        let embedding_dim = self.embedding_dim;

        let stream = files.then(move |file_with_embeddings| {
            let file_path = file_with_embeddings.file_path.clone();
            let metadata = file_with_embeddings.metadata.clone();

            async move {
                // Collect all embedded chunks
                let embedded_chunks: Vec<EmbeddedChunk> = file_with_embeddings.embedded_chunks
                    .filter_map(|result| async move {
                        match result {
                            Ok(chunk) => Some(chunk),
                            Err(e) => {
                                warn!("Failed to process chunk: {}", e);
                                None
                            }
                        }
                    })
                    .collect()
                    .await;

                if embedded_chunks.is_empty() {
                    return None;
                }

                // Calculate weights based on token counts
                let mut weights = Vec::new();
                let mut total_weight = 0.0;
                let mut full_content = String::new();

                for embedded_chunk in &embedded_chunks {
                    let text = match &embedded_chunk.chunk {
                        Chunk::Semantic(sc) => &sc.text,
                        Chunk::Text(sc) => &sc.text,
                    };

                    // Approximate token count (rough estimate: ~4 chars per token)
                    let token_count = (text.len() as f32 / 4.0).max(1.0);
                    weights.push(token_count);
                    total_weight += token_count;

                    // Concatenate content
                    if !full_content.is_empty() {
                        full_content.push('\n');
                    }
                    full_content.push_str(text);
                }

                // Normalize weights
                for weight in &mut weights {
                    *weight /= total_weight;
                }

                // Compute weighted average embedding
                let mut aggregated_embedding = vec![0.0; embedding_dim];

                for (i, embedded_chunk) in embedded_chunks.iter().enumerate() {
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

                // Create the document
                let mut doc = CodeDocument::new(file_path, full_content);
                doc.file_size = metadata.size;
                doc.last_modified = metadata.modified
                    .duration_since(std::time::UNIX_EPOCH)
                    .ok()
                    .and_then(|d| {
                        let secs = d.as_secs() as i64;
                        let nanos = d.subsec_nanos();
                        chrono::DateTime::from_timestamp(secs, nanos).map(|dt| dt.naive_utc())
                    })
                    .unwrap_or_else(|| chrono::Utc::now().naive_utc());
                doc.update_embedding(aggregated_embedding);

                Some(doc)
            }
        })
        .filter_map(|x| async move { x });

        Box::pin(stream)
    }
}

impl Default for WeightedAverageDocumentBuilder {
    fn default() -> Self {
        Self::new(512) // Default embedding dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::stream;
    use crate::pipeline::EmbeddedChunk;
    use breeze_chunkers::{Chunk, SemanticChunk, ChunkMetadata};
    use std::time::SystemTime;

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
        let builder = WeightedAverageDocumentBuilder::new(3);

        // Create test file with embeddings
        let chunk1 = create_test_chunk("short", vec![1.0, 0.0, 0.0]); // weight ~1.25
        let chunk2 = create_test_chunk("this is a longer chunk", vec![0.0, 1.0, 0.0]); // weight ~5.5
        let chunk3 = create_test_chunk("medium chunk", vec![0.0, 0.0, 1.0]); // weight ~3.0

        let file = ProjectFileWithEmbeddings {
            file_path: "test.txt".to_string(),
            metadata: breeze_chunkers::FileMetadata {
                size: 100,
                modified: SystemTime::now(),
                line_count: 3,
                primary_language: Some("text".to_string()),
                content_hash: "test".to_string(),
                is_binary: false,
            },
            embedded_chunks: Box::pin(stream::iter(vec![
                Ok(chunk1),
                Ok(chunk2),
                Ok(chunk3),
            ])),
        };

        let mut documents = builder.build_documents(Box::pin(stream::once(async { file })));
        let doc = documents.next().await.unwrap();

        // Verify document properties
        assert_eq!(doc.file_path, "test.txt");
        assert_eq!(doc.content, "short\nthis is a longer chunk\nmedium chunk");
        assert_eq!(doc.content_embedding.len(), 3);

        // The weighted average should favor the longer chunk
        // Total weight: 1.25 + 5.5 + 3.0 = 9.75
        // Weights: [0.128, 0.564, 0.308]
        // Expected embedding: [0.128, 0.564, 0.308]
        assert!(doc.content_embedding[0] < 0.2); // Should be small
        assert!(doc.content_embedding[1] > 0.5); // Should be largest
        assert!(doc.content_embedding[2] > 0.2 && doc.content_embedding[2] < 0.4); // Should be medium

        // Sum should be approximately 1.0
        let sum: f32 = doc.content_embedding.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn test_empty_chunks() {
        let builder = WeightedAverageDocumentBuilder::new(3);

        let file = ProjectFileWithEmbeddings {
            file_path: "empty.txt".to_string(),
            metadata: breeze_chunkers::FileMetadata {
                size: 0,
                modified: SystemTime::now(),
                line_count: 0,
                primary_language: Some("text".to_string()),
                content_hash: "empty".to_string(),
                is_binary: false,
            },
            embedded_chunks: Box::pin(stream::empty()),
        };

        let mut documents = builder.build_documents(Box::pin(stream::once(async { file })));
        let result = documents.next().await;

        assert!(result.is_none());
    }
}
