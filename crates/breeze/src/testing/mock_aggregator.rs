use async_stream::stream;
use futures_util::StreamExt;

use crate::{models::CodeDocument, pipeline::{Aggregator, BoxStream, EmbeddingBatch}};


/// Mock aggregator that combines embeddings per file
pub struct MockAggregator;

impl Aggregator for MockAggregator {
    fn aggregate(&self, embeddings: BoxStream<EmbeddingBatch>) -> BoxStream<CodeDocument> {
        Box::pin(stream! {
            let mut file_embeddings: std::collections::HashMap<String, Vec<(Vec<f32>, usize)>> = 
                std::collections::HashMap::new();
            
            let mut embeddings = embeddings;
            while let Some(batch) = embeddings.next().await {
                for (embedding, metadata) in batch.embeddings.into_iter().zip(batch.metadata) {
                    file_embeddings
                        .entry(metadata.file_path.clone())
                        .or_default()
                        .push((embedding, metadata.token_count));
                }
            }
            
            // Aggregate embeddings per file
            for (file_path, embeddings) in file_embeddings {
                if embeddings.is_empty() {
                    continue;
                }
                
                let dimension = embeddings[0].0.len();
                let total_tokens: usize = embeddings.iter().map(|(_, tokens)| tokens).sum();
                
                // Weighted average
                let mut final_embedding = vec![0.0; dimension];
                for (embedding, token_count) in &embeddings {
                    let weight = *token_count as f32 / total_tokens as f32;
                    for (i, val) in embedding.iter().enumerate() {
                        final_embedding[i] += val * weight;
                    }
                }
                
                let mut doc = CodeDocument::new(
                    file_path.clone(),
                    format!("// Mock content for {}\n// {} chunks, {} tokens", 
                        file_path, embeddings.len(), total_tokens)
                );
                doc.update_embedding(final_embedding);
                yield doc;
            }
        })
    }
}