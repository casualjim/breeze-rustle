use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::array::Float32Array;
use futures::TryStreamExt;
use lancedb::Table;
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::rerankers::rrf;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::info;

use crate::embeddings::EmbeddingProvider;
use crate::models::CodeDocument;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
  pub id: String,
  pub file_path: String,
  pub content: String,
  pub content_hash: [u8; 32],
  pub relevance_score: f32,
  pub file_size: u64,
  pub last_modified: chrono::NaiveDateTime,
  pub indexed_at: chrono::NaiveDateTime,
}

/// Perform hybrid search combining vector and FTS
pub async fn hybrid_search(
  table: Arc<RwLock<Table>>,
  embedding_provider: Arc<dyn EmbeddingProvider>,
  query: &str,
  limit: usize,
) -> Result<Vec<SearchResult>> {
  info!(query = %query, limit = limit, "Performing hybrid search");

  // Generate embedding for query
  let query_embedding = embedding_provider
    .embed(&[crate::embeddings::EmbeddingInput {
      text: query,
      token_count: None,
    }])
    .await
    .map_err(|e| anyhow!("Failed to embed query: {}", e))?;

  if query_embedding.is_empty() {
    return Err(anyhow!("No embedding generated for query"));
  }

  let query_vector = query_embedding[0].as_slice();

  // Execute hybrid search
  let table = table.read().await;

  // Build hybrid query with vector search and FTS
  let mut results = table
    .query()
    .full_text_search(FullTextSearchQuery::new(query.to_string()))
    .nearest_to(query_vector)?
    .column("content_embedding")
    .rerank(Arc::new(rrf::RRFReranker::new(60.0)))
    .limit(limit)
    .execute()
    .await?;

  // Collect results from RecordBatch stream
  let mut search_results = Vec::new();

  while let Some(batch) = results
    .try_next()
    .await
    .map_err(|e| anyhow!("Error reading results: {}", e))?
  {
    // Try to get distance scores if available
    let distances = batch
      .column_by_name("_distance")
      .and_then(|col| col.as_any().downcast_ref::<Float32Array>());

    // Process each row
    for row in 0..batch.num_rows() {
      let doc = CodeDocument::from_record_batch(&batch, row)
        .map_err(|e| anyhow!("Failed to convert row {}: {}", row, e))?;

      // Get the actual distance/score if available
      // Note: Lower distance = better match in vector search
      // We convert to a score where higher = better
      let relevance_score = if let Some(distances) = distances {
        let distance = distances.value(row);
        // Convert distance to score: closer = higher score
        // Using 1/(1+distance) to get a score between 0 and 1
        1.0 / (1.0 + distance)
      } else {
        // If no distance column, just use 0.0 to indicate no score
        0.0
      };

      search_results.push(SearchResult {
        id: doc.id,
        file_path: doc.file_path,
        content: doc.content,
        content_hash: doc.content_hash,
        relevance_score,
        file_size: doc.file_size,
        last_modified: doc.last_modified,
        indexed_at: doc.indexed_at,
      });
    }
  }

  Ok(search_results)
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::models::CodeDocument;
  use lancedb::arrow::IntoArrow;
  use std::collections::HashSet;
  use tempfile::TempDir;

  struct MockEmbeddingProvider {
    embedding_dim: usize,
  }

  impl MockEmbeddingProvider {
    fn new(embedding_dim: usize) -> Self {
      Self { embedding_dim }
    }
  }

  #[async_trait::async_trait]
  impl EmbeddingProvider for MockEmbeddingProvider {
    async fn embed(
      &self,
      inputs: &[crate::embeddings::EmbeddingInput<'_>],
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
      Ok(
        inputs
          .iter()
          .map(|input| {
            // Generate a deterministic embedding based on text content
            let mut embedding = vec![0.0; self.embedding_dim];
            let hash = crate::models::CodeDocument::compute_hash(input.text);
            for (i, &byte) in hash.iter().enumerate() {
              if i < self.embedding_dim {
                embedding[i] = (byte as f32) / 255.0;
              }
            }
            embedding
          })
          .collect(),
      )
    }

    fn embedding_dim(&self) -> usize {
      self.embedding_dim
    }

    fn context_length(&self) -> usize {
      8192 // Mock context length
    }

    fn create_batching_strategy(&self) -> Box<dyn crate::embeddings::batching::BatchingStrategy> {
      Box::new(crate::embeddings::batching::LocalBatchingStrategy::new(
        100, // batch size
      ))
    }

    fn tokenizer(&self) -> Option<Arc<tokenizers::Tokenizer>> {
      None // Mock provider doesn't use a tokenizer
    }
  }

  async fn setup_test_table() -> (TempDir, Arc<RwLock<Table>>) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.lance");

    let connection = lancedb::connect(db_path.to_str().unwrap())
      .execute()
      .await
      .unwrap();

    let embedding_dim = 384;
    let table = CodeDocument::ensure_table(&connection, "test_embeddings", embedding_dim)
      .await
      .unwrap();

    // Insert test documents
    let test_documents = vec![
      create_test_document(
        "src/main.rs",
        "fn main() {\n    println!(\"Hello, world!\");\n}",
        embedding_dim,
      ),
      create_test_document(
        "src/lib.rs",
        "pub fn search(query: &str) -> Vec<String> {\n    // Search implementation\n    vec![]\n}",
        embedding_dim,
      ),
      create_test_document(
        "tests/test_search.rs",
        "#[test]\nfn test_search() {\n    let results = search(\"test\");\n    assert!(results.is_empty());\n}",
        embedding_dim,
      ),
      create_test_document(
        "README.md",
        "# Search Library\n\nA simple search implementation in Rust.",
        embedding_dim,
      ),
    ];

    for doc in test_documents {
      // Use the IntoArrow trait from CodeDocument
      let arrow_data = doc.into_arrow().unwrap();
      table.add(arrow_data).execute().await.unwrap();
    }

    (temp_dir, Arc::new(RwLock::new(table)))
  }

  fn create_test_document(file_path: &str, content: &str, embedding_dim: usize) -> CodeDocument {
    let mut doc = CodeDocument::new(file_path.to_string(), content.to_string());
    // Generate a deterministic embedding based on content
    let hash = CodeDocument::compute_hash(content);
    let mut embedding = vec![0.0; embedding_dim];
    for (i, &byte) in hash.iter().enumerate() {
      if i < embedding_dim {
        embedding[i] = (byte as f32) / 255.0;
      }
    }
    doc.update_embedding(embedding);
    doc
  }

  #[tokio::test]
  async fn test_hybrid_search_basic() {
    let (_temp_dir, table) = setup_test_table().await;
    let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

    let results = hybrid_search(table, embedding_provider, "search", 10)
      .await
      .unwrap();

    assert!(!results.is_empty());

    // Check that we get relevant results containing "search"
    let file_paths: HashSet<String> = results
      .iter()
      .map(|result| result.file_path.clone())
      .collect();

    // We expect at least lib.rs and test_search.rs to match
    assert!(file_paths.contains("src/lib.rs"));
    assert!(file_paths.contains("tests/test_search.rs"));
  }

  #[tokio::test]
  async fn test_hybrid_search_limit() {
    let (_temp_dir, table) = setup_test_table().await;
    let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

    let results = hybrid_search(table, embedding_provider, "test", 2)
      .await
      .unwrap();

    assert_eq!(results.len(), 2);
  }

  #[tokio::test]
  async fn test_hybrid_search_no_results() {
    let (_temp_dir, table) = setup_test_table().await;
    let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

    let results = hybrid_search(table, embedding_provider, "nonexistent", 10)
      .await
      .unwrap();

    // Hybrid search might still return results based on vector similarity
    // even if FTS doesn't match, so we just verify it completes without error
    assert!(results.len() <= 4); // We have 4 documents total
  }

  #[tokio::test]
  async fn test_hybrid_search_returns_complete_results() {
    let (_temp_dir, table) = setup_test_table().await;
    let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

    let results = hybrid_search(table, embedding_provider, "main", 10)
      .await
      .unwrap();

    assert!(!results.is_empty());

    // Verify all fields are populated
    for result in results {
      assert!(!result.id.is_empty());
      assert!(!result.file_path.is_empty());
      assert!(!result.content.is_empty());
      assert_ne!(result.content_hash, [0u8; 32]);
      assert!(result.file_size > 0);
      // Relevance score might be 0 if no distance column
      assert!(result.relevance_score >= 0.0);
    }
  }

  #[tokio::test]
  async fn test_search_result_fields() {
    let (_temp_dir, table) = setup_test_table().await;
    let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

    let results = hybrid_search(table, embedding_provider, "fn", 1)
      .await
      .unwrap();

    assert_eq!(results.len(), 1);
    let result = &results[0];

    // Verify the new fields are present
    assert!(!result.id.is_empty());
    assert!(uuid::Uuid::parse_str(&result.id).is_ok()); // Valid UUID
    assert_ne!(result.content_hash, [0u8; 32]); // Non-zero hash
    assert!(result.last_modified <= chrono::Utc::now().naive_utc());
    assert!(result.indexed_at <= chrono::Utc::now().naive_utc());
  }
}
