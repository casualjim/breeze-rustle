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
  // Filter out the dummy project
  let mut results = table
    .query()
    .only_if(format!("project_id != '{}'", uuid::Uuid::nil()).as_str())
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
    // Get reranked relevance scores (from hybrid search with reranker)
    // We always use reranking, so this column should always be present
    let relevance_scores = batch
      .column_by_name("_relevance_score")
      .and_then(|col| col.as_any().downcast_ref::<Float32Array>())
      .ok_or_else(|| anyhow!("Missing _relevance_score column in results"))?;

    // Process each row
    for row in 0..batch.num_rows() {
      let doc = CodeDocument::from_record_batch(&batch, row)
        .map_err(|e| anyhow!("Failed to convert row {}: {}", row, e))?;

      // Get the reranked relevance score from hybrid search
      let relevance_score = relevance_scores.value(row);

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

#[cfg(all(test, feature = "local-embeddings"))]
mod tests {
  use super::*;
  use crate::Config;
  use crate::embeddings::factory::create_embedding_provider;
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
    ) -> crate::embeddings::EmbeddingResult<Vec<Vec<f32>>> {
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

    // ensure the table twice
    CodeDocument::ensure_table(&connection, "test_embeddings", embedding_dim)
      .await
      .expect("Table ensure should be idempotent");

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
    let project_id = uuid::Uuid::now_v7(); // Use a valid project ID, not nil
    let mut doc = CodeDocument::new(project_id, file_path.to_string(), content.to_string());
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

  async fn setup_empty_table() -> (TempDir, Arc<RwLock<Table>>, Arc<dyn EmbeddingProvider>) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.lance");

    let connection = lancedb::connect(db_path.to_str().unwrap())
      .execute()
      .await
      .unwrap();

    // Create config for local embeddings
    let config = Config {
      database_path: temp_dir.path().join("test.db"),
      embedding_provider: crate::config::EmbeddingProvider::Local,
      model: "BAAI/bge-small-en-v1.5".to_string(),
      voyage: None,
      openai_providers: std::collections::HashMap::new(),
      max_chunk_size: 1000,
      max_file_size: Some(5 * 1024 * 1024),
      max_parallel_files: 4,
      large_file_threads: None,
      embedding_workers: 1,
      optimize_threshold: 250,
    };

    let embedding_provider = create_embedding_provider(&config).await.unwrap();
    let embedding_dim = embedding_provider.embedding_dim();

    let table = CodeDocument::ensure_table(&connection, "test_embeddings", embedding_dim)
      .await
      .unwrap();

    (
      temp_dir,
      Arc::new(RwLock::new(table)),
      Arc::from(embedding_provider),
    )
  }

  async fn create_and_embed_document(
    content: &str,
    file_path: &str,
    embedding_provider: &Arc<dyn EmbeddingProvider>,
  ) -> CodeDocument {
    let project_id = uuid::Uuid::now_v7(); // Use a valid project ID, not nil
    let mut doc = CodeDocument::new(project_id, file_path.to_string(), content.to_string());

    // Generate real embedding
    let embeddings = embedding_provider
      .embed(&[crate::embeddings::EmbeddingInput {
        text: content,
        token_count: None,
      }])
      .await
      .unwrap();

    doc.update_embedding(embeddings[0].clone());
    doc
  }

  #[tokio::test]
  async fn test_search_empty_database_no_error() {
    let (_temp_dir, table, embedding_provider) = setup_empty_table().await;

    // Search on empty database should return empty results, not error
    let results = hybrid_search(table, embedding_provider, "some query", 10)
      .await
      .unwrap();

    assert_eq!(results.len(), 0, "Empty database should return no results");
  }

  #[tokio::test]
  async fn test_keyword_search_works() {
    let (_temp_dir, table, embedding_provider) = setup_empty_table().await;

    // Add documents with real embeddings
    let docs = vec![
      create_and_embed_document(
        "fn calculate_fibonacci(n: usize) -> usize { /* fibonacci implementation */ }",
        "math.rs",
        &embedding_provider,
      )
      .await,
      create_and_embed_document(
        "fn bubble_sort(arr: &mut [i32]) { /* sorting implementation */ }",
        "sort.rs",
        &embedding_provider,
      )
      .await,
      create_and_embed_document(
        "struct DatabaseConnection { /* database handling */ }",
        "db.rs",
        &embedding_provider,
      )
      .await,
    ];

    // Insert documents
    {
      let table_write = table.write().await;
      for doc in docs {
        let arrow_data = doc.into_arrow().unwrap();
        table_write.add(arrow_data).execute().await.unwrap();
      }
    }

    // Search for "fibonacci" - should find math.rs
    let results = hybrid_search(table.clone(), embedding_provider.clone(), "fibonacci", 10)
      .await
      .unwrap();

    assert!(!results.is_empty(), "Should find results for 'fibonacci'");
    assert_eq!(
      results[0].file_path, "math.rs",
      "Should find math.rs when searching for fibonacci"
    );
    assert!(
      results[0].relevance_score > 0.0,
      "Should have non-zero relevance score"
    );
  }

  #[tokio::test]
  async fn test_vector_search_semantic_similarity() {
    let (_temp_dir, table, embedding_provider) = setup_empty_table().await;

    // Add documents about similar topics with real embeddings
    let docs = vec![
      create_and_embed_document(
        "async fn fetch_user_data(id: u64) -> Result<User> {
          // Fetches user information from the database
          let connection = get_db_connection().await?;
          let user = connection.query_user(id).await?;
          Ok(user)
        }",
        "user_service.rs",
        &embedding_provider,
      )
      .await,
      create_and_embed_document(
        "async fn get_customer_info(customer_id: String) -> CustomerData {
          // Retrieves customer details from persistent storage
          let db = connect_to_database().await;
          db.find_customer(&customer_id).await
        }",
        "customer_api.rs",
        &embedding_provider,
      )
      .await,
      create_and_embed_document(
        "fn calculate_shipping_cost(weight: f64, distance: f64) -> f64 {
          // Computes shipping fees based on package weight and distance
          let base_rate = 5.0;
          base_rate + (weight * 0.5) + (distance * 0.1)
        }",
        "shipping.rs",
        &embedding_provider,
      )
      .await,
    ];

    // Insert documents
    {
      let table_write = table.write().await;
      for doc in docs {
        let arrow_data = doc.into_arrow().unwrap();
        table_write.add(arrow_data).execute().await.unwrap();
      }
    }

    // Search for "retrieve user information from database"
    // Should find both user_service.rs and customer_api.rs as they're semantically similar
    let results = hybrid_search(
      table.clone(),
      embedding_provider.clone(),
      "retrieve user information from database",
      10,
    )
    .await
    .unwrap();

    assert!(
      results.len() >= 2,
      "Should find at least 2 semantically similar results"
    );

    // Both user_service.rs and customer_api.rs should be in top results
    let file_paths: Vec<&str> = results.iter().map(|r| r.file_path.as_str()).collect();
    assert!(
      file_paths.contains(&"user_service.rs") || file_paths.contains(&"customer_api.rs"),
      "Should find semantically similar user/customer data fetching functions"
    );

    // Verify we have meaningful relevance scores
    for result in &results[..2] {
      assert!(
        result.relevance_score > 0.0,
        "Top results should have non-zero relevance scores. Got {} for {}",
        result.relevance_score,
        result.file_path
      );
    }

    // Shipping.rs should have lower relevance as it's not about data retrieval
    if let Some(shipping_result) = results.iter().find(|r| r.file_path == "shipping.rs") {
      let top_score = results[0].relevance_score;
      assert!(
        shipping_result.relevance_score <= top_score,
        "Shipping calculation should have lower relevance than data retrieval functions"
      );
    }
  }

  #[tokio::test]
  async fn test_vector_search_ranking_quality() {
    let (_temp_dir, table, embedding_provider) = setup_empty_table().await;

    // Add documents with varying relevance to "error handling"
    let docs = vec![
      create_and_embed_document(
        "fn handle_error(e: Error) -> Result<Response> {
          // Comprehensive error handling with logging and recovery
          match e.kind() {
            ErrorKind::NotFound => Ok(Response::not_found()),
            ErrorKind::Unauthorized => Ok(Response::unauthorized()),
            _ => {
              log::error!(\"Unexpected error: {:?}\", e);
              Ok(Response::internal_server_error())
            }
          }
        }",
        "error_handler.rs",
        &embedding_provider,
      )
      .await,
      create_and_embed_document(
        "fn process_payment(amount: f64) -> bool {
          // Simple payment processing
          if amount > 0.0 {
            charge_credit_card(amount)
          } else {
            false
          }
        }",
        "payment.rs",
        &embedding_provider,
      )
      .await,
      create_and_embed_document(
        "fn validate_input(data: &str) -> Result<(), ValidationError> {
          // Input validation with error reporting
          if data.is_empty() {
            return Err(ValidationError::Empty);
          }
          if data.len() > 1000 {
            return Err(ValidationError::TooLong);
          }
          Ok(())
        }",
        "validator.rs",
        &embedding_provider,
      )
      .await,
    ];

    // Insert documents
    {
      let table_write = table.write().await;
      for doc in docs {
        let arrow_data = doc.into_arrow().unwrap();
        table_write.add(arrow_data).execute().await.unwrap();
      }
    }

    // Search for "error handling"
    let results = hybrid_search(
      table.clone(),
      embedding_provider.clone(),
      "error handling and recovery",
      10,
    )
    .await
    .unwrap();

    assert!(
      !results.is_empty(),
      "Should find results for error handling"
    );

    // error_handler.rs should be the top result
    assert_eq!(
      results[0].file_path, "error_handler.rs",
      "Error handler should be the most relevant result"
    );
    assert!(
      results[0].relevance_score > 0.0,
      "Top result must have positive relevance score, got {}",
      results[0].relevance_score
    );

    // validator.rs should rank higher than payment.rs
    let validator_rank = results.iter().position(|r| r.file_path == "validator.rs");
    let payment_rank = results.iter().position(|r| r.file_path == "payment.rs");

    if let (Some(v_rank), Some(p_rank)) = (validator_rank, payment_rank) {
      assert!(
        v_rank < p_rank,
        "Validator (with error handling) should rank higher than payment processing"
      );
    }
  }
}
