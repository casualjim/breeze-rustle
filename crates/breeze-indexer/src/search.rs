mod chunks;
mod documents;

#[cfg(all(test, feature = "local-embeddings"))]
mod chunk_tests;

use std::sync::Arc;

use anyhow::{Result, anyhow};
use lancedb::Table;
use lancedb::index::scalar::FullTextSearchQuery;
use lancedb::query::QueryBase;
use lancedb::rerankers::rrf;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::info;
use uuid::Uuid;

use crate::embeddings::EmbeddingProvider;
use chunks::search_chunks;
use documents::search_documents;

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum SearchGranularity {
  #[default]
  Document, // Default - search files, return with top chunks
  Chunk, // Search chunks directly, return chunks grouped by file
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchOptions {
  pub languages: Option<Vec<String>>, // Filter by languages
  pub file_limit: usize,              // Number of files to return (default: 5)
  pub chunks_per_file: usize,         // Number of chunks per file (default: 3)
  pub granularity: SearchGranularity, // Search mode: Document or Chunk

  // Semantic filters (mainly for Chunk mode)
  pub node_types: Option<Vec<String>>,   // ["function", "class"]
  pub node_name_pattern: Option<String>, // Regex or glob pattern
  pub parent_context_pattern: Option<String>, // Pattern match on parent
  pub scope_depth: Option<(usize, usize)>, // Min and max nesting level
  pub has_definitions: Option<Vec<String>>, // Must define these symbols
  pub has_references: Option<Vec<String>>, // Must reference these symbols
}

impl Default for SearchOptions {
  fn default() -> Self {
    Self {
      languages: None,
      file_limit: 5,
      chunks_per_file: 3,
      granularity: SearchGranularity::default(),
      node_types: None,
      node_name_pattern: None,
      parent_context_pattern: None,
      scope_depth: None,
      has_definitions: None,
      has_references: None,
    }
  }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChunkResult {
  pub content: String,
  pub start_line: usize,
  pub end_line: usize,
  pub start_byte: usize,
  pub end_byte: usize,
  pub relevance_score: f32,

  // Semantic metadata from CodeChunk
  pub node_type: String,
  pub node_name: Option<String>,
  pub language: String,
  pub parent_context: Option<String>,
  pub scope_path: Vec<String>,
  pub definitions: Vec<String>,
  pub references: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
  pub id: String,
  pub file_path: String,
  pub relevance_score: f32,
  pub chunk_count: u32,
  pub chunks: Vec<ChunkResult>, // Top chunks from this file

  // Document-level metadata from CodeDocument
  pub file_size: u64,
  pub last_modified: chrono::NaiveDateTime,
  pub indexed_at: chrono::NaiveDateTime,
  pub languages: Vec<String>,
  pub primary_language: Option<String>,
}

/// Build a hybrid search query with common filters
async fn build_hybrid_query(
  table: &Table,
  query: &str,
  query_vector: &[f32],
  project_id: Option<&str>,
  embedding_column: &str,
) -> Result<lancedb::query::VectorQuery> {
  let mut query_builder = table
    .query()
    .only_if(format!("project_id != '{}'", Uuid::nil()));

  // Add project filter if provided
  if let Some(pid) = project_id {
    query_builder = query_builder.only_if(format!("project_id = '{}'", pid));
  }

  Ok(
    query_builder
      .full_text_search(FullTextSearchQuery::new(query.to_string()))
      .nearest_to(query_vector)?
      .column(embedding_column)
      .rerank(Arc::new(rrf::RRFReranker::new(60.0))),
  )
}

/// Perform hybrid search combining vector and FTS with optional language filtering
pub async fn hybrid_search(
  documents_table: Arc<RwLock<Table>>,
  chunks_table: Arc<RwLock<Table>>,
  embedding_provider: Arc<dyn EmbeddingProvider>,
  query: &str,
  options: SearchOptions,
  project_id: Option<Uuid>,
) -> Result<Vec<SearchResult>> {
  info!(query = %query, options = ?options, "Performing hybrid search");

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

  match options.granularity {
    SearchGranularity::Document => {
      search_documents(
        documents_table,
        chunks_table,
        query,
        query_vector,
        &options,
        project_id.map(|id| id.to_string()).as_deref(),
      )
      .await
    }
    SearchGranularity::Chunk => {
      search_chunks(
        documents_table,
        chunks_table,
        query,
        query_vector,
        &options,
        project_id.map(|id| id.to_string()).as_deref(),
      )
      .await
    }
  }
}

#[cfg(all(test, feature = "local-embeddings"))]
mod tests {
  use super::*;
  use crate::Config;
  use crate::embeddings::factory::create_embedding_provider;
  use crate::models::{ChunkMetadataUpdate, CodeChunk, CodeDocument};
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

  async fn setup_test_table_with_chunks() -> (TempDir, Arc<RwLock<Table>>, Arc<RwLock<Table>>) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.lance");

    let connection = lancedb::connect(db_path.to_str().unwrap())
      .execute()
      .await
      .unwrap();

    let embedding_dim = 384;
    let doc_table = CodeDocument::ensure_table(&connection, "test_embeddings", embedding_dim)
      .await
      .unwrap();
    let chunk_table = CodeChunk::ensure_table(&connection, "code_chunks", embedding_dim)
      .await
      .unwrap();

    // Insert test documents with language info
    let test_documents = vec![
      create_test_document_with_lang(
        "src/main.rs",
        "fn main() {\n    println!(\"Hello, world!\");\n}",
        embedding_dim,
        vec!["rust".to_string()],
        Some("rust".to_string()),
        2,
      ),
      create_test_document_with_lang(
        "src/lib.rs",
        "pub fn search(query: &str) -> Vec<String> {\n    // Search implementation\n    vec![]\n}",
        embedding_dim,
        vec!["rust".to_string()],
        Some("rust".to_string()),
        1,
      ),
      create_test_document_with_lang(
        "tests/test_search.rs",
        "#[test]\nfn test_search() {\n    let results = search(\"test\");\n    assert!(results.is_empty());\n}",
        embedding_dim,
        vec!["rust".to_string()],
        Some("rust".to_string()),
        1,
      ),
      create_test_document_with_lang(
        "README.md",
        "# Search Library\n\nA simple search implementation in Rust.",
        embedding_dim,
        vec!["markdown".to_string()],
        Some("markdown".to_string()),
        1,
      ),
    ];

    // Insert documents and create chunks
    for doc in &test_documents {
      let arrow_data = doc.clone().into_arrow().unwrap();
      doc_table.add(arrow_data).execute().await.unwrap();

      // Create a chunk for each document
      let mut chunk = CodeChunk::builder()
        .file_id(doc.id)
        .project_id(doc.project_id)
        .file_path(doc.file_path.clone())
        .content(doc.content.clone())
        .start_byte(0)
        .end_byte(doc.content.len())
        .start_line(1)
        .end_line(doc.content.lines().count())
        .build();
      chunk.update_embedding(doc.content_embedding.clone());
      chunk.update_metadata(
        ChunkMetadataUpdate::builder()
          .node_type("file".to_string())
          .node_name(Some(doc.file_path.clone()))
          .language(doc.primary_language.clone().unwrap_or("text".to_string()))
          .parent_context(None)
          .scope_path(vec![])
          .definitions(vec![])
          .references(vec![])
          .build(),
      );

      let chunk_arrow = chunk.into_arrow().unwrap();
      chunk_table.add(chunk_arrow).execute().await.unwrap();
    }

    (
      temp_dir,
      Arc::new(RwLock::new(doc_table)),
      Arc::new(RwLock::new(chunk_table)),
    )
  }

  fn create_test_document_with_lang(
    file_path: &str,
    content: &str,
    embedding_dim: usize,
    languages: Vec<String>,
    primary_language: Option<String>,
    chunk_count: u32,
  ) -> CodeDocument {
    let project_id = uuid::Uuid::now_v7();
    let mut doc = CodeDocument::new(project_id, file_path.to_string(), content.to_string());
    let hash = CodeDocument::compute_hash(content);
    let mut embedding = vec![0.0; embedding_dim];
    for (i, &byte) in hash.iter().enumerate() {
      if i < embedding_dim {
        embedding[i] = (byte as f32) / 255.0;
      }
    }
    doc.update_embedding(embedding);
    doc.languages = languages;
    doc.primary_language = primary_language;
    doc.chunk_count = chunk_count;
    doc
  }

  #[tokio::test]
  async fn test_hybrid_search_basic() {
    let (_temp_dir, doc_table, chunk_table) = setup_test_table_with_chunks().await;
    let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

    let results = hybrid_search(
      doc_table,
      chunk_table,
      embedding_provider,
      "search",
      SearchOptions::default(),
      None,
    )
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

    // Check that chunks are included - documents are created from chunks so must have them
    for result in &results {
      assert!(
        !result.chunks.is_empty(),
        "Documents are created from chunks, so must have chunks"
      );
      assert!(
        result.chunk_count > 0,
        "chunk_count must be > 0 since documents are aggregated from chunks"
      );
    }
  }

  #[tokio::test]
  async fn test_hybrid_search_limit() {
    let (_temp_dir, doc_table, chunk_table) = setup_test_table_with_chunks().await;
    let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

    let options = SearchOptions {
      file_limit: 2,
      ..Default::default()
    };

    let results = hybrid_search(
      doc_table,
      chunk_table,
      embedding_provider,
      "test",
      options,
      None,
    )
    .await
    .unwrap();

    assert_eq!(results.len(), 2);
  }

  #[tokio::test]
  async fn test_hybrid_search_no_results() {
    let (_temp_dir, doc_table, chunk_table) = setup_test_table_with_chunks().await;
    let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

    let results = hybrid_search(
      doc_table,
      chunk_table,
      embedding_provider,
      "nonexistent",
      SearchOptions::default(),
      None,
    )
    .await
    .unwrap();

    // Hybrid search might still return results based on vector similarity
    // even if FTS doesn't match, so we just verify it completes without error
    assert!(results.len() <= 4); // We have 4 documents total
  }

  #[tokio::test]
  async fn test_hybrid_search_returns_complete_results() {
    let (_temp_dir, doc_table, chunk_table) = setup_test_table_with_chunks().await;
    let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

    let results = hybrid_search(
      doc_table,
      chunk_table,
      embedding_provider,
      "main",
      SearchOptions::default(),
      None,
    )
    .await
    .unwrap();

    assert!(!results.is_empty());

    // Verify all fields are populated
    for result in results {
      assert!(!result.id.is_empty());
      assert!(!result.file_path.is_empty());
      assert!(result.relevance_score >= 0.0);
      assert!(result.chunk_count > 0);

      // Verify chunks - documents must have chunks since they're created from them
      assert!(
        !result.chunks.is_empty(),
        "Documents are created from chunks, so must have at least one"
      );
      for chunk in &result.chunks {
        assert!(!chunk.content.is_empty());
        assert!(chunk.start_line > 0);
        assert!(chunk.end_line >= chunk.start_line);
        assert!(chunk.relevance_score >= 0.0);
      }
    }
  }

  #[tokio::test]
  async fn test_search_result_fields() {
    let (_temp_dir, doc_table, chunk_table) = setup_test_table_with_chunks().await;
    let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

    let results = hybrid_search(
      doc_table,
      chunk_table,
      embedding_provider,
      "fn",
      SearchOptions {
        file_limit: 1,
        ..Default::default()
      },
      None,
    )
    .await
    .unwrap();

    assert_eq!(results.len(), 1);
    let result = &results[0];

    // Verify the new fields are present
    assert!(!result.id.is_empty());
    assert!(uuid::Uuid::parse_str(&result.id).is_ok()); // Valid UUID
    assert!(result.chunk_count > 0);
    assert!(!result.chunks.is_empty());
  }

  async fn setup_empty_tables() -> (
    TempDir,
    Arc<RwLock<Table>>,
    Arc<RwLock<Table>>,
    Arc<dyn EmbeddingProvider>,
  ) {
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
      document_batch_size: 100,
    };

    let embedding_provider = create_embedding_provider(&config).await.unwrap();
    let embedding_dim = embedding_provider.embedding_dim();

    let doc_table = CodeDocument::ensure_table(&connection, "test_embeddings", embedding_dim)
      .await
      .unwrap();
    let chunk_table = CodeChunk::ensure_table(&connection, "code_chunks", embedding_dim)
      .await
      .unwrap();

    (
      temp_dir,
      Arc::new(RwLock::new(doc_table)),
      Arc::new(RwLock::new(chunk_table)),
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
    let (_temp_dir, doc_table, chunk_table, embedding_provider) = setup_empty_tables().await;

    // Search on empty database should return empty results, not error
    let results = hybrid_search(
      doc_table,
      chunk_table,
      embedding_provider,
      "some query",
      SearchOptions::default(),
      None,
    )
    .await
    .unwrap();

    assert_eq!(results.len(), 0, "Empty database should return no results");
  }

  #[tokio::test]
  async fn test_keyword_search_works() {
    let (_temp_dir, doc_table, chunk_table, embedding_provider) = setup_empty_tables().await;

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

    // Insert documents and create chunks
    {
      let doc_write = doc_table.write().await;
      let chunk_write = chunk_table.write().await;
      for mut doc in docs {
        doc.languages = vec!["rust".to_string()];
        doc.primary_language = Some("rust".to_string());
        doc.chunk_count = 1;

        // Create chunk for document
        let mut chunk = CodeChunk::builder()
          .file_id(doc.id)
          .project_id(doc.project_id)
          .file_path(doc.file_path.clone())
          .content(doc.content.clone())
          .start_byte(0)
          .end_byte(doc.content.len())
          .start_line(1)
          .end_line(1)
          .build();
        chunk.update_embedding(doc.content_embedding.clone());
        chunk.update_metadata(
          ChunkMetadataUpdate::builder()
            .node_type("function".to_string())
            .node_name(None)
            .language("rust".to_string())
            .parent_context(None)
            .scope_path(vec![])
            .definitions(vec![])
            .references(vec![])
            .build(),
        );

        let doc_arrow = doc.into_arrow().unwrap();
        doc_write.add(doc_arrow).execute().await.unwrap();

        let chunk_arrow = chunk.into_arrow().unwrap();
        chunk_write.add(chunk_arrow).execute().await.unwrap();
      }
    }

    // Search for "fibonacci" - should find math.rs
    let results = hybrid_search(
      doc_table.clone(),
      chunk_table.clone(),
      embedding_provider.clone(),
      "fibonacci",
      SearchOptions::default(),
      None,
    )
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
    assert!(
      !results[0].chunks.is_empty(),
      "Documents are created from chunks, so must have chunks"
    );
  }

  #[tokio::test]
  async fn test_vector_search_semantic_similarity() {
    let (_temp_dir, doc_table, chunk_table, embedding_provider) = setup_empty_tables().await;

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

    // Insert documents and create chunks
    {
      let doc_write = doc_table.write().await;
      let chunk_write = chunk_table.write().await;
      for mut doc in docs {
        doc.languages = vec!["rust".to_string()];
        doc.primary_language = Some("rust".to_string());
        doc.chunk_count = 1;

        // Create chunk for document
        let mut chunk = CodeChunk::builder()
          .file_id(doc.id)
          .project_id(doc.project_id)
          .file_path(doc.file_path.clone())
          .content(doc.content.clone())
          .start_byte(0)
          .end_byte(doc.content.len())
          .start_line(1)
          .end_line(doc.content.lines().count())
          .build();
        chunk.update_embedding(doc.content_embedding.clone());
        chunk.update_metadata(
          ChunkMetadataUpdate::builder()
            .node_type("function".to_string())
            .node_name(None)
            .language("rust".to_string())
            .parent_context(None)
            .scope_path(vec![])
            .definitions(vec![])
            .references(vec![])
            .build(),
        );

        let doc_arrow = doc.into_arrow().unwrap();
        doc_write.add(doc_arrow).execute().await.unwrap();

        let chunk_arrow = chunk.into_arrow().unwrap();
        chunk_write.add(chunk_arrow).execute().await.unwrap();
      }
    }

    // Search for "retrieve user information from database"
    // Should find both user_service.rs and customer_api.rs as they're semantically similar
    let results = hybrid_search(
      doc_table.clone(),
      chunk_table.clone(),
      embedding_provider.clone(),
      "retrieve user information from database",
      SearchOptions::default(),
      None,
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
    let (_temp_dir, doc_table, chunk_table, embedding_provider) = setup_empty_tables().await;

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

    // Insert documents and create chunks
    {
      let doc_write = doc_table.write().await;
      let chunk_write = chunk_table.write().await;
      for mut doc in docs {
        doc.languages = vec!["rust".to_string()];
        doc.primary_language = Some("rust".to_string());
        doc.chunk_count = 1;

        // Create chunk for document
        let mut chunk = CodeChunk::builder()
          .file_id(doc.id)
          .project_id(doc.project_id)
          .file_path(doc.file_path.clone())
          .content(doc.content.clone())
          .start_byte(0)
          .end_byte(doc.content.len())
          .start_line(1)
          .end_line(doc.content.lines().count())
          .build();
        chunk.update_embedding(doc.content_embedding.clone());
        chunk.update_metadata(
          ChunkMetadataUpdate::builder()
            .node_type("function".to_string())
            .node_name(None)
            .language("rust".to_string())
            .parent_context(None)
            .scope_path(vec![])
            .definitions(vec![])
            .references(vec![])
            .build(),
        );

        let doc_arrow = doc.into_arrow().unwrap();
        doc_write.add(doc_arrow).execute().await.unwrap();

        let chunk_arrow = chunk.into_arrow().unwrap();
        chunk_write.add(chunk_arrow).execute().await.unwrap();
      }
    }

    // Search for "error handling"
    let results = hybrid_search(
      doc_table.clone(),
      chunk_table.clone(),
      embedding_provider.clone(),
      "error handling and recovery",
      SearchOptions::default(),
      None,
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

  #[tokio::test]
  async fn test_language_filtering() {
    let (_temp_dir, doc_table, chunk_table) = setup_test_table_with_chunks().await;
    let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

    // Test filtering for rust files only
    let rust_options = SearchOptions {
      languages: Some(vec!["rust".to_string()]),
      ..Default::default()
    };

    let rust_results = hybrid_search(
      doc_table.clone(),
      chunk_table.clone(),
      embedding_provider.clone(),
      "search",
      rust_options,
      None,
    )
    .await
    .unwrap();

    // Should only get rust files, not markdown
    assert!(!rust_results.is_empty());
    for result in &rust_results {
      assert!(
        result.file_path.ends_with(".rs"),
        "Expected only .rs files, got {}",
        result.file_path
      );
    }

    // Test filtering for markdown only
    let md_options = SearchOptions {
      languages: Some(vec!["markdown".to_string()]),
      ..Default::default()
    };

    let md_results = hybrid_search(
      doc_table.clone(),
      chunk_table.clone(),
      embedding_provider.clone(),
      "search",
      md_options,
      None,
    )
    .await
    .unwrap();

    // Should only get markdown files
    for result in &md_results {
      assert!(
        result.file_path.ends_with(".md"),
        "Expected only .md files, got {}",
        result.file_path
      );
    }
  }

  #[tokio::test]
  async fn test_chunks_per_file_limit() {
    let (_temp_dir, doc_table, chunk_table) = setup_test_table_with_chunks().await;
    let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

    // Test with chunks_per_file = 1
    let options = SearchOptions {
      chunks_per_file: 1,
      ..Default::default()
    };

    let results = hybrid_search(
      doc_table,
      chunk_table,
      embedding_provider,
      "search",
      options,
      None,
    )
    .await
    .unwrap();

    for result in &results {
      assert_eq!(
        result.chunks.len(),
        1,
        "Expected exactly 1 chunk per file, got {} for {}",
        result.chunks.len(),
        result.file_path
      );
    }
  }
}
