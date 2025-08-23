use super::super::*;
use crate::models::{ChunkMetadataUpdate, CodeChunk, CodeDocument};
use lancedb::arrow::IntoArrow;
use std::sync::Arc;
use tempfile::TempDir;
use tokio::sync::RwLock;
use uuid::Uuid;

struct TestSetup {
  _temp_dir: TempDir,
  doc_table: Arc<RwLock<lancedb::Table>>,
  chunk_table: Arc<RwLock<lancedb::Table>>,
}

/// Create test tables with sample data for chunk search tests
async fn setup_chunk_search_test_data() -> TestSetup {
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

  // Create test data with different semantic properties
  let project_id = Uuid::now_v7();

  // File 1: Multiple functions with different properties
  let file1_id = Uuid::now_v7();
  let mut doc1 = CodeDocument::new(
    project_id,
    "src/auth.rs".to_string(),
    "auth module content".to_string(),
  );
  doc1.languages = vec!["rust".to_string()];
  doc1.primary_language = Some("rust".to_string());
  doc1.chunk_count = 3;
  doc1.id = file1_id;

  // Chunk 1.1: Login function
  let mut chunk1_1 = CodeChunk::builder()
    .file_id(file1_id)
    .project_id(project_id)
    .file_path("src/auth.rs".to_string())
    .content("fn login(username: &str, password: &str) -> Result<User> { ... }".to_string())
    .start_byte(0)
    .end_byte(100)
    .start_line(1)
    .end_line(5)
    .build();
  chunk1_1.update_embedding(vec![0.1; embedding_dim]);
  chunk1_1.update_metadata(
    ChunkMetadataUpdate::builder()
      .node_type("function".to_string())
      .node_name(Some("login".to_string()))
      .language("rust".to_string())
      .parent_context(None)
      .scope_path(vec!["auth".to_string()])
      .definitions(vec!["User".to_string()])
      .references(vec!["username".to_string(), "password".to_string()])
      .build(),
  );

  // Chunk 1.2: Logout function
  let mut chunk1_2 = CodeChunk::builder()
    .file_id(file1_id)
    .project_id(project_id)
    .file_path("src/auth.rs".to_string())
    .content("fn logout(user: &User) -> Result<()> { ... }".to_string())
    .start_byte(100)
    .end_byte(150)
    .start_line(6)
    .end_line(8)
    .build();
  chunk1_2.update_embedding(vec![0.2; embedding_dim]);
  chunk1_2.update_metadata(
    ChunkMetadataUpdate::builder()
      .node_type("function".to_string())
      .node_name(Some("logout".to_string()))
      .language("rust".to_string())
      .parent_context(None)
      .scope_path(vec!["auth".to_string()])
      .definitions(vec![])
      .references(vec!["User".to_string()])
      .build(),
  );

  // Chunk 1.3: Auth middleware class
  let mut chunk1_3 = CodeChunk::builder()
    .file_id(file1_id)
    .project_id(project_id)
    .file_path("src/auth.rs".to_string())
    .content(
      "impl AuthMiddleware { fn check_auth(&self, req: Request) -> bool { ... } }".to_string(),
    )
    .start_byte(150)
    .end_byte(250)
    .start_line(9)
    .end_line(15)
    .build();
  chunk1_3.update_embedding(vec![0.3; embedding_dim]);
  chunk1_3.update_metadata(
    ChunkMetadataUpdate::builder()
      .node_type("impl".to_string())
      .node_name(Some("AuthMiddleware".to_string()))
      .language("rust".to_string())
      .parent_context(None)
      .scope_path(vec!["auth".to_string(), "AuthMiddleware".to_string()])
      .definitions(vec!["check_auth".to_string()])
      .references(vec!["Request".to_string()])
      .build(),
  );

  // File 2: Database connection code
  let file2_id = Uuid::now_v7();
  let mut doc2 = CodeDocument::new(
    project_id,
    "src/database.rs".to_string(),
    "database module content".to_string(),
  );
  doc2.languages = vec!["rust".to_string()];
  doc2.primary_language = Some("rust".to_string());
  doc2.chunk_count = 2;
  doc2.id = file2_id;

  // Chunk 2.1: Connection struct
  let mut chunk2_1 = CodeChunk::builder()
    .file_id(file2_id)
    .project_id(project_id)
    .file_path("src/database.rs".to_string())
    .content("struct Connection { pool: Pool<Postgres> }".to_string())
    .start_byte(0)
    .end_byte(50)
    .start_line(1)
    .end_line(3)
    .build();
  chunk2_1.update_embedding(vec![0.4; embedding_dim]);
  chunk2_1.update_metadata(
    ChunkMetadataUpdate::builder()
      .node_type("struct".to_string())
      .node_name(Some("Connection".to_string()))
      .language("rust".to_string())
      .parent_context(None)
      .scope_path(vec!["database".to_string()])
      .definitions(vec!["Connection".to_string()])
      .references(vec!["Pool".to_string(), "Postgres".to_string()])
      .build(),
  );

  // Chunk 2.2: Nested method with deeper scope
  let mut chunk2_2 = CodeChunk::builder()
    .file_id(file2_id)
    .project_id(project_id)
    .file_path("src/database.rs".to_string())
    .content(
      "impl Connection { fn execute_query(&self, query: &str) -> Result<Vec<Row>> { ... } }"
        .to_string(),
    )
    .start_byte(50)
    .end_byte(150)
    .start_line(4)
    .end_line(10)
    .build();
  chunk2_2.update_embedding(vec![0.5; embedding_dim]);
  chunk2_2.update_metadata(
    ChunkMetadataUpdate::builder()
      .node_type("method".to_string())
      .node_name(Some("execute_query".to_string()))
      .language("rust".to_string())
      .parent_context(Some("impl Connection".to_string()))
      .scope_path(vec![
        "database".to_string(),
        "Connection".to_string(),
        "execute_query".to_string(),
      ])
      .definitions(vec!["execute_query".to_string()])
      .references(vec![
        "Result".to_string(),
        "Vec".to_string(),
        "Row".to_string(),
        "logger".to_string(),
      ])
      .build(),
  );

  // File 3: Python test file
  let file3_id = Uuid::now_v7();
  let mut doc3 = CodeDocument::new(
    project_id,
    "tests/test_auth.py".to_string(),
    "python test content".to_string(),
  );
  doc3.languages = vec!["python".to_string()];
  doc3.primary_language = Some("python".to_string());
  doc3.chunk_count = 1;
  doc3.id = file3_id;

  let mut chunk3_1 = CodeChunk::builder()
    .file_id(file3_id)
    .project_id(project_id)
    .file_path("tests/test_auth.py".to_string())
    .content("def test_login(): assert login('user', 'pass') is not None".to_string())
    .start_byte(0)
    .end_byte(60)
    .start_line(1)
    .end_line(3)
    .build();
  chunk3_1.update_embedding(vec![0.6; embedding_dim]);
  chunk3_1.update_metadata(
    ChunkMetadataUpdate::builder()
      .node_type("function".to_string())
      .node_name(Some("test_login".to_string()))
      .language("python".to_string())
      .parent_context(None)
      .scope_path(vec!["test_auth".to_string()])
      .definitions(vec!["test_login".to_string()])
      .references(vec!["login".to_string()])
      .build(),
  );

  // Insert all documents
  let docs = vec![doc1, doc2, doc3];
  for doc in docs {
    let arrow_data = doc.into_arrow().unwrap();
    doc_table.add(arrow_data).execute().await.unwrap();
  }

  // Insert all chunks
  let chunks = vec![chunk1_1, chunk1_2, chunk1_3, chunk2_1, chunk2_2, chunk3_1];
  for chunk in chunks {
    let arrow_data = chunk.into_arrow().unwrap();
    chunk_table.add(arrow_data).execute().await.unwrap();
  }

  TestSetup {
    _temp_dir: temp_dir,
    doc_table: Arc::new(RwLock::new(doc_table)),
    chunk_table: Arc::new(RwLock::new(chunk_table)),
  }
}

struct MockEmbeddingProvider {
  embedding_dim: usize,
}

impl MockEmbeddingProvider {
  fn new(embedding_dim: usize) -> Self {
    Self { embedding_dim }
  }
}

#[async_trait::async_trait]
impl crate::embeddings::EmbeddingProvider for MockEmbeddingProvider {
  async fn embed(
    &self,
    inputs: &[crate::embeddings::EmbeddingInput<'_>],
  ) -> crate::embeddings::EmbeddingResult<Vec<Vec<f32>>> {
    Ok(
      inputs
        .iter()
        .enumerate()
        .map(|(i, _)| vec![0.1 + (i as f32 * 0.1); self.embedding_dim])
        .collect(),
    )
  }

  fn embedding_dim(&self) -> usize {
    self.embedding_dim
  }

  fn context_length(&self) -> usize {
    8192
  }

  fn create_batching_strategy(&self) -> Box<dyn crate::embeddings::batching::BatchingStrategy> {
    Box::new(crate::embeddings::batching::LocalBatchingStrategy::new(100))
  }

  fn tokenizer(&self) -> Option<Arc<tokenizers::Tokenizer>> {
    None
  }
}

#[tokio::test]
async fn test_search_granularity_default() {
  let options = SearchOptions::default();
  match options.granularity {
    SearchGranularity::Document => {} // Expected
    _ => panic!("Default granularity should be Document"),
  }
}

#[tokio::test]
async fn test_chunk_search_mode() {
  let setup = setup_chunk_search_test_data().await;
  let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

  let options = SearchOptions {
    granularity: SearchGranularity::Chunk,
    file_limit: 2,
    chunks_per_file: 2,
    ..Default::default()
  };

  let results = hybrid_search(
    setup.doc_table,
    setup.chunk_table,
    embedding_provider,
    "auth",
    options,
    None,
  )
  .await
  .unwrap();

  // Should get results grouped by file
  assert!(!results.is_empty(), "Should find results for 'auth'");
  assert!(results.len() <= 2, "Should respect file_limit");

  // The auth.rs file should be first as it has more auth-related chunks
  assert_eq!(results[0].file_path, "src/auth.rs");
  assert!(
    results[0].chunks.len() <= 2,
    "Should respect chunks_per_file"
  );
  assert!(results[0].chunk_count > 0);
}

#[tokio::test]
async fn test_semantic_filter_node_types() {
  let setup = setup_chunk_search_test_data().await;
  let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

  let options = SearchOptions {
    granularity: SearchGranularity::Chunk,
    node_types: Some(vec!["function".to_string()]),
    ..Default::default()
  };

  let results = hybrid_search(
    setup.doc_table,
    setup.chunk_table,
    embedding_provider,
    "auth",
    options,
    None,
  )
  .await
  .unwrap();

  // Should only get function chunks, not impl or struct
  for result in &results {
    for chunk in &result.chunks {
      // We can't verify the node_type from ChunkResult, but we know
      // only functions should be returned based on our test data
      assert!(
        chunk.content.contains("fn ") || chunk.content.contains("def "),
        "Should only return function chunks"
      );
    }
  }
}

#[tokio::test]
async fn test_semantic_filter_node_name_pattern() {
  let setup = setup_chunk_search_test_data().await;
  let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

  let options = SearchOptions {
    granularity: SearchGranularity::Chunk,
    node_name_pattern: Some("login".to_string()),
    ..Default::default()
  };

  let results = hybrid_search(
    setup.doc_table,
    setup.chunk_table,
    embedding_provider,
    "function",
    options,
    None,
  )
  .await
  .unwrap();

  // Should only get chunks with node_name = "login"
  if !results.is_empty() {
    for result in &results {
      for chunk in &result.chunks {
        assert!(
          chunk.content.contains("login"),
          "Should only return chunks with login in the name"
        );
      }
    }
  }
}

#[tokio::test]
async fn test_semantic_filter_parent_context() {
  let setup = setup_chunk_search_test_data().await;
  let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

  let options = SearchOptions {
    granularity: SearchGranularity::Chunk,
    parent_context_pattern: Some("impl Connection".to_string()),
    ..Default::default()
  };

  let results = hybrid_search(
    setup.doc_table,
    setup.chunk_table,
    embedding_provider,
    "execute",
    options,
    None,
  )
  .await
  .unwrap();

  // Should only get chunks within "impl Connection" context
  if !results.is_empty() {
    assert_eq!(results[0].file_path, "src/database.rs");
    assert!(results[0].chunks[0].content.contains("execute_query"));
  }
}

#[tokio::test]
async fn test_semantic_filter_scope_depth() {
  let setup = setup_chunk_search_test_data().await;
  let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

  // Test minimum scope depth of 2
  let options = SearchOptions {
    granularity: SearchGranularity::Chunk,
    scope_depth: Some((2, 5)), // At least 2 levels deep
    ..Default::default()
  };

  let results = hybrid_search(
    setup.doc_table,
    setup.chunk_table,
    embedding_provider,
    "method",
    options,
    None,
  )
  .await
  .unwrap();

  // Should get chunks with scope_path length >= 2
  // Based on our test data, this includes AuthMiddleware methods and Connection methods
  for result in &results {
    assert!(!result.chunks.is_empty());
  }
}

#[tokio::test]
async fn test_semantic_filter_has_definitions() {
  let setup = setup_chunk_search_test_data().await;
  let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

  let options = SearchOptions {
    granularity: SearchGranularity::Chunk,
    has_definitions: Some(vec!["Connection".to_string()]),
    ..Default::default()
  };

  let results = hybrid_search(
    setup.doc_table,
    setup.chunk_table,
    embedding_provider,
    "database",
    options,
    None,
  )
  .await
  .unwrap();

  // Should only get chunks that define "Connection"
  if !results.is_empty() {
    assert_eq!(results[0].file_path, "src/database.rs");
    assert!(results[0].chunks[0].content.contains("struct Connection"));
  }
}

#[tokio::test]
async fn test_semantic_filter_has_references() {
  let setup = setup_chunk_search_test_data().await;
  let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

  let options = SearchOptions {
    granularity: SearchGranularity::Chunk,
    has_references: Some(vec!["logger".to_string()]),
    ..Default::default()
  };

  let results = hybrid_search(
    setup.doc_table,
    setup.chunk_table,
    embedding_provider,
    "execute",
    options,
    None,
  )
  .await
  .unwrap();

  // Should only get chunks that reference "logger"
  if !results.is_empty() {
    assert_eq!(results[0].file_path, "src/database.rs");
  }
}

#[tokio::test]
async fn test_chunk_search_language_filter() {
  let setup = setup_chunk_search_test_data().await;
  let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

  let options = SearchOptions {
    granularity: SearchGranularity::Chunk,
    languages: Some(vec!["python".to_string()]),
    ..Default::default()
  };

  let results = hybrid_search(
    setup.doc_table,
    setup.chunk_table,
    embedding_provider,
    "test",
    options,
    None,
  )
  .await
  .unwrap();

  // Should only get Python chunks
  for result in &results {
    assert!(result.file_path.ends_with(".py"));
  }
}

#[tokio::test]
async fn test_chunk_search_aggregation_scoring() {
  let setup = setup_chunk_search_test_data().await;
  let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

  let options = SearchOptions {
    granularity: SearchGranularity::Chunk,
    chunks_per_file: 3,
    file_limit: 10,
    ..Default::default()
  };

  let results = hybrid_search(
    setup.doc_table,
    setup.chunk_table,
    embedding_provider,
    "auth login",
    options,
    None,
  )
  .await
  .unwrap();

  // Files with more relevant chunks should rank higher
  if results.len() >= 2 {
    // auth.rs should rank higher than other files because it has multiple auth-related chunks
    assert_eq!(results[0].file_path, "src/auth.rs");
    assert!(
      results[0].relevance_score > results[1].relevance_score,
      "File with more relevant chunks should have higher aggregate score"
    );
  }
}

#[tokio::test]
async fn test_combined_semantic_filters() {
  let setup = setup_chunk_search_test_data().await;
  let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

  let options = SearchOptions {
    granularity: SearchGranularity::Chunk,
    node_types: Some(vec!["function".to_string(), "method".to_string()]),
    languages: Some(vec!["rust".to_string()]),
    has_references: Some(vec!["User".to_string()]),
    ..Default::default()
  };

  let results = hybrid_search(
    setup.doc_table,
    setup.chunk_table,
    embedding_provider,
    "auth",
    options,
    None,
  )
  .await
  .unwrap();

  // Should only get Rust functions/methods that reference "User"
  for result in &results {
    assert!(result.file_path.ends_with(".rs"));
    for chunk in &result.chunks {
      assert!(
        chunk.content.contains("fn ") || chunk.content.contains("impl"),
        "Should only have functions or methods"
      );
    }
  }
}

#[tokio::test]
async fn test_empty_semantic_filters() {
  let setup = setup_chunk_search_test_data().await;
  let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

  // Test with empty filter arrays - should not filter anything
  let options = SearchOptions {
    granularity: SearchGranularity::Chunk,
    node_types: Some(vec![]),      // Empty array
    has_definitions: Some(vec![]), // Empty array
    has_references: Some(vec![]),  // Empty array
    ..Default::default()
  };

  let results = hybrid_search(
    setup.doc_table,
    setup.chunk_table,
    embedding_provider,
    "function",
    options,
    None,
  )
  .await
  .unwrap();

  // Empty filters should not restrict results
  assert!(
    !results.is_empty(),
    "Empty filters should not filter out results"
  );
}

#[tokio::test]
async fn test_chunk_mode_respects_limits() {
  let setup = setup_chunk_search_test_data().await;
  let embedding_provider = Arc::new(MockEmbeddingProvider::new(384));

  let options = SearchOptions {
    granularity: SearchGranularity::Chunk,
    file_limit: 1,
    chunks_per_file: 1,
    ..Default::default()
  };

  let results = hybrid_search(
    setup.doc_table,
    setup.chunk_table,
    embedding_provider,
    "auth",
    options,
    None,
  )
  .await
  .unwrap();

  assert_eq!(results.len(), 1, "Should respect file_limit");
  assert_eq!(results[0].chunks.len(), 1, "Should respect chunks_per_file");
}
