use blake3::Hasher;
#[cfg(test)]
use std::path::{Path, PathBuf};
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq)]
pub struct CodeDocument {
  pub id: String,
  pub file_path: String,
  pub content: String,
  pub content_hash: [u8; 32],
  pub content_embedding: Vec<f32>,
  pub file_size: u64,
  pub last_modified: chrono::NaiveDateTime,
  pub indexed_at: chrono::NaiveDateTime,
}

impl CodeDocument {
  pub fn new(file_path: String, content: String) -> Self {
    let id = Uuid::now_v7().to_string();
    let content_hash = Self::compute_hash(content.as_str());
    let file_size = content.len() as u64;
    let last_modified = chrono::Utc::now().naive_utc();
    let indexed_at = last_modified;

    Self {
      id,
      file_path,
      content,
      content_hash,
      content_embedding: Vec::new(),
      file_size,
      last_modified,
      indexed_at,
    }
  }

  /// Create the Arrow schema for CodeDocument with the given embedding dimensions
  pub fn schema(embedding_dim: usize) -> arrow::datatypes::Schema {
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};

    let fields = vec![
      Field::new("id", DataType::Utf8, false),
      Field::new("file_path", DataType::Utf8, false),
      Field::new("content", DataType::Utf8, false),
      Field::new("content_hash", DataType::FixedSizeBinary(32), false),
      Field::new(
        "content_embedding",
        DataType::FixedSizeList(
          std::sync::Arc::new(Field::new("item", DataType::Float32, true)),
          embedding_dim as i32,
        ),
        false,
      ),
      Field::new("file_size", DataType::UInt64, false),
      Field::new(
        "last_modified",
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        false,
      ),
      Field::new(
        "indexed_at",
        DataType::Timestamp(TimeUnit::Nanosecond, None),
        false,
      ),
    ];

    Schema::new(fields)
  }

  /// Create a LanceDB table for storing CodeDocuments
  /// LanceDB requires at least one batch to create a table, so we create a dummy row and delete it
  pub async fn create_table(
    connection: &lancedb::Connection,
    table_name: &str,
    embedding_dim: usize,
  ) -> lancedb::Result<lancedb::Table> {
    use arrow::array::*;
    use arrow::datatypes::{DataType, Field};
    use arrow::record_batch::{RecordBatch, RecordBatchIterator};

    let schema = std::sync::Arc::new(Self::schema(embedding_dim));

    // Create dummy data - LanceDB requires at least one batch
    let id_array = StringArray::from(vec!["00000000-0000-0000-0000-000000000000"]);
    let file_path_array = StringArray::from(vec!["__dummy__.txt"]);
    let content_array = StringArray::from(vec!["dummy"]);

    let mut content_hash_builder = FixedSizeBinaryBuilder::with_capacity(1, 32);
    content_hash_builder.append_value([0u8; 32]).unwrap();
    let content_hash_array = content_hash_builder.finish();

    // Create dummy embedding
    let embedding_array = Float32Array::from(vec![0.0; embedding_dim]);
    let embedding_field = std::sync::Arc::new(Field::new("item", DataType::Float32, true));
    let embedding_list = FixedSizeListArray::try_new(
      embedding_field,
      embedding_dim as i32,
      std::sync::Arc::new(embedding_array),
      None,
    )
    .map_err(|e| lancedb::Error::Arrow { source: e })?;

    let file_size_array = UInt64Array::from(vec![0u64]);
    let last_modified_array = TimestampNanosecondArray::from(vec![0i64]);
    let indexed_at_array = TimestampNanosecondArray::from(vec![0i64]);

    let batch = RecordBatch::try_new(
      schema.clone(),
      vec![
        std::sync::Arc::new(id_array),
        std::sync::Arc::new(file_path_array),
        std::sync::Arc::new(content_array),
        std::sync::Arc::new(content_hash_array),
        std::sync::Arc::new(embedding_list),
        std::sync::Arc::new(file_size_array),
        std::sync::Arc::new(last_modified_array),
        std::sync::Arc::new(indexed_at_array),
      ],
    )
    .map_err(|e| lancedb::Error::Arrow { source: e })?;

    let batch_iter = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);

    // Create table
    let table = connection
      .create_table(table_name, Box::new(batch_iter))
      .execute()
      .await?;

    // Delete the dummy row
    table
      .delete("id = '00000000-0000-0000-0000-000000000000'")
      .await?;

    Ok(table)
  }

  /// Ensure a table exists - open if it exists, create if it doesn't
  pub async fn ensure_table(
    connection: &lancedb::Connection,
    table_name: &str,
    embedding_dim: usize,
  ) -> lancedb::Result<lancedb::Table> {
    // Try to open the table first
    match connection.open_table(table_name).execute().await {
      Ok(table) => Ok(table),
      Err(e) => {
        // Check if it's a table not found error
        match &e {
          lancedb::Error::TableNotFound { .. } => {
            // Create the table
            Self::create_table(connection, table_name, embedding_dim).await
          }
          _ => Err(e), // Propagate other errors
        }
      }
    }
  }

  pub fn compute_hash(content: &str) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(content.as_bytes());
    let hash = hasher.finalize();
    let mut result = [0u8; 32];
    result.copy_from_slice(hash.as_bytes());
    result
  }

  pub fn update_content_hash(&mut self, hash: [u8; 32]) {
    self.content_hash = hash;
  }

  pub fn update_embedding(&mut self, embedding: Vec<f32>) {
    self.content_embedding = embedding;
    self.indexed_at = chrono::Utc::now().naive_utc();
  }

  /// Read file content and compute hash in a single pass
  #[cfg(test)]
  pub async fn from_file(file_path: impl Into<PathBuf>) -> std::io::Result<Self> {
    let path: PathBuf = file_path.into();
    let path_str = path.to_string_lossy().to_string();

    // Get file metadata
    let metadata = tokio::fs::metadata(&path).await?;
    let file_size = metadata.len();
    let last_modified = metadata
      .modified()
      .ok()
      .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
      .and_then(|d| {
        let secs = d.as_secs() as i64;
        let nanos = d.subsec_nanos();
        chrono::DateTime::from_timestamp(secs, nanos).map(|dt| dt.naive_utc())
      })
      .unwrap_or_else(|| chrono::Utc::now().naive_utc());

    // Read file and compute hash in one pass
    let (content, hash) = Self::read_and_hash(&path).await?;

    let id = Uuid::now_v7().to_string();
    let indexed_at = chrono::Utc::now().naive_utc();

    Ok(Self {
      id,
      file_path: path_str,
      content,
      content_hash: hash,
      content_embedding: Vec::new(),
      file_size,
      last_modified,
      indexed_at,
    })
  }

  /// Read file content and compute hash in a single pass
  #[cfg(test)]
  async fn read_and_hash(path: &Path) -> std::io::Result<(String, [u8; 32])> {
    use tokio::io::AsyncReadExt;

    let mut file = tokio::fs::File::open(path).await?;
    let mut content = Vec::new();
    let mut hasher = Hasher::new();

    // Read file in chunks and update hash
    let mut buffer = vec![0u8; 1024 * 1024]; // 1MB chunks
    loop {
      let n = file.read(&mut buffer).await?;
      if n == 0 {
        break;
      }
      hasher.update(&buffer[..n]);
      content.extend_from_slice(&buffer[..n]);
    }

    let hash = hasher.finalize();
    let mut hash_bytes = [0u8; 32];
    hash_bytes.copy_from_slice(hash.as_bytes());

    let content_string = String::from_utf8_lossy(&content).to_string();

    Ok((content_string, hash_bytes))
  }
}

// Implement IntoArrow for CodeDocument to enable LanceDB persistence
impl lancedb::arrow::IntoArrow for CodeDocument {
  fn into_arrow(self) -> lancedb::Result<Box<dyn arrow::array::RecordBatchReader + Send>> {
    use arrow::array::*;
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    let embedding_dim = self.content_embedding.len();
    let schema = Arc::new(CodeDocument::schema(embedding_dim));

    // Build arrays for single document
    let id_array = StringArray::from(vec![self.id.as_str()]);
    let file_path_array = StringArray::from(vec![self.file_path.as_str()]);
    let content_array = StringArray::from(vec![self.content.as_str()]);

    let content_hash_builder = FixedSizeBinaryBuilder::with_capacity(1, 32);
    let mut content_hash_array = content_hash_builder;
    content_hash_array.append_value(self.content_hash).unwrap();
    let content_hash_array = content_hash_array.finish();

    // Create embedding array
    let embedding_array = Float32Array::from(self.content_embedding);
    let embedding_field = Arc::new(arrow::datatypes::Field::new(
      "item",
      arrow::datatypes::DataType::Float32,
      true,
    ));
    let embedding_list = FixedSizeListArray::try_new(
      embedding_field,
      embedding_dim as i32,
      Arc::new(embedding_array),
      None,
    )
    .map_err(|e| lancedb::Error::Arrow { source: e })?;

    let file_size_array = UInt64Array::from(vec![self.file_size]);

    // Convert timestamps
    let last_modified_ns = self
      .last_modified
      .and_utc()
      .timestamp_nanos_opt()
      .unwrap_or(0);
    let indexed_at_ns = self.indexed_at.and_utc().timestamp_nanos_opt().unwrap_or(0);
    let last_modified_array = TimestampNanosecondArray::from(vec![last_modified_ns]);
    let indexed_at_array = TimestampNanosecondArray::from(vec![indexed_at_ns]);

    // Create the record batch
    let batch = RecordBatch::try_new(
      schema.clone(),
      vec![
        Arc::new(id_array),
        Arc::new(file_path_array),
        Arc::new(content_array),
        Arc::new(content_hash_array),
        Arc::new(embedding_list),
        Arc::new(file_size_array),
        Arc::new(last_modified_array),
        Arc::new(indexed_at_array),
      ],
    )
    .map_err(|e| lancedb::Error::Arrow { source: e })?;

    // Return as RecordBatchReader
    Ok(Box::new(arrow::record_batch::RecordBatchIterator::new(
      vec![Ok(batch)].into_iter(),
      schema,
    )))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use std::fs;
  use tempfile::TempDir;

  #[test]
  fn test_compute_hash() {
    let content = "Hello, world!";
    let hash1 = CodeDocument::compute_hash(content);
    let hash2 = CodeDocument::compute_hash(content);

    // Same content should produce same hash
    assert_eq!(hash1, hash2);

    // Different content should produce different hash
    let hash3 = CodeDocument::compute_hash("Different content");
    assert_ne!(hash1, hash3);
  }

  #[tokio::test]
  async fn test_from_file() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test.py");
    let content = "def hello():\n    print('Hello, world!')";
    fs::write(&file_path, content).unwrap();

    let doc = CodeDocument::from_file(&file_path).await.unwrap();

    assert_eq!(doc.file_path, file_path.to_string_lossy());
    assert_eq!(doc.content, content);
    assert_eq!(doc.file_size, content.len() as u64);
    assert_eq!(doc.content_hash, CodeDocument::compute_hash(content));
    assert!(doc.content_embedding.is_empty());
    assert!(!doc.id.is_empty());
  }

  #[tokio::test]
  async fn test_from_file_not_found() {
    let result = CodeDocument::from_file("/non/existent/file.txt").await;
    assert!(result.is_err());
  }

  #[tokio::test]
  async fn test_read_and_hash_consistency() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("consistency.txt");
    let content = "Test content for consistency check";
    fs::write(&file_path, content).unwrap();

    // Read and hash through from_file
    let doc = CodeDocument::from_file(&file_path).await.unwrap();

    // Compute hash separately
    let expected_hash = CodeDocument::compute_hash(content);

    assert_eq!(doc.content_hash, expected_hash);
  }

  #[tokio::test]
  async fn test_large_file_handling() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("large.txt");

    // Create a 2MB file
    let chunk = "a".repeat(1024); // 1KB
    let content = chunk.repeat(2048); // 2MB
    fs::write(&file_path, &content).unwrap();

    let doc = CodeDocument::from_file(&file_path).await.unwrap();

    assert_eq!(doc.content, content);
    assert_eq!(doc.file_size, content.len() as u64);
  }

  #[tokio::test]
  async fn test_utf8_handling() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("utf8.txt");
    let content = "Hello 世界! 🌍 Здравствуй мир!";
    fs::write(&file_path, content).unwrap();

    let doc = CodeDocument::from_file(&file_path).await.unwrap();

    assert_eq!(doc.content, content);
  }

  #[test]
  fn test_update_embedding() {
    let mut doc = CodeDocument::new("test.py".to_string(), "content".to_string());

    let initial_indexed_at = doc.indexed_at;

    // Small delay to ensure time difference
    std::thread::sleep(std::time::Duration::from_millis(10));

    let embedding = vec![1.0, 2.0, 3.0];
    doc.update_embedding(embedding.clone());

    assert_eq!(doc.content_embedding, embedding);
    assert!(doc.indexed_at > initial_indexed_at);
  }

  #[test]
  fn test_new_document() {
    let file_path = "test.py".to_string();
    let content = "def main(): pass".to_string();

    let doc = CodeDocument::new(file_path.clone(), content.clone());

    assert_eq!(doc.file_path, file_path);
    assert_eq!(doc.content, content);
    assert_eq!(doc.file_size, content.len() as u64);
    assert_eq!(doc.content_hash, CodeDocument::compute_hash(&content));
    assert!(doc.content_embedding.is_empty());
    assert!(!doc.id.is_empty());

    // Verify it's a valid UUID
    assert!(uuid::Uuid::parse_str(&doc.id).is_ok());
  }
}
