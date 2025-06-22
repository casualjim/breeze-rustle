use blake3::Hasher;
use lancedb::index::Index;
use lancedb::index::vector::IvfFlatIndexBuilder;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use tracing::debug;
use uuid::Uuid;

/// Reserved ID for the dummy document used to initialize LanceDB tables
/// This document must never be deleted to preserve indices
pub const DUMMY_DOCUMENT_ID: &str = "00000000-0000-0000-0000-000000000000";

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
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
      ),
      Field::new(
        "indexed_at",
        DataType::Timestamp(TimeUnit::Microsecond, None),
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
    let id_array = StringArray::from(vec![DUMMY_DOCUMENT_ID]);
    let file_path_array = StringArray::from(vec!["__lancedb_dummy__.txt"]);
    let content_array = StringArray::from(vec![
      "LanceDB requires at least one document to create a table with proper schema",
    ]);

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
    let last_modified_array = arrow::array::TimestampMicrosecondArray::from(vec![0i64]);
    let indexed_at_array = arrow::array::TimestampMicrosecondArray::from(vec![0i64]);

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

    // Create indices on new table (while dummy row exists)
    Self::ensure_indices(&table).await?;

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
      Ok(table) => {
        // allow for extra indices to be created if they don't exist
        Self::ensure_indices(&table).await?;
        Ok(table)
      }
      Err(e) => {
        // Check if it's a table not found error
        match &e {
          lancedb::Error::TableNotFound { .. } => {
            // Create the table
            let table = Self::create_table(connection, table_name, embedding_dim).await?;

            Ok(table)
          }
          _ => Err(e), // Propagate other errors
        }
      }
    }
  }

  /// Ensure indices exist on content, file_path, and content_hash fields
  async fn ensure_indices(table: &lancedb::Table) -> lancedb::Result<()> {
    table
      .create_index(
        &["content_embedding"],
        Index::IvfFlat(IvfFlatIndexBuilder::default()),
      )
      .execute()
      .await?;

    // Create FTS index on content field for full-text search
    table
      .create_index(&["content"], Index::FTS(Default::default()))
      .execute()
      .await?;
    debug!("Created FTS index on content field");

    // Create Auto index on file_path for exact match lookups
    table
      .create_index(&["file_path"], Index::Auto)
      .execute()
      .await?;
    debug!("Created index on file_path field");

    // Create Auto index on content_hash for hash-based lookups
    table
      .create_index(&["content_hash"], Index::Auto)
      .execute()
      .await?;
    debug!("Created index on content_hash field");

    Ok(())
  }

  /// Convert a RecordBatch row to CodeDocument
  pub fn from_record_batch(
    batch: &arrow::record_batch::RecordBatch,
    row: usize,
  ) -> Result<Self, lancedb::Error> {
    use arrow::array::*;

    if row >= batch.num_rows() {
      return Err(lancedb::Error::Runtime {
        message: format!(
          "Row index {} out of bounds (batch has {} rows)",
          row,
          batch.num_rows()
        ),
      });
    }

    let id = batch
      .column_by_name("id")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid id column".to_string(),
      })?
      .value(row)
      .to_string();

    let file_path = batch
      .column_by_name("file_path")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid file_path column".to_string(),
      })?
      .value(row)
      .to_string();

    let content = batch
      .column_by_name("content")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid content column".to_string(),
      })?
      .value(row)
      .to_string();

    let content_hash_arr = batch
      .column_by_name("content_hash")
      .and_then(|col| col.as_any().downcast_ref::<FixedSizeBinaryArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid content_hash column".to_string(),
      })?;

    let mut content_hash = [0u8; 32];
    content_hash.copy_from_slice(content_hash_arr.value(row));

    // Extract embedding
    let embedding_list = batch
      .column_by_name("content_embedding")
      .and_then(|col| col.as_any().downcast_ref::<FixedSizeListArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid content_embedding column".to_string(),
      })?;

    let embedding_value = embedding_list.value(row);
    let embedding_values = embedding_value
      .as_any()
      .downcast_ref::<Float32Array>()
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Invalid embedding array type".to_string(),
      })?;

    let content_embedding: Vec<f32> = (0..embedding_values.len())
      .map(|i| embedding_values.value(i))
      .collect();

    let file_size = batch
      .column_by_name("file_size")
      .and_then(|col| col.as_any().downcast_ref::<UInt64Array>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid file_size column".to_string(),
      })?
      .value(row);

    let last_modified = batch
      .column_by_name("last_modified")
      .and_then(|col| {
        col
          .as_any()
          .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
      })
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid last_modified column".to_string(),
      })?
      .value(row);

    let indexed_at = batch
      .column_by_name("indexed_at")
      .and_then(|col| {
        col
          .as_any()
          .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
      })
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid indexed_at column".to_string(),
      })?
      .value(row);

    Ok(Self {
      id,
      file_path,
      content,
      content_hash,
      content_embedding,
      file_size,
      last_modified: chrono::DateTime::from_timestamp_micros(last_modified)
        .unwrap_or_default()
        .naive_utc(),
      indexed_at: chrono::DateTime::from_timestamp_micros(indexed_at)
        .unwrap_or_default()
        .naive_utc(),
    })
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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FileOperation {
  Add,
  Update,
  Delete,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct FileChange {
  pub path: PathBuf,
  pub operation: FileOperation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TaskType {
  FullIndex,
  PartialUpdate { changes: BTreeSet<FileChange> },
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct IndexTask {
  pub id: Uuid,
  pub project_id: Uuid,
  pub path: String,
  pub task_type: TaskType,
  pub status: TaskStatus,
  pub created_at: chrono::NaiveDateTime,
  pub started_at: Option<chrono::NaiveDateTime>,
  pub completed_at: Option<chrono::NaiveDateTime>,
  pub error: Option<String>,
  pub files_indexed: Option<usize>,
  pub merged_into: Option<Uuid>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum TaskStatus {
  Pending,
  Running,
  Completed,
  Failed,
  Merged,
}

impl std::fmt::Display for TaskStatus {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      TaskStatus::Pending => write!(f, "pending"),
      TaskStatus::Running => write!(f, "running"),
      TaskStatus::Completed => write!(f, "completed"),
      TaskStatus::Failed => write!(f, "failed"),
      TaskStatus::Merged => write!(f, "merged"),
    }
  }
}

impl std::str::FromStr for TaskStatus {
  type Err = String;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    match s {
      "pending" => Ok(TaskStatus::Pending),
      "running" => Ok(TaskStatus::Running),
      "completed" => Ok(TaskStatus::Completed),
      "failed" => Ok(TaskStatus::Failed),
      "merged" => Ok(TaskStatus::Merged),
      _ => Err(format!("Invalid task status: {}", s)),
    }
  }
}

impl IndexTask {
  pub fn new(project_id: Uuid, path: &Path) -> Self {
    Self {
      id: Uuid::now_v7(),
      project_id,
      path: path.to_path_buf().to_string_lossy().to_string(),
      task_type: TaskType::FullIndex,
      status: TaskStatus::Pending,
      created_at: chrono::Utc::now().naive_utc(),
      started_at: None,
      completed_at: None,
      error: None,
      files_indexed: None,
      merged_into: None,
    }
  }

  pub fn new_partial(project_id: Uuid, path: &Path, changes: BTreeSet<FileChange>) -> Self {
    Self {
      id: Uuid::now_v7(),
      project_id,
      path: path.to_path_buf().to_string_lossy().to_string(),
      task_type: TaskType::PartialUpdate { changes },
      status: TaskStatus::Pending,
      created_at: chrono::Utc::now().naive_utc(),
      started_at: None,
      completed_at: None,
      error: None,
      files_indexed: None,
      merged_into: None,
    }
  }

  /// Create the Arrow schema for IndexTask
  pub fn schema() -> arrow::datatypes::Schema {
    Self::arrow_schema()
  }

  /// Create the Arrow schema for IndexTask
  pub fn arrow_schema() -> arrow::datatypes::Schema {
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};

    let fields = vec![
      Field::new("id", DataType::Utf8, false),
      Field::new("project_id", DataType::Utf8, false), // UUID as string
      Field::new("path", DataType::Utf8, false),
      Field::new("task_type", DataType::Utf8, false), // JSON serialized
      Field::new("status", DataType::Utf8, false),
      Field::new(
        "created_at",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
      ),
      Field::new(
        "started_at",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        true,
      ),
      Field::new(
        "completed_at",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        true,
      ),
      Field::new("error", DataType::Utf8, true),
      Field::new("files_indexed", DataType::UInt64, true),
      Field::new("merged_into", DataType::Utf8, true), // UUID as string
    ];

    Schema::new(fields)
  }

  /// Create a LanceDB table for storing IndexTasks
  pub async fn create_table(
    connection: &lancedb::Connection,
    table_name: &str,
  ) -> lancedb::Result<lancedb::Table> {
    use arrow::array::*;
    use arrow::record_batch::{RecordBatch, RecordBatchIterator};

    let schema = std::sync::Arc::new(Self::schema());

    // Create dummy data - LanceDB requires at least one batch
    let id_array = StringArray::from(vec!["00000000-0000-0000-0000-000000000000"]);
    let dummy_uuid = Uuid::nil();
    let project_id_array = StringArray::from(vec![dummy_uuid.to_string()]);
    let path_array = StringArray::from(vec!["/__dummy__"]);
    let task_type_array =
      StringArray::from(vec![serde_json::to_string(&TaskType::FullIndex).unwrap()]);
    let status_array = StringArray::from(vec!["pending"]);

    let created_at_array = arrow::array::TimestampMicrosecondArray::from(vec![0i64]);
    let started_at_array = arrow::array::TimestampMicrosecondArray::from(vec![None as Option<i64>]);
    let completed_at_array =
      arrow::array::TimestampMicrosecondArray::from(vec![None as Option<i64>]);
    let error_array = StringArray::from(vec![None as Option<&str>]);
    let files_indexed_array = UInt64Array::from(vec![None as Option<u64>]);
    let merged_into_array = StringArray::from(vec![None as Option<&str>]);

    let batch = RecordBatch::try_new(
      schema.clone(),
      vec![
        std::sync::Arc::new(id_array),
        std::sync::Arc::new(project_id_array),
        std::sync::Arc::new(path_array),
        std::sync::Arc::new(task_type_array),
        std::sync::Arc::new(status_array),
        std::sync::Arc::new(created_at_array),
        std::sync::Arc::new(started_at_array),
        std::sync::Arc::new(completed_at_array),
        std::sync::Arc::new(error_array),
        std::sync::Arc::new(files_indexed_array),
        std::sync::Arc::new(merged_into_array),
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

    // Create indices
    Self::create_indices(&table).await;

    Ok(table)
  }

  /// Create indices on the table
  async fn create_indices(table: &lancedb::Table) {
    use lancedb::index::Index;
    use tracing::debug;

    // Create Auto index on project_id for project-based queries
    match table
      .create_index(&["project_id"], Index::Auto)
      .execute()
      .await
    {
      Ok(_) => {
        debug!("Created index on project_id field");
      }
      Err(e) => {
        debug!(
          "Index on project_id might already exist or creation failed: {}",
          e
        );
      }
    }

    // Create Auto index on path for path-based queries
    match table.create_index(&["path"], Index::Auto).execute().await {
      Ok(_) => {
        debug!("Created index on path field");
      }
      Err(e) => {
        debug!(
          "Index on path might already exist or creation failed: {}",
          e
        );
      }
    }
  }

  /// Ensure a table exists - open if it exists, create if it doesn't
  pub async fn ensure_table(
    connection: &lancedb::Connection,
    table_name: &str,
  ) -> lancedb::Result<lancedb::Table> {
    // Try to open the table first
    match connection.open_table(table_name).execute().await {
      Ok(table) => Ok(table),
      Err(e) => {
        // Check if it's a table not found error
        match &e {
          lancedb::Error::TableNotFound { .. } => {
            // Create the table
            Self::create_table(connection, table_name).await
          }
          _ => Err(e), // Propagate other errors
        }
      }
    }
  }

  /// Convert RecordBatch row to IndexTask
  pub fn from_record_batch(
    batch: &arrow::record_batch::RecordBatch,
    row: usize,
  ) -> Result<Self, lancedb::Error> {
    use arrow::array::*;

    let id_array = batch
      .column_by_name("id")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Invalid id column".to_string(),
      })?;

    let id_str = id_array.value(row);
    let id = Uuid::parse_str(id_str).map_err(|e| lancedb::Error::Runtime {
      message: format!("Invalid UUID in id column: {}", e),
    })?;

    let project_id_array = batch
      .column_by_name("project_id")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Invalid project_id column".to_string(),
      })?;

    let project_id_str = project_id_array.value(row);
    let project_id = Uuid::parse_str(project_id_str).map_err(|e| lancedb::Error::Runtime {
      message: format!("Invalid UUID in project_id column: {}", e),
    })?;

    let path_array = batch
      .column_by_name("path")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Invalid path column".to_string(),
      })?;

    let task_type_array = batch
      .column_by_name("task_type")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Invalid task_type column".to_string(),
      })?;

    let task_type: TaskType =
      serde_json::from_str(task_type_array.value(row)).map_err(|e| lancedb::Error::Runtime {
        message: format!("Failed to parse task_type JSON: {}", e),
      })?;

    let status_array = batch
      .column_by_name("status")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Invalid status column".to_string(),
      })?;

    let status = status_array
      .value(row)
      .parse()
      .map_err(|_| lancedb::Error::Runtime {
        message: format!("Invalid status value: {}", status_array.value(row)),
      })?;

    let created_at_array = batch
      .column_by_name("created_at")
      .and_then(|col| col.as_any().downcast_ref::<TimestampMicrosecondArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Invalid created_at column".to_string(),
      })?;

    let started_at_array = batch
      .column_by_name("started_at")
      .and_then(|col| col.as_any().downcast_ref::<TimestampMicrosecondArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Invalid started_at column".to_string(),
      })?;

    let completed_at_array = batch
      .column_by_name("completed_at")
      .and_then(|col| col.as_any().downcast_ref::<TimestampMicrosecondArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Invalid completed_at column".to_string(),
      })?;

    let error_array = batch
      .column_by_name("error")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Invalid error column".to_string(),
      })?;

    let files_indexed_array = batch
      .column_by_name("files_indexed")
      .and_then(|col| col.as_any().downcast_ref::<UInt64Array>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Invalid files_indexed column".to_string(),
      })?;

    let merged_into_array = batch
      .column_by_name("merged_into")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Invalid merged_into column".to_string(),
      })?;

    let merged_into = if merged_into_array.is_null(row) {
      None
    } else {
      Some(
        Uuid::parse_str(merged_into_array.value(row)).map_err(|e| lancedb::Error::Runtime {
          message: format!("Invalid UUID in merged_into column: {}", e),
        })?,
      )
    };

    Ok(IndexTask {
      id,
      project_id,
      path: path_array.value(row).to_string(),
      task_type,
      status,
      created_at: chrono::DateTime::from_timestamp_micros(created_at_array.value(row))
        .ok_or_else(|| lancedb::Error::Runtime {
          message: "Invalid created_at timestamp".to_string(),
        })?
        .naive_utc(),
      started_at: if started_at_array.is_null(row) {
        None
      } else {
        Some(
          chrono::DateTime::from_timestamp_micros(started_at_array.value(row))
            .ok_or_else(|| lancedb::Error::Runtime {
              message: "Invalid started_at timestamp".to_string(),
            })?
            .naive_utc(),
        )
      },
      completed_at: if completed_at_array.is_null(row) {
        None
      } else {
        Some(
          chrono::DateTime::from_timestamp_micros(completed_at_array.value(row))
            .ok_or_else(|| lancedb::Error::Runtime {
              message: "Invalid completed_at timestamp".to_string(),
            })?
            .naive_utc(),
        )
      },
      error: if error_array.is_null(row) {
        None
      } else {
        Some(error_array.value(row).to_string())
      },
      files_indexed: if files_indexed_array.is_null(row) {
        None
      } else {
        Some(files_indexed_array.value(row) as usize)
      },
      merged_into,
    })
  }
}

// Implement IntoArrow for IndexTask to enable LanceDB persistence
impl lancedb::arrow::IntoArrow for IndexTask {
  fn into_arrow(self) -> lancedb::Result<Box<dyn arrow::array::RecordBatchReader + Send>> {
    use arrow::array::*;
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    let schema = Arc::new(IndexTask::schema());

    // Build arrays for single task
    let id_array = StringArray::from(vec![self.id.to_string()]);
    let project_id_array = StringArray::from(vec![self.project_id.to_string()]);
    let path_array = StringArray::from(vec![self.path.as_str()]);
    let task_type_array = StringArray::from(vec![serde_json::to_string(&self.task_type).unwrap()]);
    let status_array = StringArray::from(vec![self.status.to_string()]);

    // Convert timestamps to microseconds
    let created_at_us = self.created_at.and_utc().timestamp_micros();
    let created_at_array = arrow::array::TimestampMicrosecondArray::from(vec![created_at_us]);

    let started_at_array = arrow::array::TimestampMicrosecondArray::from(vec![
      self.started_at.map(|t| t.and_utc().timestamp_micros()),
    ]);

    let completed_at_array = arrow::array::TimestampMicrosecondArray::from(vec![
      self.completed_at.map(|t| t.and_utc().timestamp_micros()),
    ]);

    let error_array = StringArray::from(vec![self.error.as_deref()]);
    let files_indexed_array = UInt64Array::from(vec![self.files_indexed.map(|n| n as u64)]);
    let merged_into_array =
      StringArray::from(vec![self.merged_into.map(|u| u.to_string()).as_deref()]);

    // Create the record batch
    let batch = RecordBatch::try_new(
      schema.clone(),
      vec![
        Arc::new(id_array),
        Arc::new(project_id_array),
        Arc::new(path_array),
        Arc::new(task_type_array),
        Arc::new(status_array),
        Arc::new(created_at_array),
        Arc::new(started_at_array),
        Arc::new(completed_at_array),
        Arc::new(error_array),
        Arc::new(files_indexed_array),
        Arc::new(merged_into_array),
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

    // Convert timestamps to microseconds
    let last_modified_us = self.last_modified.and_utc().timestamp_micros();
    let indexed_at_us = self.indexed_at.and_utc().timestamp_micros();
    let last_modified_array = arrow::array::TimestampMicrosecondArray::from(vec![last_modified_us]);
    let indexed_at_array = arrow::array::TimestampMicrosecondArray::from(vec![indexed_at_us]);

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Project {
  pub id: Uuid,
  pub name: String,
  pub directory: String,
  pub description: Option<String>,
  pub created_at: chrono::NaiveDateTime,
  pub updated_at: chrono::NaiveDateTime,
}

impl Project {
  pub fn new(name: String, directory: String, description: Option<String>) -> Result<Self, String> {
    // Validate directory exists
    let path = Path::new(&directory);
    if !path.exists() {
      return Err(format!("Directory does not exist: {}", directory));
    }
    if !path.is_dir() {
      return Err(format!("Path is not a directory: {}", directory));
    }

    let now = chrono::Utc::now().naive_utc();
    Ok(Self {
      id: Uuid::now_v7(),
      name,
      directory,
      description,
      created_at: now,
      updated_at: now,
    })
  }

  /// Create the Arrow schema for Project
  pub fn schema() -> arrow::datatypes::Schema {
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};

    let fields = vec![
      Field::new("id", DataType::Utf8, false), // UUID as string
      Field::new("name", DataType::Utf8, false),
      Field::new("directory", DataType::Utf8, false),
      Field::new("description", DataType::Utf8, true),
      Field::new(
        "created_at",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
      ),
      Field::new(
        "updated_at",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
      ),
    ];

    Schema::new(fields)
  }

  /// Create a LanceDB table for storing Projects
  pub async fn create_table(
    connection: &lancedb::Connection,
    table_name: &str,
  ) -> lancedb::Result<lancedb::Table> {
    use arrow::array::*;
    use arrow::record_batch::{RecordBatch, RecordBatchIterator};

    let schema = std::sync::Arc::new(Self::schema());

    // Create dummy data - LanceDB requires at least one batch
    let dummy_uuid = Uuid::nil();
    let id_array = StringArray::from(vec![dummy_uuid.to_string()]);
    let name_array = StringArray::from(vec!["__dummy__"]);
    let directory_array = StringArray::from(vec!["/__dummy__"]);
    let description_array = StringArray::from(vec![None as Option<&str>]);
    let created_at_array = arrow::array::TimestampMicrosecondArray::from(vec![0i64]);
    let updated_at_array = arrow::array::TimestampMicrosecondArray::from(vec![0i64]);

    let batch = RecordBatch::try_new(
      schema.clone(),
      vec![
        std::sync::Arc::new(id_array),
        std::sync::Arc::new(name_array),
        std::sync::Arc::new(directory_array),
        std::sync::Arc::new(description_array),
        std::sync::Arc::new(created_at_array),
        std::sync::Arc::new(updated_at_array),
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
    table.delete(&format!("id = '{}'", dummy_uuid)).await?;

    Ok(table)
  }

  /// Ensure a table exists - open if it exists, create if it doesn't
  pub async fn ensure_table(
    connection: &lancedb::Connection,
    table_name: &str,
  ) -> lancedb::Result<lancedb::Table> {
    // Try to open the table first
    match connection.open_table(table_name).execute().await {
      Ok(table) => Ok(table),
      Err(e) => {
        // Check if it's a table not found error
        match &e {
          lancedb::Error::TableNotFound { .. } => {
            // Create the table
            Self::create_table(connection, table_name).await
          }
          _ => Err(e), // Propagate other errors
        }
      }
    }
  }

  /// Convert a RecordBatch row to Project
  pub fn from_record_batch(
    batch: &arrow::record_batch::RecordBatch,
    row: usize,
  ) -> Result<Self, lancedb::Error> {
    use arrow::array::*;

    if row >= batch.num_rows() {
      return Err(lancedb::Error::Runtime {
        message: format!(
          "Row index {} out of bounds (batch has {} rows)",
          row,
          batch.num_rows()
        ),
      });
    }

    let id_str = batch
      .column_by_name("id")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid id column".to_string(),
      })?
      .value(row);

    let id = Uuid::parse_str(id_str).map_err(|e| lancedb::Error::Runtime {
      message: format!("Invalid UUID string: {}", e),
    })?;

    let name = batch
      .column_by_name("name")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid name column".to_string(),
      })?
      .value(row)
      .to_string();

    let directory = batch
      .column_by_name("directory")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid directory column".to_string(),
      })?
      .value(row)
      .to_string();

    let description_array = batch
      .column_by_name("description")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid description column".to_string(),
      })?;

    let description = if description_array.is_null(row) {
      None
    } else {
      Some(description_array.value(row).to_string())
    };

    let created_at = batch
      .column_by_name("created_at")
      .and_then(|col| {
        col
          .as_any()
          .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
      })
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid created_at column".to_string(),
      })?
      .value(row);

    let updated_at = batch
      .column_by_name("updated_at")
      .and_then(|col| {
        col
          .as_any()
          .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
      })
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid updated_at column".to_string(),
      })?
      .value(row);

    Ok(Self {
      id,
      name,
      directory,
      description,
      created_at: chrono::DateTime::from_timestamp_micros(created_at)
        .unwrap_or_default()
        .naive_utc(),
      updated_at: chrono::DateTime::from_timestamp_micros(updated_at)
        .unwrap_or_default()
        .naive_utc(),
    })
  }
}

// Implement IntoArrow for Project to enable LanceDB persistence
impl lancedb::arrow::IntoArrow for Project {
  fn into_arrow(self) -> lancedb::Result<Box<dyn arrow::array::RecordBatchReader + Send>> {
    use arrow::array::*;
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    let schema = Arc::new(Project::schema());

    // Build arrays for single project
    let id_array = StringArray::from(vec![self.id.to_string()]);

    let name_array = StringArray::from(vec![self.name.as_str()]);
    let directory_array = StringArray::from(vec![self.directory.as_str()]);
    let description_array = StringArray::from(vec![self.description.as_deref()]);

    // Convert timestamps to microseconds
    let created_at_us = self.created_at.and_utc().timestamp_micros();
    let updated_at_us = self.updated_at.and_utc().timestamp_micros();
    let created_at_array = arrow::array::TimestampMicrosecondArray::from(vec![created_at_us]);
    let updated_at_array = arrow::array::TimestampMicrosecondArray::from(vec![updated_at_us]);

    // Create the record batch
    let batch = RecordBatch::try_new(
      schema.clone(),
      vec![
        Arc::new(id_array),
        Arc::new(name_array),
        Arc::new(directory_array),
        Arc::new(description_array),
        Arc::new(created_at_array),
        Arc::new(updated_at_array),
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

  #[test]
  fn test_index_task_new() {
    let project_id = Uuid::now_v7();
    let path = Path::new("/test/project");
    let task = IndexTask::new(project_id, path);

    assert_eq!(task.project_id, project_id);
    assert_eq!(task.path, path.to_str().unwrap());
    assert_eq!(task.status, TaskStatus::Pending);
    assert!(task.started_at.is_none());
    assert!(task.completed_at.is_none());
    assert!(task.error.is_none());
    assert!(task.files_indexed.is_none());

    let now = chrono::Utc::now().naive_utc();
    let diff = now - task.created_at;
    assert!(diff.num_seconds() < 1);
  }

  #[test]
  fn test_task_status_display() {
    assert_eq!(TaskStatus::Pending.to_string(), "pending");
    assert_eq!(TaskStatus::Running.to_string(), "running");
    assert_eq!(TaskStatus::Completed.to_string(), "completed");
    assert_eq!(TaskStatus::Failed.to_string(), "failed");
  }

  #[test]
  fn test_task_status_from_str() {
    assert_eq!(
      "pending".parse::<TaskStatus>().unwrap(),
      TaskStatus::Pending
    );
    assert_eq!(
      "running".parse::<TaskStatus>().unwrap(),
      TaskStatus::Running
    );
    assert_eq!(
      "completed".parse::<TaskStatus>().unwrap(),
      TaskStatus::Completed
    );
    assert_eq!("failed".parse::<TaskStatus>().unwrap(), TaskStatus::Failed);

    assert!("invalid".parse::<TaskStatus>().is_err());
  }

  #[test]
  fn test_index_task_schema() {
    let schema = IndexTask::schema();

    assert_eq!(schema.fields().len(), 11);

    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
      field_names,
      vec![
        "id",
        "project_id",
        "path",
        "task_type",
        "status",
        "created_at",
        "started_at",
        "completed_at",
        "error",
        "files_indexed",
        "merged_into"
      ]
    );

    assert!(!schema.field(0).is_nullable()); // id
    assert!(!schema.field(1).is_nullable()); // project_id
    assert!(!schema.field(2).is_nullable()); // path
    assert!(!schema.field(3).is_nullable()); // task_type
    assert!(!schema.field(4).is_nullable()); // status
    assert!(!schema.field(5).is_nullable()); // created_at
    assert!(schema.field(6).is_nullable()); // started_at
    assert!(schema.field(7).is_nullable()); // completed_at
    assert!(schema.field(8).is_nullable()); // error
    assert!(schema.field(9).is_nullable()); // files_indexed
    assert!(schema.field(10).is_nullable()); // merged_into
  }

  #[test]
  fn test_project_new() {
    let temp_dir = TempDir::new().unwrap();
    let test_dir = temp_dir.path().join("test_project");
    std::fs::create_dir(&test_dir).unwrap();

    let project = Project::new(
      "Test Project".to_string(),
      test_dir.to_str().unwrap().to_string(),
      Some("A test project".to_string()),
    )
    .unwrap();

    assert_eq!(project.name, "Test Project");
    assert_eq!(project.directory, test_dir.to_str().unwrap());
    assert_eq!(project.description, Some("A test project".to_string()));
    assert!(!project.id.is_nil());
    assert_eq!(project.created_at, project.updated_at);
  }

  #[test]
  fn test_project_new_invalid_directory() {
    let result = Project::new(
      "Test Project".to_string(),
      "/non/existent/directory".to_string(),
      None,
    );

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("does not exist"));
  }

  #[test]
  fn test_project_new_not_directory() {
    let temp_dir = TempDir::new().unwrap();
    let test_file = temp_dir.path().join("not_a_dir.txt");
    std::fs::write(&test_file, "content").unwrap();

    let result = Project::new(
      "Test Project".to_string(),
      test_file.to_str().unwrap().to_string(),
      None,
    );

    assert!(result.is_err());
    assert!(result.unwrap_err().contains("not a directory"));
  }

  #[test]
  fn test_project_schema() {
    let schema = Project::schema();

    assert_eq!(schema.fields().len(), 6);

    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
      field_names,
      vec![
        "id",
        "name",
        "directory",
        "description",
        "created_at",
        "updated_at"
      ]
    );

    assert!(!schema.field(0).is_nullable()); // id
    assert!(!schema.field(1).is_nullable()); // name
    assert!(!schema.field(2).is_nullable()); // directory
    assert!(schema.field(3).is_nullable()); // description
    assert!(!schema.field(4).is_nullable()); // created_at
    assert!(!schema.field(5).is_nullable()); // updated_at
  }
}
