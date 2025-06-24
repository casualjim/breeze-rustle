use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use uuid::Uuid;

/// A record of an embedding failure that needs to be retried
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct RetryError {
  pub timestamp: chrono::NaiveDateTime,
  pub error: String,
}

/// Batch of files that failed embedding and need retry
#[derive(Debug, Clone)]
pub struct FailedEmbeddingBatch {
  pub id: Uuid,
  pub task_id: Uuid,
  pub project_id: Uuid,
  pub project_path: String,
  pub failed_files: BTreeSet<String>,
  pub errors: Vec<RetryError>,
  pub retry_count: i32,
  pub retry_after: chrono::NaiveDateTime,
  pub created_at: chrono::NaiveDateTime,
}

impl FailedEmbeddingBatch {
  /// Create a new failed embedding batch
  pub fn new(
    task_id: Uuid,
    project_id: Uuid,
    project_path: String,
    failed_files: BTreeSet<String>,
    error: String,
  ) -> Self {
    let now = chrono::Utc::now().naive_utc();
    Self {
      id: Uuid::now_v7(),
      task_id,
      project_id,
      project_path,
      failed_files,
      errors: vec![RetryError {
        timestamp: now,
        error,
      }],
      retry_count: 0,
      retry_after: Self::calculate_next_retry_after(0),
      created_at: now,
    }
  }

  /// Calculate when the next retry should happen based on retry count
  pub fn calculate_next_retry_after(retry_count: i32) -> chrono::NaiveDateTime {
    let minutes = match retry_count {
      0 => 1,  // First retry after 1 minute
      1 => 5,  // Second retry after 5 minutes
      2 => 10, // Third retry after 10 minutes
      3 => 20, // Fourth retry after 20 minutes
      4 => 40, // Fifth retry after 40 minutes
      _ => 60, // Every hour thereafter
    };
    chrono::Utc::now().naive_utc() + chrono::Duration::minutes(minutes as i64)
  }

  /// Update for next retry attempt
  pub fn update_for_retry(&mut self, error: String) {
    self.retry_count += 1;
    self.retry_after = Self::calculate_next_retry_after(self.retry_count);
    self.errors.push(RetryError {
      timestamp: chrono::Utc::now().naive_utc(),
      error,
    });
  }

  /// Create the Arrow schema for FailedEmbeddingBatch
  pub fn schema() -> arrow::datatypes::Schema {
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};

    let fields = vec![
      Field::new("id", DataType::Utf8, false),
      Field::new("task_id", DataType::Utf8, false),
      Field::new("project_id", DataType::Utf8, false),
      Field::new("project_path", DataType::Utf8, false),
      Field::new(
        "failed_files",
        DataType::List(std::sync::Arc::new(Field::new(
          "item",
          DataType::Utf8,
          true,
        ))),
        false,
      ),
      Field::new("errors", DataType::Utf8, false), // JSON serialized
      Field::new("retry_count", DataType::Int32, false),
      Field::new(
        "retry_after",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
      ),
      Field::new(
        "created_at",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
      ),
    ];

    Schema::new(fields)
  }

  /// Create a LanceDB table for storing FailedEmbeddingBatch
  pub async fn create_table(
    connection: &lancedb::Connection,
    table_name: &str,
  ) -> lancedb::Result<lancedb::Table> {
    use arrow::array::*;
    use arrow::record_batch::{RecordBatch, RecordBatchIterator};

    let schema = std::sync::Arc::new(Self::schema());

    // Create dummy data
    let id_array = StringArray::from(vec!["00000000-0000-0000-0000-000000000000"]);
    let task_id_array = StringArray::from(vec![Uuid::nil().to_string()]);
    let project_id_array = StringArray::from(vec![Uuid::nil().to_string()]);
    let project_path_array = StringArray::from(vec!["/__dummy__"]);

    // Create empty list array for failed_files
    let mut failed_files_builder = ListBuilder::new(StringBuilder::new());
    failed_files_builder.append(true); // Append empty list
    let failed_files_array = failed_files_builder.finish();

    let errors_array = StringArray::from(vec!["[]"]); // Empty JSON array
    let retry_count_array = Int32Array::from(vec![0]);
    let retry_after_array = TimestampMicrosecondArray::from(vec![0i64]);
    let created_at_array = TimestampMicrosecondArray::from(vec![0i64]);

    let batch = RecordBatch::try_new(
      schema.clone(),
      vec![
        std::sync::Arc::new(id_array),
        std::sync::Arc::new(task_id_array),
        std::sync::Arc::new(project_id_array),
        std::sync::Arc::new(project_path_array),
        std::sync::Arc::new(failed_files_array),
        std::sync::Arc::new(errors_array),
        std::sync::Arc::new(retry_count_array),
        std::sync::Arc::new(retry_after_array),
        std::sync::Arc::new(created_at_array),
      ],
    )
    .map_err(|e| lancedb::Error::Arrow { source: e })?;

    let batch_iter = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);

    // Create table
    let table = connection
      .create_table(table_name, Box::new(batch_iter))
      .execute()
      .await?;

    // Create indices
    table
      .create_index(&["retry_after"], lancedb::index::Index::Auto)
      .execute()
      .await?;

    Ok(table)
  }

  /// Ensure a table exists - open if it exists, create if it doesn't
  pub async fn ensure_table(
    connection: &lancedb::Connection,
    table_name: &str,
  ) -> lancedb::Result<lancedb::Table> {
    match connection.open_table(table_name).execute().await {
      Ok(table) => Ok(table),
      Err(e) => match &e {
        lancedb::Error::TableNotFound { .. } => Self::create_table(connection, table_name).await,
        _ => Err(e),
      },
    }
  }

  /// Convert from RecordBatch
  pub fn from_record_batch(
    batch: &arrow::record_batch::RecordBatch,
    row: usize,
  ) -> lancedb::Result<Self> {
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

    // Extract id
    let id_str = batch
      .column_by_name("id")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid id column".to_string(),
      })?
      .value(row);
    let id = Uuid::parse_str(id_str).map_err(|e| lancedb::Error::Runtime {
      message: format!("Invalid UUID: {}", e),
    })?;

    // Extract task_id
    let task_id_str = batch
      .column_by_name("task_id")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid task_id column".to_string(),
      })?
      .value(row);
    let task_id = Uuid::parse_str(task_id_str).map_err(|e| lancedb::Error::Runtime {
      message: format!("Invalid task_id UUID: {}", e),
    })?;

    // Extract project_id
    let project_id_str = batch
      .column_by_name("project_id")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid project_id column".to_string(),
      })?
      .value(row);
    let project_id = Uuid::parse_str(project_id_str).map_err(|e| lancedb::Error::Runtime {
      message: format!("Invalid project_id UUID: {}", e),
    })?;

    // Extract project_path
    let project_path = batch
      .column_by_name("project_path")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid project_path column".to_string(),
      })?
      .value(row)
      .to_string();

    // Extract failed_files from ListArray
    let failed_files_list = batch
      .column_by_name("failed_files")
      .and_then(|col| col.as_any().downcast_ref::<ListArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid failed_files column".to_string(),
      })?;

    let mut failed_files = BTreeSet::new();
    let values = failed_files_list.values();
    let string_array = values
      .as_any()
      .downcast_ref::<StringArray>()
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "failed_files values are not strings".to_string(),
      })?;

    let start = failed_files_list.value_offsets()[row] as usize;
    let end = failed_files_list.value_offsets()[row + 1] as usize;
    for i in start..end {
      failed_files.insert(string_array.value(i).to_string());
    }

    // Extract errors (JSON)
    let errors_json = batch
      .column_by_name("errors")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid errors column".to_string(),
      })?
      .value(row);
    let errors: Vec<RetryError> =
      serde_json::from_str(errors_json).map_err(|e| lancedb::Error::Runtime {
        message: format!("Failed to parse errors JSON: {}", e),
      })?;

    // Extract retry_count
    let retry_count = batch
      .column_by_name("retry_count")
      .and_then(|col| col.as_any().downcast_ref::<Int32Array>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid retry_count column".to_string(),
      })?
      .value(row);

    // Extract timestamps
    let retry_after_micros = batch
      .column_by_name("retry_after")
      .and_then(|col| col.as_any().downcast_ref::<TimestampMicrosecondArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid retry_after column".to_string(),
      })?
      .value(row);
    let retry_after = chrono::DateTime::from_timestamp_micros(retry_after_micros)
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Invalid retry_after timestamp".to_string(),
      })?
      .naive_utc();

    let created_at_micros = batch
      .column_by_name("created_at")
      .and_then(|col| col.as_any().downcast_ref::<TimestampMicrosecondArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid created_at column".to_string(),
      })?
      .value(row);
    let created_at = chrono::DateTime::from_timestamp_micros(created_at_micros)
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Invalid created_at timestamp".to_string(),
      })?
      .naive_utc();

    Ok(Self {
      id,
      task_id,
      project_id,
      project_path,
      failed_files,
      errors,
      retry_count,
      retry_after,
      created_at,
    })
  }
}

// Implement IntoArrow for FailedEmbeddingBatch
impl lancedb::arrow::IntoArrow for FailedEmbeddingBatch {
  fn into_arrow(self) -> lancedb::Result<Box<dyn arrow::array::RecordBatchReader + Send>> {
    use arrow::array::*;
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    let schema = Arc::new(Self::schema());

    // Build arrays
    let id_array = StringArray::from(vec![self.id.to_string()]);
    let task_id_array = StringArray::from(vec![self.task_id.to_string()]);
    let project_id_array = StringArray::from(vec![self.project_id.to_string()]);
    let project_path_array = StringArray::from(vec![self.project_path.as_str()]);

    // Build failed_files list array
    let mut failed_files_builder = ListBuilder::new(StringBuilder::new());
    for file in &self.failed_files {
      failed_files_builder.values().append_value(file);
    }
    failed_files_builder.append(true);
    let failed_files_array = failed_files_builder.finish();

    // Serialize errors to JSON
    let errors_json = serde_json::to_string(&self.errors).map_err(|e| lancedb::Error::Runtime {
      message: format!("Failed to serialize errors: {}", e),
    })?;
    let errors_array = StringArray::from(vec![errors_json.as_str()]);

    let retry_count_array = Int32Array::from(vec![self.retry_count]);

    let retry_after_us = self.retry_after.and_utc().timestamp_micros();
    let retry_after_array = TimestampMicrosecondArray::from(vec![retry_after_us]);

    let created_at_us = self.created_at.and_utc().timestamp_micros();
    let created_at_array = TimestampMicrosecondArray::from(vec![created_at_us]);

    // Create the record batch
    let batch = RecordBatch::try_new(
      schema.clone(),
      vec![
        Arc::new(id_array),
        Arc::new(task_id_array),
        Arc::new(project_id_array),
        Arc::new(project_path_array),
        Arc::new(failed_files_array),
        Arc::new(errors_array),
        Arc::new(retry_count_array),
        Arc::new(retry_after_array),
        Arc::new(created_at_array),
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
  use arrow::array::{Array, Int32Array, ListArray, StringArray};
  use futures::stream::TryStreamExt;
  use lancedb::arrow::IntoArrow;
  use lancedb::query::ExecutableQuery;

  #[test]
  fn test_failed_embedding_batch_new() {
    let task_id = Uuid::now_v7();
    let project_id = Uuid::now_v7();
    let project_path = "/path/to/project".to_string();
    let mut failed_files = BTreeSet::new();
    failed_files.insert("file1.rs".to_string());
    failed_files.insert("file2.rs".to_string());
    let error = "Failed to connect to embedding service".to_string();

    let batch = FailedEmbeddingBatch::new(
      task_id,
      project_id,
      project_path.clone(),
      failed_files.clone(),
      error.clone(),
    );

    assert_eq!(batch.task_id, task_id);
    assert_eq!(batch.project_id, project_id);
    assert_eq!(batch.project_path, project_path);
    assert_eq!(batch.failed_files, failed_files);
    assert_eq!(batch.retry_count, 0);
    assert_eq!(batch.errors.len(), 1);
    assert_eq!(batch.errors[0].error, error);
    assert!(batch.retry_after > batch.created_at);
  }

  #[test]
  fn test_calculate_next_retry_after() {
    let base_time = chrono::Utc::now().naive_utc();

    // Test retry intervals
    let intervals = vec![
      (0, 1),   // First retry after 1 minute
      (1, 5),   // Second retry after 5 minutes
      (2, 10),  // Third retry after 10 minutes
      (3, 20),  // Fourth retry after 20 minutes
      (4, 40),  // Fifth retry after 40 minutes
      (5, 60),  // Every hour thereafter
      (10, 60), // Still every hour
    ];

    for (retry_count, expected_minutes) in intervals {
      let retry_time = FailedEmbeddingBatch::calculate_next_retry_after(retry_count);
      let diff = retry_time - base_time;
      let diff_minutes = diff.num_minutes();

      // Allow 1 minute tolerance due to timing
      assert!(
        diff_minutes >= expected_minutes - 1 && diff_minutes <= expected_minutes + 1,
        "Retry count {} should wait {} minutes, got {} minutes",
        retry_count,
        expected_minutes,
        diff_minutes
      );
    }
  }

  #[test]
  fn test_update_for_retry() {
    let task_id = Uuid::now_v7();
    let project_id = Uuid::now_v7();
    let mut failed_files = BTreeSet::new();
    failed_files.insert("file1.rs".to_string());

    let mut batch = FailedEmbeddingBatch::new(
      task_id,
      project_id,
      "/project".to_string(),
      failed_files,
      "Initial error".to_string(),
    );

    let initial_retry_after = batch.retry_after;
    let initial_error_count = batch.errors.len();

    // Update for retry
    batch.update_for_retry("Second error".to_string());

    assert_eq!(batch.retry_count, 1);
    assert_eq!(batch.errors.len(), initial_error_count + 1);
    assert_eq!(batch.errors[1].error, "Second error");
    assert!(batch.retry_after > initial_retry_after);

    // Update again
    batch.update_for_retry("Third error".to_string());

    assert_eq!(batch.retry_count, 2);
    assert_eq!(batch.errors.len(), initial_error_count + 2);
    assert_eq!(batch.errors[2].error, "Third error");
  }

  #[tokio::test]
  async fn test_schema() {
    let schema = FailedEmbeddingBatch::schema();
    assert_eq!(schema.fields().len(), 9);

    // Check field names and types
    assert_eq!(schema.field(0).name(), "id");
    assert_eq!(schema.field(1).name(), "task_id");
    assert_eq!(schema.field(2).name(), "project_id");
    assert_eq!(schema.field(3).name(), "project_path");
    assert_eq!(schema.field(4).name(), "failed_files");
    assert_eq!(schema.field(5).name(), "errors");
    assert_eq!(schema.field(6).name(), "retry_count");
    assert_eq!(schema.field(7).name(), "retry_after");
    assert_eq!(schema.field(8).name(), "created_at");
  }

  #[tokio::test]
  async fn test_create_table() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let db_path = temp_dir.path().to_str().unwrap();
    let conn = lancedb::connect(db_path).execute().await.unwrap();

    let table = FailedEmbeddingBatch::create_table(&conn, "test_failed_embeddings")
      .await
      .unwrap();

    // Verify table was created
    let table_names = conn.table_names().execute().await.unwrap();
    assert!(table_names.contains(&"test_failed_embeddings".to_string()));

    // Verify we can query the table (should have dummy row)
    let count = table.count_rows(None).await.unwrap();
    assert_eq!(count, 1, "Should have dummy row");

    // Verify dummy row has expected values
    let stream = table.query().execute().await.unwrap();
    let batches: Vec<_> = stream.try_collect().await.unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);

    // Check dummy row id
    let id_col = batches[0]
      .column_by_name("id")
      .unwrap()
      .as_any()
      .downcast_ref::<StringArray>()
      .unwrap();
    assert_eq!(id_col.value(0), "00000000-0000-0000-0000-000000000000");
  }

  #[tokio::test]
  async fn test_ensure_table() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let db_path = temp_dir.path().to_str().unwrap();
    let conn = lancedb::connect(db_path).execute().await.unwrap();

    // First call creates table
    let table1 = FailedEmbeddingBatch::ensure_table(&conn, "test_failed_embeddings")
      .await
      .unwrap();
    let count1 = table1.count_rows(None).await.unwrap();
    assert_eq!(count1, 1, "Should have dummy row");

    // Second call opens existing table
    let table2 = FailedEmbeddingBatch::ensure_table(&conn, "test_failed_embeddings")
      .await
      .unwrap();
    let count2 = table2.count_rows(None).await.unwrap();
    assert_eq!(count2, 1, "Should still have only dummy row");
  }

  #[test]
  fn test_into_arrow() {
    let task_id = Uuid::now_v7();
    let project_id = Uuid::now_v7();
    let mut failed_files = BTreeSet::new();
    failed_files.insert("file1.rs".to_string());
    failed_files.insert("file2.rs".to_string());

    let mut batch = FailedEmbeddingBatch::new(
      task_id,
      project_id,
      "/project".to_string(),
      failed_files,
      "Test error".to_string(),
    );

    // Add another error
    batch.update_for_retry("Retry error".to_string());

    let reader = batch.into_arrow().unwrap();
    let schema = reader.schema();
    assert_eq!(schema.fields().len(), 9);

    // Collect batches
    let batches: Vec<_> = reader.collect();
    assert_eq!(batches.len(), 1);

    let record_batch = batches[0].as_ref().unwrap();
    assert_eq!(record_batch.num_rows(), 1);
    assert_eq!(record_batch.num_columns(), 9);

    // Verify project_path
    let project_path_array = record_batch
      .column(3)
      .as_any()
      .downcast_ref::<StringArray>()
      .unwrap();
    assert_eq!(project_path_array.value(0), "/project");

    // Verify failed_files list
    let failed_files_array = record_batch
      .column(4)
      .as_any()
      .downcast_ref::<ListArray>()
      .unwrap();
    assert!(!failed_files_array.is_null(0));

    // Verify retry_count
    let retry_count_array = record_batch
      .column(6)
      .as_any()
      .downcast_ref::<Int32Array>()
      .unwrap();
    assert_eq!(retry_count_array.value(0), 1);
  }

  #[tokio::test]
  async fn test_from_record_batch() {
    let task_id = Uuid::now_v7();
    let project_id = Uuid::now_v7();
    let mut failed_files = BTreeSet::new();
    failed_files.insert("test1.rs".to_string());
    failed_files.insert("test2.rs".to_string());

    let original = FailedEmbeddingBatch::new(
      task_id,
      project_id,
      "/test/path".to_string(),
      failed_files.clone(),
      "Original error".to_string(),
    );

    // Convert to Arrow and back
    let reader = original.clone().into_arrow().unwrap();
    let batches: Vec<_> = reader.collect();
    let record_batch = batches[0].as_ref().unwrap();

    let reconstructed = FailedEmbeddingBatch::from_record_batch(record_batch, 0).unwrap();

    assert_eq!(reconstructed.id, original.id);
    assert_eq!(reconstructed.task_id, original.task_id);
    assert_eq!(reconstructed.project_id, original.project_id);
    assert_eq!(reconstructed.project_path, original.project_path);
    assert_eq!(reconstructed.failed_files, original.failed_files);
    assert_eq!(reconstructed.retry_count, original.retry_count);
    assert_eq!(reconstructed.errors.len(), original.errors.len());
    assert_eq!(reconstructed.errors[0].error, original.errors[0].error);
  }

  #[tokio::test]
  async fn test_from_record_batch_out_of_bounds() {
    let batch = FailedEmbeddingBatch::new(
      Uuid::now_v7(),
      Uuid::now_v7(),
      "/test".to_string(),
      BTreeSet::new(),
      "Error".to_string(),
    );

    let reader = batch.into_arrow().unwrap();
    let batches: Vec<_> = reader.collect();
    let record_batch = batches[0].as_ref().unwrap();

    // Try to access out of bounds row
    let result = FailedEmbeddingBatch::from_record_batch(record_batch, 1);
    assert!(result.is_err());
  }

  #[test]
  fn test_retry_error_serialization() {
    let error = RetryError {
      timestamp: chrono::Utc::now().naive_utc(),
      error: "Test error message".to_string(),
    };

    // Test serialization
    let json = serde_json::to_string(&error).unwrap();
    assert!(json.contains("Test error message"));

    // Test deserialization
    let deserialized: RetryError = serde_json::from_str(&json).unwrap();
    assert_eq!(deserialized.error, error.error);
  }
}
