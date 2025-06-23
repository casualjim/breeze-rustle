use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use tracing::debug;
use uuid::Uuid;

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
  PartiallyCompleted,
}

impl std::fmt::Display for TaskStatus {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    match self {
      TaskStatus::Pending => write!(f, "pending"),
      TaskStatus::Running => write!(f, "running"),
      TaskStatus::Completed => write!(f, "completed"),
      TaskStatus::Failed => write!(f, "failed"),
      TaskStatus::Merged => write!(f, "merged"),
      TaskStatus::PartiallyCompleted => write!(f, "partially_completed"),
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
      "partially_completed" => Ok(TaskStatus::PartiallyCompleted),
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
#[cfg(test)]
mod tests {
  use super::*;

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
}
