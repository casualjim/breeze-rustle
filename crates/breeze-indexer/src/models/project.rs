use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::Duration;
use strum::{Display, EnumString};
use uuid::Uuid;

#[derive(
  Debug,
  Clone,
  Copy,
  PartialEq,
  Serialize,
  Deserialize,
  Display,
  EnumString
)]
pub enum ProjectStatus {
  Active,
  PendingDeletion,
  Deleted,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Project {
  pub id: Uuid,
  pub name: String,
  pub directory: String,
  pub description: Option<String>,
  pub status: ProjectStatus,
  pub deletion_requested_at: Option<chrono::NaiveDateTime>,
  pub created_at: chrono::NaiveDateTime,
  pub updated_at: chrono::NaiveDateTime,
  pub rescan_interval: Option<Duration>,
  pub last_indexed_at: Option<chrono::NaiveDateTime>,
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
      status: ProjectStatus::Active,
      deletion_requested_at: None,
      created_at: now,
      updated_at: now,
      rescan_interval: None,
      last_indexed_at: None,
    })
  }

  pub fn with_rescan_interval(mut self, rescan_interval: Option<Duration>) -> Self {
    self.rescan_interval = rescan_interval;
    self
  }

  /// Create the Arrow schema for Project
  pub fn schema() -> arrow::datatypes::Schema {
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};

    let fields = vec![
      Field::new("id", DataType::Utf8, false), // UUID as string
      Field::new("name", DataType::Utf8, false),
      Field::new("directory", DataType::Utf8, false),
      Field::new("description", DataType::Utf8, true),
      Field::new("status", DataType::Utf8, false), // ProjectStatus as string
      Field::new(
        "deletion_requested_at",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        true,
      ),
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
      Field::new(
        "rescan_interval",
        DataType::Duration(TimeUnit::Microsecond),
        true,
      ),
      Field::new(
        "last_indexed_at",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        true,
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
    let status_array = StringArray::from(vec!["Active"]);
    let deletion_requested_at_array =
      arrow::array::TimestampMicrosecondArray::from(vec![None as Option<i64>]);
    let created_at_array = arrow::array::TimestampMicrosecondArray::from(vec![0i64]);
    let updated_at_array = arrow::array::TimestampMicrosecondArray::from(vec![0i64]);

    // Add missing rescan_interval and last_indexed_at arrays for dummy row
    let rescan_interval_array =
      arrow::array::DurationMicrosecondArray::from(vec![None as Option<i64>]);
    let last_indexed_at_array =
      arrow::array::TimestampMicrosecondArray::from(vec![None as Option<i64>]);

    let batch = RecordBatch::try_new(
      schema.clone(),
      vec![
        std::sync::Arc::new(id_array),
        std::sync::Arc::new(name_array),
        std::sync::Arc::new(directory_array),
        std::sync::Arc::new(description_array),
        std::sync::Arc::new(status_array),
        std::sync::Arc::new(deletion_requested_at_array),
        std::sync::Arc::new(created_at_array),
        std::sync::Arc::new(updated_at_array),
        std::sync::Arc::new(rescan_interval_array),
        std::sync::Arc::new(last_indexed_at_array),
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

    let rescan_interval_array = batch
      .column_by_name("rescan_interval")
      .and_then(|col| col.as_any().downcast_ref::<DurationMicrosecondArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid rescan_interval column".to_string(),
      })?;

    let rescan_interval = if rescan_interval_array.is_null(row) {
      None
    } else {
      Some(Duration::from_micros(
        rescan_interval_array.value(row) as u64
      ))
    };

    let last_indexed_at_array = batch
      .column_by_name("last_indexed_at")
      .and_then(|col| {
        col
          .as_any()
          .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
      })
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid last_indexed_at column".to_string(),
      })?;

    let last_indexed_at = if last_indexed_at_array.is_null(row) {
      None
    } else {
      chrono::DateTime::from_timestamp_micros(last_indexed_at_array.value(row))
        .map(|dt| dt.naive_utc())
    };

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

    let status_str = batch
      .column_by_name("status")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid status column".to_string(),
      })?
      .value(row);

    let status = status_str
      .parse::<ProjectStatus>()
      .map_err(|_| lancedb::Error::Runtime {
        message: format!("Invalid project status: {}", status_str),
      })?;

    let deletion_requested_at_array = batch
      .column_by_name("deletion_requested_at")
      .and_then(|col| {
        col
          .as_any()
          .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
      })
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid deletion_requested_at column".to_string(),
      })?;

    let deletion_requested_at = if deletion_requested_at_array.is_null(row) {
      None
    } else {
      chrono::DateTime::from_timestamp_micros(deletion_requested_at_array.value(row))
        .map(|dt| dt.naive_utc())
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
      status,
      deletion_requested_at,
      created_at: chrono::DateTime::from_timestamp_micros(created_at)
        .unwrap_or_default()
        .naive_utc(),
      updated_at: chrono::DateTime::from_timestamp_micros(updated_at)
        .unwrap_or_default()
        .naive_utc(),
      rescan_interval,
      last_indexed_at,
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
    let status_array = StringArray::from(vec![self.status.to_string()]);

    let deletion_requested_at_array = if let Some(dt) = self.deletion_requested_at {
      arrow::array::TimestampMicrosecondArray::from(vec![Some(dt.and_utc().timestamp_micros())])
    } else {
      arrow::array::TimestampMicrosecondArray::from(vec![None as Option<i64>])
    };

    // Convert timestamps to microseconds
    let created_at_us = self.created_at.and_utc().timestamp_micros();
    let updated_at_us = self.updated_at.and_utc().timestamp_micros();
    let created_at_array = arrow::array::TimestampMicrosecondArray::from(vec![created_at_us]);
    let updated_at_array = arrow::array::TimestampMicrosecondArray::from(vec![updated_at_us]);

    let rescan_interval_array = if let Some(interval) = self.rescan_interval {
      DurationMicrosecondArray::from(vec![Some(interval.as_micros() as i64)])
    } else {
      DurationMicrosecondArray::from(vec![None as Option<i64>])
    };

    let last_indexed_at_array = if let Some(dt) = self.last_indexed_at {
      let ts = dt.and_utc().timestamp_micros();
      arrow::array::TimestampMicrosecondArray::from(vec![Some(ts)])
    } else {
      arrow::array::TimestampMicrosecondArray::from(vec![None])
    };

    // Create the record batch
    let batch = RecordBatch::try_new(
      schema.clone(),
      vec![
        Arc::new(id_array),
        Arc::new(name_array),
        Arc::new(directory_array),
        Arc::new(description_array),
        Arc::new(status_array),
        Arc::new(deletion_requested_at_array),
        Arc::new(created_at_array),
        Arc::new(updated_at_array),
        Arc::new(rescan_interval_array),
        Arc::new(last_indexed_at_array),
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
  use tempfile::TempDir;

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

    assert_eq!(schema.fields().len(), 10);

    let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
    assert_eq!(
      field_names,
      vec![
        "id",
        "name",
        "directory",
        "description",
        "status",
        "deletion_requested_at",
        "created_at",
        "updated_at",
        "rescan_interval",
        "last_indexed_at",
      ]
    );

    assert!(!schema.field(0).is_nullable()); // id
    assert!(!schema.field(1).is_nullable()); // name
    assert!(!schema.field(2).is_nullable()); // directory
    assert!(schema.field(3).is_nullable()); // description
    assert!(!schema.field(4).is_nullable()); // status
    assert!(schema.field(5).is_nullable()); // deletion_requested_at
    assert!(!schema.field(6).is_nullable()); // created_at
    assert!(!schema.field(7).is_nullable()); // updated_at
    assert!(schema.field(8).is_nullable()); // rescan_interval
    assert!(schema.field(9).is_nullable()); // last_indexed_at
  }
}
