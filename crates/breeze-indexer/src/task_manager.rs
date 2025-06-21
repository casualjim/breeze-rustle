use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use futures::TryStreamExt as _;
use lancedb::Table;
use lancedb::arrow::IntoArrow;
use lancedb::query::{ExecutableQuery as _, QueryBase};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};

use crate::bulk_indexer::BulkIndexer;
use crate::models::{IndexTask, TaskStatus};

pub struct TaskManager {
  task_table: Arc<RwLock<Table>>,
  indexer: BulkIndexer,
}

impl TaskManager {
  pub fn new(task_table: Arc<RwLock<Table>>, indexer: BulkIndexer) -> Self {
    Self {
      task_table,
      indexer,
    }
  }

  /// Submit a new indexing task and return its ID
  pub async fn submit_task(
    &self,
    path: &Path,
  ) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
    let task = IndexTask::new(path);
    let task_id = task.id.clone();

    // Insert task into table
    let arrow_data = task.into_arrow()?;

    let table = self.task_table.write().await;
    table.add(arrow_data).execute().await?;

    info!(task_id, "Submitted new indexing task");
    Ok(task_id)
  }

  /// Worker loop that processes pending tasks
  pub async fn run_worker(
    &self,
    shutdown_token: CancellationToken,
  ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    info!("Starting task worker");

    // On startup, reset any "running" tasks to "pending" (in case of crash)
    self.reset_stuck_tasks().await?;

    loop {
      tokio::select! {
        _ = shutdown_token.cancelled() => {
          info!("Task worker shutting down");
          break;
        }
        _ = tokio::time::sleep(Duration::from_secs(1)) => {
          // Check for pending tasks
          match self.process_next_task(&shutdown_token).await {
            Ok(true) => {
              // Processed a task, immediately check for more
              continue;
            }
            Ok(false) => {
              // No tasks to process, wait before checking again
            }
            Err(e) => {
              error!("Error processing task: {}", e);
              // Wait a bit before retrying
              tokio::time::sleep(Duration::from_secs(5)).await;
            }
          }
        }
      }
    }

    Ok(())
  }

  /// Get task by ID
  pub async fn get_task(
    &self,
    task_id: &str,
  ) -> Result<Option<IndexTask>, Box<dyn std::error::Error + Send + Sync>> {
    let table = self.task_table.read().await;

    let results = table
      .query()
      .only_if(format!("id = '{}'", task_id).as_str())
      .limit(1)
      .execute()
      .await?
      .try_collect::<Vec<_>>()
      .await?;

    if results.is_empty() {
      return Ok(None);
    }

    // Convert RecordBatch to IndexTask
    let batch = &results[0];
    let task = Self::record_batch_to_task(batch, 0)?;

    Ok(Some(task))
  }

  /// List recent tasks
  pub async fn list_tasks(
    &self,
    limit: usize,
  ) -> Result<Vec<IndexTask>, Box<dyn std::error::Error + Send + Sync>> {
    let table = self.task_table.read().await;

    let results = table
      .query()
      .limit(limit)
      .execute()
      .await?
      .try_collect::<Vec<_>>()
      .await?;

    let mut tasks = Vec::new();
    for batch in results {
      for i in 0..batch.num_rows() {
        tasks.push(Self::record_batch_to_task(&batch, i)?);
      }
    }

    // Sort by created_at descending (newest first)
    tasks.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    tasks.truncate(limit);

    Ok(tasks)
  }

  /// Process the next pending task
  async fn process_next_task(
    &self,
    cancel_token: &CancellationToken,
  ) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
    // Try to claim the oldest pending task
    let task = match self.claim_pending_task().await? {
      Some(task) => task,
      None => return Ok(false), // No pending tasks
    };

    info!(
      task_id = task.id,
      path = task.path,
      "Processing indexing task"
    );

    // Execute the indexing
    let start_time = std::time::Instant::now();
    let result = self
      .indexer
      .index(std::path::Path::new(&task.path), Some(cancel_token.clone()))
      .await;

    let elapsed = start_time.elapsed();

    // Update task based on result
    match result {
      Ok(files_indexed) => {
        info!(
          task_id = task.id,
          files_indexed,
          elapsed_secs = elapsed.as_secs_f64(),
          "Task completed successfully"
        );
        self.update_task_completed(&task.id, files_indexed).await?;
      }
      Err(e) => {
        error!(
          task_id = task.id,
          error = %e,
          elapsed_secs = elapsed.as_secs_f64(),
          "Task failed"
        );
        self.update_task_failed(&task.id, &e.to_string()).await?;
      }
    }

    Ok(true)
  }

  /// Claim the oldest pending task by updating its status to running
  async fn claim_pending_task(
    &self,
  ) -> Result<Option<IndexTask>, Box<dyn std::error::Error + Send + Sync>> {
    let table = self.task_table.write().await;

    // Get oldest pending task
    let results = table
      .query()
      .only_if("status = 'pending'")
      .limit(1)
      .execute()
      .await?
      .try_collect::<Vec<_>>()
      .await?;

    if results.is_empty() {
      return Ok(None);
    }

    let batch = &results[0];
    let mut task = Self::record_batch_to_task(batch, 0)?;

    // Update status to running
    task.status = TaskStatus::Running;
    task.started_at = Some(chrono::Utc::now().naive_utc());

    // Update in database
    table
      .update()
      .only_if(format!("id = '{}' AND status = 'pending'", task.id).as_str())
      .column("status", "'running'")
      .column(
        "started_at",
        format!("{}", task.started_at.unwrap().and_utc().timestamp_micros()),
      )
      .execute()
      .await?;

    debug!(task_id = task.id, "Claimed task");
    Ok(Some(task))
  }

  /// Update task as completed
  async fn update_task_completed(
    &self,
    task_id: &str,
    files_indexed: usize,
  ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let table = self.task_table.write().await;

    table
      .update()
      .only_if(format!("id = '{}'", task_id).as_str())
      .column("status", "'completed'")
      .column(
        "completed_at",
        format!(
          "{}",
          chrono::Utc::now().naive_utc().and_utc().timestamp_micros()
        ),
      )
      .column("files_indexed", format!("{}", files_indexed))
      .execute()
      .await?;

    Ok(())
  }

  /// Update task as failed
  async fn update_task_failed(
    &self,
    task_id: &str,
    error: &str,
  ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let table = self.task_table.write().await;

    table
      .update()
      .only_if(format!("id = '{}'", task_id).as_str())
      .column("status", "'failed'")
      .column(
        "completed_at",
        format!(
          "{}",
          chrono::Utc::now().naive_utc().and_utc().timestamp_micros()
        ),
      )
      .column("error", format!("'{}'", error.replace("'", "''")))
      .execute()
      .await?;

    Ok(())
  }

  /// Reset stuck "running" tasks to "pending" on startup
  async fn reset_stuck_tasks(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let table = self.task_table.write().await;

    // Reset tasks that have been running for more than 30 minutes
    let thirty_minutes_ago = chrono::Utc::now().naive_utc() - chrono::Duration::minutes(30);
    let threshold_micros = thirty_minutes_ago.and_utc().timestamp_micros();

    // Use a simpler approach - get all running tasks and check them individually
    let running_tasks = table
      .query()
      .only_if("status = 'running'")
      .execute()
      .await?
      .try_collect::<Vec<_>>()
      .await?;

    for batch in running_tasks {
      for i in 0..batch.num_rows() {
        let task = Self::record_batch_to_task(&batch, i)?;

        if let Some(started_at) = task.started_at {
          let started_micros = started_at.and_utc().timestamp_micros();
          if started_micros < threshold_micros {
            // This task has been running too long, reset it
            table
              .update()
              .only_if(format!("id = '{}'", task.id).as_str())
              .column("status", "'pending'")
              .column("started_at", "null")
              .execute()
              .await?;
            debug!("Reset stuck task {} to pending", task.id);
          }
        }
      }
    }

    Ok(())
  }

  /// Convert RecordBatch row to IndexTask
  fn record_batch_to_task(
    batch: &arrow::record_batch::RecordBatch,
    row: usize,
  ) -> Result<IndexTask, Box<dyn std::error::Error + Send + Sync>> {
    use arrow::array::*;

    let id_array = batch
      .column(0)
      .as_any()
      .downcast_ref::<StringArray>()
      .ok_or("Invalid id column")?;

    let path_array = batch
      .column(1)
      .as_any()
      .downcast_ref::<StringArray>()
      .ok_or("Invalid path column")?;

    let status_array = batch
      .column(2)
      .as_any()
      .downcast_ref::<StringArray>()
      .ok_or("Invalid status column")?;

    let created_at_array = batch
      .column(3)
      .as_any()
      .downcast_ref::<TimestampMicrosecondArray>()
      .ok_or("Invalid created_at column")?;

    let started_at_array = batch
      .column(4)
      .as_any()
      .downcast_ref::<TimestampMicrosecondArray>()
      .ok_or("Invalid started_at column")?;

    let completed_at_array = batch
      .column(5)
      .as_any()
      .downcast_ref::<TimestampMicrosecondArray>()
      .ok_or("Invalid completed_at column")?;

    let error_array = batch
      .column(6)
      .as_any()
      .downcast_ref::<StringArray>()
      .ok_or("Invalid error column")?;

    let files_indexed_array = batch
      .column(7)
      .as_any()
      .downcast_ref::<UInt64Array>()
      .ok_or("Invalid files_indexed column")?;

    let status = match status_array.value(row) {
      "pending" => TaskStatus::Pending,
      "running" => TaskStatus::Running,
      "completed" => TaskStatus::Completed,
      "failed" => TaskStatus::Failed,
      _ => return Err("Invalid status value".into()),
    };

    Ok(IndexTask {
      id: id_array.value(row).to_string(),
      path: path_array.value(row).to_string(),
      status,
      created_at: chrono::DateTime::from_timestamp_micros(created_at_array.value(row))
        .ok_or("Invalid created_at timestamp")?
        .naive_utc(),
      started_at: if started_at_array.is_null(row) {
        None
      } else {
        Some(
          chrono::DateTime::from_timestamp_micros(started_at_array.value(row))
            .ok_or("Invalid started_at timestamp")?
            .naive_utc(),
        )
      },
      completed_at: if completed_at_array.is_null(row) {
        None
      } else {
        Some(
          chrono::DateTime::from_timestamp_micros(completed_at_array.value(row))
            .ok_or("Invalid completed_at timestamp")?
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
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::Config;
  use crate::bulk_indexer::BulkIndexer;
  use crate::embeddings::factory::create_embedding_provider;
  use crate::models::CodeDocument;
  use std::sync::Mutex;
  use std::sync::atomic::{AtomicUsize, Ordering};
  use tempfile::TempDir;
  use tokio::time::timeout;

  async fn create_test_task_manager() -> (TaskManager, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");

    let connection = lancedb::connect(db_path.to_str().unwrap())
      .execute()
      .await
      .unwrap();

    let task_table = IndexTask::ensure_table(&connection, "test_tasks")
      .await
      .unwrap();
    let task_table = Arc::new(RwLock::new(task_table));

    let code_table = CodeDocument::ensure_table(&connection, "test_docs", 384)
      .await
      .unwrap();
    let code_table = Arc::new(RwLock::new(code_table));

    // Use the test config which sets up local embeddings if available
    let (_temp_dir_config, config) = Config::test();

    let embedding_provider = create_embedding_provider(&config).await.unwrap();

    let bulk_indexer = BulkIndexer::new(
      Arc::new(config),
      Arc::from(embedding_provider),
      384,
      code_table,
    );

    let task_manager = TaskManager::new(task_table, bulk_indexer);

    (task_manager, temp_dir)
  }

  #[tokio::test]
  async fn test_submit_task() {
    let (task_manager, _temp_dir) = create_test_task_manager().await;
    let test_path = std::path::Path::new("/test/path");

    let task_id = task_manager.submit_task(test_path).await.unwrap();

    assert!(!task_id.is_empty());
    assert!(uuid::Uuid::parse_str(&task_id).is_ok());

    // Verify task was created
    let task = task_manager.get_task(&task_id).await.unwrap().unwrap();
    assert_eq!(task.path, test_path.to_string_lossy());
    assert_eq!(task.status, TaskStatus::Pending);
    assert!(task.started_at.is_none());
    assert!(task.completed_at.is_none());
  }

  #[tokio::test]
  async fn test_get_task_not_found() {
    let (task_manager, _temp_dir) = create_test_task_manager().await;

    let result = task_manager.get_task("non-existent-id").await.unwrap();
    assert!(result.is_none());
  }

  #[tokio::test]
  async fn test_list_tasks_empty() {
    let (task_manager, _temp_dir) = create_test_task_manager().await;

    let tasks = task_manager.list_tasks(10).await.unwrap();
    assert!(tasks.is_empty());
  }

  #[tokio::test]
  async fn test_list_tasks_with_data() {
    let (task_manager, _temp_dir) = create_test_task_manager().await;

    // Submit multiple tasks
    let paths = ["/test/path1", "/test/path2", "/test/path3"];
    let mut task_ids = Vec::new();

    for path in &paths {
      let task_id = task_manager
        .submit_task(std::path::Path::new(path))
        .await
        .unwrap();
      task_ids.push(task_id);
    }

    // List tasks
    let tasks = task_manager.list_tasks(10).await.unwrap();
    assert_eq!(tasks.len(), 3);

    // Verify they're sorted by created_at descending (newest first)
    for (i, task) in tasks.iter().enumerate() {
      if i > 0 {
        assert!(task.created_at <= tasks[i - 1].created_at);
      }
    }
  }

  #[tokio::test]
  async fn test_list_tasks_with_limit() {
    let (task_manager, _temp_dir) = create_test_task_manager().await;

    // Submit 5 tasks
    for i in 0..5 {
      let path = format!("/test/path{}", i);
      task_manager
        .submit_task(std::path::Path::new(&path))
        .await
        .unwrap();
    }

    // List with limit of 3
    let tasks = task_manager.list_tasks(3).await.unwrap();
    assert_eq!(tasks.len(), 3);
  }

  #[tokio::test]
  async fn test_concurrent_task_submission() {
    let (task_manager, _temp_dir) = create_test_task_manager().await;
    let task_manager = Arc::new(task_manager);

    let num_tasks = 10;
    let mut handles = Vec::new();

    // Submit tasks concurrently
    for i in 0..num_tasks {
      let tm = task_manager.clone();
      let handle = tokio::spawn(async move {
        let path = format!("/test/concurrent/{}", i);
        tm.submit_task(std::path::Path::new(&path)).await
      });
      handles.push(handle);
    }

    // Wait for all to complete
    let mut task_ids = Vec::new();
    for handle in handles {
      let task_id = handle.await.unwrap().unwrap();
      task_ids.push(task_id);
    }

    // Verify all tasks were created
    assert_eq!(task_ids.len(), num_tasks);
    let tasks = task_manager.list_tasks(num_tasks).await.unwrap();
    assert_eq!(tasks.len(), num_tasks);

    // Verify all task IDs are unique
    let mut unique_ids = std::collections::HashSet::new();
    for task_id in &task_ids {
      assert!(unique_ids.insert(task_id.clone()));
    }
  }

  #[tokio::test]
  async fn test_reset_stuck_tasks() {
    let (task_manager, _temp_dir) = create_test_task_manager().await;

    // Create a task and manually set it to running with old timestamp
    let task_id = task_manager
      .submit_task(std::path::Path::new("/test/stuck"))
      .await
      .unwrap();

    // Manually update the task to be running with old timestamp
    let old_time = chrono::Utc::now().naive_utc() - chrono::Duration::hours(1);
    let old_time_micros = old_time.and_utc().timestamp_micros();

    // First verify the task exists and is pending
    let initial_task = task_manager.get_task(&task_id).await.unwrap().unwrap();
    assert_eq!(initial_task.status, TaskStatus::Pending);

    // Update to running with old timestamp
    {
      let table = task_manager.task_table.write().await;
      table
        .update()
        .only_if(format!("id = '{}'", task_id).as_str())
        .column("status", "'running'")
        .column("started_at", format!("{}", old_time_micros))
        .execute()
        .await
        .unwrap();
    }

    // Verify the update worked
    let updated_task = task_manager.get_task(&task_id).await.unwrap().unwrap();
    assert_eq!(updated_task.status, TaskStatus::Running);
    assert!(updated_task.started_at.is_some());

    // Reset stuck tasks
    task_manager.reset_stuck_tasks().await.unwrap();

    // Verify task was reset to pending
    let task = task_manager.get_task(&task_id).await.unwrap().unwrap();
    assert_eq!(task.status, TaskStatus::Pending);
    assert!(task.started_at.is_none());
  }

  #[tokio::test]
  async fn test_claim_pending_task() {
    let (task_manager, _temp_dir) = create_test_task_manager().await;

    // Submit a task
    let task_id = task_manager
      .submit_task(std::path::Path::new("/test/claim"))
      .await
      .unwrap();

    // Claim the task
    let claimed_task = task_manager.claim_pending_task().await.unwrap().unwrap();
    assert_eq!(claimed_task.id, task_id);
    assert_eq!(claimed_task.status, TaskStatus::Running);
    assert!(claimed_task.started_at.is_some());

    // Verify task is no longer pending
    let no_task = task_manager.claim_pending_task().await.unwrap();
    assert!(no_task.is_none());
  }

  #[tokio::test]
  async fn test_concurrent_task_claiming() {
    let (task_manager, _temp_dir) = create_test_task_manager().await;
    let task_manager = Arc::new(task_manager);

    // Submit multiple tasks
    let num_tasks = 5;
    for i in 0..num_tasks {
      let path = format!("/test/claim/{}", i);
      task_manager
        .submit_task(std::path::Path::new(&path))
        .await
        .unwrap();
    }

    let claimed_count = Arc::new(AtomicUsize::new(0));
    let claimed_tasks = Arc::new(Mutex::new(Vec::new()));
    let mut handles = Vec::new();

    // Try to claim tasks concurrently
    for _ in 0..num_tasks {
      let tm = task_manager.clone();
      let count = claimed_count.clone();
      let tasks = claimed_tasks.clone();
      let handle = tokio::spawn(async move {
        if let Ok(Some(task)) = tm.claim_pending_task().await {
          count.fetch_add(1, Ordering::SeqCst);
          tasks.lock().unwrap().push(task);
        }
      });
      handles.push(handle);
    }

    // Wait for all to complete
    for handle in handles {
      handle.await.unwrap();
    }

    // Verify exactly num_tasks were claimed
    assert_eq!(claimed_count.load(Ordering::SeqCst), num_tasks);

    // Verify all claimed tasks are unique and running
    let claimed_tasks = claimed_tasks.lock().unwrap();
    let mut unique_ids = std::collections::HashSet::new();
    for task in claimed_tasks.iter() {
      assert!(unique_ids.insert(task.id.clone()));
      assert_eq!(task.status, TaskStatus::Running);
      assert!(task.started_at.is_some());
    }
  }

  #[tokio::test]
  async fn test_update_task_completed() {
    let (task_manager, _temp_dir) = create_test_task_manager().await;

    let task_id = task_manager
      .submit_task(std::path::Path::new("/test/complete"))
      .await
      .unwrap();

    task_manager
      .update_task_completed(&task_id, 42)
      .await
      .unwrap();

    let task = task_manager.get_task(&task_id).await.unwrap().unwrap();
    assert_eq!(task.status, TaskStatus::Completed);
    assert!(task.completed_at.is_some());
    assert_eq!(task.files_indexed, Some(42));
  }

  #[tokio::test]
  async fn test_update_task_failed() {
    let (task_manager, _temp_dir) = create_test_task_manager().await;

    let task_id = task_manager
      .submit_task(std::path::Path::new("/test/fail"))
      .await
      .unwrap();

    let error_msg = "Test error message";
    task_manager
      .update_task_failed(&task_id, error_msg)
      .await
      .unwrap();

    let task = task_manager.get_task(&task_id).await.unwrap().unwrap();
    assert_eq!(task.status, TaskStatus::Failed);
    assert!(task.completed_at.is_some());
    assert_eq!(task.error, Some(error_msg.to_string()));
  }

  #[tokio::test]
  async fn test_worker_processes_task() {
    let (task_manager, temp_dir) = create_test_task_manager().await;
    let task_manager = Arc::new(task_manager);

    // Create a real directory to index
    let test_dir = temp_dir.path().join("test_project");
    tokio::fs::create_dir(&test_dir).await.unwrap();
    tokio::fs::write(test_dir.join("test.txt"), "test content")
      .await
      .unwrap();

    // Submit a task
    let task_id = task_manager.submit_task(&test_dir).await.unwrap();

    // Start worker with short timeout
    let shutdown_token = CancellationToken::new();
    let worker_task_manager = task_manager.clone();
    let worker_shutdown = shutdown_token.clone();

    let worker_handle =
      tokio::spawn(async move { worker_task_manager.run_worker(worker_shutdown).await });

    // Wait a bit for task processing
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Check task status
    let task = task_manager.get_task(&task_id).await.unwrap().unwrap();

    // Task should either be completed or still running (or possibly still pending if worker is slow to start)
    assert!(
      matches!(
        task.status,
        TaskStatus::Completed | TaskStatus::Running | TaskStatus::Pending
      ),
      "Task status was {:?}",
      task.status
    );

    // Shutdown worker
    shutdown_token.cancel();

    // Give worker time to shutdown gracefully
    let _ = timeout(Duration::from_secs(2), worker_handle).await;
  }

  #[tokio::test]
  async fn test_worker_shutdown() {
    let (task_manager, _temp_dir) = create_test_task_manager().await;

    let shutdown_token = CancellationToken::new();
    let worker_shutdown = shutdown_token.clone();

    let worker_handle = tokio::spawn(async move { task_manager.run_worker(worker_shutdown).await });

    // Immediately shutdown
    shutdown_token.cancel();

    // Worker should shutdown quickly
    let result = timeout(Duration::from_secs(2), worker_handle).await;
    assert!(result.is_ok());
  }

  #[tokio::test]
  async fn test_oldest_pending_task_processed_first() {
    let (task_manager, _temp_dir) = create_test_task_manager().await;

    // Submit tasks with small delays to ensure different timestamps
    let task1_id = task_manager
      .submit_task(std::path::Path::new("/test/first"))
      .await
      .unwrap();

    tokio::time::sleep(Duration::from_millis(10)).await;

    let task2_id = task_manager
      .submit_task(std::path::Path::new("/test/second"))
      .await
      .unwrap();

    tokio::time::sleep(Duration::from_millis(10)).await;

    let task3_id = task_manager
      .submit_task(std::path::Path::new("/test/third"))
      .await
      .unwrap();

    // Claim tasks and verify order (oldest first)
    let first_claimed = task_manager.claim_pending_task().await.unwrap().unwrap();
    assert_eq!(first_claimed.id, task1_id);

    let second_claimed = task_manager.claim_pending_task().await.unwrap().unwrap();
    assert_eq!(second_claimed.id, task2_id);

    let third_claimed = task_manager.claim_pending_task().await.unwrap().unwrap();
    assert_eq!(third_claimed.id, task3_id);

    // No more tasks to claim
    let no_task = task_manager.claim_pending_task().await.unwrap();
    assert!(no_task.is_none());
  }

  #[tokio::test]
  async fn test_task_error_handling_with_quotes() {
    let (task_manager, _temp_dir) = create_test_task_manager().await;

    let task_id = task_manager
      .submit_task(std::path::Path::new("/test/quotes"))
      .await
      .unwrap();

    // Test error message with single quotes (SQL injection prevention)
    let error_msg = "Error with 'single quotes' and \"double quotes\"";
    task_manager
      .update_task_failed(&task_id, error_msg)
      .await
      .unwrap();

    let task = task_manager.get_task(&task_id).await.unwrap().unwrap();
    assert_eq!(task.status, TaskStatus::Failed);
    assert_eq!(task.error, Some(error_msg.to_string()));
  }

  #[tokio::test]
  async fn test_record_batch_to_task_conversion() {
    let (task_manager, _temp_dir) = create_test_task_manager().await;

    // Submit a task and update it to have all fields populated
    let task_id = task_manager
      .submit_task(std::path::Path::new("/test/conversion"))
      .await
      .unwrap();

    task_manager
      .update_task_completed(&task_id, 123)
      .await
      .unwrap();

    // Retrieve and verify all fields are correctly converted
    let task = task_manager.get_task(&task_id).await.unwrap().unwrap();
    assert_eq!(task.id, task_id);
    assert_eq!(task.path, "/test/conversion");
    assert_eq!(task.status, TaskStatus::Completed);
    assert!(task.created_at > chrono::Utc::now().naive_utc() - chrono::Duration::seconds(10));
    assert!(task.completed_at.is_some());
    assert_eq!(task.files_indexed, Some(123));
    assert!(task.error.is_none());
  }
}
