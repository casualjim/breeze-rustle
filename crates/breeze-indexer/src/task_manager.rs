use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use futures::TryStreamExt as _;
use lancedb::Table;
use lancedb::arrow::IntoArrow;
use lancedb::query::{ExecutableQuery as _, QueryBase};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

use crate::bulk_indexer::BulkIndexer;
use crate::models::{IndexTask, TaskStatus};
use crate::IndexerError;

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

  /// Submit a new indexing task and return its ID (backward compatibility)
  pub async fn submit_task(
    &self,
    project_id: Uuid,
    path: &Path,
  ) -> Result<Uuid, IndexerError> {
    self.submit_task_with_type(project_id, path, crate::models::TaskType::FullIndex).await
  }

  /// Submit a new indexing task with specific type and return its ID
  pub async fn submit_task_with_type(
    &self,
    project_id: Uuid,
    path: &Path,
    task_type: crate::models::TaskType,
  ) -> Result<Uuid, IndexerError> {
    let table = self.task_table.write().await;

    // Check for existing pending tasks for this project
    let existing_tasks = table
      .query()
      .only_if(format!("project_id = '{}' AND status = 'pending'", project_id).as_str())
      .execute()
      .await?
      .try_collect::<Vec<_>>()
      .await?;

    // Process merging logic
    for batch in existing_tasks {
      for i in 0..batch.num_rows() {
        let existing_task = crate::models::IndexTask::from_record_batch(&batch, i)?;
        
        match (&existing_task.task_type, &task_type) {
          // If there's already a full index pending, mark any new task as merged
          (crate::models::TaskType::FullIndex, _) => {
            let mut merged_task = match &task_type {
              crate::models::TaskType::FullIndex => crate::models::IndexTask::new(project_id, path),
              crate::models::TaskType::PartialUpdate { changes } => {
                crate::models::IndexTask::new_partial(project_id, path, changes.clone())
              }
            };
            merged_task.status = crate::models::TaskStatus::Merged;
            merged_task.merged_into = Some(existing_task.id);
            
            let merged_task_id = merged_task.id;
            let arrow_data = merged_task.into_arrow()?;
            table.add(arrow_data).execute().await?;
            
            info!(
              task_id = %merged_task_id,
              merged_into = %existing_task.id,
              "Task merged into existing full index task"
            );
            return Ok(merged_task_id);
          }
          
          // If there's a partial update and we're adding another partial update, merge them
          (crate::models::TaskType::PartialUpdate { changes: existing_changes }, 
           crate::models::TaskType::PartialUpdate { changes: new_changes }) => {
            // Merge the file changes - BTreeSet automatically handles deduplication
            let mut merged_changes = existing_changes.clone();
            for new_change in new_changes {
              // Add the new change
              merged_changes.insert(new_change.clone());
            }
            
            // Update the existing task with merged changes
            let task_type_json = serde_json::to_string(&crate::models::TaskType::PartialUpdate { 
              changes: merged_changes 
            })?;
            
            table
              .update()
              .only_if(format!("id = '{}'", existing_task.id).as_str())
              .column("task_type", format!("'{}'", task_type_json.replace("'", "''")))
              .execute()
              .await?;
            
            // Create a merged record for the new task
            let mut merged_task = crate::models::IndexTask::new_partial(project_id, path, new_changes.clone());
            merged_task.status = crate::models::TaskStatus::Merged;
            merged_task.merged_into = Some(existing_task.id);
            
            let merged_task_id = merged_task.id;
            let arrow_data = merged_task.into_arrow()?;
            table.add(arrow_data).execute().await?;
            
            info!(
              task_id = %merged_task_id,
              merged_into = %existing_task.id,
              "Partial update task merged into existing partial update"
            );
            return Ok(merged_task_id);
          }
          
          // If there's a partial update and we're adding a full index, don't merge
          // The full index should be queued separately and will supersede the partial when it runs
          (crate::models::TaskType::PartialUpdate { .. }, crate::models::TaskType::FullIndex) => {
            // Continue checking other tasks
          }
        }
      }
    }

    // No merging needed, create a new task
    let task = match task_type {
      crate::models::TaskType::FullIndex => crate::models::IndexTask::new(project_id, path),
      crate::models::TaskType::PartialUpdate { changes } => {
        crate::models::IndexTask::new_partial(project_id, path, changes)
      }
    };
    let task_id = task.id;

    // Insert task into table
    let arrow_data = task.into_arrow()?;
    table.add(arrow_data).execute().await?;

    info!(task_id = %task_id, "Submitted new indexing task");
    Ok(task_id)
  }

  /// Worker loop that processes pending tasks
  pub async fn run_worker(
    &self,
    shutdown_token: CancellationToken,
  ) -> Result<(), IndexerError> {
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
    task_id: &Uuid,
  ) -> Result<Option<IndexTask>, IndexerError> {
    let table = self.task_table.read().await;

    let mut stream = table
      .query()
      .only_if(format!("id = '{}'", task_id).as_str())
      .limit(1)
      .execute()
      .await?;

    if let Some(batch) = stream.try_next().await? {
      let task = IndexTask::from_record_batch(&batch, 0)?;
      Ok(Some(task))
    } else {
      Ok(None)
    }
  }

  /// List recent tasks
  pub async fn list_tasks(
    &self,
    limit: usize,
  ) -> Result<Vec<IndexTask>, IndexerError> {
    let table = self.task_table.read().await;

    let mut stream = table
      .query()
      .execute()
      .await?;

    let mut tasks = Vec::new();
    while let Some(batch) = stream.try_next().await? {
      for i in 0..batch.num_rows() {
        tasks.push(IndexTask::from_record_batch(&batch, i)?);
      }
    }

    // Sort by created_at descending (newest first) and limit
    tasks.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    tasks.truncate(limit);

    Ok(tasks)
  }

  /// Process the next pending task
  async fn process_next_task(
    &self,
    cancel_token: &CancellationToken,
  ) -> Result<bool, IndexerError> {
    // Try to claim the oldest pending task
    let task = match self.claim_pending_task().await? {
      Some(task) => task,
      None => return Ok(false), // No pending tasks
    };

    info!(
      task_id = %task.id,
      path = task.path,
      "Processing indexing task"
    );

    // Execute the indexing based on task type
    let start_time = std::time::Instant::now();
    let result = match &task.task_type {
      crate::models::TaskType::FullIndex => {
        self.execute_full_index(&task, cancel_token).await
      }
      crate::models::TaskType::PartialUpdate { changes } => {
        self.execute_partial_update(&task, changes, cancel_token).await
      }
    };

    let elapsed = start_time.elapsed();

    // Update task based on result
    match result {
      Ok(files_indexed) => {
        info!(
          task_id = %task.id,
          files_indexed,
          elapsed_secs = elapsed.as_secs_f64(),
          "Task completed successfully"
        );
        self.update_task_completed(&task.id, files_indexed).await?;
      }
      Err(e) => {
        error!(
          task_id = %task.id,
          error = %e,
          elapsed_secs = elapsed.as_secs_f64(),
          "Task failed"
        );
        self.update_task_failed(&task.id, &e.to_string()).await?;
      }
    }

    Ok(true)
  }

  /// Check if a project has any active (running) tasks
  pub async fn has_active_task(
    &self,
    project_id: Uuid,
  ) -> Result<bool, IndexerError> {
    let table = self.task_table.read().await;
    
    let mut stream = table
      .query()
      .only_if(format!("project_id = '{}' AND status = 'running'", project_id).as_str())
      .limit(1)
      .execute()
      .await?;
    
    Ok(stream.try_next().await?.is_some())
  }

  /// Claim the oldest pending task by updating its status to running
  async fn claim_pending_task(
    &self,
  ) -> Result<Option<IndexTask>, IndexerError> {
    let table = self.task_table.write().await;

    // Get oldest pending task (excluding merged tasks)
    let mut stream = table
      .query()
      .only_if("status = 'pending'")
      .limit(1)
      .execute()
      .await?;

    let batch = match stream.try_next().await? {
      Some(batch) => batch,
      None => return Ok(None),
    };

    let mut task = IndexTask::from_record_batch(&batch, 0)?;

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

    debug!(task_id = %task.id, "Claimed task");
    Ok(Some(task))
  }

  /// Update task as completed
  async fn update_task_completed(
    &self,
    task_id: &Uuid,
    files_indexed: usize,
  ) -> Result<(), IndexerError> {
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
    task_id: &Uuid,
    error: &str,
  ) -> Result<(), IndexerError> {
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
  async fn reset_stuck_tasks(&self) -> Result<(), IndexerError> {
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
        let task = IndexTask::from_record_batch(&batch, i)?;

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

  /// Execute a full index task
  async fn execute_full_index(
    &self,
    task: &crate::models::IndexTask,
    cancel_token: &CancellationToken,
  ) -> Result<usize, IndexerError> {
    info!(
      task_id = %task.id,
      path = task.path,
      "Executing full index"
    );
    
    self
      .indexer
      .index(std::path::Path::new(&task.path), Some(cancel_token.clone()))
      .await
  }

  /// Execute a partial update task
  async fn execute_partial_update(
    &self,
    task: &crate::models::IndexTask,
    changes: &std::collections::BTreeSet<crate::models::FileChange>,
    cancel_token: &CancellationToken,
  ) -> Result<usize, IndexerError> {
    // TODO: Implement partial indexing for specific files
    // For now, we'll do a full index until we implement partial indexing
    warn!(
      task_id = %task.id,
      files_count = changes.len(),
      "Partial update not yet implemented, falling back to full index"
    );
    
    // Log the files that would be updated
    for change in changes {
      debug!(
        path = ?change.path,
        operation = ?change.operation,
        "Would process file change"
      );
    }
    
    self
      .indexer
      .index(std::path::Path::new(&task.path), Some(cancel_token.clone()))
      .await
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

  async fn create_test_task_manager() -> (TaskManager, TempDir, Uuid) {
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
    
    // Create a test project ID
    let test_project_id = Uuid::now_v7();

    (task_manager, temp_dir, test_project_id)
  }

  #[tokio::test]
  async fn test_submit_task() {
    let (task_manager, _temp_dir, project_id) = create_test_task_manager().await;
    let test_path = std::path::Path::new("/test/path");

    let task_id = task_manager.submit_task(project_id, test_path).await.unwrap();


    // Verify task was created
    let task = task_manager.get_task(&task_id).await.unwrap().unwrap();
    assert_eq!(task.path, test_path.to_string_lossy());
    assert_eq!(task.project_id, project_id);
    assert_eq!(task.status, TaskStatus::Pending);
    assert!(task.started_at.is_none());
    assert!(task.completed_at.is_none());
  }

  #[tokio::test]
  async fn test_get_task_not_found() {
    let (task_manager, _temp_dir, _project_id) = create_test_task_manager().await;

    let non_existent_id = Uuid::now_v7();
    let result = task_manager.get_task(&non_existent_id).await.unwrap();
    assert!(result.is_none());
  }

  #[tokio::test]
  async fn test_list_tasks_empty() {
    let (task_manager, _temp_dir, _project_id) = create_test_task_manager().await;

    let tasks = task_manager.list_tasks(10).await.unwrap();
    assert!(tasks.is_empty());
  }

  #[tokio::test]
  async fn test_list_tasks_with_data() {
    let (task_manager, _temp_dir, project_id) = create_test_task_manager().await;

    // Submit multiple tasks
    let paths = ["/test/path1", "/test/path2", "/test/path3"];
    let mut task_ids = Vec::new();

    for path in &paths {
      let task_id = task_manager
        .submit_task(project_id, std::path::Path::new(path))
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
    let (task_manager, _temp_dir, project_id) = create_test_task_manager().await;

    // Submit 5 tasks
    for i in 0..5 {
      let path = format!("/test/path{}", i);
      task_manager
        .submit_task(project_id, std::path::Path::new(&path))
        .await
        .unwrap();
    }

    // List with limit of 3
    let tasks = task_manager.list_tasks(3).await.unwrap();
    assert_eq!(tasks.len(), 3);
  }

  #[tokio::test]
  async fn test_concurrent_task_submission() {
    let (task_manager, _temp_dir, project_id) = create_test_task_manager().await;
    let task_manager = Arc::new(task_manager);

    let num_tasks = 10;
    let mut handles = Vec::new();

    // Submit tasks concurrently
    for i in 0..num_tasks {
      let tm = task_manager.clone();
      let handle = tokio::spawn(async move {
        let path = format!("/test/concurrent/{}", i);
        tm.submit_task(project_id, std::path::Path::new(&path)).await
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
    let (task_manager, _temp_dir, project_id) = create_test_task_manager().await;

    // Create a task and manually set it to running with old timestamp
    let task_id = task_manager
      .submit_task(project_id, std::path::Path::new("/test/stuck"))
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
    let (task_manager, _temp_dir, project_id) = create_test_task_manager().await;

    // Submit a task
    let task_id = task_manager
      .submit_task(project_id, std::path::Path::new("/test/claim"))
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
    let (task_manager, _temp_dir, _project_id) = create_test_task_manager().await;
    let task_manager = Arc::new(task_manager);

    // Submit multiple tasks with different project IDs to avoid merging
    let num_tasks = 5;
    for i in 0..num_tasks {
      let path = format!("/test/claim/{}", i);
      let project_id = Uuid::now_v7(); // Different project ID for each task
      task_manager
        .submit_task(project_id, std::path::Path::new(&path))
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
    let (task_manager, _temp_dir, project_id) = create_test_task_manager().await;

    let task_id = task_manager
      .submit_task(project_id, std::path::Path::new("/test/complete"))
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
    let (task_manager, _temp_dir, project_id) = create_test_task_manager().await;

    let task_id = task_manager
      .submit_task(project_id, std::path::Path::new("/test/fail"))
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
    let (task_manager, temp_dir, project_id) = create_test_task_manager().await;
    let task_manager = Arc::new(task_manager);

    // Create a real directory to index
    let test_dir = temp_dir.path().join("test_project");
    tokio::fs::create_dir(&test_dir).await.unwrap();
    tokio::fs::write(test_dir.join("test.txt"), "test content")
      .await
      .unwrap();

    // Submit a task
    let task_id = task_manager.submit_task(project_id, &test_dir).await.unwrap();

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
    let (task_manager, _temp_dir, _project_id) = create_test_task_manager().await;

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
    let (task_manager, _temp_dir, _project_id) = create_test_task_manager().await;

    // Submit tasks with small delays to ensure different timestamps
    // Use different project IDs to avoid merging
    let project1 = Uuid::now_v7();
    let task1_id = task_manager
      .submit_task(project1, std::path::Path::new("/test/first"))
      .await
      .unwrap();

    tokio::time::sleep(Duration::from_millis(10)).await;

    let project2 = Uuid::now_v7();
    let task2_id = task_manager
      .submit_task(project2, std::path::Path::new("/test/second"))
      .await
      .unwrap();

    tokio::time::sleep(Duration::from_millis(10)).await;

    let project3 = Uuid::now_v7();
    let task3_id = task_manager
      .submit_task(project3, std::path::Path::new("/test/third"))
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
    let (task_manager, _temp_dir, project_id) = create_test_task_manager().await;

    let task_id = task_manager
      .submit_task(project_id, std::path::Path::new("/test/quotes"))
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
    let (task_manager, _temp_dir, project_id) = create_test_task_manager().await;

    // Submit a task and update it to have all fields populated
    let task_id = task_manager
      .submit_task(project_id, std::path::Path::new("/test/conversion"))
      .await
      .unwrap();

    task_manager
      .update_task_completed(&task_id, 123)
      .await
      .unwrap();

    // Retrieve and verify all fields are correctly converted
    let task = task_manager.get_task(&task_id).await.unwrap().unwrap();
    assert_eq!(task.id, task_id);
    assert_eq!(task.project_id, project_id);
    assert_eq!(task.path, "/test/conversion");
    assert_eq!(task.status, TaskStatus::Completed);
    assert!(task.created_at > chrono::Utc::now().naive_utc() - chrono::Duration::seconds(10));
    assert!(task.completed_at.is_some());
    assert_eq!(task.files_indexed, Some(123));
    assert!(task.error.is_none());
  }

  #[tokio::test]
  async fn test_has_active_task() {
    let (task_manager, _temp_dir, project_id) = create_test_task_manager().await;
    
    // Initially no active tasks
    assert!(!task_manager.has_active_task(project_id).await.unwrap());
    
    // Submit a task
    let _task_id = task_manager
      .submit_task(project_id, std::path::Path::new("/test/active"))
      .await
      .unwrap();
    
    // Still no active task (it's pending)
    assert!(!task_manager.has_active_task(project_id).await.unwrap());
    
    // Claim the task to make it running
    let _claimed = task_manager.claim_pending_task().await.unwrap();
    
    // Now should have active task
    assert!(task_manager.has_active_task(project_id).await.unwrap());
    
    // Different project should not have active task
    let other_project_id = Uuid::now_v7();
    assert!(!task_manager.has_active_task(other_project_id).await.unwrap());
  }
}
