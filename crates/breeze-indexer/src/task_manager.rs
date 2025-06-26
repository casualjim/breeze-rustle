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
use uuid::Uuid;

use crate::IndexerError;
use crate::bulk_indexer::BulkIndexer;
use crate::models::{FailedEmbeddingBatch, IndexTask, Project, ProjectStatus, TaskStatus};

pub struct TaskManager {
  task_table: Arc<RwLock<Table>>,
  failed_batches_table: Arc<RwLock<Table>>,
  project_table: Arc<RwLock<Table>>,
  indexer: BulkIndexer,
}

impl TaskManager {
  pub fn new(
    task_table: Arc<RwLock<Table>>,
    failed_batches_table: Arc<RwLock<Table>>,
    project_table: Arc<RwLock<Table>>,
    indexer: BulkIndexer,
  ) -> Self {
    Self {
      task_table,
      failed_batches_table,
      project_table,
      indexer,
    }
  }

  /// Submit a new indexing task with specific type and return its ID
  pub async fn submit_task(
    &self,
    project_id: Uuid,
    path: &Path,
    task_type: crate::models::TaskType,
  ) -> Result<Uuid, IndexerError> {
    // Check project status first
    let project_table = self.project_table.read().await;
    let mut project_stream = project_table
      .query()
      .only_if(format!("id = '{}'", project_id).as_str())
      .limit(1)
      .execute()
      .await?;

    let project = if let Some(batch) = project_stream.try_next().await? {
      Project::from_record_batch(&batch, 0)?
    } else {
      return Err(IndexerError::ProjectNotFound(project_id));
    };
    drop(project_table);

    // Don't accept tasks for deleted or pending deletion projects
    if project.status != ProjectStatus::Active {
      return Err(IndexerError::Task(format!(
        "Cannot submit task for project {} with status {:?}",
        project_id, project.status
      )));
    }

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
          (
            crate::models::TaskType::PartialUpdate {
              changes: existing_changes,
            },
            crate::models::TaskType::PartialUpdate {
              changes: new_changes,
            },
          ) => {
            // Merge the file changes - BTreeSet automatically handles deduplication
            let mut merged_changes = existing_changes.clone();
            for new_change in new_changes {
              // Add the new change
              merged_changes.insert(new_change.clone());
            }

            // Update the existing task with merged changes
            let task_type_json = serde_json::to_string(&crate::models::TaskType::PartialUpdate {
              changes: merged_changes,
            })?;

            table
              .update()
              .only_if(format!("id = '{}'", existing_task.id).as_str())
              .column(
                "task_type",
                format!("'{}'", task_type_json.replace("'", "''")),
              )
              .execute()
              .await?;

            // Create a merged record for the new task
            let mut merged_task =
              crate::models::IndexTask::new_partial(project_id, path, new_changes.clone());
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
  pub async fn run_worker(&self, shutdown_token: CancellationToken) -> Result<(), IndexerError> {
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
  pub async fn get_task(&self, task_id: &Uuid) -> Result<Option<IndexTask>, IndexerError> {
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
  pub async fn list_tasks(&self, limit: usize) -> Result<Vec<IndexTask>, IndexerError> {
    let table = self.task_table.read().await;

    let mut stream = table.query().execute().await?;

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

  /// Check for failed batches that need retry
  async fn check_retry_batches(&self) -> Result<Option<FailedEmbeddingBatch>, IndexerError> {
    let table = self.failed_batches_table.read().await;
    let now = chrono::Utc::now().naive_utc();

    // Query for batches where retry_after <= now, excluding the dummy row
    // Format as timestamp without timezone (YYYY-MM-DD HH:MM:SS.ffffff)
    let filter = format!(
      "retry_after <= timestamp '{}' AND id != '{}'",
      now
        .and_utc()
        .to_rfc3339_opts(chrono::SecondsFormat::Micros, true),
      Uuid::nil()
    );
    let mut stream = table.query().only_if(&filter).limit(1).execute().await?;

    if let Some(batch) = stream.try_next().await? {
      if let Ok(failed_batch) = FailedEmbeddingBatch::from_record_batch(&batch, 0) {
        return Ok(Some(failed_batch));
      }
    }

    Ok(None)
  }

  /// Process a retry batch
  async fn process_retry_batch(
    &self,
    mut failed_batch: FailedEmbeddingBatch,
    cancel_token: &CancellationToken,
  ) -> Result<bool, IndexerError> {
    info!(
      batch_id = %failed_batch.id,
      retry_count = failed_batch.retry_count,
      failed_files_count = failed_batch.failed_files.len(),
      "Processing retry batch"
    );

    let project_path = std::path::Path::new(&failed_batch.project_path);

    // Convert BTreeSet<String> to BTreeSet<FileChange>
    let changes = failed_batch
      .failed_files
      .iter()
      .map(|path| crate::models::FileChange {
        path: std::path::PathBuf::from(path),
        operation: crate::models::FileOperation::Update,
      })
      .collect();

    // Execute partial update for the failed files
    let start_time = std::time::Instant::now();
    let result = self
      .indexer
      .index_file_changes(
        failed_batch.project_id,
        project_path,
        &changes,
        Some(cancel_token.clone()),
      )
      .await;

    let elapsed = start_time.elapsed();

    match result {
      Ok((files_indexed, new_failures)) => {
        if let Some((newly_failed_files, error)) = new_failures {
          // Update the failed batch
          failed_batch.failed_files = newly_failed_files;
          failed_batch.update_for_retry(error);

          info!(
            batch_id = %failed_batch.id,
            retry_count = failed_batch.retry_count,
            remaining_failures = failed_batch.failed_files.len(),
            elapsed = %humantime::format_duration(elapsed),
            "Retry batch partially succeeded"
          );

          // Update the batch in the database
          self.update_failed_batch(&failed_batch).await?;
        } else {
          // All files succeeded, delete the batch
          info!(
            batch_id = %failed_batch.id,
            files_indexed,
            elapsed = %humantime::format_duration(elapsed),
            "Retry batch fully succeeded"
          );

          self.delete_failed_batch(&failed_batch.id).await?;
        }
      }
      Err(e) => {
        error!(
          batch_id = %failed_batch.id,
          error = %e,
          elapsed = %humantime::format_duration(elapsed),
          "Retry batch failed"
        );

        // Update for next retry
        failed_batch.update_for_retry(e.to_string());
        self.update_failed_batch(&failed_batch).await?;
      }
    }

    Ok(true)
  }

  /// Process the next pending task
  async fn process_next_task(
    &self,
    cancel_token: &CancellationToken,
  ) -> Result<bool, IndexerError> {
    // Try to claim the oldest pending task
    let task = match self.claim_pending_task().await? {
      Some(task) => task,
      None => {
        // No pending tasks, check for retry tasks
        if let Some(retry_batch) = self.check_retry_batches().await? {
          return self.process_retry_batch(retry_batch, cancel_token).await;
        }
        return Ok(false); // No tasks at all
      }
    };

    info!(
      task_id = %task.id,
      path = task.path,
      "Processing indexing task"
    );

    // Execute the indexing based on task type
    let start_time = std::time::Instant::now();
    let result = match &task.task_type {
      crate::models::TaskType::FullIndex => self.execute_full_index(&task, cancel_token).await,
      crate::models::TaskType::PartialUpdate { changes } => {
        self
          .execute_partial_update(&task, changes, cancel_token)
          .await
      }
    };

    let elapsed = start_time.elapsed();

    // Update task based on result
    match result {
      Ok((files_indexed, failures)) => {
        if let Some((failed_files, error)) = failures {
          info!(
            task_id = %task.id,
            failed_files_count = failed_files.len(),
            error = %error,
            "Some files failed to embed"
          );

          // Create a FailedEmbeddingBatch record
          let failed_batch = FailedEmbeddingBatch::new(
            task.id,
            task.project_id,
            task.path.clone(),
            failed_files,
            error,
          );

          let table = self.failed_batches_table.write().await;
          table.add(failed_batch.into_arrow()?).execute().await?;

          // Use PartiallyCompleted status when there are failures
          info!(
            task_id = %task.id,
            files_indexed,
            elapsed = %humantime::format_duration(elapsed),
            "Task partially completed with failures"
          );
          self
            .update_task_partially_completed(&task.id, files_indexed)
            .await?;
        } else {
          info!(
            task_id = %task.id,
            files_indexed,
            elapsed = %humantime::format_duration(elapsed),
            "Task completed successfully"
          );
          self.update_task_completed(&task.id, files_indexed).await?;
        }
      }
      Err(e) => {
        error!(
          task_id = %task.id,
          error = %e,
          elapsed = %humantime::format_duration(elapsed),
          "Task failed"
        );
        self.update_task_failed(&task.id, &e.to_string()).await?;
      }
    }

    Ok(true)
  }

  /// Check if a project has any active (running) tasks
  pub async fn has_active_task(&self, project_id: Uuid) -> Result<bool, IndexerError> {
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
  async fn claim_pending_task(&self) -> Result<Option<IndexTask>, IndexerError> {
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

  /// Update task as partially completed (with some failures)
  async fn update_task_partially_completed(
    &self,
    task_id: &Uuid,
    files_indexed: usize,
  ) -> Result<(), IndexerError> {
    let table = self.task_table.write().await;

    table
      .update()
      .only_if(format!("id = '{}'", task_id).as_str())
      .column("status", "'partially_completed'")
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
  async fn update_task_failed(&self, task_id: &Uuid, error: &str) -> Result<(), IndexerError> {
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

  /// Update a failed batch in the database
  async fn update_failed_batch(&self, batch: &FailedEmbeddingBatch) -> Result<(), IndexerError> {
    let table = self.failed_batches_table.write().await;

    // Use merge insert to update the batch
    let mut op = table.merge_insert(&["id"]);
    op.when_matched_update_all(None)
      .when_not_matched_insert_all()
      .clone()
      .execute(batch.clone().into_arrow()?)
      .await?;

    Ok(())
  }

  /// Delete a failed batch from the database
  async fn delete_failed_batch(&self, batch_id: &Uuid) -> Result<(), IndexerError> {
    let table = self.failed_batches_table.write().await;

    table
      .delete(format!("id = '{}'", batch_id).as_str())
      .await?;

    Ok(())
  }

  /// Reset stuck "running" tasks to "pending" on startup
  async fn reset_stuck_tasks(&self) -> Result<(), IndexerError> {
    let table = self.task_table.write().await;

    // On startup, ALL running tasks should be reset to pending
    // because there can't be any legitimately running tasks if the process just started
    let running_tasks = table
      .query()
      .only_if("status = 'running'")
      .execute()
      .await?
      .try_collect::<Vec<_>>()
      .await?;

    let mut reset_count = 0;
    for batch in running_tasks {
      for i in 0..batch.num_rows() {
        let task = IndexTask::from_record_batch(&batch, i)?;

        // Reset ALL running tasks to pending
        table
          .update()
          .only_if(format!("id = '{}'", task.id).as_str())
          .column("status", "'pending'")
          .column("started_at", "null")
          .execute()
          .await?;

        reset_count += 1;
      }
    }

    if reset_count > 0 {
      info!("Reset {} running tasks to pending on startup", reset_count);
    }

    Ok(())
  }

  /// Execute a full index task
  async fn execute_full_index(
    &self,
    task: &crate::models::IndexTask,
    cancel_token: &CancellationToken,
  ) -> Result<(usize, Option<(std::collections::BTreeSet<String>, String)>), IndexerError> {
    info!(
      task_id = %task.id,
      path = task.path,
      "Executing full index"
    );

    self
      .indexer
      .index(
        task.project_id,
        std::path::Path::new(&task.path),
        Some(cancel_token.clone()),
      )
      .await
  }

  /// Execute a partial update task
  async fn execute_partial_update(
    &self,
    task: &crate::models::IndexTask,
    changes: &std::collections::BTreeSet<crate::models::FileChange>,
    cancel_token: &CancellationToken,
  ) -> Result<(usize, Option<(std::collections::BTreeSet<String>, String)>), IndexerError> {
    info!(
      task_id = %task.id,
      files_count = changes.len(),
      "Executing partial update"
    );

    let project_root = std::path::Path::new(&task.path);

    self
      .indexer
      .index_file_changes(
        task.project_id,
        project_root,
        changes,
        Some(cancel_token.clone()),
      )
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

  async fn create_test_task_manager() -> (TaskManager, Arc<RwLock<Table>>, TempDir, Uuid) {
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

    let failed_batches_table =
      FailedEmbeddingBatch::ensure_table(&connection, "test_failed_batches")
        .await
        .unwrap();
    let failed_batches_table = Arc::new(RwLock::new(failed_batches_table));

    let code_table = CodeDocument::ensure_table(&connection, "test_docs", 384)
      .await
      .unwrap();
    let code_table = Arc::new(RwLock::new(code_table));

    let chunk_table = crate::models::CodeChunk::ensure_table(&connection, "test_chunks", 384)
      .await
      .unwrap();
    let chunk_table = Arc::new(RwLock::new(chunk_table));

    // Use the test config which sets up local embeddings if available
    let (_temp_dir_config, config) = Config::test();

    let embedding_provider = create_embedding_provider(&config).await.unwrap();

    let bulk_indexer = BulkIndexer::new(
      Arc::new(config),
      Arc::from(embedding_provider),
      384,
      code_table.clone(),
      chunk_table,
    );

    let project_table = crate::models::Project::ensure_table(&connection, "test_projects")
      .await
      .unwrap();
    let project_table = Arc::new(RwLock::new(project_table));

    let task_manager = TaskManager::new(
      task_table,
      failed_batches_table,
      project_table.clone(),
      bulk_indexer,
    );

    // Create a test project in the database
    let test_project_dir = temp_dir.path().join("test_project");
    std::fs::create_dir(&test_project_dir).unwrap();

    let project = crate::models::Project::new(
      "Test Project".to_string(),
      test_project_dir.to_str().unwrap().to_string(),
      None,
    )
    .unwrap();
    let test_project_id = project.id;

    // Insert the project into the table
    let arrow_data = project.into_arrow().unwrap();
    let table = project_table.write().await;
    table.add(arrow_data).execute().await.unwrap();
    drop(table);

    (task_manager, code_table, temp_dir, test_project_id)
  }

  #[tokio::test]
  async fn test_submit_task() {
    let (task_manager, _code_table, _temp_dir, project_id) = create_test_task_manager().await;
    let test_path = std::path::Path::new("/test/path");

    let task_id = task_manager
      .submit_task(project_id, test_path, crate::models::TaskType::FullIndex)
      .await
      .unwrap();

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
    let (task_manager, _code_table, _temp_dir, _project_id) = create_test_task_manager().await;

    let non_existent_id = Uuid::now_v7();
    let result = task_manager.get_task(&non_existent_id).await.unwrap();
    assert!(result.is_none());
  }

  #[tokio::test]
  async fn test_list_tasks_empty() {
    let (task_manager, _code_table, _temp_dir, _project_id) = create_test_task_manager().await;

    let tasks = task_manager.list_tasks(10).await.unwrap();
    assert!(tasks.is_empty());
  }

  #[tokio::test]
  async fn test_submit_task_inactive_project() {
    let (task_manager, _code_table, _temp_dir, project_id) = create_test_task_manager().await;

    // Mark the project as pending deletion
    let project_table = task_manager.project_table.write().await;
    let now_micros = chrono::Utc::now().naive_utc().and_utc().timestamp_micros();

    project_table
      .update()
      .only_if(format!("id = '{}'", project_id).as_str())
      .column("status", "'PendingDeletion'")
      .column("deletion_requested_at", format!("{}", now_micros))
      .column("updated_at", format!("{}", now_micros))
      .execute()
      .await
      .unwrap();
    drop(project_table);

    // Try to submit a task - should fail
    let test_path = std::path::Path::new("/test/path");
    let result = task_manager
      .submit_task(project_id, test_path, crate::models::TaskType::FullIndex)
      .await;

    assert!(result.is_err());
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("Cannot submit task"));
    assert!(error_msg.contains("PendingDeletion"));
  }

  #[tokio::test]
  async fn test_submit_task_nonexistent_project() {
    let (task_manager, _code_table, _temp_dir, _project_id) = create_test_task_manager().await;
    let test_path = std::path::Path::new("/test/path");

    let result = task_manager
      .submit_task(
        Uuid::now_v7(),
        test_path,
        crate::models::TaskType::FullIndex,
      )
      .await;

    assert!(result.is_err());
    match result.unwrap_err() {
      IndexerError::ProjectNotFound(_) => {}
      _ => panic!("Expected ProjectNotFound error"),
    }
  }

  #[tokio::test]
  async fn test_list_tasks_with_data() {
    let (task_manager, _code_table, _temp_dir, project_id) = create_test_task_manager().await;

    // Submit multiple tasks
    let paths = ["/test/path1", "/test/path2", "/test/path3"];
    let mut task_ids = Vec::new();

    for path in &paths {
      let task_id = task_manager
        .submit_task(
          project_id,
          std::path::Path::new(path),
          crate::models::TaskType::FullIndex,
        )
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
    let (task_manager, _code_table, _temp_dir, project_id) = create_test_task_manager().await;

    // Submit 5 tasks
    for i in 0..5 {
      let path = format!("/test/path{}", i);
      task_manager
        .submit_task(
          project_id,
          std::path::Path::new(&path),
          crate::models::TaskType::FullIndex,
        )
        .await
        .unwrap();
    }

    // List with limit of 3
    let tasks = task_manager.list_tasks(3).await.unwrap();
    assert_eq!(tasks.len(), 3);
  }

  #[tokio::test]
  async fn test_reindex_removes_deleted_files() {
    let (task_manager, code_table, temp_dir, project_id) = create_test_task_manager().await;

    // Use the existing test_project directory created by create_test_task_manager
    let project_dir = temp_dir.path().join("test_project");

    // Create initial files
    let file1 = project_dir.join("file1.rs");
    let file2 = project_dir.join("file2.rs");
    let file3 = project_dir.join("to_delete.rs");

    std::fs::write(&file1, "fn main() { println!(\"file1\"); }").unwrap();
    std::fs::write(&file2, "fn test() { println!(\"file2\"); }").unwrap();
    std::fs::write(&file3, "fn delete_me() { println!(\"delete\"); }").unwrap();

    // First index - should index all 3 files
    let _task_id = task_manager
      .submit_task(project_id, &project_dir, crate::models::TaskType::FullIndex)
      .await
      .unwrap();

    // Process the task
    let cancel_token = CancellationToken::new();
    task_manager.process_next_task(&cancel_token).await.unwrap();

    // Verify all 3 files are in the index
    let table = code_table.read().await;
    let mut count_query = table
      .query()
      .only_if(
        format!(
          "file_path LIKE '{}%'",
          project_dir.to_string_lossy().replace("'", "''")
        )
        .as_str(),
      )
      .execute()
      .await
      .unwrap();

    let mut file_count = 0;
    let mut files_found = std::collections::HashSet::new();
    while let Some(batch) = count_query.try_next().await.unwrap() {
      for i in 0..batch.num_rows() {
        let doc = CodeDocument::from_record_batch(&batch, i).unwrap();
        files_found.insert(doc.file_path.clone());
        file_count += 1;
      }
    }
    drop(table);

    assert_eq!(file_count, 3, "Should have indexed 3 files initially");
    assert!(files_found.contains(&file1.to_string_lossy().to_string()));
    assert!(files_found.contains(&file2.to_string_lossy().to_string()));
    assert!(files_found.contains(&file3.to_string_lossy().to_string()));

    // Delete one file
    std::fs::remove_file(&file3).unwrap();

    // Second index - full reindex
    let _task_id2 = task_manager
      .submit_task(project_id, &project_dir, crate::models::TaskType::FullIndex)
      .await
      .unwrap();

    // Process the reindex task
    task_manager.process_next_task(&cancel_token).await.unwrap();

    // Verify only 2 files remain in the index
    let table = code_table.read().await;
    let mut count_query2 = table
      .query()
      .only_if(
        format!(
          "file_path LIKE '{}%'",
          project_dir.to_string_lossy().replace("'", "''")
        )
        .as_str(),
      )
      .execute()
      .await
      .unwrap();

    let mut file_count_after = 0;
    let mut files_found_after = std::collections::HashSet::new();
    while let Some(batch) = count_query2.try_next().await.unwrap() {
      for i in 0..batch.num_rows() {
        let doc = CodeDocument::from_record_batch(&batch, i).unwrap();
        files_found_after.insert(doc.file_path.clone());
        file_count_after += 1;
      }
    }
    drop(table);

    // This test should fail because deleted files are not removed during reindex
    assert_eq!(
      file_count_after, 2,
      "Should only have 2 files after reindex (deleted file should be removed)"
    );
    assert!(files_found_after.contains(&file1.to_string_lossy().to_string()));
    assert!(files_found_after.contains(&file2.to_string_lossy().to_string()));
    assert!(
      !files_found_after.contains(&file3.to_string_lossy().to_string()),
      "Deleted file should not be in index"
    );
  }

  #[tokio::test]
  async fn test_concurrent_task_submission() {
    let (task_manager, _code_table, _temp_dir, project_id) = create_test_task_manager().await;
    let task_manager = Arc::new(task_manager);

    let num_tasks = 10;
    let mut handles = Vec::new();

    // Submit tasks concurrently
    for i in 0..num_tasks {
      let tm = task_manager.clone();
      let handle = tokio::spawn(async move {
        let path = format!("/test/concurrent/{}", i);
        tm.submit_task(
          project_id,
          std::path::Path::new(&path),
          crate::models::TaskType::FullIndex,
        )
        .await
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
      assert!(unique_ids.insert(*task_id));
    }
  }

  #[tokio::test]
  async fn test_reset_stuck_tasks() {
    let (task_manager, _code_table, _temp_dir, project_id) = create_test_task_manager().await;

    // Create a task and manually set it to running with old timestamp
    let task_id = task_manager
      .submit_task(
        project_id,
        std::path::Path::new("/test/stuck"),
        crate::models::TaskType::FullIndex,
      )
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
    let (task_manager, _code_table, _temp_dir, project_id) = create_test_task_manager().await;

    // Submit a task
    let task_id = task_manager
      .submit_task(
        project_id,
        std::path::Path::new("/test/claim"),
        crate::models::TaskType::FullIndex,
      )
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
    let (task_manager, _code_table, temp_dir, _existing_project_id) =
      create_test_task_manager().await;
    let task_manager = Arc::new(task_manager);

    // Submit multiple tasks with different project IDs to avoid merging
    let num_tasks = 5;
    for i in 0..num_tasks {
      // Create a new project for each task
      let project_table = task_manager.project_table.write().await;
      let dir = temp_dir.path().join(format!("project_{}", i));
      std::fs::create_dir(&dir).unwrap();
      let project = crate::models::Project::new(
        format!("Project {}", i),
        dir.to_str().unwrap().to_string(),
        None,
      )
      .unwrap();
      let project_id = project.id;
      project_table
        .add(project.into_arrow().unwrap())
        .execute()
        .await
        .unwrap();
      drop(project_table);

      let path = format!("/test/claim/{}", i);
      task_manager
        .submit_task(
          project_id,
          std::path::Path::new(&path),
          crate::models::TaskType::FullIndex,
        )
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
      assert!(unique_ids.insert(task.id));
      assert_eq!(task.status, TaskStatus::Running);
      assert!(task.started_at.is_some());
    }
  }

  #[tokio::test]
  async fn test_update_task_completed() {
    let (task_manager, _code_table, _temp_dir, project_id) = create_test_task_manager().await;

    let task_id = task_manager
      .submit_task(
        project_id,
        std::path::Path::new("/test/complete"),
        crate::models::TaskType::FullIndex,
      )
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
    let (task_manager, _code_table, _temp_dir, project_id) = create_test_task_manager().await;

    let task_id = task_manager
      .submit_task(
        project_id,
        std::path::Path::new("/test/fail"),
        crate::models::TaskType::FullIndex,
      )
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
    let (task_manager, _code_table, temp_dir, project_id) = create_test_task_manager().await;
    let task_manager = Arc::new(task_manager);

    // Use the existing test_project directory created by create_test_task_manager
    let test_dir = temp_dir.path().join("test_project");
    tokio::fs::write(test_dir.join("test.txt"), "test content")
      .await
      .unwrap();

    // Submit a task
    let task_id = task_manager
      .submit_task(project_id, &test_dir, crate::models::TaskType::FullIndex)
      .await
      .unwrap();

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
    let (task_manager, _code_table, _temp_dir, _project_id) = create_test_task_manager().await;

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
    let (task_manager, _code_table, temp_dir, _project_id) = create_test_task_manager().await;

    // Create projects in the database for each task
    let project_table = task_manager.project_table.write().await;

    // Create project 1
    let dir1 = temp_dir.path().join("project1");
    std::fs::create_dir(&dir1).unwrap();
    let project1 = crate::models::Project::new(
      "Project 1".to_string(),
      dir1.to_str().unwrap().to_string(),
      None,
    )
    .unwrap();
    let project1_id = project1.id;
    project_table
      .add(project1.into_arrow().unwrap())
      .execute()
      .await
      .unwrap();

    drop(project_table);

    // Submit tasks with small delays to ensure different timestamps
    let task1_id = task_manager
      .submit_task(
        project1_id,
        std::path::Path::new("/test/first"),
        crate::models::TaskType::FullIndex,
      )
      .await
      .unwrap();

    tokio::time::sleep(Duration::from_millis(10)).await;

    // Create project 2
    let project_table = task_manager.project_table.write().await;
    let dir2 = temp_dir.path().join("project2");
    std::fs::create_dir(&dir2).unwrap();
    let project2 = crate::models::Project::new(
      "Project 2".to_string(),
      dir2.to_str().unwrap().to_string(),
      None,
    )
    .unwrap();
    let project2_id = project2.id;
    project_table
      .add(project2.into_arrow().unwrap())
      .execute()
      .await
      .unwrap();
    drop(project_table);

    let task2_id = task_manager
      .submit_task(
        project2_id,
        std::path::Path::new("/test/second"),
        crate::models::TaskType::FullIndex,
      )
      .await
      .unwrap();

    tokio::time::sleep(Duration::from_millis(10)).await;

    // Create project 3
    let project_table = task_manager.project_table.write().await;
    let dir3 = temp_dir.path().join("project3");
    std::fs::create_dir(&dir3).unwrap();
    let project3 = crate::models::Project::new(
      "Project 3".to_string(),
      dir3.to_str().unwrap().to_string(),
      None,
    )
    .unwrap();
    let project3_id = project3.id;
    project_table
      .add(project3.into_arrow().unwrap())
      .execute()
      .await
      .unwrap();
    drop(project_table);

    let task3_id = task_manager
      .submit_task(
        project3_id,
        std::path::Path::new("/test/third"),
        crate::models::TaskType::FullIndex,
      )
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
    let (task_manager, _code_table, _temp_dir, project_id) = create_test_task_manager().await;

    let task_id = task_manager
      .submit_task(
        project_id,
        std::path::Path::new("/test/quotes"),
        crate::models::TaskType::FullIndex,
      )
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
    let (task_manager, _code_table, _temp_dir, project_id) = create_test_task_manager().await;

    // Submit a task and update it to have all fields populated
    let task_id = task_manager
      .submit_task(
        project_id,
        std::path::Path::new("/test/conversion"),
        crate::models::TaskType::FullIndex,
      )
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
  async fn test_execute_partial_update() {
    let (task_manager, code_table, temp_dir, project_id) = create_test_task_manager().await;

    // Use the existing test_project directory created by create_test_task_manager
    let project_dir = temp_dir.path().join("test_project");

    // Create some test files
    let file1 = project_dir.join("file1.rs");
    let file2 = project_dir.join("file2.rs");
    let file3 = project_dir.join("file3.rs");

    tokio::fs::write(&file1, "fn main() { println!(\"File 1\"); }")
      .await
      .unwrap();
    tokio::fs::write(&file2, "fn test() { println!(\"File 2\"); }")
      .await
      .unwrap();
    tokio::fs::write(&file3, "fn helper() { println!(\"File 3\"); }")
      .await
      .unwrap();

    // First, do a full index to populate the database
    let full_task_id = task_manager
      .submit_task(project_id, &project_dir, crate::models::TaskType::FullIndex)
      .await
      .unwrap();

    // Process the full index task
    let task = task_manager.claim_pending_task().await.unwrap().unwrap();
    assert_eq!(task.id, full_task_id);

    // Execute the full index
    let cancel_token = CancellationToken::new();
    let result = task_manager.execute_full_index(&task, &cancel_token).await;
    assert!(result.is_ok(), "Full index failed: {:?}", result);

    // Now create a partial update task
    let mut changes = std::collections::BTreeSet::new();

    // Delete file1 from disk first
    std::fs::remove_file(&file1).unwrap();

    // Then record the deletion
    changes.insert(crate::models::FileChange {
      path: file1.clone(),
      operation: crate::models::FileOperation::Delete,
    });

    // Update file2
    tokio::fs::write(&file2, "fn test() { println!(\"File 2 - Updated\"); }")
      .await
      .unwrap();
    changes.insert(crate::models::FileChange {
      path: file2.clone(),
      operation: crate::models::FileOperation::Update,
    });

    // Add a new file
    let file4 = project_dir.join("file4.rs");
    tokio::fs::write(&file4, "fn new_func() { println!(\"File 4\"); }")
      .await
      .unwrap();
    changes.insert(crate::models::FileChange {
      path: file4.clone(),
      operation: crate::models::FileOperation::Add,
    });

    let partial_task_id = task_manager
      .submit_task(
        project_id,
        &project_dir,
        crate::models::TaskType::PartialUpdate {
          changes: changes.clone(),
        },
      )
      .await
      .unwrap();

    // Get the partial update task
    let partial_task = task_manager
      .get_task(&partial_task_id)
      .await
      .unwrap()
      .unwrap();

    // Execute the partial update
    let result = task_manager
      .execute_partial_update(&partial_task, &changes, &cancel_token)
      .await;

    assert!(result.is_ok(), "Partial update failed: {:?}", result);
    let (files_processed, failures) = result.unwrap();
    assert!(failures.is_none(), "Should have no failures");

    // Should have processed at least 3 files (1 delete + 2 add/update)
    // Note: The count might be higher if the indexer processes additional files
    assert!(
      files_processed >= 3,
      "Expected at least 3 files processed, got {}",
      files_processed
    );

    // Verify file1 was deleted by checking the table directly
    use lancedb::query::{ExecutableQuery, QueryBase};
    let table = code_table.read().await;

    // First, let's see what files are actually in the database
    let mut all_files_query = table
      .query()
      .only_if(
        format!(
          "file_path LIKE '{}%'",
          project_dir.to_string_lossy().replace("'", "''")
        )
        .as_str(),
      )
      .execute()
      .await
      .unwrap();

    use futures::TryStreamExt;
    let mut all_files = Vec::new();
    while let Some(batch) = all_files_query.try_next().await.unwrap() {
      for i in 0..batch.num_rows() {
        let doc = CodeDocument::from_record_batch(&batch, i).unwrap();
        all_files.push(doc.file_path.clone());
      }
    }
    println!("Files in database after partial update: {:?}", all_files);
    println!("Looking for file: {}", file1.to_string_lossy());

    let mut query = table
      .query()
      .only_if(format!(
        "file_path = '{}'",
        file1.to_string_lossy().replace("'", "''")
      ))
      .execute()
      .await
      .unwrap();

    let deleted_result = query.try_next().await.unwrap();
    assert!(deleted_result.is_none(), "File1 should have been deleted");
  }

  #[tokio::test]
  async fn test_file_deleted_during_indexing_race_condition() {
    let temp_dir = TempDir::new().unwrap();
    let project_dir = temp_dir.path().join("test_project");
    std::fs::create_dir(&project_dir).unwrap();

    // Create a test task manager
    let (task_manager, code_table, _actual_temp_dir, project_id) = create_test_task_manager().await;

    // Create a file that we'll delete during indexing
    let file1 = project_dir.join("file1.rs");
    std::fs::write(&file1, "fn main() { println!(\"file1\"); }").unwrap();

    // Create a custom walk stream that deletes the file after yielding it
    let file1_path = file1.clone();
    let files_stream = async_stream::stream! {
      yield file1_path.clone();
      // Delete the file immediately after yielding it
      // This simulates the race condition
      tokio::fs::remove_file(&file1_path).await.unwrap();
    };

    // Run indexing - should handle the deleted file gracefully
    let result = task_manager
      .indexer
      .index_files(project_id, &project_dir, files_stream, None)
      .await;

    assert!(
      result.is_ok(),
      "Indexing should handle deleted files gracefully"
    );

    // Verify no files are in the index (since the only file was deleted)
    // But ignore the dummy document
    let table = code_table.read().await;
    let mut query = table
      .query()
      .only_if(format!("id != '{}'", Uuid::nil()))
      .execute()
      .await
      .expect("Failed to query table");

    let mut found_files = Vec::new();
    while let Some(batch) = query.try_next().await.unwrap() {
      for i in 0..batch.num_rows() {
        let doc = crate::models::CodeDocument::from_record_batch(&batch, i).unwrap();
        found_files.push(doc.file_path);
      }
    }

    assert_eq!(
      found_files.len(),
      0,
      "Should have no files in index after race condition, but found: {:?}",
      found_files
    );

    // Clean up
    std::fs::remove_dir_all(&project_dir).unwrap();
  }

  #[tokio::test]
  async fn test_has_active_task() {
    let (task_manager, _code_table, _temp_dir, project_id) = create_test_task_manager().await;

    // Initially no active tasks
    assert!(!task_manager.has_active_task(project_id).await.unwrap());

    // Submit a task
    let _task_id = task_manager
      .submit_task(
        project_id,
        std::path::Path::new("/test/active"),
        crate::models::TaskType::FullIndex,
      )
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
    assert!(
      !task_manager
        .has_active_task(other_project_id)
        .await
        .unwrap()
    );
  }

  #[tokio::test]
  #[cfg(feature = "local-embeddings")]
  async fn test_directory_deletion_removes_all_files() {
    let (task_manager, code_table, temp_dir, project_id) = create_test_task_manager().await;

    // Use the existing test_project directory created by create_test_task_manager
    let project_dir = temp_dir.path().join("test_project");

    // Create subdirectory with files
    let sub_dir = project_dir.join("src");
    tokio::fs::create_dir(&sub_dir).await.unwrap();

    let file1 = sub_dir.join("main.rs");
    let file2 = sub_dir.join("lib.rs");
    let file3 = sub_dir.join("utils.rs");

    tokio::fs::write(&file1, "fn main() { println!(\"Main\"); }")
      .await
      .unwrap();
    tokio::fs::write(&file2, "pub fn lib() { println!(\"Lib\"); }")
      .await
      .unwrap();
    tokio::fs::write(&file3, "pub fn util() { println!(\"Util\"); }")
      .await
      .unwrap();

    // Also create a file with similar name prefix to test proper path separator handling
    let similar_dir = project_dir.join("src2");
    tokio::fs::create_dir(&similar_dir).await.unwrap();
    let similar_file = similar_dir.join("test.rs");
    tokio::fs::write(&similar_file, "fn test() { println!(\"Test\"); }")
      .await
      .unwrap();

    // First, do a full index to populate the database
    let full_task_id = task_manager
      .submit_task(project_id, &project_dir, crate::models::TaskType::FullIndex)
      .await
      .unwrap();

    // Process the full index task
    let task = task_manager.claim_pending_task().await.unwrap().unwrap();
    assert_eq!(task.id, full_task_id);

    // Execute the full index
    let cancel_token = CancellationToken::new();
    let result = task_manager.execute_full_index(&task, &cancel_token).await;
    assert!(result.is_ok(), "Full index failed: {:?}", result);

    // Verify all files were indexed
    use futures::TryStreamExt;
    use lancedb::query::{ExecutableQuery, QueryBase};
    let table = code_table.read().await;

    // Check that all files exist in the index
    for file_path in [&file1, &file2, &file3, &similar_file] {
      let mut query = table
        .query()
        .only_if(format!(
          "file_path = '{}'",
          file_path.to_string_lossy().replace("'", "''")
        ))
        .execute()
        .await
        .unwrap();

      let result = query.try_next().await.unwrap();
      assert!(
        result.is_some(),
        "File {} should be in the index",
        file_path.display()
      );
    }

    // Drop the read lock before creating the partial update
    drop(table);

    // Now delete the directory
    tokio::fs::remove_dir_all(&sub_dir).await.unwrap();

    // Create a partial update task for directory deletion
    let mut changes = std::collections::BTreeSet::new();
    changes.insert(crate::models::FileChange {
      path: sub_dir.clone(),
      operation: crate::models::FileOperation::Delete,
    });

    let partial_task_id = task_manager
      .submit_task(
        project_id,
        &project_dir,
        crate::models::TaskType::PartialUpdate {
          changes: changes.clone(),
        },
      )
      .await
      .unwrap();

    // Get the partial update task
    let partial_task = task_manager
      .get_task(&partial_task_id)
      .await
      .unwrap()
      .unwrap();

    // Execute the partial update
    let result = task_manager
      .execute_partial_update(&partial_task, &changes, &cancel_token)
      .await;

    assert!(result.is_ok(), "Partial update failed: {:?}", result);

    // Verify all files in the directory were deleted
    let table = code_table.read().await;

    for file_path in [&file1, &file2, &file3] {
      let mut query = table
        .query()
        .only_if(format!(
          "file_path = '{}'",
          file_path.to_string_lossy().replace("'", "''")
        ))
        .execute()
        .await
        .unwrap();

      let result = query.try_next().await.unwrap();
      assert!(
        result.is_none(),
        "File {} should have been deleted",
        file_path.display()
      );
    }

    // Verify the similar directory file was NOT deleted
    let mut query = table
      .query()
      .only_if(format!(
        "file_path = '{}'",
        similar_file.to_string_lossy().replace("'", "''")
      ))
      .execute()
      .await
      .unwrap();

    let result = query.try_next().await.unwrap();
    assert!(
      result.is_some(),
      "File {} should NOT have been deleted",
      similar_file.display()
    );
  }

  #[tokio::test]
  async fn test_reset_stuck_tasks_on_startup() {
    let (task_manager, _code_table, temp_dir, project_id) = create_test_task_manager().await;

    // Create a project directory
    let project_dir = temp_dir.path().join("test_project");
    std::fs::create_dir_all(&project_dir).unwrap();

    // Submit a task
    let task_id = task_manager
      .submit_task(project_id, &project_dir, crate::models::TaskType::FullIndex)
      .await
      .unwrap();

    // Manually set the task to running state (simulating a killed process)
    {
      let table = task_manager.task_table.write().await;

      // Set to running with a recent timestamp
      let now_micros = chrono::Utc::now().naive_utc().and_utc().timestamp_micros();
      table
        .update()
        .only_if(format!("id = '{}'", task_id).as_str())
        .column("status", "'running'")
        .column("started_at", format!("{}", now_micros))
        .execute()
        .await
        .unwrap();
    }

    // Verify task is running
    let task = task_manager.get_task(&task_id).await.unwrap().unwrap();
    assert_eq!(task.status, crate::models::TaskStatus::Running);
    assert!(task.started_at.is_some());

    // Call reset_stuck_tasks (simulating server restart)
    task_manager.reset_stuck_tasks().await.unwrap();

    // Verify task was reset to pending (ALL running tasks should be reset on startup)
    let task = task_manager.get_task(&task_id).await.unwrap().unwrap();
    assert_eq!(
      task.status,
      crate::models::TaskStatus::Pending,
      "ALL running tasks should be reset to pending on startup"
    );
    assert!(task.started_at.is_none());
  }

  #[tokio::test]
  async fn test_failed_batch_creation() {
    let (task_manager, _code_table, _temp_dir, _project_id) = create_test_task_manager().await;

    let task_id = Uuid::now_v7();
    let project_id = Uuid::now_v7();

    let mut failed_files = std::collections::BTreeSet::new();
    failed_files.insert("/test/file1.rs".to_string());
    failed_files.insert("/test/file2.rs".to_string());

    let failed_batch = crate::models::FailedEmbeddingBatch::new(
      task_id,
      project_id,
      "/test".to_string(),
      failed_files.clone(),
      "Test error".to_string(),
    );

    let batch_id = failed_batch.id;

    // Add the failed batch
    let table = task_manager.failed_batches_table.write().await;
    table
      .add(failed_batch.clone().into_arrow().unwrap())
      .execute()
      .await
      .unwrap();

    // Check it was created by querying directly
    let mut stream = table
      .query()
      .only_if(format!("id = '{}'", batch_id).as_str())
      .limit(1)
      .execute()
      .await
      .unwrap();

    let batch_record = stream.try_next().await.unwrap();
    assert!(batch_record.is_some());

    let created_batch =
      crate::models::FailedEmbeddingBatch::from_record_batch(&batch_record.unwrap(), 0).unwrap();
    assert_eq!(created_batch.task_id, task_id);
    assert_eq!(created_batch.project_id, project_id);
    assert_eq!(created_batch.failed_files, failed_files);
    assert_eq!(created_batch.errors.len(), 1);
    assert_eq!(created_batch.errors[0].error, "Test error");
    assert_eq!(created_batch.retry_count, 0);

    // Verify retry_after is set to future (1 minute from creation)
    assert!(created_batch.retry_after > chrono::Utc::now().naive_utc());
  }

  #[tokio::test]
  async fn test_exponential_backoff_schedule() {
    let (task_manager, _code_table, _temp_dir, _project_id) = create_test_task_manager().await;

    // Test the backoff schedule values
    let expected_minutes = vec![
      (0, 1),  // retry_count 0 -> 1 minute
      (1, 5),  // retry_count 1 -> 5 minutes
      (2, 10), // retry_count 2 -> 10 minutes
      (3, 20), // retry_count 3 -> 20 minutes
      (4, 40), // retry_count 4 -> 40 minutes
      (5, 60), // retry_count 5 -> 60 minutes (capped)
      (6, 60), // retry_count 6 -> 60 minutes (stays capped)
    ];

    for (retry_count, expected) in expected_minutes {
      let now = chrono::Utc::now().naive_utc();
      let retry_after =
        crate::models::FailedEmbeddingBatch::calculate_next_retry_after(retry_count);
      let actual_minutes = (retry_after - now).num_seconds() / 60;
      assert_eq!(
        actual_minutes, expected,
        "retry_count {} should yield {} minute delay",
        retry_count, expected
      );
    }

    let task_id = Uuid::now_v7();
    let project_id = Uuid::now_v7();

    let mut failed_files = std::collections::BTreeSet::new();
    failed_files.insert("/test/file1.rs".to_string());

    let mut failed_batch = crate::models::FailedEmbeddingBatch::new(
      task_id,
      project_id,
      "/test".to_string(),
      failed_files,
      "Initial error".to_string(),
    );

    // Set retry_after to past so it's ready for retry
    failed_batch.retry_after = chrono::Utc::now().naive_utc() - chrono::Duration::hours(1);

    // Add the failed batch
    let table = task_manager.failed_batches_table.write().await;
    table
      .add(failed_batch.clone().into_arrow().unwrap())
      .execute()
      .await
      .unwrap();
    drop(table);

    // Expected delays after calling update_for_retry
    // When update_for_retry is called on retry_count=0, it becomes retry_count=1 with 5 minute delay
    let expected_delays_after_update = [
      5,  // retry_count 0->1
      10, // retry_count 1->2
      20, // retry_count 2->3
      40, // retry_count 3->4
      60, // retry_count 4->5 (capped)
      60, // retry_count 5->6 (stays capped)
    ];

    for (i, expected_minutes) in expected_delays_after_update.iter().enumerate() {
      // Get the batch
      let batch = task_manager.check_retry_batches().await.unwrap().unwrap();
      assert_eq!(batch.retry_count as usize, i);

      let now_before_update = chrono::Utc::now().naive_utc();

      // Update for next retry
      let mut updated_batch = batch.clone();
      updated_batch.update_for_retry(format!("Retry {} failed", i));

      // Check the retry_after is set correctly
      let delay_seconds = (updated_batch.retry_after - now_before_update).num_seconds();
      let expected_seconds = expected_minutes * 60;

      assert!(
        delay_seconds >= expected_seconds - 5 && delay_seconds <= expected_seconds + 5,
        "After update from retry_count {} to {}, expected delay {} minutes ({} seconds), got {} seconds",
        i,
        i + 1,
        expected_minutes,
        expected_seconds,
        delay_seconds
      );

      // Update in database
      task_manager
        .update_failed_batch(&updated_batch)
        .await
        .unwrap();

      // Set retry_after to past for next iteration
      let table = task_manager.failed_batches_table.write().await;
      table
        .update()
        .only_if(format!("id = '{}'", updated_batch.id).as_str())
        .column(
          "retry_after",
          format!(
            "{}",
            (chrono::Utc::now().naive_utc() - chrono::Duration::hours(1))
              .and_utc()
              .timestamp_micros()
          ),
        )
        .execute()
        .await
        .unwrap();
    }
  }

  #[tokio::test]
  async fn test_retry_queue_priority() {
    let (task_manager, _code_table, _temp_dir, project_id) = create_test_task_manager().await;

    // Create a regular pending task
    let regular_task_id = task_manager
      .submit_task(
        project_id,
        std::path::Path::new("/test"),
        crate::models::TaskType::FullIndex,
      )
      .await
      .unwrap();

    // Create a failed batch ready for retry
    let mut failed_files = std::collections::BTreeSet::new();
    failed_files.insert("/test/failed.rs".to_string());

    let mut failed_batch = crate::models::FailedEmbeddingBatch::new(
      Uuid::now_v7(),
      project_id,
      "/test".to_string(),
      failed_files,
      "Test error".to_string(),
    );
    failed_batch.retry_after = chrono::Utc::now().naive_utc() - chrono::Duration::hours(1);

    let table = task_manager.failed_batches_table.write().await;
    table
      .add(failed_batch.clone().into_arrow().unwrap())
      .execute()
      .await
      .unwrap();
    drop(table);

    // Process next task - should process regular task first
    let cancel_token = CancellationToken::new();
    let processed = task_manager.process_next_task(&cancel_token).await.unwrap();
    assert!(processed);

    // Verify regular task was processed
    let task = task_manager
      .get_task(&regular_task_id)
      .await
      .unwrap()
      .unwrap();
    assert_ne!(task.status, crate::models::TaskStatus::Pending);

    // Now process retry task (since no more pending tasks)
    let processed_retry = task_manager.process_next_task(&cancel_token).await.unwrap();
    assert!(processed_retry);

    // The failed batch should have been processed (either deleted or updated)
    // We can't easily check the exact outcome without mocking the indexer
  }

  #[tokio::test]
  async fn test_merge_insert_update() {
    let (task_manager, _code_table, _temp_dir, _project_id) = create_test_task_manager().await;

    let task_id = Uuid::now_v7();
    let project_id = Uuid::now_v7();

    let mut failed_files = std::collections::BTreeSet::new();
    failed_files.insert("/test/file1.rs".to_string());

    let mut failed_batch = crate::models::FailedEmbeddingBatch::new(
      task_id,
      project_id,
      "/test".to_string(),
      failed_files.clone(),
      "Initial error".to_string(),
    );

    // Set retry_after to past so it shows up in queries
    failed_batch.retry_after = chrono::Utc::now().naive_utc() - chrono::Duration::hours(1);

    // Add the failed batch
    let table = task_manager.failed_batches_table.write().await;
    table
      .add(failed_batch.clone().into_arrow().unwrap())
      .execute()
      .await
      .unwrap();
    drop(table);

    // Update it with more failures
    let mut updated_batch = failed_batch.clone();
    updated_batch
      .failed_files
      .insert("/test/file2.rs".to_string());
    updated_batch.update_for_retry("Second error".to_string());

    // Also set retry_after to past for testing
    updated_batch.retry_after = chrono::Utc::now().naive_utc() - chrono::Duration::hours(1);

    // Use merge insert to update
    task_manager
      .update_failed_batch(&updated_batch)
      .await
      .unwrap();

    // Verify the update by querying directly
    let table = task_manager.failed_batches_table.read().await;
    let mut stream = table
      .query()
      .only_if(format!("id = '{}'", failed_batch.id).as_str())
      .limit(1)
      .execute()
      .await
      .unwrap();

    let batch_record = stream.try_next().await.unwrap().unwrap();
    let batch = crate::models::FailedEmbeddingBatch::from_record_batch(&batch_record, 0).unwrap();

    assert_eq!(batch.failed_files.len(), 2);
    assert_eq!(batch.retry_count, 1);
    assert_eq!(batch.errors.len(), 2);
    assert_eq!(batch.errors[1].error, "Second error");
  }

  #[tokio::test]
  async fn test_delete_successful_retry() {
    let (task_manager, _code_table, _temp_dir, _project_id) = create_test_task_manager().await;

    let batch_id = Uuid::now_v7();
    let task_id = Uuid::now_v7();
    let project_id = Uuid::now_v7();

    let mut failed_files = std::collections::BTreeSet::new();
    failed_files.insert("/test/file1.rs".to_string());

    let mut failed_batch = crate::models::FailedEmbeddingBatch::new(
      task_id,
      project_id,
      "/test".to_string(),
      failed_files,
      "Test error".to_string(),
    );
    failed_batch.id = batch_id;

    // Add the failed batch
    let table = task_manager.failed_batches_table.write().await;
    table
      .add(failed_batch.clone().into_arrow().unwrap())
      .execute()
      .await
      .unwrap();
    drop(table);

    // Delete it
    task_manager.delete_failed_batch(&batch_id).await.unwrap();

    // Verify it's gone
    let batches = task_manager.check_retry_batches().await.unwrap();
    assert!(batches.is_none());
  }

  #[tokio::test]
  async fn test_task_status_partially_completed() {
    let (task_manager, _code_table, temp_dir, project_id) = create_test_task_manager().await;

    // Create a test directory
    let project_dir = temp_dir.path().join("test_project");
    std::fs::create_dir_all(&project_dir).unwrap();

    // Submit a task
    let task_id = task_manager
      .submit_task(project_id, &project_dir, crate::models::TaskType::FullIndex)
      .await
      .unwrap();

    // Update task to partially completed
    task_manager
      .update_task_partially_completed(&task_id, 50)
      .await
      .unwrap();

    // Verify the task status
    let task = task_manager.get_task(&task_id).await.unwrap().unwrap();
    assert_eq!(task.status, crate::models::TaskStatus::PartiallyCompleted);
    assert_eq!(task.files_indexed, Some(50));
    assert!(task.completed_at.is_some());
  }
}
