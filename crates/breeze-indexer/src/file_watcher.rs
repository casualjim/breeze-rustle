use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use breeze_chunkers::CandidateMatcher;
use notify::{Config, EventKind, RecursiveMode};
use notify_debouncer_full::{DebounceEventResult, Debouncer, FileIdMap};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};
use uuid::Uuid;

use crate::task_manager::TaskManager;

/// Manages file watching for a project
pub struct ProjectWatcher {
  project_id: Uuid,
  project_path: PathBuf,
  task_manager: Arc<TaskManager>,
  candidate_matcher: Arc<CandidateMatcher>,
  _debouncer: Arc<RwLock<Debouncer<notify::RecommendedWatcher, FileIdMap>>>,
}

impl ProjectWatcher {
  /// Create a new project watcher
  pub async fn new<P: AsRef<Path>>(
    project_id: Uuid,
    project_path: P,
    task_manager: Arc<TaskManager>,
    max_file_size: Option<u64>,
  ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
    let project_path = project_path.as_ref();
    let project_path_buf = project_path.to_path_buf();

    // Create candidate matcher
    let candidate_matcher = Arc::new(CandidateMatcher::new(project_path, max_file_size)?);
    let candidate_matcher_clone = candidate_matcher.clone();

    // Clone references for the event handler
    let task_manager_clone = task_manager.clone();
    let project_id_clone = project_id;
    let project_path_clone = project_path_buf.clone();

    // Create debouncer with 1 second timeout
    let poll_interval = if cfg!(test) {
      Duration::from_millis(100)
    } else {
      Duration::from_secs(3600) // 1 hour
    };

    let config = Config::default()
      .with_poll_interval(poll_interval)
      .with_compare_contents(true);

    let mut debouncer =
      notify_debouncer_full::new_debouncer_opt(
        Duration::from_secs(30),
        Some(Duration::from_secs(2)),
        move |result: DebounceEventResult| {
          match result {
            Ok(events) => {
              let candidate_matcher = candidate_matcher_clone.clone();
              let task_manager = task_manager_clone.clone();
              let project_id = project_id_clone;
              let project_path = project_path_clone.clone();

              // Collect all file changes
              let mut changes = std::collections::BTreeSet::new();

              for event in events {
                // Check all paths in the event
                for path in &event.event.paths {
                  // Skip directories
                  if path.is_dir() {
                    continue;
                  }

                  match &event.event.kind {
                    EventKind::Create(_) | EventKind::Modify(_) => {
                      // Check if this is a candidate for indexing
                      if candidate_matcher.matches(path) {
                        changes.insert(crate::models::FileChange {
                          path: path.clone(),
                          operation: crate::models::FileOperation::Update,
                        });
                      }
                    }
                    EventKind::Remove(_) => {
                      // For deletions, we don't need to check if it's a candidate
                      // We just need to remove it from the index
                      changes.insert(crate::models::FileChange {
                        path: path.clone(),
                        operation: crate::models::FileOperation::Delete,
                      });
                    }
                    _ => {} // Ignore other event types
                  }
                }
              }

              if !changes.is_empty() {
                // Process in a detached task since this callback is synchronous
                let rt = tokio::runtime::Handle::try_current();
                if let Ok(handle) = rt {
                  handle.spawn(async move {
                    match task_manager
                      .submit_task_with_type(
                        project_id,
                        &project_path,
                        crate::models::TaskType::PartialUpdate { changes },
                      )
                      .await
                    {
                      Ok(task_id) => {
                        info!(task_id = %task_id, "Submitted partial update task for file changes");
                      }
                      Err(e) => {
                        error!("Failed to submit partial update task: {}", e);
                      }
                    }
                  });
                } else {
                  // Fallback: spawn a thread to run the async task
                  std::thread::spawn(move || {
                    let rt = tokio::runtime::Runtime::new().unwrap();
                    rt.block_on(async {
                    match task_manager.submit_task_with_type(
                      project_id,
                      &project_path,
                      crate::models::TaskType::PartialUpdate { changes },
                    ).await {
                      Ok(task_id) => {
                        info!(task_id = %task_id, "Submitted partial update task for file changes");
                      }
                      Err(e) => {
                        error!("Failed to submit partial update task: {}", e);
                      }
                    }
                  });
                  });
                }
              }
            }
            Err(errors) => {
              for error in errors {
                error!("File watcher error: {:?}", error);
              }
            }
          }
        },
        notify_debouncer_full::FileIdMap::new(),
        config,
      )?;

    // Start watching
    debouncer.watch(project_path, RecursiveMode::Recursive)?;

    Ok(Self {
      project_id,
      project_path: project_path_buf,
      task_manager,
      candidate_matcher,
      _debouncer: Arc::new(RwLock::new(debouncer)),
    })
  }

  /// Get the project ID
  pub fn project_id(&self) -> Uuid {
    self.project_id
  }

  /// Get the project path
  pub fn project_path(&self) -> &Path {
    &self.project_path
  }

  /// Wait for shutdown signal
  pub async fn wait_for_shutdown(&self, shutdown: CancellationToken) {
    shutdown.cancelled().await;
    info!("File watcher shutting down for project {}", self.project_id);
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::Config;
  use crate::bulk_indexer::BulkIndexer;
  use crate::embeddings::factory::create_embedding_provider;
  use crate::models::{CodeDocument, IndexTask};
  use tempfile::TempDir;

  async fn create_test_watcher() -> (ProjectWatcher, TempDir, Uuid) {
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

    let (_temp_dir_config, config) = Config::test();
    let embedding_provider = create_embedding_provider(&config).await.unwrap();

    let bulk_indexer = BulkIndexer::new(
      Arc::new(config),
      Arc::from(embedding_provider),
      384,
      code_table,
    );

    let task_manager = Arc::new(TaskManager::new(task_table, bulk_indexer));

    let project_id = Uuid::now_v7();
    let project_dir = temp_dir.path().join("project");
    std::fs::create_dir(&project_dir).unwrap();

    let watcher = ProjectWatcher::new(project_id, &project_dir, task_manager, None)
      .await
      .unwrap();

    (watcher, temp_dir, project_id)
  }

  #[tokio::test]
  async fn test_file_watcher_creation() {
    let (watcher, _temp_dir, project_id) = create_test_watcher().await;

    assert_eq!(watcher.project_id(), project_id);
  }

  #[tokio::test]
  async fn test_file_watch_debouncing() {
    let (watcher, temp_dir, _project_id) = create_test_watcher().await;

    // Give the watcher time to initialize
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Create multiple files in quick succession
    let project_dir = temp_dir.path().join("project");
    for i in 0..5 {
      let test_file = project_dir.join(format!("test{}.rs", i));
      std::fs::write(&test_file, format!("fn main() {{ println!(\"{}\"); }}", i)).unwrap();
      tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Wait for debouncing to complete
    tokio::time::sleep(Duration::from_secs(2)).await;

    // Check that only one task was submitted (would need to expose task count from task_manager to test properly)
    // For now, just ensure no panic occurred
  }
}
