use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use breeze_chunkers::CandidateMatcher;
use notify::{
  Config, EventKind, RecursiveMode,
  event::{CreateKind, DataChange, ModifyKind},
};
use notify_debouncer_full::{DebounceEventResult, Debouncer, FileIdMap};
use sysinfo::{Pid, System};
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{debug, error, info};
use uuid::Uuid;

use crate::{IndexerError, task_manager::TaskManager};

/// Manages file watching for a project
pub struct ProjectWatcher {
  project_id: Uuid,

  _debouncer: Arc<RwLock<Debouncer<notify::RecommendedWatcher, FileIdMap>>>,
}

impl ProjectWatcher {
  /// Create a new project watcher
  pub async fn new<P: AsRef<Path>>(
    project_id: Uuid,
    project_path: P,
    task_manager: Arc<TaskManager>,
    max_file_size: Option<u64>,
  ) -> Result<Self, IndexerError> {
    let project_path = project_path.as_ref();
    let project_path_buf = project_path.to_path_buf();

    // Create candidate matcher
    let candidate_matcher = Arc::new(
      CandidateMatcher::new(project_path, max_file_size)
        .map_err(|e| IndexerError::Config(e.to_string()))?,
    );
    let candidate_matcher_clone = candidate_matcher.clone();

    // Clone references for the event handler
    let task_manager_clone = task_manager.clone();
    let project_id_clone = project_id;
    let project_path_clone = project_path_buf.clone();

    // // Create debouncer with 1 second timeout
    // let poll_interval = if cfg!(test) {
    //   Duration::from_millis(100)
    // } else {
    //   Duration::from_secs(3600) // 1 hour
    // };

    let debounce_timeout = if cfg!(test) {
      Duration::from_millis(500) // Much faster for tests
    } else {
      Duration::from_secs(30)
    };

    let tick_interval = if cfg!(test) {
      Duration::from_millis(100)
    } else {
      Duration::from_secs(2)
    };

    let config = Config::default();
    // .with_poll_interval(poll_interval)
    // .with_compare_contents(true);

    // Added logging for diagnosis
    // debug!(
    //   "Watcher config: poll_interval={:?}, compare_contents=false",
    //   poll_interval
    // );

    let mut debouncer =
      notify_debouncer_full::new_debouncer_opt(
        debounce_timeout,
        Some(tick_interval),
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
                  match &event.event.kind {
                    EventKind::Create(CreateKind::File)
                    | EventKind::Modify(ModifyKind::Data(DataChange::Content | DataChange::Size)) =>
                    {
                      // Check if this is a candidate for indexing
                      if candidate_matcher.matches(path) {
                        changes.insert(crate::models::FileChange {
                          path: path.clone(),
                          operation: crate::models::FileOperation::Update,
                        });
                      }
                    }
                    EventKind::Remove(_) => {
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
                      .submit_task(
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
                    match task_manager.submit_task(
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

    // Added logging to monitor open files - requires sysinfo crate
    // Note: Add sysinfo to Cargo.toml dependencies for this to work
    let system = System::new_all();
    let pid = std::process::id();
    if let Some(process) = system.process(Pid::from_u32(pid)) {
      debug!(
        "Open files after watch setup: {}",
        process.open_files().unwrap_or_default()
      );
    }

    Ok(Self {
      project_id,

      _debouncer: Arc::new(RwLock::new(debouncer)),
    })
  }

  /// Get the project ID
  #[cfg(test)]
  pub fn project_id(&self) -> Uuid {
    self.project_id
  }

  /// Wait for shutdown signal
  pub async fn wait_for_shutdown(&self, shutdown: CancellationToken) {
    shutdown.cancelled().await;
    info!("File watcher shutting down for project {}", self.project_id);

    // Added logging for shutdown FD check
    let system = System::new_all();
    let pid = std::process::id();
    if let Some(process) = system.process(Pid::from_u32(pid)) {
      debug!(
        "Open files at shutdown: {}",
        process.open_files().unwrap_or_default()
      );
    }
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

    let chunk_table = crate::models::CodeChunk::ensure_table(&connection, "test_chunks", 384)
      .await
      .unwrap();
    let chunk_table = Arc::new(RwLock::new(chunk_table));

    let failed_batches_table =
      crate::models::FailedEmbeddingBatch::ensure_table(&connection, "test_failed_batches")
        .await
        .unwrap();
    let failed_batches_table = Arc::new(RwLock::new(failed_batches_table));

    let (_temp_dir_config, config) = Config::test();
    let embedding_provider = create_embedding_provider(&config).await.unwrap();

    let project_table = crate::models::Project::ensure_table(&connection, "test_projects")
      .await
      .unwrap();
    let project_table = Arc::new(RwLock::new(project_table));

    let bulk_indexer = BulkIndexer::new(
      Arc::new(config),
      Arc::from(embedding_provider),
      384,
      code_table.clone(),
      chunk_table,
      project_table,
    );

    let project_table = crate::models::Project::ensure_table(&connection, "test_projects")
      .await
      .unwrap();
    let project_table = Arc::new(RwLock::new(project_table));

    let task_manager = Arc::new(TaskManager::new(
      task_table,
      failed_batches_table,
      project_table,
      bulk_indexer,
    ));

    let project_id = Uuid::now_v7();
    let project_dir = temp_dir.path().join("project");
    tokio::fs::create_dir(&project_dir).await.unwrap();

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
  async fn test_file_creation_and_modification() {
    let (_watcher, temp_dir, _project_id) = create_test_watcher().await;
    let project_dir = temp_dir.path().join("project");

    // Wait for watcher to initialize
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Create a file
    let test_file = project_dir.join("test.rs");
    tokio::fs::write(&test_file, "fn main() {}").await.unwrap();

    // Wait for event to be processed
    tokio::time::sleep(Duration::from_millis(700)).await;

    // Modify the file
    tokio::fs::write(&test_file, "fn main() { println!(\"Hello\"); }")
      .await
      .unwrap();

    // Wait for event to be processed
    tokio::time::sleep(Duration::from_millis(700)).await;

    // Delete the file
    tokio::fs::remove_file(&test_file).await.unwrap();

    // Wait for event to be processed
    tokio::time::sleep(Duration::from_millis(700)).await;
  }

  #[tokio::test]
  async fn test_directory_deletion_bug() {
    let (_watcher, temp_dir, _project_id) = create_test_watcher().await;
    let project_dir = temp_dir.path().join("project");

    // Wait for watcher to initialize
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Create a directory with multiple files
    let sub_dir = project_dir.join("src");
    tokio::fs::create_dir(&sub_dir).await.unwrap();

    let files = vec![
      sub_dir.join("main.rs"),
      sub_dir.join("lib.rs"),
      sub_dir.join("utils.rs"),
    ];

    for file in &files {
      tokio::fs::write(file, "// test content").await.unwrap();
    }

    // Wait for creation events
    tokio::time::sleep(Duration::from_millis(700)).await;

    // Delete the entire directory
    tokio::fs::remove_dir_all(&sub_dir).await.unwrap();

    // Wait for deletion events
    tokio::time::sleep(Duration::from_millis(700)).await;

    // BUG: Currently, only the directory deletion event is triggered,
    // but the files inside are not marked for deletion in the index
  }

  #[tokio::test]
  async fn test_nested_directory_operations() {
    let (_watcher, temp_dir, _project_id) = create_test_watcher().await;
    let project_dir = temp_dir.path().join("project");

    // Wait for watcher to initialize
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Create nested directory structure
    let src_dir = project_dir.join("src");
    let models_dir = src_dir.join("models");
    let utils_dir = src_dir.join("utils");

    tokio::fs::create_dir(&src_dir).await.unwrap();
    tokio::fs::create_dir(&models_dir).await.unwrap();
    tokio::fs::create_dir(&utils_dir).await.unwrap();

    // Add files to each directory
    tokio::fs::write(src_dir.join("main.rs"), "fn main() {}")
      .await
      .unwrap();
    tokio::fs::write(models_dir.join("user.rs"), "struct User {}")
      .await
      .unwrap();
    tokio::fs::write(models_dir.join("post.rs"), "struct Post {}")
      .await
      .unwrap();
    tokio::fs::write(utils_dir.join("helpers.rs"), "fn helper() {}")
      .await
      .unwrap();

    // Wait for creation events
    tokio::time::sleep(Duration::from_millis(700)).await;

    // Delete a subdirectory
    tokio::fs::remove_dir_all(&models_dir).await.unwrap();

    // Wait for deletion events
    tokio::time::sleep(Duration::from_millis(700)).await;
  }

  #[tokio::test]
  async fn test_file_move_operations() {
    let (_watcher, temp_dir, _project_id) = create_test_watcher().await;
    let project_dir = temp_dir.path().join("project");

    // Wait for watcher to initialize
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Create source and destination directories
    let src_dir = project_dir.join("src");
    let dest_dir = project_dir.join("dest");
    tokio::fs::create_dir(&src_dir).await.unwrap();
    tokio::fs::create_dir(&dest_dir).await.unwrap();

    // Create a file
    let src_file = src_dir.join("test.rs");
    tokio::fs::write(&src_file, "fn test() {}").await.unwrap();

    // Wait for creation
    tokio::time::sleep(Duration::from_millis(700)).await;

    // Move the file
    let dest_file = dest_dir.join("test.rs");
    tokio::fs::rename(&src_file, &dest_file).await.unwrap();

    // Wait for move events (should see delete + create)
    tokio::time::sleep(Duration::from_millis(700)).await;
  }

  #[tokio::test]
  async fn test_ignored_files() {
    let (_watcher, temp_dir, _project_id) = create_test_watcher().await;
    let project_dir = temp_dir.path().join("project");

    // Wait for watcher to initialize
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Create files that should be ignored
    let ignored_files = vec![
      project_dir.join(".git").join("config"),
      project_dir.join("target").join("debug").join("binary"),
      project_dir
        .join("node_modules")
        .join("package")
        .join("index.js"),
      project_dir.join(".DS_Store"),
    ];

    // Create directories first
    tokio::fs::create_dir_all(project_dir.join(".git"))
      .await
      .unwrap();
    tokio::fs::create_dir_all(project_dir.join("target").join("debug"))
      .await
      .unwrap();
    tokio::fs::create_dir_all(project_dir.join("node_modules").join("package"))
      .await
      .unwrap();

    // Create the ignored files
    for file in &ignored_files {
      tokio::fs::write(file, "ignored content").await.unwrap();
    }

    // Create a file that should be watched
    let watched_file = project_dir.join("main.rs");
    tokio::fs::write(&watched_file, "fn main() {}")
      .await
      .unwrap();

    // Wait for events
    tokio::time::sleep(Duration::from_millis(700)).await;

    // Only the watched file should trigger events, not the ignored ones
  }

  #[tokio::test]
  async fn test_large_file_handling() {
    let (_watcher, temp_dir, _project_id) = create_test_watcher().await;
    let project_dir = temp_dir.path().join("project");

    // Wait for watcher to initialize
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Create a large file (over default max_file_size)
    let large_file = project_dir.join("large.bin");
    let large_content = vec![0u8; 100 * 1024 * 1024]; // 100MB
    tokio::fs::write(&large_file, large_content).await.unwrap();

    // Create a normal file
    let normal_file = project_dir.join("normal.rs");
    tokio::fs::write(&normal_file, "fn main() {}")
      .await
      .unwrap();

    // Wait for events
    tokio::time::sleep(Duration::from_millis(700)).await;

    // Only the normal file should be indexed
  }

  #[tokio::test]
  async fn test_file_watch_debouncing() {
    let (_watcher, temp_dir, _project_id) = create_test_watcher().await;

    // Give the watcher time to initialize
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Create multiple files in quick succession
    let project_dir = temp_dir.path().join("project");
    for i in 0..5 {
      let test_file = project_dir.join(format!("test{}.rs", i));
      tokio::fs::write(&test_file, format!("fn main() {{ println!(\"{}\"); }}", i))
        .await
        .unwrap();
      tokio::time::sleep(Duration::from_millis(50)).await;
    }

    // Wait for debouncing to complete
    tokio::time::sleep(Duration::from_millis(700)).await;

    // All files should be batched into a single task submission
  }
}
