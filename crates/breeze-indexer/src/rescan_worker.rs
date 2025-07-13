use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::sleep;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

use crate::{IndexerError, ProjectManager, TaskManager};

/// Starts the rescan worker loop
///
/// This function runs a background task that periodically checks for projects
/// that need rescanning and submits indexing tasks for them.
///
/// # Arguments
///
/// * `project_manager` - Shared reference to the ProjectManager
/// * `task_manager` - Shared reference to the TaskManager
/// * `interval` - Duration between rescan cycles
/// * `shutdown_token` - Token to signal worker shutdown
pub fn start_rescan_worker(
  project_manager: Arc<ProjectManager>,
  task_manager: Arc<TaskManager>,
  interval: Duration,
  shutdown_token: CancellationToken,
) {
  tokio::spawn(async move {
    while !shutdown_token.is_cancelled() {
      // Run rescan cycle
      if let Err(e) = run_rescan_cycle(&project_manager, &task_manager).await {
        error!("Rescan cycle failed: {}", e);
      }

      // Wait for interval or cancellation
      tokio::select! {
          _ = sleep(interval) => {},
          _ = shutdown_token.cancelled() => break,
      }
    }
    info!("Rescan worker shut down");
  });
}

/// Runs a single rescan cycle
///
/// This function:
/// 1. Finds projects that need rescanning
/// 2. Submits indexing tasks for each project
/// 3. Handles errors gracefully with proper logging
async fn run_rescan_cycle(
  project_manager: &ProjectManager,
  task_manager: &TaskManager,
) -> Result<(), IndexerError> {
  let now = chrono::Utc::now();
  let projects = project_manager.find_projects_needing_rescan(now).await?;

  for project in projects {
    let project_id = project.id;
    let project_path = project.directory.clone();

    info!(
        project_id = %project_id,
        project_name = %project.name,
        "Submitting periodic rescan task for project"
    );

    // Submit a full index task for the rescan
    // The TaskManager's existing deduplication logic will handle preventing duplicate tasks
    match task_manager
      .submit_task(
        project_id,
        Path::new(&project_path),
        crate::models::TaskType::FullIndex,
      )
      .await
    {
      Ok(task_id) => {
        info!(
            project_id = %project_id,
            task_id = %task_id,
            "Successfully submitted rescan task"
        );
      }
      Err(e) => {
        error!(
            project_id = %project_id,
            error = %e,
            "Failed to submit rescan task"
        );
        // Continue with other projects even if one fails
      }
    }
  }

  Ok(())
}
