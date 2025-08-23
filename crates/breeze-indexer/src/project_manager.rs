use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use uuid::Uuid;

use futures::TryStreamExt;
use lancedb::Table;
use lancedb::arrow::IntoArrow;
use lancedb::query::{ExecutableQuery, QueryBase};
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::{
  IndexerError,
  models::{Project, ProjectStatus},
  task_manager::TaskManager,
};

pub struct ProjectManager {
  project_table: Arc<RwLock<Table>>,
  task_manager: Arc<TaskManager>,
}

impl ProjectManager {
  pub fn new(project_table: Arc<RwLock<Table>>, task_manager: Arc<TaskManager>) -> Self {
    Self {
      project_table,
      task_manager,
    }
  }

  /// Canonicalize a path for consistent lookups
  /// - Requires absolute path
  /// - Remove trailing slash
  /// - Convert to string
  fn canonicalize_path<P: AsRef<Path>>(path: P) -> Result<String, IndexerError> {
    let path = path.as_ref();

    if !path.is_absolute() {
      return Err(IndexerError::PathNotAbsolute(path.display().to_string()));
    }

    // If the full path doesn't exist (e.g., a file inside a dir),
    // canonicalize the nearest existing ancestor and append the tail
    let mut existing_ancestor = Some(path);
    while let Some(p) = existing_ancestor {
      if p.exists() {
        break;
      }
      existing_ancestor = p.parent();
    }

    let normalized = if let Some(ancestor) = existing_ancestor {
      // Canonicalize the ancestor (this resolves symlinks like /var -> /private/var on macOS)
      let ancestor_canon = ancestor
        .canonicalize()
        .unwrap_or_else(|_| ancestor.to_path_buf());
      // Append the remaining components (if any)
      let tail = path.strip_prefix(ancestor).unwrap_or(path);
      ancestor_canon.join(tail)
    } else {
      // Should not happen for absolute paths (at least '/' exists),
      // but fall back to the original path just in case
      path.to_path_buf()
    };

    // Convert to string and remove trailing slash
    let mut path_str = normalized.to_string_lossy().to_string();
    if path_str.ends_with('/') && path_str.len() > 1 {
      path_str.pop();
    }

    Ok(path_str)
  }

  /// Find a project by directory path
  pub async fn find_by_path<P: AsRef<Path>>(
    &self,
    path: P,
  ) -> Result<Option<Project>, IndexerError> {
    let canonical_path = Self::canonicalize_path(path)?;

    // Fetch all projects and find the longest directory that is a prefix of the provided path
    let table = self.project_table.read().await;
    let mut stream = table.query().execute().await?;

    let mut projects: Vec<Project> = Vec::new();
    while let Some(batch) = stream.try_next().await? {
      for i in 0..batch.num_rows() {
        projects.push(Project::from_record_batch(&batch, i)?);
      }
    }

    // Sort by directory length descending to prefer the most specific (deepest) path
    projects.sort_by(|a, b| b.directory.len().cmp(&a.directory.len()));

    let needle = Path::new(&canonical_path);
    for project in projects.into_iter() {
      let proj_path = Path::new(&project.directory);
      if needle.starts_with(proj_path) {
        return Ok(Some(project));
      }
    }

    Ok(None)
  }

  /// Create a new project
  pub async fn create_project(
    &self,
    name: String,
    directory: String,
    description: Option<String>,
    rescan_interval: Option<Duration>,
  ) -> Result<Project, IndexerError> {
    // Canonicalize the directory path (validates it's absolute)
    let canonical_directory = Self::canonicalize_path(&directory)?;

    // Check if a project already exists for this directory
    if let Some(existing) = self.get_by_directory(&canonical_directory).await? {
      return Err(IndexerError::ProjectAlreadyExists {
        directory: canonical_directory,
        existing_id: existing.id,
      });
    }

    // Create the project (validates directory exists and is a directory)
    let project = Project::new(name, canonical_directory, description)
      .map(|p| p.with_rescan_interval(rescan_interval))
      .map_err(IndexerError::Config)?;

    // Insert into table
    let arrow_data = project.clone().into_arrow()?;

    let table = self.project_table.write().await;
    table.add(arrow_data).execute().await?;
    drop(table); // Release the lock before calling task_manager

    info!(project_id = %project.id, "Created new project");

    // Automatically submit an indexing task for the new project
    match self
      .task_manager
      .submit_task(
        project.id,
        Path::new(&project.directory),
        crate::models::TaskType::FullIndex,
      )
      .await
    {
      Ok(task_id) => {
        info!(project_id = %project.id, task_id = %task_id, "Automatically submitted indexing task for new project");
      }
      Err(e) => {
        // Log the error but don't fail the project creation
        tracing::error!(project_id = %project.id, error = %e, "Failed to submit automatic indexing task");
      }
    }

    Ok(project)
  }

  /// Get a project by ID
  pub async fn get_project(&self, id: Uuid) -> Result<Option<Project>, IndexerError> {
    let table = self.project_table.read().await;

    let mut stream = table
      .query()
      .only_if(format!("id = '{}'", id).as_str())
      .limit(1)
      .execute()
      .await?;

    if let Some(batch) = stream.try_next().await? {
      let project = Project::from_record_batch(&batch, 0)?;
      Ok(Some(project))
    } else {
      Ok(None)
    }
  }

  /// List all projects
  pub async fn list_projects(&self) -> Result<Vec<Project>, IndexerError> {
    let table = self.project_table.read().await;

    let mut stream = table.query().execute().await?;

    let mut projects = Vec::new();
    while let Some(batch) = stream.try_next().await? {
      for i in 0..batch.num_rows() {
        projects.push(Project::from_record_batch(&batch, i)?);
      }
    }

    // Sort by created_at descending (newest first)
    projects.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    Ok(projects)
  }

  /// Update a project
  pub async fn update_project(
    &self,
    id: Uuid,
    name: Option<String>,
    description: Option<Option<String>>,
    rescan_interval: Option<Duration>,
  ) -> Result<Option<Project>, IndexerError> {
    // First check if project exists
    let existing = self.get_project(id).await?;
    if existing.is_none() {
      return Ok(None);
    }

    let table = self.project_table.write().await;

    // Build update query
    let mut update = table.update();
    update = update.only_if(format!("id = '{}'", id).as_str());

    if let Some(new_name) = name {
      update = update.column("name", format!("'{}'", new_name.replace("'", "''")));
    }

    if let Some(new_description) = description {
      update = update.column(
        "description",
        match new_description {
          Some(desc) => format!("'{}'", desc.replace("'", "''")),
          None => "null".to_string(),
        },
      );
    }

    if let Some(interval) = rescan_interval {
      update = update.column("rescan_interval", (interval.as_nanos() as i64).to_string());
    }

    // Always update updated_at
    let now_micros = chrono::Utc::now().naive_utc().and_utc().timestamp_micros();
    update = update.column("updated_at", format!("{}", now_micros));

    update.execute().await?;

    // Return the updated project
    drop(table);
    self.get_project(id).await
  }

  /// Mark a project for deletion - atomically sets status to PendingDeletion
  pub async fn mark_for_deletion(&self, id: Uuid) -> Result<Option<Project>, IndexerError> {
    // First check if project exists and is not already deleted
    let existing = self.get_project(id).await?;
    let existing = match existing {
      Some(p) => p,
      None => return Ok(None),
    };

    // Don't mark if already deleted or pending deletion
    if existing.status == ProjectStatus::Deleted
      || existing.status == ProjectStatus::PendingDeletion
    {
      info!(project_id = %id, status = ?existing.status, "Project already marked for deletion");
      return Ok(Some(existing));
    }

    let table = self.project_table.write().await;

    // Update status to PendingDeletion and set deletion_requested_at
    let now = chrono::Utc::now().naive_utc();
    let now_micros = now.and_utc().timestamp_micros();

    table
      .update()
      .only_if(format!("id = '{}' AND status = 'Active'", id).as_str())
      .column("status", "'PendingDeletion'")
      .column("deletion_requested_at", format!("{}", now_micros))
      .column("updated_at", format!("{}", now_micros))
      .execute()
      .await?;

    info!(project_id = %id, "Marked project for deletion");

    // Return the updated project
    drop(table);
    self.get_project(id).await
  }

  /// Delete a project
  pub async fn delete_project(&self, id: Uuid) -> Result<bool, IndexerError> {
    let table = self.project_table.write().await;

    // Check if exists first
    let mut stream = table
      .query()
      .only_if(format!("id = '{}'", id).as_str())
      .limit(1)
      .execute()
      .await?;

    if stream.try_next().await?.is_none() {
      return Ok(false);
    }

    // Delete the project
    table.delete(&format!("id = '{}'", id)).await?;

    debug!(project_id = %id, "Deleted project");
    Ok(true)
  }

  /// Check if a directory is already associated with a project
  pub async fn get_by_directory<P: AsRef<Path>>(
    &self,
    directory: P,
  ) -> Result<Option<Project>, IndexerError> {
    // Normalize the input directory to match stored canonical form
    let canonical_dir = Self::canonicalize_path(directory)?;

    let table = self.project_table.read().await;

    let mut stream = table
      .query()
      .only_if(format!("directory = '{}'", canonical_dir.replace("'", "''")).as_str())
      .limit(1)
      .execute()
      .await?;

    if let Some(batch) = stream.try_next().await? {
      let project = Project::from_record_batch(&batch, 0)?;
      Ok(Some(project))
    } else {
      Ok(None)
    }
  }

  /// Find projects that need periodic rescanning using a database query.
  /// This performs interval arithmetic in the database to find projects that are due for rescanning.
  pub async fn find_projects_needing_rescan(
    &self,
    now: chrono::DateTime<chrono::Utc>,
  ) -> Result<Vec<Project>, IndexerError> {
    let project_table = self.project_table.read().await;
    let now_str = now.to_rfc3339();

    // This query finds projects that are active, enabled for rescanning, and either
    // have never been indexed (last_indexed_at is NULL) or are due for a scan.
    // `last_indexed_at + rescan_interval` performs interval arithmetic in the DB.
    let filter = format!(
      "status = 'Active' AND rescan_interval IS NOT NULL AND (last_indexed_at IS NULL OR (last_indexed_at + rescan_interval) <= timestamp '{}')",
      now_str
    );

    let mut stream = project_table.query().only_if(&filter).execute().await?;
    let mut projects = Vec::new();

    while let Some(batch) = stream.try_next().await? {
      for i in 0..batch.num_rows() {
        let project = Project::from_record_batch(&batch, i)?;
        projects.push(project);
      }
    }

    Ok(projects)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use tempfile::TempDir;

  async fn create_test_project_manager() -> (ProjectManager, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");

    let connection = lancedb::connect(db_path.to_str().unwrap())
      .execute()
      .await
      .unwrap();

    let project_table = Project::ensure_table(&connection, "test_projects")
      .await
      .unwrap();
    let project_table = Arc::new(RwLock::new(project_table));

    // Create a dummy TaskManager for tests
    // We'll use the crate::models imports for task_manager
    let task_table = crate::models::IndexTask::ensure_table(&connection, "test_tasks")
      .await
      .unwrap();
    let code_table = crate::models::CodeDocument::ensure_table(&connection, "test_code", 1536)
      .await
      .unwrap();

    let task_table = Arc::new(RwLock::new(task_table));
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

    // Create a minimal config and embedding provider for BulkIndexer
    let config = crate::Config {
      database_path: db_path.clone(),
      embedding_provider: crate::config::EmbeddingProvider::Local,
      model: "BAAI/bge-small-en-v1.5".to_string(),
      voyage: None,
      openai_providers: std::collections::HashMap::new(),
      max_chunk_size: 1000,
      max_file_size: Some(1024 * 1024),
      max_parallel_files: 4,
      large_file_threads: None,
      embedding_workers: 1,
      optimize_threshold: 250,
      document_batch_size: 100,
    };

    let embedding_provider = crate::embeddings::factory::create_embedding_provider(&config)
      .await
      .unwrap();

    let bulk_indexer = crate::bulk_indexer::BulkIndexer::new(
      Arc::new(config),
      Arc::from(embedding_provider),
      384, // BAAI/bge-small-en-v1.5 has 384 dimensions
      code_table.clone(),
      chunk_table,
      project_table.clone(),
    );

    let task_manager = Arc::new(TaskManager::new(
      task_table,
      failed_batches_table,
      project_table.clone(),
      bulk_indexer,
    ));
    let project_manager = ProjectManager::new(project_table, task_manager);

    (project_manager, temp_dir)
  }

  #[tokio::test]
  async fn test_create_project() {
    let (project_manager, temp_dir) = create_test_project_manager().await;

    // Create a test directory
    let test_dir = temp_dir.path().join("test_project");
    std::fs::create_dir(&test_dir).unwrap();

    let project = project_manager
      .create_project(
        "Test Project".to_string(),
        test_dir.to_str().unwrap().to_string(),
        Some("A test project".to_string()),
        None,
      )
      .await
      .unwrap();

    assert_eq!(project.name, "Test Project");
    // Check that the directory ends with the test directory name (handles symlink resolution)
    assert!(project.directory.ends_with("test_project"));
    assert_eq!(project.description, Some("A test project".to_string()));
    assert!(!project.id.is_nil());
  }

  #[tokio::test]
  async fn test_create_project_invalid_directory() {
    let (project_manager, _temp_dir) = create_test_project_manager().await;

    let result = project_manager
      .create_project(
        "Test Project".to_string(),
        "/non/existent/directory".to_string(),
        None,
        None,
      )
      .await;

    assert!(result.is_err());
    // The error could be from path normalization or Project::new validation
    let error_msg = result.unwrap_err().to_string();
    assert!(error_msg.contains("does not exist") || error_msg.contains("No such file"));
  }

  #[tokio::test]
  async fn test_get_project() {
    let (project_manager, temp_dir) = create_test_project_manager().await;

    // Create a test directory
    let test_dir = temp_dir.path().join("test_project");
    std::fs::create_dir(&test_dir).unwrap();

    let created_project = project_manager
      .create_project(
        "Test Project".to_string(),
        test_dir.to_str().unwrap().to_string(),
        None,
        None,
      )
      .await
      .unwrap();

    // Get the project
    let retrieved_project = project_manager
      .get_project(created_project.id)
      .await
      .unwrap()
      .unwrap();

    assert_eq!(retrieved_project.id, created_project.id);
    assert_eq!(retrieved_project.name, created_project.name);
    assert_eq!(retrieved_project.directory, created_project.directory);
  }

  #[tokio::test]
  async fn test_get_project_not_found() {
    let (project_manager, _temp_dir) = create_test_project_manager().await;

    let result = project_manager.get_project(Uuid::now_v7()).await.unwrap();
    assert!(result.is_none());
  }

  #[tokio::test]
  async fn test_list_projects() {
    let (project_manager, temp_dir) = create_test_project_manager().await;

    // Create multiple projects
    let mut project_ids = Vec::new();
    for i in 0..3 {
      let test_dir = temp_dir.path().join(format!("project_{}", i));
      std::fs::create_dir(&test_dir).unwrap();

      let project = project_manager
        .create_project(
          format!("Project {}", i),
          test_dir.to_str().unwrap().to_string(),
          None,
          None,
        )
        .await
        .unwrap();
      project_ids.push(project.id);
    }

    let projects = project_manager.list_projects().await.unwrap();
    assert_eq!(projects.len(), 3);

    // Verify they're sorted by created_at descending
    for i in 1..projects.len() {
      assert!(projects[i - 1].created_at >= projects[i].created_at);
    }
  }

  #[tokio::test]
  async fn test_update_project() {
    let (project_manager, temp_dir) = create_test_project_manager().await;

    // Create a project
    let test_dir = temp_dir.path().join("test_project");
    std::fs::create_dir(&test_dir).unwrap();

    let project = project_manager
      .create_project(
        "Original Name".to_string(),
        test_dir.to_str().unwrap().to_string(),
        Some("Original description".to_string()),
        None,
      )
      .await
      .unwrap();

    let original_updated_at = project.updated_at;

    // Small delay to ensure time difference
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    // Update the project
    let updated_project = project_manager
      .update_project(
        project.id,
        Some("New Name".to_string()),
        Some(Some("New description".to_string())),
        None,
      )
      .await
      .unwrap()
      .unwrap();

    assert_eq!(updated_project.name, "New Name");
    assert_eq!(
      updated_project.description,
      Some("New description".to_string())
    );
    assert!(updated_project.updated_at > original_updated_at);
  }

  #[tokio::test]
  async fn test_delete_project() {
    let (project_manager, temp_dir) = create_test_project_manager().await;

    // Create a project
    let test_dir = temp_dir.path().join("test_project");
    std::fs::create_dir(&test_dir).unwrap();

    let project = project_manager
      .create_project(
        "To Delete".to_string(),
        test_dir.to_str().unwrap().to_string(),
        None,
        None,
      )
      .await
      .unwrap();

    // Delete it
    let deleted = project_manager.delete_project(project.id).await.unwrap();
    assert!(deleted);

    // Verify it's gone
    let result = project_manager.get_project(project.id).await.unwrap();
    assert!(result.is_none());

    // Try to delete again
    let deleted_again = project_manager.delete_project(project.id).await.unwrap();
    assert!(!deleted_again);
  }

  #[tokio::test]
  async fn test_find_by_directory() {
    let (project_manager, temp_dir) = create_test_project_manager().await;

    // Create a project
    let test_dir = temp_dir.path().join("test_project");
    std::fs::create_dir(&test_dir).unwrap();
    let dir_path = test_dir.to_str().unwrap().to_string();

    let project = project_manager
      .create_project("Test Project".to_string(), dir_path.clone(), None, None)
      .await
      .unwrap();

    // Find by directory
    let found = project_manager
      .get_by_directory(&dir_path)
      .await
      .unwrap()
      .unwrap();

    assert_eq!(found.id, project.id);
    // The stored directory should be the normalized version of the original path
    assert_eq!(found.directory, project.directory);

    // Try non-existent directory
    let not_found = project_manager
      .get_by_directory("/non/existent")
      .await
      .unwrap();
    assert!(not_found.is_none());
  }

  #[tokio::test]
  async fn test_mark_for_deletion() {
    let (project_manager, temp_dir) = create_test_project_manager().await;

    // Create a project
    let test_dir = temp_dir.path().join("test_project");
    std::fs::create_dir(&test_dir).unwrap();

    let project = project_manager
      .create_project(
        "Test Project".to_string(),
        test_dir.to_str().unwrap().to_string(),
        None,
        None,
      )
      .await
      .unwrap();

    // Mark it for deletion
    let marked_project = project_manager
      .mark_for_deletion(project.id)
      .await
      .unwrap()
      .unwrap();

    assert_eq!(marked_project.status, ProjectStatus::PendingDeletion);
    assert!(marked_project.deletion_requested_at.is_some());
    assert!(marked_project.updated_at > project.updated_at);

    // Try to mark again - should return the same status
    let marked_again = project_manager
      .mark_for_deletion(project.id)
      .await
      .unwrap()
      .unwrap();

    assert_eq!(marked_again.status, ProjectStatus::PendingDeletion);
    assert_eq!(
      marked_again.deletion_requested_at,
      marked_project.deletion_requested_at
    );
  }

  #[tokio::test]
  async fn test_mark_for_deletion_nonexistent() {
    let (project_manager, _temp_dir) = create_test_project_manager().await;

    let result = project_manager
      .mark_for_deletion(Uuid::now_v7())
      .await
      .unwrap();
    assert!(result.is_none());
  }

  #[tokio::test]
  async fn test_create_project_duplicate_directory() {
    let (project_manager, temp_dir) = create_test_project_manager().await;

    // Create a test directory
    let test_dir = temp_dir.path().join("test_project");
    std::fs::create_dir(&test_dir).unwrap();
    let dir_path = test_dir.to_str().unwrap().to_string();

    // Create first project
    let project1 = project_manager
      .create_project("Project 1".to_string(), dir_path.clone(), None, None)
      .await
      .unwrap();

    // Try to create another project with the same directory
    let result = project_manager
      .create_project("Project 2".to_string(), dir_path.clone(), None, None)
      .await;

    assert!(result.is_err());
    match result.unwrap_err() {
      IndexerError::ProjectAlreadyExists {
        directory,
        existing_id,
      } => {
        assert!(directory.ends_with("test_project"));
        assert_eq!(existing_id, project1.id);
      }
      _ => panic!("Expected ProjectAlreadyExists error"),
    }

    // Verify only one project exists
    let projects = project_manager.list_projects().await.unwrap();
    assert_eq!(projects.len(), 1);
    assert_eq!(projects[0].id, project1.id);
  }

  /// Test database query logic for finding projects that need rescanning
  #[tokio::test]
  async fn test_find_projects_needing_rescan_query() {
    let (project_manager, temp_dir) = create_test_project_manager().await;

    // Create test projects with different rescanning configurations
    let test_project_dir = temp_dir.path().join("rescan_test");
    std::fs::create_dir(&test_project_dir).unwrap();

    let now = chrono::Utc::now();
    let one_day_ago = now.naive_utc() - chrono::Duration::days(1);

    // Project 1: Should rescan (never indexed, rescan enabled)
    let project1 = Project::new(
      "Never Indexed Project".to_string(),
      test_project_dir.to_str().unwrap().to_string(),
      None,
    )
    .unwrap()
    .with_rescan_interval(Some(std::time::Duration::from_secs(3600))); // 1 hour

    // Project 2: Should rescan (last indexed more than interval ago)
    let mut project2 = Project::new(
      "Old Indexed Project".to_string(),
      test_project_dir.to_str().unwrap().to_string(),
      None,
    )
    .unwrap()
    .with_rescan_interval(Some(std::time::Duration::from_secs(3600))); // 1 hour
    project2.last_indexed_at = Some(one_day_ago); // Indexed 1 day ago, should rescan

    // Project 3: Should NOT rescan (last indexed recently)
    let mut project3 = Project::new(
      "Recently Indexed Project".to_string(),
      test_project_dir.to_str().unwrap().to_string(),
      None,
    )
    .unwrap()
    .with_rescan_interval(Some(std::time::Duration::from_secs(3600))); // 1 hour
    project3.last_indexed_at = Some(now.naive_utc() - chrono::Duration::seconds(3599)); // Indexed recently, should NOT rescan yet

    // Project 4: Should NOT rescan (rescanning disabled)
    let project4 = Project::new(
      "Rescanning Disabled Project".to_string(),
      test_project_dir.to_str().unwrap().to_string(),
      None,
    )
    .unwrap(); // No rescan_interval set (None)

    // Insert projects into the database
    let table = project_manager.project_table.write().await;
    for project in [&project1, &project2, &project3, &project4] {
      table
        .add(project.clone().into_arrow().unwrap())
        .execute()
        .await
        .unwrap();
    }
    drop(table);

    // Test the query logic directly
    let projects_needing_rescan = project_manager
      .find_projects_needing_rescan(now)
      .await
      .unwrap();

    // Debug: Print what we found
    println!(
      "Found {} projects needing rescan:",
      projects_needing_rescan.len()
    );
    for project in &projects_needing_rescan {
      println!(
        "- Project '{}' (ID: {}), last_indexed_at: {:?}, rescan_interval: {:?}",
        project.name, project.id, project.last_indexed_at, project.rescan_interval
      );
    }

    // Should have 2 projects: project1 (never indexed) and project2 (old indexed)
    assert_eq!(
      projects_needing_rescan.len(),
      2,
      "Should find 2 projects needing rescan"
    );

    // Verify the projects are the correct ones
    let mut found_project_ids: Vec<Uuid> = projects_needing_rescan.iter().map(|p| p.id).collect();
    found_project_ids.sort();
    let mut expected_project_ids = vec![project1.id, project2.id];
    expected_project_ids.sort();

    assert_eq!(
      found_project_ids, expected_project_ids,
      "Should find never-indexed and old-indexed projects"
    );
  }

  /// Test task submission logic for rescanning projects
  #[tokio::test]
  async fn test_rescanning_task_submission() {
    let (project_manager, temp_dir) = create_test_project_manager().await;

    // Create test projects with different rescanning configurations
    let test_project_dir = temp_dir.path().join("rescan_test");
    std::fs::create_dir(&test_project_dir).unwrap();

    let now = chrono::Utc::now();
    let one_day_ago = now.naive_utc() - chrono::Duration::days(1);

    // Create only projects that should be rescanned for simplicity
    let project1 = Project::new(
      "Never Indexed Project".to_string(),
      test_project_dir.to_str().unwrap().to_string(),
      None,
    )
    .unwrap()
    .with_rescan_interval(Some(std::time::Duration::from_secs(3600))); // 1 hour

    let mut project2 = Project::new(
      "Old Indexed Project".to_string(),
      test_project_dir.to_str().unwrap().to_string(),
      None,
    )
    .unwrap()
    .with_rescan_interval(Some(std::time::Duration::from_secs(3600))); // 1 hour
    project2.last_indexed_at = Some(one_day_ago); // Indexed 1 day ago, should rescan

    // Insert projects into the database
    let table = project_manager.project_table.write().await;
    for project in [&project1, &project2] {
      table
        .add(project.clone().into_arrow().unwrap())
        .execute()
        .await
        .unwrap();
    }
    drop(table);

    // Manually find projects needing rescan and submit tasks, mimicking rescan_worker
    let projects_to_rescan = project_manager
      .find_projects_needing_rescan(now)
      .await
      .unwrap();
    for project in projects_to_rescan {
      project_manager
        .task_manager
        .submit_task(
          project.id,
          Path::new(&project.directory),
          crate::models::TaskType::FullIndex,
        )
        .await
        .unwrap();
    }

    // Verify that the correct tasks were submitted
    let tasks = project_manager.task_manager.list_tasks(10).await.unwrap();
    assert_eq!(tasks.len(), 2, "Should have submitted 2 rescan tasks");

    // Verify task types are FullIndex
    for task in &tasks {
      assert_eq!(
        task.task_type,
        crate::models::TaskType::FullIndex,
        "Rescan tasks should be FullIndex type"
      );
      assert_eq!(
        task.status,
        crate::models::TaskStatus::Pending,
        "Rescan tasks should be pending"
      );
    }

    // Verify the tasks are for the correct projects
    let mut task_project_ids: Vec<Uuid> = tasks.iter().map(|t| t.project_id).collect();
    task_project_ids.sort();
    let mut expected_project_ids = vec![project1.id, project2.id];
    expected_project_ids.sort();

    assert_eq!(
      task_project_ids, expected_project_ids,
      "Tasks should be submitted for the correct projects"
    );
  }

  #[tokio::test]
  async fn test_rescanning_interval_calculations() {
    let (project_manager, temp_dir) = create_test_project_manager().await;

    let test_project_dir = temp_dir.path().join("interval_test");
    std::fs::create_dir(&test_project_dir).unwrap();

    // Test different rescan intervals
    let intervals = vec![
      (std::time::Duration::from_secs(60), "1 minute"),
      (std::time::Duration::from_secs(3600), "1 hour"),
      (std::time::Duration::from_secs(86400), "1 day"),
      (std::time::Duration::from_secs(604800), "1 week"),
    ];

    for (interval, description) in intervals {
      let now = chrono::Utc::now();
      let last_indexed_at = now.naive_utc()
        - chrono::Duration::from_std(interval).unwrap()
        - chrono::Duration::minutes(1);

      let mut project = Project::new(
        format!("Test Project {}", description),
        test_project_dir.to_str().unwrap().to_string(),
        None,
      )
      .unwrap()
      .with_rescan_interval(Some(interval));
      project.last_indexed_at = Some(last_indexed_at);

      // Insert project
      let table = project_manager.project_table.write().await;
      table
        .add(project.clone().into_arrow().unwrap())
        .execute()
        .await
        .unwrap();
      drop(table);

      // Query for projects requiring rescan using the same logic as ProjectManager
      let table = project_manager.project_table.read().await;
      let now_str = now.to_rfc3339();
      let filter = format!(
        "status = 'Active' AND rescan_interval IS NOT NULL AND (last_indexed_at IS NULL OR (last_indexed_at + rescan_interval) <= timestamp '{}')",
        now_str
      );

      let mut stream = table.query().only_if(&filter).execute().await.unwrap();
      let mut found_project = false;

      while let Some(batch) = stream.try_next().await.unwrap() {
        for i in 0..batch.num_rows() {
          let found = Project::from_record_batch(&batch, i).unwrap();
          if found.id == project.id {
            found_project = true;
            break;
          }
        }
      }
      drop(table);

      assert!(
        found_project,
        "Project with {} interval should be found for rescanning",
        description
      );

      // Clean up for next iteration
      let table = project_manager.project_table.write().await;
      table
        .delete(&format!("id = '{}'", project.id))
        .await
        .unwrap();
      drop(table);
    }
  }

  #[tokio::test]
  async fn test_rescanning_edge_cases() {
    let (project_manager, temp_dir) = create_test_project_manager().await;

    let test_project_dir = temp_dir.path().join("edge_test");
    std::fs::create_dir(&test_project_dir).unwrap();

    // Test Case 1: Project with PendingDeletion status (should be ignored)
    let mut project_pending_deletion = Project::new(
      "Pending Deletion Project".to_string(),
      test_project_dir.to_str().unwrap().to_string(),
      None,
    )
    .unwrap()
    .with_rescan_interval(Some(std::time::Duration::from_secs(60)));
    project_pending_deletion.status = ProjectStatus::PendingDeletion;
    project_pending_deletion.deletion_requested_at = Some(chrono::Utc::now().naive_utc());

    // Test Case 2: Project with Deleted status (should be ignored)
    let mut project_deleted = Project::new(
      "Deleted Project".to_string(),
      test_project_dir.to_str().unwrap().to_string(),
      None,
    )
    .unwrap()
    .with_rescan_interval(Some(std::time::Duration::from_secs(60)));
    project_deleted.status = ProjectStatus::Deleted;

    // Test Case 3: Active project with NULL rescan_interval (should be ignored)
    let project_no_interval = Project::new(
      "No Interval Project".to_string(),
      test_project_dir.to_str().unwrap().to_string(),
      None,
    )
    .unwrap(); // rescan_interval is None by default

    // Test Case 4: Active project with very large interval (should be ignored)
    let mut project_large_interval = Project::new(
      "Large Interval Project".to_string(),
      test_project_dir.to_str().unwrap().to_string(),
      None,
    )
    .unwrap()
    .with_rescan_interval(Some(std::time::Duration::from_secs(365 * 24 * 3600))); // 1 year
    project_large_interval.last_indexed_at =
      Some(chrono::Utc::now().naive_utc() - chrono::Duration::hours(1));

    // Insert all projects
    let table = project_manager.project_table.write().await;
    for project in [
      &project_pending_deletion,
      &project_deleted,
      &project_no_interval,
      &project_large_interval,
    ] {
      table
        .add(project.clone().into_arrow().unwrap())
        .execute()
        .await
        .unwrap();
    }
    drop(table);

    // Manually find projects needing rescan, mimicking rescan_worker
    let projects_to_rescan = project_manager
      .find_projects_needing_rescan(chrono::Utc::now())
      .await
      .unwrap();

    // Verify no tasks were submitted (because no projects should need rescanning)
    assert_eq!(
      projects_to_rescan.len(),
      0,
      "No projects should need rescanning for edge cases"
    );
    let tasks = project_manager.task_manager.list_tasks(10).await.unwrap();
    assert_eq!(
      tasks.len(),
      0,
      "No rescan tasks should be submitted for edge cases"
    );
  }

  #[tokio::test]
  async fn test_find_by_path_longest_prefix() {
    let (project_manager, temp_dir) = create_test_project_manager().await;

    // Create a small directory tree
    let root = temp_dir.path().join("prefix_root");
    std::fs::create_dir(&root).unwrap();
    let dir_a = root.join("a");
    std::fs::create_dir(&dir_a).unwrap();
    let dir_ab = dir_a.join("b");
    std::fs::create_dir(&dir_ab).unwrap();

    // Create two projects, one nested inside the other
    let proj_a = project_manager
      .create_project(
        "Proj A".to_string(),
        dir_a.to_str().unwrap().to_string(),
        None,
        None,
      )
      .await
      .unwrap();
    let proj_ab = project_manager
      .create_project(
        "Proj AB".to_string(),
        dir_ab.to_str().unwrap().to_string(),
        None,
        None,
      )
      .await
      .unwrap();

    // A deeper path under the more specific project directory
    let deep_path = dir_ab.join("c").join("file.txt");
    let found = project_manager
      .find_by_path(&deep_path)
      .await
      .unwrap()
      .expect("should find a project");

    // Should choose the longest matching prefix (proj_ab)
    assert_eq!(found.id, proj_ab.id);
    assert_ne!(found.id, proj_a.id);
  }

  #[tokio::test]
  async fn test_find_by_path_no_match() {
    let (project_manager, temp_dir) = create_test_project_manager().await;

    // Create one project
    let dir_a = temp_dir.path().join("some_project");
    std::fs::create_dir(&dir_a).unwrap();
    project_manager
      .create_project(
        "Some Project".to_string(),
        dir_a.to_str().unwrap().to_string(),
        None,
        None,
      )
      .await
      .unwrap();

    // Query with an absolute path outside any project roots
    let unrelated = temp_dir.path().join("unrelated").join("path");
    // no need to create; path just needs to be absolute
    let result = project_manager.find_by_path(&unrelated).await.unwrap();
    assert!(result.is_none());
  }

  #[tokio::test]
  async fn test_find_by_path_requires_absolute() {
    let (project_manager, _temp_dir) = create_test_project_manager().await;

    // Relative path should error
    let rel = Path::new("relative/path");
    let err = project_manager
      .find_by_path(rel)
      .await
      .expect_err("expected error");
    match err {
      IndexerError::PathNotAbsolute(msg) => {
        assert!(msg.contains("relative/path"));
      }
      other => panic!("unexpected error: {other:?}"),
    }
  }
}
