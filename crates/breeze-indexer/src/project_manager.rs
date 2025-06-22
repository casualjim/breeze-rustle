use std::path::Path;
use std::sync::Arc;
use uuid::Uuid;

use futures::TryStreamExt;
use lancedb::Table;
use lancedb::arrow::IntoArrow;
use lancedb::query::{ExecutableQuery, QueryBase};
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::{IndexerError, models::Project, task_manager::TaskManager};

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

    // Try to canonicalize for consistent lookups
    let canonical = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());

    // Convert to string and remove trailing slash
    let mut path_str = canonical.to_string_lossy().to_string();
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

    let table = self.project_table.read().await;

    let mut stream = table
      .query()
      .only_if(format!("directory = '{}'", canonical_path.replace("'", "''")).as_str())
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

  /// Create a new project
  pub async fn create_project(
    &self,
    name: String,
    directory: String,
    description: Option<String>,
  ) -> Result<Project, IndexerError> {
    // Canonicalize the directory path (validates it's absolute)
    let canonical_directory = Self::canonicalize_path(&directory)?;

    // Create the project (validates directory exists and is a directory)
    let project =
      Project::new(name, canonical_directory, description).map_err(IndexerError::Config)?;

    // Insert into table
    let arrow_data = project.clone().into_arrow()?;

    let table = self.project_table.write().await;
    table.add(arrow_data).execute().await?;

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

    // Always update updated_at
    let now_micros = chrono::Utc::now().naive_utc().and_utc().timestamp_micros();
    update = update.column("updated_at", format!("{}", now_micros));

    update.execute().await?;

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
  pub async fn find_by_directory(&self, directory: &str) -> Result<Option<Project>, IndexerError> {
    let table = self.project_table.read().await;

    let mut stream = table
      .query()
      .only_if(format!("directory = '{}'", directory.replace("'", "''")).as_str())
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
    };

    let embedding_provider = crate::embeddings::factory::create_embedding_provider(&config)
      .await
      .unwrap();

    let bulk_indexer = crate::bulk_indexer::BulkIndexer::new(
      Arc::new(config),
      Arc::from(embedding_provider),
      384, // BAAI/bge-small-en-v1.5 has 384 dimensions
      code_table.clone(),
    );

    let task_manager = Arc::new(TaskManager::new(task_table, bulk_indexer));
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
      .create_project("Test Project".to_string(), dir_path.clone(), None)
      .await
      .unwrap();

    // Find by directory
    let found = project_manager
      .find_by_path(&dir_path)
      .await
      .unwrap()
      .unwrap();

    assert_eq!(found.id, project.id);
    // The stored directory should be the normalized version of the original path
    assert_eq!(found.directory, project.directory);

    // Try non-existent directory
    let not_found = project_manager.find_by_path("/non/existent").await.unwrap();
    assert!(not_found.is_none());
  }
}
