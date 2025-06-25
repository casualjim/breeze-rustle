use reqwest::Client;
use tokio_util::sync::CancellationToken;
use tracing::{info, instrument};

use crate::config::Config;
pub use breeze_server::types::{
  ChunkResult, CreateProjectRequest, Project, SearchRequest, SearchResult, Task, TaskStatus,
  TaskSubmittedResponse, UpdateProjectRequest,
};

pub struct App {
  client: Client,
  api_base: String,
}

impl App {
  /// Create a new App instance with the given configuration
  #[instrument(skip(config, _shutdown_token))]
  pub async fn new(
    config: Config,
    _shutdown_token: CancellationToken,
  ) -> Result<Self, Box<dyn std::error::Error>> {
    info!("Initializing Breeze app");

    let client = Client::builder()
      .timeout(std::time::Duration::from_secs(30))
      .build()?;

    let api_base = format!("http://localhost:{}/api/v1", config.server.ports.http);

    Ok(Self { client, api_base })
  }

  // Project management methods
  pub async fn create_project(
    &self,
    name: String,
    directory: String,
    description: Option<String>,
  ) -> Result<Project, Box<dyn std::error::Error>> {
    let req = CreateProjectRequest {
      name,
      directory,
      description,
    };

    let response = self
      .client
      .post(format!("{}/projects", self.api_base))
      .json(&req)
      .send()
      .await?;

    if response.status().is_success() {
      Ok(response.json().await?)
    } else {
      Err(format!("Failed to create project: {}", response.status()).into())
    }
  }

  pub async fn list_projects(&self) -> Result<Vec<Project>, Box<dyn std::error::Error>> {
    let response = self
      .client
      .get(format!("{}/projects", self.api_base))
      .send()
      .await?;

    if response.status().is_success() {
      Ok(response.json().await?)
    } else {
      Err(format!("Failed to list projects: {}", response.status()).into())
    }
  }

  pub async fn get_project(&self, id: &str) -> Result<Project, Box<dyn std::error::Error>> {
    let response = self
      .client
      .get(format!("{}/projects/{}", self.api_base, id))
      .send()
      .await?;

    if response.status().is_success() {
      Ok(response.json().await?)
    } else if response.status() == 404 {
      Err("Project not found".into())
    } else {
      Err(format!("Failed to get project: {}", response.status()).into())
    }
  }

  pub async fn update_project(
    &self,
    id: &str,
    name: Option<String>,
    description: Option<String>,
  ) -> Result<Project, Box<dyn std::error::Error>> {
    let req = UpdateProjectRequest {
      name,
      description: description.map(Some),
    };

    let response = self
      .client
      .put(format!("{}/projects/{}", self.api_base, id))
      .json(&req)
      .send()
      .await?;

    if response.status().is_success() {
      Ok(response.json().await?)
    } else if response.status() == 404 {
      Err("Project not found".into())
    } else {
      Err(format!("Failed to update project: {}", response.status()).into())
    }
  }

  pub async fn delete_project(&self, id: &str) -> Result<(), Box<dyn std::error::Error>> {
    let response = self
      .client
      .delete(format!("{}/projects/{}", self.api_base, id))
      .send()
      .await?;

    if response.status().is_success() || response.status() == 204 {
      Ok(())
    } else if response.status() == 404 {
      Err("Project not found".into())
    } else {
      Err(format!("Failed to delete project: {}", response.status()).into())
    }
  }

  pub async fn index_project(
    &self,
    id: &str,
  ) -> Result<TaskSubmittedResponse, Box<dyn std::error::Error>> {
    let response = self
      .client
      .post(format!("{}/projects/{}/index", self.api_base, id))
      .send()
      .await?;

    if response.status().is_success() || response.status() == 202 {
      Ok(response.json().await?)
    } else {
      Err(format!("Failed to index project: {}", response.status()).into())
    }
  }

  // Task management methods
  pub async fn get_task(&self, id: &str) -> Result<Task, Box<dyn std::error::Error>> {
    let response = self
      .client
      .get(format!("{}/tasks/{}", self.api_base, id))
      .send()
      .await?;

    if response.status().is_success() {
      Ok(response.json().await?)
    } else if response.status() == 404 {
      Err("Task not found".into())
    } else {
      Err(format!("Failed to get task: {}", response.status()).into())
    }
  }

  pub async fn list_tasks(
    &self,
    limit: Option<usize>,
  ) -> Result<Vec<Task>, Box<dyn std::error::Error>> {
    let mut url = format!("{}/tasks", self.api_base);
    if let Some(l) = limit {
      url.push_str(&format!("?limit={}", l));
    }

    let response = self.client.get(url).send().await?;

    if response.status().is_success() {
      Ok(response.json().await?)
    } else {
      Err(format!("Failed to list tasks: {}", response.status()).into())
    }
  }

  // Search method
  pub async fn search(
    &self,
    req: SearchRequest,
  ) -> Result<Vec<SearchResult>, Box<dyn std::error::Error>> {
    let response = self
      .client
      .post(format!("{}/search", self.api_base))
      .json(&req)
      .send()
      .await?;

    if response.status().is_success() {
      Ok(response.json().await?)
    } else {
      Err(format!("Failed to search: {}", response.status()).into())
    }
  }
}
