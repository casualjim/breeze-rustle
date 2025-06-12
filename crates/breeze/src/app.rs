use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::config::Config;
use crate::embeddings::{
    loader::load_embedder,
    sentence_transformer::SentenceTransformerEmbedder,
    models::ModelType,
};
use crate::models::CodeDocument;
use crate::indexer::Indexer;

pub struct App {
    config: Config,
    embedder: SentenceTransformerEmbedder,
    table: Arc<RwLock<lancedb::Table>>,
}

impl App {
    /// Create a new App instance with the given configuration
    pub async fn new(config: Config) -> Result<Self, Box<dyn std::error::Error>> {
        // Create LanceDB connection
        let connection = lancedb::connect(config.database_path.to_str()
            .ok_or("Invalid database path")?)
            .execute()
            .await?;

        // Load embedder based on config
        let model_type = match config.model.as_str() {
            "ibm-granite/granite-embedding-125m-english" => ModelType::Granite,
            "jinaai/jina-embeddings-v2-base-code" => ModelType::JinaCodeV2,
            "sentence-transformers/all-MiniLM-L6-v2" => ModelType::AllMiniLM,
            _ => return Err(format!("Unsupported local model: {}. Supported models are: {}, {}, {}",
                config.model,
                ModelType::Granite.model_id(),
                ModelType::JinaCodeV2.model_id(),
                ModelType::AllMiniLM.model_id()
            ).into()),
        };
        let embedder = load_embedder(model_type).await?;

        // Ensure table exists
        let embedding_dim = embedder.embedding_dim();
        let table = CodeDocument::ensure_table(
            &connection,
            &config.table_name,
            embedding_dim,
        ).await?;

        Ok(Self {
            config: config.clone(),
            embedder,
            table: Arc::new(RwLock::new(table)),
        })
    }

    /// Index a repository at the given path
    pub async fn index(&self, path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
        // Create an indexer with our resources
        let indexer = Indexer::new(
            &self.config,
            &self.embedder,
            self.table.clone(),
        );

        // Run the indexing
        indexer.index(path).await
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_app_index() {
        // Create test config
        let mut config = Config::default();

        // Use temp directory for database
        let temp_db = tempdir().unwrap();
        config.database_path = temp_db.path().to_path_buf();

        // Create app
        let app = App::new(config).await.unwrap();

        // Create test repository
        let test_repo = tempdir().unwrap();
        let test_file = test_repo.path().join("main.py");
        std::fs::write(&test_file, "def hello_world():\n    print('Hello, world!')").unwrap();

        // Index the repository
        let count = app.index(test_repo.path()).await.unwrap();
        assert_eq!(count, 1);


    }
}
