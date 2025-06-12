use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;

use futures_util::StreamExt;
use lancedb::Table;
use tokio::sync::RwLock;

use crate::aggregator::FileAggregator;
use crate::batcher::PassthroughBatcher;
use crate::config::Config;
use crate::converter::BufferedRecordBatchConverter;
use crate::embeddings::sentence_transformer::SentenceTransformerEmbedder;
use crate::models::CodeDocument;
use crate::pipeline::{Aggregator, Batcher, Embedder, PathWalker, RecordBatchConverter, Sink};
use crate::sinks::lancedb_sink::LanceDbSink;
use crate::walker::ProjectWalker;

pub struct Indexer<'a> {
  config: &'a Config,
  embedder: &'a SentenceTransformerEmbedder,
  table: Arc<RwLock<Table>>,
}

impl<'a> Indexer<'a> {
  /// Create a new pipeline with external resources
  pub fn new(
    config: &'a Config,
    embedder: &'a SentenceTransformerEmbedder,
    table: Arc<RwLock<Table>>,
  ) -> Self {
    Self {
      config,
      embedder,
      table,
    }
  }

  /// Run the indexing pipeline
  pub async fn index(&self, path: &Path) -> Result<usize, Box<dyn std::error::Error>> {
    // Create walk options from config
    use breeze_chunkers::{Tokenizer, WalkOptions};

    // Use the tokenizer from the embedding model for consistent tokenization
    let tokenizer = self.embedder.tokenizer();
    let walk_options = WalkOptions {
      max_chunk_size: self.config.max_chunk_size,
      tokenizer: Tokenizer::PreloadedHuggingFace(tokenizer),
      max_parallel: self.config.max_parallel_files,
      max_file_size: self.config.max_file_size,
    };

    let walker = ProjectWalker::new(walk_options);
    let batcher = PassthroughBatcher::single();
    let aggregator = FileAggregator::new();

    let embedding_dim = self.embedder.embedding_dim();
    let schema = Arc::new(CodeDocument::schema(embedding_dim));
    let converter = BufferedRecordBatchConverter::<CodeDocument>::new(
      NonZeroUsize::new(100).unwrap(), // Buffer size for record batches
      schema.clone(),
    );

    let sink = LanceDbSink::new(self.table.clone());

    // Connect the pipeline stages
    let chunks = walker.walk(path);
    let batches = batcher.batch(chunks);
    let embeddings = self.embedder.embed(batches);
    let documents = aggregator.aggregate(embeddings);
    let record_batches = converter.convert(documents);

    // Process through sink
    let mut sink_stream = sink.sink(record_batches);

    while let Some(()) = sink_stream.next().await {
      // Sink processes batches
    }

    // Get the actual count from the table
    let table = self.table.read().await;
    let count = table.count_rows(None).await?;

    Ok(count)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::embeddings::{loader::load_embedder, models::ModelType};
  use tempfile::tempdir;

  #[tokio::test]
  async fn test_indexer() {
    let config = Config::default();

    // Create embedder
    let embedder = load_embedder(ModelType::AllMiniLM).await.unwrap();
    let embedding_dim = embedder.embedding_dim();

    // Create temporary directory for LanceDB
    let temp_db = tempdir().unwrap();
    let connection = lancedb::connect(temp_db.path().to_str().unwrap())
      .execute()
      .await
      .unwrap();

    // Create table
    let table = CodeDocument::ensure_table(&connection, "test_table", embedding_dim)
      .await
      .unwrap();

    // Create pipeline with external resources
    let pipeline = Indexer::new(&config, &embedder, Arc::new(RwLock::new(table)));

    // Create test file
    let test_dir = tempdir().unwrap();
    let test_file = test_dir.path().join("test.py");
    std::fs::write(&test_file, "def hello():\n    print('world')").unwrap();

    // Run indexing
    let count = pipeline.index(test_dir.path()).await.unwrap();
    assert_eq!(count, 1);
  }
}
