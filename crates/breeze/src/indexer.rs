use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use breeze_chunkers::{Tokenizer, WalkOptions};
use futures_util::StreamExt;
use lancedb::Table;
use tokio::sync::RwLock;
use tracing::{debug, info};

use crate::config::Config;
use crate::converter::BufferedRecordBatchConverter;
use crate::document_builder::WeightedAverageDocumentBuilder;
use crate::embeddings::tei::TEIEmbedder;
use crate::models::CodeDocument;
use crate::pipeline::{DocumentBuilder, Embedder, PathWalker, RecordBatchConverter, Sink};
use crate::sinks::lancedb_sink::LanceDbSink;
use crate::walker::ProjectWalker;

pub struct Indexer<'a> {
  config: &'a Config,
  embedder: &'a TEIEmbedder,
  table: Arc<RwLock<Table>>,
}

impl<'a> Indexer<'a> {
  /// Create a new pipeline with external resources
  pub fn new(
    config: &'a Config,
    embedder: &'a TEIEmbedder,
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
    let start_time = Instant::now();
    info!(
      path = %path.display(),
      "Starting indexing"
    );

    // Use the tokenizer from the embedding model for consistent tokenization
    let tokenizer = self.embedder.tokenizer();
    let walk_options = WalkOptions {
      max_chunk_size: self.config.max_chunk_size,
      tokenizer: Tokenizer::PreloadedHuggingFace(tokenizer),
      max_parallel: self.config.max_parallel_files,
      max_file_size: self.config.max_file_size,
    };

    debug!(
      max_chunk_size = self.config.max_chunk_size,
      max_parallel_files = self.config.max_parallel_files,
      max_file_size = ?self.config.max_file_size,
      "Walk options configured"
    );

    let walker = ProjectWalker::new(walk_options);

    let embedding_dim = self.embedder.embedding_dim();
    info!(
      embedding_dim = embedding_dim,
      "Embedding model initialized"
    );

    let document_builder = WeightedAverageDocumentBuilder::new(embedding_dim);
    
    let converter = BufferedRecordBatchConverter::<CodeDocument>::default()
      .with_schema(Arc::new(CodeDocument::schema(embedding_dim)));

    let sink = LanceDbSink::new(self.table.clone());

    // Counters for progress tracking
    let files_processed = Arc::new(AtomicUsize::new(0));
    let files_with_embeddings_processed = Arc::new(AtomicUsize::new(0));
    let documents_processed = Arc::new(AtomicUsize::new(0));
    let batches_written = Arc::new(AtomicUsize::new(0));

    // Connect the pipeline stages with logging
    let files_counter = files_processed.clone();
    let files = walker.walk(path).inspect(move |file| {
      let count = files_counter.fetch_add(1, Ordering::Relaxed) + 1;
      info!(
        file_number = count,
        file_path = %file.file_path,
        file_size = file.metadata.size,
        line_count = file.metadata.line_count,
        language = ?file.metadata.primary_language,
        "Processing file"
      );
    });

    let files_with_embeddings_counter = files_with_embeddings_processed.clone();
    let files_with_embeddings = self.embedder.embed(Box::pin(files)).inspect(move |file| {
      let count = files_with_embeddings_counter.fetch_add(1, Ordering::Relaxed) + 1;
      debug!(
        files_embedded = count,
        file_path = %file.file_path,
        "File chunks embedded"
      );
    });

    let docs_counter = documents_processed.clone();
    let documents = document_builder.build_documents(Box::pin(files_with_embeddings)).inspect(move |doc| {
      let count = docs_counter.fetch_add(1, Ordering::Relaxed) + 1;
      if count % 10 == 0 {
        debug!(
          documents_processed = count,
          current_file = %doc.file_path,
          "Document building progress"
        );
      }
    });

    let record_batches = converter.convert(Box::pin(documents));

    // Process through sink
    let batches_counter = batches_written.clone();
    let mut sink_stream = sink.sink(record_batches).inspect(move |_| {
      let count = batches_counter.fetch_add(1, Ordering::Relaxed) + 1;
      debug!(
        batch_number = count,
        "Written batch to LanceDB"
      );
    });

    while let Some(()) = sink_stream.next().await {
      // Sink processes batches
    }

    // Get the actual count from the table
    let table = self.table.read().await;
    let count = table.count_rows(None).await?;

    let elapsed = start_time.elapsed();
    let total_files = files_processed.load(Ordering::Relaxed);
    let total_files_embedded = files_with_embeddings_processed.load(Ordering::Relaxed);
    let total_documents = documents_processed.load(Ordering::Relaxed);
    let total_batches = batches_written.load(Ordering::Relaxed);

    info!(
      elapsed_seconds = elapsed.as_secs_f64(),
      files_processed = total_files,
      files_embedded = total_files_embedded,
      documents_created = total_documents,
      batches_written = total_batches,
      documents_in_index = count,
      "Indexing completed"
    );

    Ok(count)
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::embeddings::loader::load_tei_embedder;
  use tempfile::tempdir;

  #[tokio::test]
  async fn test_indexer() {
    let config = Config::default();

    // Create embedder
    let embedder = load_tei_embedder(&config.model, "float32", None).await.unwrap();
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
