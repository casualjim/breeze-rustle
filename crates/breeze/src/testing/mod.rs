pub mod mock_converter;
pub mod mock_document_builder;
pub mod mock_embedder;
pub mod mock_indexer;
pub mod mock_sink;

pub use mock_converter::MockRecordBatchConverter;
pub use mock_document_builder::MockDocumentBuilder;
pub use mock_embedder::MockEmbedder;
pub use mock_indexer::MockPathWalker;
pub use mock_sink::MockSink;

#[cfg(test)]
mod tests {
  use std::num::NonZeroUsize;
  use std::path::Path;
  use std::sync::Arc;

  use crate::config::Config;
  use crate::models::CodeDocument;
  use crate::pipeline::*;

  use super::*;
  use futures_util::StreamExt;

  #[tokio::test]
  async fn test_complete_ingestion_pipeline() {
    let path = Path::new("/test/repo");

    // Create pipeline stages
    let walker = MockPathWalker::new(2, 2); // 2 files, 2 chunks per file
    let embedder = MockEmbedder::new(384);
    let document_builder = MockDocumentBuilder::new();
    let converter = MockRecordBatchConverter::<CodeDocument>::new(
      NonZeroUsize::new(1).unwrap(), // Batch size of 1 to ensure 2 batches
      Arc::new(CodeDocument::schema(384)),
    );
    let sink = MockSink::new();

    // Connect the pipeline: PathWalker -> Embedder -> DocumentBuilder -> RecordBatchConverter -> Sink
    let files = walker.walk(path);
    let files_with_embeddings = embedder.embed(files);
    let documents = document_builder.build_documents(files_with_embeddings);
    let record_batches = converter.convert(documents);

    // Store batches in sink
    let mut sink_stream = sink.sink(record_batches);

    // Process the entire pipeline
    let mut batch_count = 0;
    while let Some(_) = sink_stream.next().await {
      batch_count += 1;
      // Each iteration processes batches through the sink
    }
    println!("Received {} batches", batch_count);

    // Verify results
    assert_eq!(sink.batch_count(), 2, "Should have received 2 batches (batch size 1)");
    assert_eq!(sink.row_count(), 2, "Should have processed 2 rows total (one per file)");

    let schema = sink.get_schema().expect("Should have schema");
    assert!(schema.field_with_name("id").is_ok());
    assert!(schema.field_with_name("file_path").is_ok());
    assert!(schema.field_with_name("content_embedding").is_ok());
  }
}
