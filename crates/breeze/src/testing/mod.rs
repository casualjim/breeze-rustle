pub mod mock_indexer;
pub mod mock_batcher;
pub mod mock_embedder;
pub mod mock_aggregator;
pub mod mock_sink;
pub mod mock_converter;

pub use mock_indexer::MockPathWalker;
pub use mock_batcher::MockBatcher;
pub use mock_embedder::MockEmbedder;
pub use mock_aggregator::MockAggregator;
pub use mock_sink::MockSink;
pub use mock_converter::MockRecordBatchConverter;

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::num::NonZeroUsize;
    use std::sync::Arc;

    use crate::config::Config;
    use crate::pipeline::*;
    use crate::models::CodeDocument;

    use super::*;
    use futures_util::StreamExt;
    
    #[tokio::test]
    async fn test_complete_ingestion_pipeline() {
        let path = Path::new("/test/repo");
        
        // Create pipeline stages
        let walker = MockPathWalker::default();
        let batcher = MockBatcher::new(2);
        let embedder = MockEmbedder::new(384);
        let aggregator = MockAggregator;
        let converter = MockRecordBatchConverter::<CodeDocument>::new(
            NonZeroUsize::new(10).unwrap(),
            Arc::new(CodeDocument::schema(384))
        );
        let sink = MockSink::new();

        // Connect the pipeline
        let chunks = walker.walk(path);
        let batches = batcher.batch(chunks);
        let embeddings = embedder.embed(batches);
        let documents = aggregator.aggregate(embeddings);
        let record_batches = converter.convert(documents);
        
        // Store batches in sink
        let mut sink_stream = sink.sink(record_batches);
        
        // Process the entire pipeline
        while let Some(_) = sink_stream.next().await {
            // Each iteration processes batches through the sink
        }
        
        // Verify results
        assert_eq!(sink.batch_count(), 1, "Should have received 1 batch");
        assert_eq!(sink.row_count(), 3, "Should have processed 3 rows total");
        
        let schema = sink.get_schema().expect("Should have schema");
        assert!(schema.field_with_name("id").is_ok());
        assert!(schema.field_with_name("file_path").is_ok());
        assert!(schema.field_with_name("content_embedding").is_ok());
    }
}