pub mod mock_indexer;
pub mod mock_batcher;
pub mod mock_embedder;
pub mod mock_aggregator;
pub mod mock_sink;

pub use mock_indexer::MockPathWalker;
pub use mock_batcher::MockBatcher;
pub use mock_embedder::MockEmbedder;
pub use mock_aggregator::MockAggregator;
pub use mock_sink::MockSink;

#[cfg(test)]
mod tests {
    use std::path::Path;

    use crate::config::Config;
    use crate::pipeline::*;

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
        let sink = MockSink::new();

        // Connect the pipeline
        let chunks = walker.walk(path);
        let batches = batcher.batch(chunks);
        let embeddings = embedder.embed(batches);
        let documents = aggregator.aggregate(embeddings);
        
        // Store documents in sink
        let mut sink_stream = sink.sink(documents);
        
        // Process the entire pipeline
        while let Some(_) = sink_stream.next().await {
            // Each iteration processes one document through the sink
        }
        
        // Verify results
        let stored_docs = sink.stored_documents();
        assert_eq!(stored_docs.len(), 3, "Should have stored 3 documents");
        
        // Verify document properties
        for (i, doc) in stored_docs.iter().enumerate() {
            println!("Stored document {}: {} ({} bytes)", i, doc.file_path, doc.file_size);
            assert_eq!(doc.content_embedding.len(), 384, "Embedding should have 384 dimensions");
            assert!(!doc.id.is_empty(), "Document should have an ID");
            assert!(!doc.content.is_empty(), "Document should have content");
        }
        
        // Verify specific documents
        assert!(stored_docs.iter().any(|d| d.file_path == "/test/repo/file0.rs"));
        assert!(stored_docs.iter().any(|d| d.file_path == "/test/repo/file1.rs"));
        assert!(stored_docs.iter().any(|d| d.file_path == "/test/repo/file2.rs"));
    }
}