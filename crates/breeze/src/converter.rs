use std::sync::Arc;
use std::num::NonZeroUsize;
use std::marker::PhantomData;
use std::pin::Pin;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use arrow::compute::concat_batches;
use futures_util::StreamExt;
use lancedb::arrow::{RecordBatchStream, IntoArrow};

use crate::pipeline::{BoxStream, RecordBatchConverter};

/// Generic converter that converts streams of T to Arrow RecordBatch streams
/// where T implements IntoArrow trait from LanceDB
pub struct BufferedRecordBatchConverter<T> {
    batch_size: NonZeroUsize,
    schema: Arc<arrow::datatypes::Schema>,
    _phantom: PhantomData<T>,
}

impl<T> BufferedRecordBatchConverter<T> {
    pub fn new(batch_size: NonZeroUsize, schema: Arc<arrow::datatypes::Schema>) -> Self {
        Self { 
            batch_size, 
            schema,
            _phantom: PhantomData,
        }
    }
    
    
    /// Convert a batch of items to RecordBatchReader
    fn items_to_batch_reader(&self, items: Vec<T>) -> lancedb::Result<Box<dyn arrow::array::RecordBatchReader + Send>>
    where
        T: IntoArrow + 'static,
    {
        if items.is_empty() {
            // Return empty iterator with the correct schema
            return Ok(Box::new(RecordBatchIterator::new(
                vec![].into_iter().map(Ok),
                self.schema.clone(),
            )));
        }
        
        // Convert each item to batches and collect all
        let mut all_batches = Vec::new();
        for item in items {
            let reader = item.into_arrow()?;
            for batch_result in reader {
                let batch = batch_result.map_err(|e| lancedb::Error::Arrow { source: e })?;
                all_batches.push(Ok(batch));
            }
        }
        
        // Return a RecordBatchIterator with all the batches
        Ok(Box::new(RecordBatchIterator::new(
            all_batches.into_iter(),
            self.schema.clone(),
        )))
    }
}

impl<T> Clone for BufferedRecordBatchConverter<T> {
    fn clone(&self) -> Self {
        Self {
            batch_size: self.batch_size,
            schema: self.schema.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T> RecordBatchConverter<T> for BufferedRecordBatchConverter<T>
where
    T: IntoArrow + Send + 'static,
{
    fn convert(&self, items: BoxStream<T>) -> Pin<Box<dyn RecordBatchStream + Send>> {
        let batch_size = self.batch_size.get();
        let converter = self.clone();
        let schema = self.schema.clone();
        
        // Convert items to batches using a simpler approach
        let batch_stream = items
            .ready_chunks(batch_size)
            .map(move |items| {
                let converter = converter.clone();
                // Convert to RecordBatchReader then collect all batches
                match converter.items_to_batch_reader(items) {
                    Ok(reader) => {
                        // Collect all batches from the reader
                        let batches: Vec<Result<RecordBatch, arrow::error::ArrowError>> = reader.collect();
                        
                        if batches.is_empty() {
                            Ok(RecordBatch::new_empty(converter.schema.clone()))
                        } else {
                            // Collect all successful batches
                            let all_batches: Result<Vec<_>, _> = batches.into_iter().collect();
                            match all_batches {
                                Ok(batches) => {
                                    // Concatenate all batches into a single batch
                                    concat_batches(&converter.schema, &batches)
                                        .map_err(|e| lancedb::Error::Arrow { source: e })
                                }
                                Err(e) => Err(lancedb::Error::Arrow { source: e })
                            }
                        }
                    }
                    Err(e) => Err(lancedb::Error::Arrow { 
                        source: arrow::error::ArrowError::from_external_error(Box::new(e))
                    })
                }
            })
            .boxed();
        
        // Return as a SimpleRecordBatchStream
        Box::pin(lancedb::arrow::SimpleRecordBatchStream::new(batch_stream, schema))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::CodeDocument;
    use futures_util::stream;
    use uuid::Uuid;
    
    fn create_test_document(file_path: &str) -> CodeDocument {
        CodeDocument {
            id: Uuid::now_v7().to_string(),
            file_path: file_path.to_string(),
            content: format!("Content of {}", file_path),
            content_hash: [0u8; 32],
            content_embedding: vec![1.0, 2.0, 3.0],
            file_size: 100,
            last_modified: chrono::Utc::now().naive_utc(),
            indexed_at: chrono::Utc::now().naive_utc(),
        }
    }
    
    #[tokio::test]
    async fn test_buffered_conversion() {
        let schema = Arc::new(CodeDocument::schema(3));
        let converter = BufferedRecordBatchConverter::<CodeDocument>::new(
            NonZeroUsize::new(2).unwrap(), 
            schema
        );
        
        let docs = vec![
            create_test_document("file1.py"),
            create_test_document("file2.py"),
            create_test_document("file3.py"),
        ];
        
        let stream = stream::iter(docs).boxed();
        let mut batch_stream = converter.convert(stream);
        
        // Collect all batches
        let mut batches = Vec::new();
        while let Some(result) = batch_stream.next().await {
            let batch = result.unwrap();
            batches.push(batch);
        }
        
        // Should have 2 batches: [2 docs, 1 doc]
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].num_rows(), 2);
        assert_eq!(batches[1].num_rows(), 1);
    }
    
    #[tokio::test]
    async fn test_schema_consistency() {
        let schema = Arc::new(CodeDocument::schema(384));
        let converter = BufferedRecordBatchConverter::<CodeDocument>::new(
            NonZeroUsize::new(10).unwrap(), 
            schema.clone()
        );
        let docs = vec![create_test_document("test.py")];
        let stream = stream::iter(docs).boxed();
        
        let batch_stream = converter.convert(stream);
        let stream_schema = batch_stream.schema();
        
        // Verify schema matches what we provided
        assert_eq!(stream_schema, schema);
        
        // Verify schema matches CodeDocument fields
        assert!(stream_schema.field_with_name("id").is_ok());
        assert!(stream_schema.field_with_name("file_path").is_ok());
        assert!(stream_schema.field_with_name("content").is_ok());
        assert!(stream_schema.field_with_name("content_hash").is_ok());
        assert!(stream_schema.field_with_name("content_embedding").is_ok());
        assert!(stream_schema.field_with_name("file_size").is_ok());
        assert!(stream_schema.field_with_name("last_modified").is_ok());
        assert!(stream_schema.field_with_name("indexed_at").is_ok());
        
        // Check embedding field is correctly configured
        let embedding_field = stream_schema.field_with_name("content_embedding").unwrap();
        match embedding_field.data_type() {
            arrow::datatypes::DataType::FixedSizeList(_, size) => assert_eq!(*size, 384),
            _ => panic!("Expected FixedSizeList for embeddings"),
        }
    }
    
    #[tokio::test]
    async fn test_empty_stream() {
        let schema = Arc::new(CodeDocument::schema(128));
        let converter = BufferedRecordBatchConverter::<CodeDocument>::new(
            NonZeroUsize::new(10).unwrap(), 
            schema
        );
        let stream = stream::empty().boxed();
        let mut batch_stream = converter.convert(stream);
        
        let result = batch_stream.next().await;
        assert!(result.is_none());
    }
}