use arrow::compute::concat_batches;
use arrow::record_batch::{RecordBatch, RecordBatchIterator};
use futures_util::StreamExt;
use lancedb::arrow::{IntoArrow, RecordBatchStream};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;
use tokio::time::interval;

use crate::pipeline::BoxStream;

/// Generic converter that converts streams of T to Arrow RecordBatch streams
/// where T implements IntoArrow trait from LanceDB
///
/// Batches writes based on:
/// - Maximum documents per batch (configurable)
/// - Timeout duration (configurable)
/// Whichever comes first
pub struct BufferedRecordBatchConverter<T> {
  batch_size: NonZeroUsize,
  timeout_seconds: u64,
  schema: Arc<arrow::datatypes::Schema>,
  _phantom: PhantomData<T>,
}

impl<T> BufferedRecordBatchConverter<T> {
  pub fn new(
    batch_size: NonZeroUsize,
    timeout_seconds: u64,
    schema: Arc<arrow::datatypes::Schema>,
  ) -> Self {
    Self {
      batch_size,
      timeout_seconds,
      schema,
      _phantom: PhantomData,
    }
  }

  /// Create a converter with default timeout of 1 second
  pub fn with_default_timeout(
    batch_size: NonZeroUsize,
    schema: Arc<arrow::datatypes::Schema>,
  ) -> Self {
    Self::new(batch_size, 1, schema)
  }

  /// Set the schema for the converter
  pub fn with_schema(mut self, schema: Arc<arrow::datatypes::Schema>) -> Self {
    self.schema = schema;
    self
  }

  /// Convert a batch of items to RecordBatchReader
  fn items_to_batch_reader(
    &self,
    items: Vec<T>,
  ) -> lancedb::Result<Box<dyn arrow::array::RecordBatchReader + Send>>
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

impl<T> Default for BufferedRecordBatchConverter<T> {
  fn default() -> Self {
    Self {
      batch_size: NonZeroUsize::new(100).unwrap(),
      timeout_seconds: 1,
      schema: Arc::new(arrow::datatypes::Schema::empty()),
      _phantom: PhantomData,
    }
  }
}

impl<T> Clone for BufferedRecordBatchConverter<T> {
  fn clone(&self) -> Self {
    Self {
      batch_size: self.batch_size,
      timeout_seconds: self.timeout_seconds,
      schema: self.schema.clone(),
      _phantom: PhantomData,
    }
  }
}

impl<T> BufferedRecordBatchConverter<T>
where
  T: IntoArrow + Send + 'static,
{
  pub fn convert(&self, items: BoxStream<T>) -> Pin<Box<dyn RecordBatchStream + Send>> {
    let converter = self.clone();
    let schema = self.schema.clone();
    let max_batch_size = self.batch_size.get();
    let timeout_duration = Duration::from_secs(self.timeout_seconds);

    // Create a stream that batches based on count or time
    let batch_stream = async_stream::stream! {
      let mut items = items;
      let mut buffer = Vec::with_capacity(max_batch_size);
      let mut timer = interval(timeout_duration);
      timer.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
      
      loop {
        tokio::select! {
          // Receive items from the stream
          maybe_item = items.next() => {
            match maybe_item {
              Some(item) => {
                buffer.push(item);
                
                // Flush if we hit the max batch size
                if buffer.len() >= max_batch_size {
                  if !buffer.is_empty() {
                    let items_to_process = std::mem::take(&mut buffer);
                    match converter.items_to_batch_reader(items_to_process) {
                      Ok(reader) => {
                        let batches: Vec<Result<RecordBatch, arrow::error::ArrowError>> = reader.collect();
                        if !batches.is_empty() {
                          let all_batches: Result<Vec<_>, _> = batches.into_iter().collect();
                          match all_batches {
                            Ok(batches) => {
                              match concat_batches(&converter.schema, &batches) {
                                Ok(batch) => yield Ok(batch),
                                Err(e) => yield Err(lancedb::Error::Arrow { source: e }),
                              }
                            }
                            Err(e) => yield Err(lancedb::Error::Arrow { source: e }),
                          }
                        }
                      }
                      Err(e) => {
                        yield Err(lancedb::Error::Arrow {
                          source: arrow::error::ArrowError::from_external_error(Box::new(e)),
                        });
                      }
                    }
                  }
                  timer.reset();
                }
              }
              None => {
                // Stream ended, flush remaining items
                if !buffer.is_empty() {
                  let items_to_process = std::mem::take(&mut buffer);
                  match converter.items_to_batch_reader(items_to_process) {
                    Ok(reader) => {
                      let batches: Vec<Result<RecordBatch, arrow::error::ArrowError>> = reader.collect();
                      if !batches.is_empty() {
                        let all_batches: Result<Vec<_>, _> = batches.into_iter().collect();
                        match all_batches {
                          Ok(batches) => {
                            match concat_batches(&converter.schema, &batches) {
                              Ok(batch) => yield Ok(batch),
                              Err(e) => yield Err(lancedb::Error::Arrow { source: e }),
                            }
                          }
                          Err(e) => yield Err(lancedb::Error::Arrow { source: e }),
                        }
                      }
                    }
                    Err(e) => {
                      yield Err(lancedb::Error::Arrow {
                        source: arrow::error::ArrowError::from_external_error(Box::new(e)),
                      });
                    }
                  }
                }
                break;
              }
            }
          }
          
          // Timeout elapsed, flush buffer
          _ = timer.tick() => {
            if !buffer.is_empty() {
              let items_to_process = std::mem::take(&mut buffer);
              match converter.items_to_batch_reader(items_to_process) {
                Ok(reader) => {
                  let batches: Vec<Result<RecordBatch, arrow::error::ArrowError>> = reader.collect();
                  if !batches.is_empty() {
                    let all_batches: Result<Vec<_>, _> = batches.into_iter().collect();
                    match all_batches {
                      Ok(batches) => {
                        match concat_batches(&converter.schema, &batches) {
                          Ok(batch) => yield Ok(batch),
                          Err(e) => yield Err(lancedb::Error::Arrow { source: e }),
                        }
                      }
                      Err(e) => yield Err(lancedb::Error::Arrow { source: e }),
                    }
                  }
                }
                Err(e) => {
                  yield Err(lancedb::Error::Arrow {
                    source: arrow::error::ArrowError::from_external_error(Box::new(e)),
                  });
                }
              }
            }
          }
        }
      }
    }.boxed();

    // Return as a SimpleRecordBatchStream
    Box::pin(lancedb::arrow::SimpleRecordBatchStream::new(
      batch_stream,
      schema,
    ))
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
    // Create converter with batch size of 2 to test batching
    let converter =
      BufferedRecordBatchConverter::<CodeDocument>::new(NonZeroUsize::new(2).unwrap(), 1, schema);

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
    let converter =
      BufferedRecordBatchConverter::<CodeDocument>::default().with_schema(schema.clone());
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
    let converter = BufferedRecordBatchConverter::<CodeDocument>::default().with_schema(schema);
    let stream = stream::empty().boxed();
    let mut batch_stream = converter.convert(stream);

    let result = batch_stream.next().await;
    assert!(result.is_none());
  }
}
