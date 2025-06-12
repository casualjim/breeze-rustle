use arrow::array::{
  FixedSizeBinaryArray, FixedSizeListArray, Float32Array, NullArray, StringArray,
  TimestampNanosecondArray, UInt64Array,
};
use arrow::record_batch::RecordBatch;
use futures_util::StreamExt;
use lancedb::arrow::{RecordBatchStream, SimpleRecordBatchStream};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::pin::Pin;
use std::sync::Arc;

use crate::pipeline::{BoxStream, RecordBatchConverter};

/// Mock converter that creates empty record batches
pub struct MockRecordBatchConverter<T> {
  batch_size: NonZeroUsize,
  schema: Arc<arrow::datatypes::Schema>,
  fail_after: Option<usize>,
  _phantom: PhantomData<T>,
}

impl<T> MockRecordBatchConverter<T> {
  pub fn new(batch_size: NonZeroUsize, schema: Arc<arrow::datatypes::Schema>) -> Self {
    Self {
      batch_size,
      schema,
      fail_after: None,
      _phantom: PhantomData,
    }
  }

  pub fn with_failure(
    batch_size: NonZeroUsize,
    schema: Arc<arrow::datatypes::Schema>,
    fail_after: usize,
  ) -> Self {
    Self {
      batch_size,
      schema,
      fail_after: Some(fail_after),
      _phantom: PhantomData,
    }
  }
}

impl<T: Send + 'static> RecordBatchConverter<T> for MockRecordBatchConverter<T> {
  fn convert(&self, items: BoxStream<T>) -> Pin<Box<dyn RecordBatchStream + Send>> {
    let batch_size = self.batch_size.get();
    let fail_after = self.fail_after;
    let mut processed = 0;

    let schema_clone = self.schema.clone();

    let batch_stream = items
      .ready_chunks(batch_size)
      .map(move |items| {
        processed += 1;

        if let Some(fail_at) = fail_after {
          if processed > fail_at {
            return Err(lancedb::Error::Arrow {
              source: arrow::error::ArrowError::InvalidArgumentError(
                "Mock converter failure".to_string(),
              ),
            });
          }
        }

        // Create a batch with mock data that has the correct number of rows
        let num_rows = items.len();

        // Create arrays for each field in the schema
        let arrays: Vec<Arc<dyn arrow::array::Array>> = schema_clone
          .fields()
          .iter()
          .map(|field| {
            match field.name().as_str() {
              "id" => Arc::new(arrow::array::StringArray::from(vec!["mock-id"; num_rows]))
                as Arc<dyn arrow::array::Array>,
              "file_path" => Arc::new(arrow::array::StringArray::from(vec!["mock-path"; num_rows]))
                as Arc<dyn arrow::array::Array>,
              "content" => Arc::new(arrow::array::StringArray::from(vec![
                "mock-content";
                num_rows
              ])) as Arc<dyn arrow::array::Array>,
              "content_hash" => Arc::new(arrow::array::FixedSizeBinaryArray::from(vec![
                [0u8; 32]
                  .as_ref();
                num_rows
              ])) as Arc<dyn arrow::array::Array>,
              "content_embedding" => {
                // Get the embedding dimension from the field type
                let dim =
                  if let arrow::datatypes::DataType::FixedSizeList(_, size) = field.data_type() {
                    *size as usize
                  } else {
                    384 // default
                  };
                let values = vec![0.0f32; num_rows * dim];
                let value_array = arrow::array::Float32Array::from(values);
                Arc::new(arrow::array::FixedSizeListArray::new(
                  Arc::new(arrow::datatypes::Field::new(
                    "item",
                    arrow::datatypes::DataType::Float32,
                    true,
                  )),
                  dim as i32,
                  Arc::new(value_array),
                  None,
                )) as Arc<dyn arrow::array::Array>
              }
              "file_size" => Arc::new(arrow::array::UInt64Array::from(vec![100u64; num_rows]))
                as Arc<dyn arrow::array::Array>,
              "last_modified" | "indexed_at" => {
                let now = chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0);
                Arc::new(arrow::array::TimestampNanosecondArray::from(vec![
                  now;
                  num_rows
                ])) as Arc<dyn arrow::array::Array>
              }
              _ => Arc::new(arrow::array::NullArray::new(num_rows)) as Arc<dyn arrow::array::Array>,
            }
          })
          .collect();

        RecordBatch::try_new(schema_clone.clone(), arrays)
          .map_err(|e| lancedb::Error::Arrow { source: e })
      })
      .boxed();

    Box::pin(SimpleRecordBatchStream::new(
      batch_stream,
      self.schema.clone(),
    ))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::models::CodeDocument;
  use futures_util::stream;
  use uuid::Uuid;

  fn create_test_document() -> CodeDocument {
    CodeDocument {
      id: Uuid::now_v7().to_string(),
      file_path: "test.py".to_string(),
      content: "test content".to_string(),
      content_hash: [0u8; 32],
      content_embedding: vec![1.0, 2.0, 3.0],
      file_size: 100,
      last_modified: chrono::Utc::now().naive_utc(),
      indexed_at: chrono::Utc::now().naive_utc(),
    }
  }

  #[tokio::test]
  async fn test_mock_converter() {
    let schema = Arc::new(CodeDocument::schema(3));
    let converter =
      MockRecordBatchConverter::<CodeDocument>::new(NonZeroUsize::new(2).unwrap(), schema);
    let docs = vec![create_test_document(); 5];
    let stream = stream::iter(docs).boxed();

    let mut batch_stream = converter.convert(stream);
    let mut count = 0;

    while let Some(result) = batch_stream.next().await {
      assert!(result.is_ok());
      count += 1;
    }

    assert_eq!(count, 3); // 5 docs / 2 per batch = 3 batches
  }

  #[tokio::test]
  async fn test_mock_converter_with_failure() {
    let schema = Arc::new(CodeDocument::schema(3));
    let converter = MockRecordBatchConverter::<CodeDocument>::with_failure(
      NonZeroUsize::new(1).unwrap(),
      schema,
      2,
    );

    let docs = vec![create_test_document(); 5];
    let stream = stream::iter(docs).boxed();

    let mut batch_stream = converter.convert(stream);
    let mut success_count = 0;
    let mut error_count = 0;

    while let Some(result) = batch_stream.next().await {
      match result {
        Ok(_) => success_count += 1,
        Err(_) => error_count += 1,
      }
    }

    assert_eq!(success_count, 2);
    assert_eq!(error_count, 3);
  }
}
