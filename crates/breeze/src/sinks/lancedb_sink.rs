use futures_util::StreamExt;
use lancedb::arrow::RecordBatchStream;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::pipeline::BoxStream;

/// LanceDB sink for persisting record batches
pub struct LanceDbSink {
  table: Arc<RwLock<lancedb::Table>>,
}

impl LanceDbSink {
  /// Create a new LanceDB sink with a shared table reference
  pub fn new(table: Arc<RwLock<lancedb::Table>>) -> Self {
    Self { table }
  }
}

impl LanceDbSink {
  pub fn sink(&self, batches: std::pin::Pin<Box<dyn RecordBatchStream + Send>>) -> BoxStream<()> {
    let table = self.table.clone();

    let stream = async_stream::stream! {
        let mut batch_count = 0;
        let mut total_rows = 0;

        futures_util::pin_mut!(batches);

        while let Some(batch_result) = batches.next().await {
            match batch_result {
                Ok(batch) => {
                    let num_rows = batch.num_rows();

                    // Get read lock to access table
                    let table_guard = table.read().await;

                    // Use merge_insert for upsert behavior
                    let mut merge = table_guard.merge_insert(&["id"]);
                    merge.when_matched_update_all(None);
                    merge.when_not_matched_insert_all();

                    // Create a single-batch iterator
                    let schema = batch.schema();
                    let batch_iter = arrow::record_batch::RecordBatchIterator::new(
                        vec![Ok(batch)].into_iter(),
                        schema
                    );

                    match merge.execute(Box::new(batch_iter)).await {
                        Ok(result) => {
                            batch_count += 1;
                            total_rows += num_rows;
                            tracing::debug!(
                                batch_number = batch_count,
                                rows = num_rows,
                                table_version = result.version,
                                total_rows = total_rows,
                                "Saved batch to LanceDB"
                            );
                            yield ();
                        }
                        Err(e) => {
                            tracing::error!(
                                error = %e,
                                batch_number = batch_count + 1,
                                rows = num_rows,
                                "Failed to save batch to LanceDB"
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "Error in record batch stream"
                    );
                }
            }
        }

        tracing::info!(
            batches_written = batch_count,
            total_rows = total_rows,
            "LanceDB sink completed"
        );
    };

    Box::pin(stream)
  }
}

impl Clone for LanceDbSink {
  fn clone(&self) -> Self {
    Self {
      table: self.table.clone(),
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::models::CodeDocument;
  use arrow::array::{
    FixedSizeBinaryArray, FixedSizeListArray, Float32Array, StringArray, TimestampNanosecondArray,
    UInt64Array,
  };
  use arrow::record_batch::RecordBatch;
  use futures_util::stream;
  use lancedb::arrow::IntoArrow;
  use lancedb::query::{ExecutableQuery, QueryBase};
  use tempfile::TempDir;

  async fn create_test_table(
    db_path: &str,
    table_name: &str,
    embedding_dim: usize,
  ) -> lancedb::Table {
    let conn = lancedb::connect(db_path).execute().await.unwrap();

    // Create table with schema
    let schema = Arc::new(CodeDocument::schema(embedding_dim));

    // Check if table exists
    let table_names = conn.table_names().execute().await.unwrap();
    if table_names.contains(&table_name.to_string()) {
      conn.open_table(table_name).execute().await.unwrap()
    } else {
      // Create a dummy document to initialize the table
      let mut dummy_doc = CodeDocument::new("dummy.py".to_string(), "dummy content".to_string());
      dummy_doc.content_embedding = vec![0.0f32; embedding_dim];

      let reader = dummy_doc.into_arrow().unwrap();
      let table = conn
        .create_table(table_name, reader)
        .execute()
        .await
        .unwrap();

      // Delete the dummy row
      table.delete("file_path = 'dummy.py'").await.unwrap();

      table
    }
  }

  fn create_test_batch(docs: Vec<CodeDocument>) -> RecordBatch {
    if docs.is_empty() {
      return RecordBatch::new_empty(Arc::new(CodeDocument::schema(384)));
    }

    let embedding_dim = docs[0].content_embedding.len();
    let schema = Arc::new(CodeDocument::schema(embedding_dim));

    // Build arrays from documents
    let ids: Vec<&str> = docs.iter().map(|d| d.id.as_str()).collect();
    let file_paths: Vec<&str> = docs.iter().map(|d| d.file_path.as_str()).collect();
    let contents: Vec<&str> = docs.iter().map(|d| d.content.as_str()).collect();
    let file_sizes: Vec<u64> = docs.iter().map(|d| d.file_size).collect();

    let id_array = StringArray::from(ids);
    let file_path_array = StringArray::from(file_paths);
    let content_array = StringArray::from(contents);

    // Create content_hash array
    let content_hashes: Vec<&[u8]> = docs.iter().map(|d| d.content_hash.as_ref()).collect();
    let content_hash_array = FixedSizeBinaryArray::from(content_hashes);

    // Create embedding array
    let all_embeddings: Vec<f32> = docs
      .iter()
      .flat_map(|d| d.content_embedding.clone())
      .collect();
    let embedding_values = Float32Array::from(all_embeddings);
    let embedding_array = FixedSizeListArray::new(
      Arc::new(arrow::datatypes::Field::new(
        "item",
        arrow::datatypes::DataType::Float32,
        true,
      )),
      embedding_dim as i32,
      Arc::new(embedding_values),
      None,
    );

    let file_size_array = UInt64Array::from(file_sizes);

    // Convert timestamps
    let last_modified: Vec<i64> = docs
      .iter()
      .map(|d| d.last_modified.and_utc().timestamp_nanos_opt().unwrap_or(0))
      .collect();
    let indexed_at: Vec<i64> = docs
      .iter()
      .map(|d| d.indexed_at.and_utc().timestamp_nanos_opt().unwrap_or(0))
      .collect();
    let last_modified_array = TimestampNanosecondArray::from(last_modified);
    let indexed_at_array = TimestampNanosecondArray::from(indexed_at);

    RecordBatch::try_new(
      schema,
      vec![
        Arc::new(id_array),
        Arc::new(file_path_array),
        Arc::new(content_array),
        Arc::new(content_hash_array),
        Arc::new(embedding_array),
        Arc::new(file_size_array),
        Arc::new(last_modified_array),
        Arc::new(indexed_at_array),
      ],
    )
    .expect("Failed to create batch")
  }

  #[tokio::test]
  async fn test_lancedb_sink_basic() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().to_str().unwrap();
    let table = create_test_table(db_path, "test_documents", 3).await;

    let sink = LanceDbSink::new(Arc::new(RwLock::new(table)));

    // Create test documents
    let mut doc1 = CodeDocument::new("file1.py".to_string(), "print('hello')".to_string());
    doc1.content_embedding = vec![1.0, 2.0, 3.0];

    let mut doc2 = CodeDocument::new("file2.py".to_string(), "def main(): pass".to_string());
    doc2.content_embedding = vec![4.0, 5.0, 6.0];

    let batch = create_test_batch(vec![doc1, doc2]);
    let batch_stream = stream::once(async { Ok(batch) });
    let record_stream = Box::pin(lancedb::arrow::SimpleRecordBatchStream::new(
      batch_stream.boxed(),
      Arc::new(CodeDocument::schema(3)),
    ));

    // Process the stream
    let mut sink_stream = sink.sink(record_stream);
    while let Some(_) = sink_stream.next().await {
      // Process each item
    }

    // Verify data was saved
    let conn = lancedb::connect(db_path).execute().await.unwrap();
    let table = conn.open_table("test_documents").execute().await.unwrap();
    let mut query_stream = table.query().execute().await.unwrap();

    let mut results = Vec::new();
    while let Some(batch) = query_stream.next().await {
      results.push(batch.unwrap());
    }

    let total_rows: usize = results.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total_rows, 2);
  }

  #[tokio::test]
  async fn test_lancedb_sink_upsert() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().to_str().unwrap();
    let table = create_test_table(db_path, "upsert_test", 3).await;

    let sink = LanceDbSink::new(Arc::new(RwLock::new(table)));

    // First insert
    let mut doc = CodeDocument::new("file.py".to_string(), "original content".to_string());
    doc.content_embedding = vec![1.0, 2.0, 3.0];
    let original_id = doc.id.clone();

    let batch1 = create_test_batch(vec![doc.clone()]);
    let batch_stream1 = stream::once(async { Ok(batch1) });
    let record_stream1 = Box::pin(lancedb::arrow::SimpleRecordBatchStream::new(
      batch_stream1.boxed(),
      Arc::new(CodeDocument::schema(3)),
    ));

    let mut sink_stream1 = sink.sink(record_stream1);
    while let Some(_) = sink_stream1.next().await {}

    // Update the same document
    doc.content = "updated content".to_string();
    doc.content_embedding = vec![4.0, 5.0, 6.0];

    let batch2 = create_test_batch(vec![doc]);
    let batch_stream2 = stream::once(async { Ok(batch2) });
    let record_stream2 = Box::pin(lancedb::arrow::SimpleRecordBatchStream::new(
      batch_stream2.boxed(),
      Arc::new(CodeDocument::schema(3)),
    ));

    let mut sink_stream2 = sink.sink(record_stream2);
    while let Some(_) = sink_stream2.next().await {}

    // Verify only one document exists with updated content
    let conn = lancedb::connect(db_path).execute().await.unwrap();
    let table = conn.open_table("upsert_test").execute().await.unwrap();
    let mut query_stream = table
      .query()
      .only_if(&format!("id = '{}'", original_id))
      .execute()
      .await
      .unwrap();

    let mut results = Vec::new();
    while let Some(batch) = query_stream.next().await {
      results.push(batch.unwrap());
    }

    assert_eq!(results.len(), 1);
    let result_batch = &results[0];
    assert_eq!(result_batch.num_rows(), 1);

    // Check content was updated
    let content_array = result_batch.column_by_name("content").unwrap();
    let content_array = content_array
      .as_any()
      .downcast_ref::<StringArray>()
      .unwrap();
    assert_eq!(content_array.value(0), "updated content");
  }
}
