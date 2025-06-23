use futures_util::StreamExt;
use lancedb::arrow::RecordBatchStream;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::pipeline::BoxStream;

/// LanceDB sink for persisting record batches
pub struct LanceDbSink {
  table: Arc<RwLock<lancedb::Table>>,
  last_optimize_version: Arc<RwLock<u64>>,
  optimize_threshold: u64,
}

impl LanceDbSink {
  /// Create a new LanceDB sink with a shared table reference
  pub fn new(
    table: Arc<RwLock<lancedb::Table>>,
    last_optimize_version: Arc<RwLock<u64>>,
    optimize_threshold: u64,
  ) -> Self {
    Self {
      table,
      last_optimize_version,
      optimize_threshold,
    }
  }
}

impl LanceDbSink {
  pub fn sink(&self, batches: std::pin::Pin<Box<dyn RecordBatchStream + Send>>) -> BoxStream<()> {
    let table = self.table.clone();
    let last_optimize_version = self.last_optimize_version.clone();
    let optimize_threshold = self.optimize_threshold;

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

        // Check if optimization is needed after stream completion
        let table_guard = table.read().await;
        match table_guard.version().await {
            Ok(current_version) => {
                let last_version = *last_optimize_version.read().await;
                if current_version.saturating_sub(last_version) > optimize_threshold {
                    tracing::info!(
                        current_version = current_version,
                        last_optimize_version = last_version,
                        threshold = optimize_threshold,
                        "Running LanceDB table optimization"
                    );

                    match table_guard.optimize(lancedb::table::OptimizeAction::All).await {
                        Ok(_) => {
                            // Update the last optimize version
                            *last_optimize_version.write().await = current_version;
                            tracing::info!(
                                new_optimize_version = current_version,
                                "LanceDB table optimization completed successfully"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                "Failed to optimize LanceDB table"
                            );
                        }
                    }
                } else {
                    tracing::debug!(
                        current_version = current_version,
                        last_optimize_version = last_version,
                        threshold = optimize_threshold,
                        "Skipping optimization, threshold not met"
                    );
                }
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Failed to get table version for optimization check"
                );
            }
        }
    };

    Box::pin(stream)
  }
}

impl Clone for LanceDbSink {
  fn clone(&self) -> Self {
    Self {
      table: self.table.clone(),
      last_optimize_version: self.last_optimize_version.clone(),
      optimize_threshold: self.optimize_threshold,
    }
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::models::CodeDocument;
  use arrow::array::StringArray;
  use arrow::record_batch::RecordBatch;
  use futures_util::stream;
  use lancedb::arrow::IntoArrow;
  use lancedb::query::{ExecutableQuery, QueryBase};
  use tempfile::TempDir;
  use uuid::Uuid;

  async fn create_test_table(
    db_path: &str,
    table_name: &str,
    embedding_dim: usize,
  ) -> lancedb::Table {
    let conn = lancedb::connect(db_path).execute().await.unwrap();

    // Check if table exists
    let table_names = conn.table_names().execute().await.unwrap();
    if table_names.contains(&table_name.to_string()) {
      conn.open_table(table_name).execute().await.unwrap()
    } else {
      // Create a dummy document to initialize the table
      let mut dummy_doc = CodeDocument::new(
        Uuid::nil(),
        "dummy.py".to_string(),
        "dummy content".to_string(),
      );
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

    // Use the IntoArrow implementation from CodeDocument
    // This creates a RecordBatchReader which we can collect into a single batch
    let readers: Vec<Box<dyn arrow::array::RecordBatchReader + Send>> = docs
      .into_iter()
      .map(|doc| doc.into_arrow().unwrap())
      .collect();

    // Collect all batches from all readers
    let mut all_batches = Vec::new();
    for mut reader in readers {
      for batch in reader.by_ref().flatten() {
        all_batches.push(batch);
      }
    }

    // Concatenate all batches into one
    if all_batches.is_empty() {
      RecordBatch::new_empty(Arc::new(CodeDocument::schema(384)))
    } else {
      let schema = all_batches[0].schema();
      arrow::compute::concat_batches(&schema, &all_batches).expect("Failed to concat batches")
    }
  }

  #[tokio::test]
  async fn test_lancedb_sink_basic() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().to_str().unwrap();
    let table = create_test_table(db_path, "test_documents", 3).await;

    let current_version = table.version().await.unwrap();
    let sink = LanceDbSink::new(
      Arc::new(RwLock::new(table)),
      Arc::new(RwLock::new(current_version)),
      250,
    );

    // Create test documents
    let mut doc1 = CodeDocument::new(
      Uuid::now_v7(),
      "file1.py".to_string(),
      "print('hello')".to_string(),
    );
    doc1.content_embedding = vec![1.0, 2.0, 3.0];

    let mut doc2 = CodeDocument::new(
      Uuid::now_v7(),
      "file2.py".to_string(),
      "def main(): pass".to_string(),
    );
    doc2.content_embedding = vec![4.0, 5.0, 6.0];

    let batch = create_test_batch(vec![doc1, doc2]);
    let batch_stream = stream::once(async { Ok(batch) });
    let record_stream = Box::pin(lancedb::arrow::SimpleRecordBatchStream::new(
      batch_stream.boxed(),
      Arc::new(CodeDocument::schema(3)),
    ));

    // Process the stream
    let mut sink_stream = sink.sink(record_stream);
    while sink_stream.next().await.is_some() {
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
  async fn test_lancedb_sink_optimization() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().to_str().unwrap();
    let table = create_test_table(db_path, "optimization_test", 3).await;

    // Get initial version
    let initial_version = table.version().await.unwrap();

    // Create sink with a low threshold for testing
    let last_optimize_version = Arc::new(RwLock::new(initial_version));
    let sink = LanceDbSink::new(
      Arc::new(RwLock::new(table)),
      last_optimize_version.clone(),
      2, // Low threshold to trigger optimization in test
    );

    // Insert multiple documents to advance the version
    for i in 0..5 {
      let mut doc = CodeDocument::new(
        Uuid::now_v7(),
        format!("file{}.py", i),
        format!("content {}", i),
      );
      doc.content_embedding = vec![i as f32, (i + 1) as f32, (i + 2) as f32];

      let batch = create_test_batch(vec![doc]);
      let batch_stream = stream::once(async { Ok(batch) });
      let record_stream = Box::pin(lancedb::arrow::SimpleRecordBatchStream::new(
        batch_stream.boxed(),
        Arc::new(CodeDocument::schema(3)),
      ));

      let mut sink_stream = sink.sink(record_stream);
      while sink_stream.next().await.is_some() {}
    }

    // Verify optimization was triggered
    let final_optimize_version = *last_optimize_version.read().await;
    assert!(
      final_optimize_version > initial_version,
      "Optimization should have updated the version. Initial: {}, Final: {}",
      initial_version,
      final_optimize_version
    );
  }

  #[tokio::test]
  async fn test_lancedb_sink_upsert() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().to_str().unwrap();
    let table = create_test_table(db_path, "upsert_test", 3).await;

    let current_version = table.version().await.unwrap();
    let sink = LanceDbSink::new(
      Arc::new(RwLock::new(table)),
      Arc::new(RwLock::new(current_version)),
      250,
    );

    // First insert
    let mut doc = CodeDocument::new(
      Uuid::now_v7(),
      "file.py".to_string(),
      "original content".to_string(),
    );
    doc.content_embedding = vec![1.0, 2.0, 3.0];
    let original_id = doc.id.clone();

    let batch1 = create_test_batch(vec![doc.clone()]);
    let batch_stream1 = stream::once(async { Ok(batch1) });
    let record_stream1 = Box::pin(lancedb::arrow::SimpleRecordBatchStream::new(
      batch_stream1.boxed(),
      Arc::new(CodeDocument::schema(3)),
    ));

    let mut sink_stream1 = sink.sink(record_stream1);
    while sink_stream1.next().await.is_some() {}

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
    while sink_stream2.next().await.is_some() {}

    // Verify only one document exists with updated content
    let conn = lancedb::connect(db_path).execute().await.unwrap();
    let table = conn.open_table("upsert_test").execute().await.unwrap();
    let mut query_stream = table
      .query()
      .only_if(format!("id = '{}'", original_id))
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
