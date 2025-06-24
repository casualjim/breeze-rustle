use futures_util::StreamExt;
use lancedb::arrow::RecordBatchStream;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::pipeline::BoxStream;

/// LanceDB sink for persisting chunk record batches
pub struct ChunkSink {
  table: Arc<RwLock<lancedb::Table>>,
  last_optimize_version: Arc<RwLock<u64>>,
  optimize_threshold: u64,
}

impl ChunkSink {
  /// Create a new chunk sink with a shared table reference
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

impl ChunkSink {
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

                    // Use merge_insert for upsert behavior based on chunk id
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
                                "Saved chunk batch to LanceDB"
                            );
                            yield ();
                        }
                        Err(e) => {
                            tracing::error!(
                                error = %e,
                                batch_number = batch_count + 1,
                                rows = num_rows,
                                "Failed to save chunk batch to LanceDB"
                            );
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        "Error in chunk record batch stream"
                    );
                }
            }
        }

        tracing::info!(
            batches_written = batch_count,
            total_rows = total_rows,
            "Chunk sink completed"
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
                        "Running chunk table optimization"
                    );

                    match table_guard.optimize(lancedb::table::OptimizeAction::All).await {
                        Ok(_) => {
                            // Update the last optimize version
                            *last_optimize_version.write().await = current_version;
                            tracing::info!(
                                new_optimize_version = current_version,
                                "Chunk table optimization completed successfully"
                            );
                        }
                        Err(e) => {
                            tracing::warn!(
                                error = %e,
                                "Failed to optimize chunk table"
                            );
                        }
                    }
                } else {
                    tracing::debug!(
                        current_version = current_version,
                        last_optimize_version = last_version,
                        threshold = optimize_threshold,
                        "Skipping chunk table optimization, threshold not met"
                    );
                }
            }
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    "Failed to get chunk table version for optimization check"
                );
            }
        }
    };

    Box::pin(stream)
  }
}

impl Clone for ChunkSink {
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
  use crate::models::{ChunkMetadataUpdate, CodeChunk};
  use arrow::array::StringArray;
  use arrow::record_batch::RecordBatch;
  use futures_util::stream;
  use lancedb::arrow::IntoArrow;
  use lancedb::query::{ExecutableQuery, QueryBase};
  use tempfile::TempDir;
  use uuid::Uuid;

  async fn create_test_chunk_table(
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
      // Create table using CodeChunk's ensure_table method
      CodeChunk::ensure_table(&conn, table_name, embedding_dim)
        .await
        .unwrap()
    }
  }

  fn create_test_batch(chunks: Vec<CodeChunk>) -> RecordBatch {
    if chunks.is_empty() {
      return RecordBatch::new_empty(Arc::new(CodeChunk::schema(384)));
    }

    // Use the IntoArrow implementation from CodeChunk
    let readers: Vec<Box<dyn arrow::array::RecordBatchReader + Send>> = chunks
      .into_iter()
      .map(|chunk| chunk.into_arrow().unwrap())
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
      RecordBatch::new_empty(Arc::new(CodeChunk::schema(384)))
    } else {
      let schema = all_batches[0].schema();
      arrow::compute::concat_batches(&schema, &all_batches).expect("Failed to concat batches")
    }
  }

  #[tokio::test]
  async fn test_chunk_sink_basic() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().to_str().unwrap();
    let table = create_test_chunk_table(db_path, "test_chunks", 3).await;

    let current_version = table.version().await.unwrap();
    let sink = ChunkSink::new(
      Arc::new(RwLock::new(table)),
      Arc::new(RwLock::new(current_version)),
      250,
    );

    // Create test chunks
    let mut chunk1 = CodeChunk::builder()
      .file_id(Uuid::now_v7())
      .project_id(Uuid::now_v7())
      .file_path("file1.py".to_string())
      .content("def hello():\n    print('hello')".to_string())
      .start_byte(0)
      .end_byte(32)
      .start_line(1)
      .end_line(2)
      .build();
    chunk1.update_embedding(vec![1.0, 2.0, 3.0]);
    chunk1.update_metadata(
      ChunkMetadataUpdate::builder()
        .node_type("function".to_string())
        .node_name(Some("hello".to_string()))
        .language("python".to_string())
        .parent_context(None)
        .scope_path(vec!["hello".to_string()])
        .definitions(vec![])
        .references(vec!["print".to_string()])
        .build(),
    );

    let mut chunk2 = CodeChunk::builder()
      .file_id(Uuid::now_v7())
      .project_id(Uuid::now_v7())
      .file_path("file2.py".to_string())
      .content("class MyClass:\n    pass".to_string())
      .start_byte(0)
      .end_byte(24)
      .start_line(1)
      .end_line(2)
      .build();
    chunk2.update_embedding(vec![4.0, 5.0, 6.0]);
    chunk2.update_metadata(
      ChunkMetadataUpdate::builder()
        .node_type("class".to_string())
        .node_name(Some("MyClass".to_string()))
        .language("python".to_string())
        .parent_context(None)
        .scope_path(vec!["MyClass".to_string()])
        .definitions(vec![])
        .references(vec![])
        .build(),
    );

    let batch = create_test_batch(vec![chunk1, chunk2]);
    let batch_stream = stream::once(async { Ok(batch) });
    let record_stream = Box::pin(lancedb::arrow::SimpleRecordBatchStream::new(
      batch_stream.boxed(),
      Arc::new(CodeChunk::schema(3)),
    ));

    // Process the stream
    let mut sink_stream = sink.sink(record_stream);
    while sink_stream.next().await.is_some() {
      // Process each item
    }

    // Verify data was saved (excluding dummy chunk)
    let conn = lancedb::connect(db_path).execute().await.unwrap();
    let table = conn.open_table("test_chunks").execute().await.unwrap();
    let count = table
      .count_rows(Some(format!("id != '{}'", Uuid::nil())))
      .await
      .unwrap();
    assert_eq!(count, 2);
  }

  #[tokio::test]
  async fn test_chunk_sink_upsert() {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().to_str().unwrap();
    let table = create_test_chunk_table(db_path, "upsert_test", 3).await;

    let current_version = table.version().await.unwrap();
    let sink = ChunkSink::new(
      Arc::new(RwLock::new(table)),
      Arc::new(RwLock::new(current_version)),
      250,
    );

    // First insert
    let mut chunk = CodeChunk::builder()
      .file_id(Uuid::now_v7())
      .project_id(Uuid::now_v7())
      .file_path("file.py".to_string())
      .content("original content".to_string())
      .start_byte(0)
      .end_byte(16)
      .start_line(1)
      .end_line(1)
      .build();
    chunk.update_embedding(vec![1.0, 2.0, 3.0]);
    let original_id = chunk.id;

    let batch1 = create_test_batch(vec![chunk.clone()]);
    let batch_stream1 = stream::once(async { Ok(batch1) });
    let record_stream1 = Box::pin(lancedb::arrow::SimpleRecordBatchStream::new(
      batch_stream1.boxed(),
      Arc::new(CodeChunk::schema(3)),
    ));

    let mut sink_stream1 = sink.sink(record_stream1);
    while sink_stream1.next().await.is_some() {}

    // Update the same chunk
    chunk.content = "updated content".to_string();
    chunk.embedding = vec![4.0, 5.0, 6.0];

    let batch2 = create_test_batch(vec![chunk]);
    let batch_stream2 = stream::once(async { Ok(batch2) });
    let record_stream2 = Box::pin(lancedb::arrow::SimpleRecordBatchStream::new(
      batch_stream2.boxed(),
      Arc::new(CodeChunk::schema(3)),
    ));

    let mut sink_stream2 = sink.sink(record_stream2);
    while sink_stream2.next().await.is_some() {}

    // Verify only one chunk exists with updated content
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
