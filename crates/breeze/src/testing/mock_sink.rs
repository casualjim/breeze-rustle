use futures_util::StreamExt;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::pipeline::{BoxStream, Sink};
use arrow::record_batch::RecordBatch;
use lancedb::arrow::RecordBatchStream;

#[derive(Clone)]
pub struct MockSink {
  batches_received: Arc<AtomicUsize>,
  rows_received: Arc<AtomicUsize>,
  stored_batches: Arc<Mutex<Vec<RecordBatch>>>,
  last_schema: Arc<Mutex<Option<arrow::datatypes::SchemaRef>>>,
}

impl Default for MockSink {
  fn default() -> Self {
    Self {
      batches_received: Arc::new(AtomicUsize::new(0)),
      rows_received: Arc::new(AtomicUsize::new(0)),
      stored_batches: Arc::new(Mutex::new(Vec::new())),
      last_schema: Arc::new(Mutex::new(None)),
    }
  }
}

impl MockSink {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn batch_count(&self) -> usize {
    self.batches_received.load(Ordering::SeqCst)
  }

  pub fn row_count(&self) -> usize {
    self.rows_received.load(Ordering::SeqCst)
  }

  pub fn stored_batches(&self) -> Vec<RecordBatch> {
    self.stored_batches.lock().unwrap().clone()
  }

  pub fn get_schema(&self) -> Option<arrow::datatypes::SchemaRef> {
    self.last_schema.lock().unwrap().clone()
  }
}

impl Sink for MockSink {
  fn sink(&self, batches: Pin<Box<dyn RecordBatchStream + Send>>) -> BoxStream<()> {
    let batches_counter = self.batches_received.clone();
    let rows_counter = self.rows_received.clone();
    let stored = self.stored_batches.clone();
    let schema_holder = self.last_schema.clone();

    Box::pin(async_stream::stream! {
        // Store the schema
        let schema = batches.schema();
        {
            let mut schema_lock = schema_holder.lock().unwrap();
            *schema_lock = Some(schema);
        }

        // Stream is already pinned, just make it mutable
        let mut batches = batches;

        while let Some(result) = batches.next().await {
            match result {
                Ok(batch) => {
                    let num_rows = batch.num_rows();
                    batches_counter.fetch_add(1, Ordering::SeqCst);
                    rows_counter.fetch_add(num_rows, Ordering::SeqCst);

                    tracing::debug!("Mock sink received batch with {} rows", num_rows);

                    // Store the batch
                    stored.lock().unwrap().push(batch);
                }
                Err(e) => {
                    tracing::error!("Mock sink received error: {}", e);
                    break;
                }
            }
        }

        tracing::info!("Mock sink completed: {} batches, {} total rows",
            batches_counter.load(Ordering::SeqCst),
            rows_counter.load(Ordering::SeqCst)
        );

        yield ();
    })
  }
}
