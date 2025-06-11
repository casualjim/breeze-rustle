use futures_util::{Stream, StreamExt};
use std::sync::{Arc, Mutex};
use std::pin::Pin;

use crate::models::CodeDocument;
use crate::pipeline::{BoxStream, Sink};

#[derive(Clone)]
pub struct MockSink {
    stored: Arc<Mutex<Vec<CodeDocument>>>,
}

impl Default for MockSink {
    fn default() -> Self {
        Self {
            stored: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl MockSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn stored_documents(&self) -> Vec<CodeDocument> {
        self.stored.lock().unwrap().clone()
    }

    pub fn document_count(&self) -> usize {
        self.stored.lock().unwrap().len()
    }
}

impl Sink for MockSink {
    fn sink(&self, documents: BoxStream<CodeDocument>) -> BoxStream<()> {
        let stored = self.stored.clone();
        
        Box::pin(async_stream::stream! {
            futures_util::pin_mut!(documents);
            
            while let Some(doc) = documents.next().await {
                println!("Storing document: {} ({} bytes)", doc.file_path, doc.file_size);
                stored.lock().unwrap().push(doc);
                yield ();
            }
        })
    }
}