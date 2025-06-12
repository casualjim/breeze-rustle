use futures_util::Stream;
use std::path::Path;
use std::pin::Pin;

use crate::models::CodeDocument;
use breeze_chunkers::ProjectChunk;
use lancedb::arrow::RecordBatchStream;

/// Type alias for a boxed stream
pub type BoxStream<T> = Pin<Box<dyn Stream<Item = T> + Send>>;

/// Represents a batch of texts ready for embedding
pub type TextBatch = Vec<ProjectChunk>;

pub type EmbeddingBatch = Vec<EmbeddingBatchItem>;

/// Represents embeddings for a batch
#[derive(Debug)]
pub struct EmbeddingBatchItem {
  pub embeddings: Vec<f32>,
  pub metadata: TextBatch,
}

/// Trait for the directory walking stage: (Path) -> Stream<ProjectChunk>
pub trait PathWalker {
  fn walk(&self, path: &Path) -> BoxStream<ProjectChunk>;
}

/// Trait for the batching stage: Stream<ProjectChunk> -> Stream<TextBatch>
pub trait Batcher {
  fn batch(&self, chunks: BoxStream<ProjectChunk>) -> BoxStream<TextBatch>;
}

/// Trait for the embedding stage: Stream<TextBatch> -> Stream<EmbeddingBatch>
pub trait Embedder {
  fn embed(&self, batches: BoxStream<TextBatch>) -> BoxStream<EmbeddingBatch>;
}

/// Trait for the aggregation stage: Stream<EmbeddingBatch> -> Stream<CodeDocument>
pub trait Aggregator {
  fn aggregate(&self, embeddings: BoxStream<EmbeddingBatch>) -> BoxStream<CodeDocument>;
}

/// Trait for converting streams of T to Arrow RecordBatchStream
pub trait RecordBatchConverter<T> {
  fn convert(&self, items: BoxStream<T>) -> Pin<Box<dyn RecordBatchStream + Send>>;
}

/// Trait for sinking RecordBatchStream to storage
pub trait Sink {
  fn sink(&self, batches: Pin<Box<dyn RecordBatchStream + Send>>) -> BoxStream<()>;
}
