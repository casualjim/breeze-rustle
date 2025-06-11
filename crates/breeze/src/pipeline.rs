use futures_util::Stream;
use std::path::Path;
use std::pin::Pin;

use crate::config::Config;
use crate::models::CodeDocument;
use breeze_chunkers::ProjectChunk;

/// Type alias for a boxed stream
pub type BoxStream<T> = Pin<Box<dyn Stream<Item = T> + Send>>;

/// Represents a batch of texts ready for embedding
#[derive(Debug, Clone)]
pub struct TextBatch {
    pub texts: Vec<String>,
    pub metadata: Vec<BatchMetadata>,
}

/// Metadata for each text in a batch
#[derive(Debug, Clone)]
pub struct BatchMetadata {
    pub file_path: String,
    pub chunk_index: usize,
    pub token_count: usize,
}

/// Represents embeddings for a batch
#[derive(Debug)]
pub struct EmbeddingBatch {
    pub embeddings: Vec<Vec<f32>>,
    pub metadata: Vec<BatchMetadata>,
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

pub trait Sink {
    fn sink(&self, documents: BoxStream<CodeDocument>) -> BoxStream<()>;
}
