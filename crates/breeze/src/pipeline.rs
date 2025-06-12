use futures_util::Stream;
use std::path::Path;
use std::pin::Pin;

use crate::models::CodeDocument;
use breeze_chunkers::{Chunk, ChunkError, ProjectFile};
use lancedb::arrow::RecordBatchStream;

/// Type alias for a boxed stream
pub type BoxStream<T> = Pin<Box<dyn Stream<Item = T> + Send>>;

/// Represents a chunk with its embedding
#[derive(Debug, Clone)]
pub struct EmbeddedChunk {
    pub chunk: Chunk,
    pub embedding: Vec<f32>,
}

/// Represents a project file with embedded chunks
pub struct ProjectFileWithEmbeddings {
    pub file_path: String,
    pub metadata: breeze_chunkers::FileMetadata,
    pub embedded_chunks: BoxStream<Result<EmbeddedChunk, ChunkError>>,
}

/// Trait for the directory walking stage: (Path) -> Stream<ProjectFile>
pub trait PathWalker {
  fn walk(&self, path: &Path) -> BoxStream<ProjectFile>;
}

/// Trait for the embedding stage: Stream<ProjectFile> -> Stream<ProjectFileWithEmbeddings>
pub trait Embedder {
  fn embed(&self, files: BoxStream<ProjectFile>) -> BoxStream<ProjectFileWithEmbeddings>;
}

/// Trait for building documents from embedded chunks
pub trait DocumentBuilder {
  fn build_documents(&self, files: BoxStream<ProjectFileWithEmbeddings>) -> BoxStream<CodeDocument>;
}

/// Trait for converting streams of T to Arrow RecordBatchStream
pub trait RecordBatchConverter<T> {
  fn convert(&self, items: BoxStream<T>) -> Pin<Box<dyn RecordBatchStream + Send>>;
}

/// Trait for sinking RecordBatchStream to storage
pub trait Sink {
  fn sink(&self, batches: Pin<Box<dyn RecordBatchStream + Send>>) -> BoxStream<()>;
}
