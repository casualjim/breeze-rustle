use breeze_chunkers::{Chunk, ProjectChunk};

/// Type alias for a boxed stream
pub type BoxStream<T> = std::pin::Pin<Box<dyn futures_util::Stream<Item = T> + Send>>;

/// Type alias for a boxed future
pub type BoxFuture<T> = std::pin::Pin<Box<dyn std::future::Future<Output = T> + Send>>;

/// Represents a chunk with its embedding
#[derive(Debug, Clone)]
pub struct EmbeddedChunk {
  pub chunk: Chunk,
  pub embedding: Vec<f32>,
}

/// A batch of chunks ready for embedding
#[derive(Debug)]
pub struct ChunkBatch {
  pub chunks: Vec<ProjectChunk>,
}

/// Embedded chunks with file path information
#[derive(Debug, Clone)]
pub struct EmbeddedChunkWithFile {
  pub file_path: String,
  pub chunk: Chunk,
  pub embedding: Vec<f32>,
}

/// File accumulator for building documents
#[derive(Debug)]
pub struct FileAccumulator {
  pub file_path: String,
  pub metadata: Option<breeze_chunkers::FileMetadata>,
  pub embedded_chunks: Vec<EmbeddedChunk>,
}

impl FileAccumulator {
  pub fn new(file_path: String) -> Self {
    Self {
      file_path,
      metadata: None,
      embedded_chunks: Vec::new(),
    }
  }

  pub fn add_chunk(&mut self, chunk: EmbeddedChunk) {
    self.embedded_chunks.push(chunk);
  }
}
