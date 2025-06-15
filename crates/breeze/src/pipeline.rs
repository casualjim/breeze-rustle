use breeze_chunkers::{Chunk, ProjectChunk};

/// Type alias for a boxed stream
pub(crate) type BoxStream<T> = std::pin::Pin<Box<dyn futures_util::Stream<Item = T> + Send>>;

/// Represents a chunk with its embedding
#[derive(Debug, Clone)]
pub(crate) struct EmbeddedChunk {
  pub chunk: Chunk,
  pub embedding: Vec<f32>,
}

/// A batch of chunks ready for embedding
#[derive(Debug)]
pub(crate) struct ChunkBatch {
  pub chunks: Vec<ProjectChunk>,
}

/// Embedded chunks with file path information
#[derive(Debug, Clone)]
pub(crate) struct EmbeddedChunkWithFile {
  pub file_path: String,
  pub chunk: Chunk,
  pub embedding: Vec<f32>,
}

/// File accumulator for building documents
#[derive(Debug)]
pub(crate) struct FileAccumulator {
  pub file_path: String,
  pub embedded_chunks: Vec<EmbeddedChunk>,
}

impl FileAccumulator {
  pub(crate) fn new(file_path: String) -> Self {
    Self {
      file_path,
      embedded_chunks: Vec::new(),
    }
  }

  pub(crate) fn add_chunk(&mut self, chunk: EmbeddedChunk) {
    self.embedded_chunks.push(chunk);
  }
}
