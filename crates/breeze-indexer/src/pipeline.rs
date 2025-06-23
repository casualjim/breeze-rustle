use breeze_chunkers::{Chunk, SemanticChunk};

/// Type alias for a boxed stream
pub(crate) type BoxStream<T> = std::pin::Pin<Box<dyn futures_util::Stream<Item = T> + Send>>;

/// Chunks that go through the embedding pipeline
#[derive(Debug, Clone)]
pub enum PipelineChunk {
  Semantic(SemanticChunk),
  Text(SemanticChunk),
  EndOfFile {
    file_path: String,
    content: String,
    content_hash: [u8; 32], // Blake3 hash
  },
}

impl PipelineChunk {
  pub fn from_chunk(chunk: Chunk) -> Option<Self> {
    match chunk {
      Chunk::Semantic(sc) => Some(PipelineChunk::Semantic(sc)),
      Chunk::Text(sc) => Some(PipelineChunk::Text(sc)),
      Chunk::EndOfFile {
        file_path,
        content,
        content_hash,
      } => Some(PipelineChunk::EndOfFile {
        file_path,
        content,
        content_hash,
      }),
      Chunk::Delete { .. } => None,
    }
  }
}

/// A project chunk that's been filtered for the pipeline
#[derive(Debug, Clone)]
pub struct PipelineProjectChunk {
  pub file_path: String,
  pub chunk: PipelineChunk,
}

/// Represents a chunk with its embedding
#[derive(Debug, Clone)]
pub(crate) struct EmbeddedChunk {
  pub chunk: PipelineChunk,
  pub embedding: Vec<f32>,
}

/// A batch of chunks ready for embedding
#[derive(Debug)]
pub(crate) struct ChunkBatch {
  pub batch_id: usize,
  pub chunks: Vec<PipelineProjectChunk>,
}

/// Embedded chunks with file path information
#[derive(Debug, Clone)]
pub(crate) enum EmbeddedChunkWithFile {
  /// Regular embedded chunk
  Embedded {
    batch_id: usize,
    file_path: String,
    chunk: PipelineChunk,
    embedding: Vec<f32>,
  },
  /// EOF marker - no embedding needed
  EndOfFile {
    batch_id: usize,
    file_path: String,
    content: String,
    content_hash: [u8; 32],
  },
  /// Batch failure notification
  BatchFailure {
    batch_id: usize,
    failed_files: std::collections::BTreeSet<String>,
    error: String,
  },
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
