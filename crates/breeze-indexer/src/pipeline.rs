use breeze_chunkers::{Chunk, SemanticChunk};
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::models::CodeChunk;

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
    expected_chunks: usize,
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
        expected_chunks,
      } => Some(PipelineChunk::EndOfFile {
        file_path,
        content,
        content_hash,
        expected_chunks,
      }),
      Chunk::Delete { .. } => None,
    }
  }
}

/// Replace all chunks for a single file (project_id + file_path) with the provided set
#[derive(Debug, Clone)]
pub struct ReplaceFileChunks {
  pub project_id: Uuid,
  pub file_path: String,
  pub chunks: Vec<CodeChunk>,
}

pub type ReplaceFileChunksSender = mpsc::Sender<ReplaceFileChunks>;

/// A project chunk that's been filtered for the pipeline
#[derive(Debug, Clone)]
pub struct PipelineProjectChunk {
  pub file_path: String,
  pub chunk: PipelineChunk,
  pub file_size: u64,
}

/// Represents a chunk with its embedding
#[derive(Debug, Clone)]
pub(crate) struct EmbeddedChunk {
  pub chunk: PipelineChunk,
  pub embedding: Vec<f32>,
}

/// A batch of chunks ready for embedding
#[derive(Debug)]
pub struct ChunkBatch {
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
    chunk: Box<PipelineChunk>,
    embedding: Vec<f32>,
  },
  /// EOF marker - no embedding needed
  EndOfFile {
    batch_id: usize,
    file_path: String,
    content: String,
    content_hash: [u8; 32],
    expected_chunks: usize,
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
  pub expected_chunks: Option<usize>,
  pub received_content_chunks: usize,
  pub has_eof: bool,
}

impl FileAccumulator {
  pub(crate) fn new(file_path: String) -> Self {
    Self {
      file_path,
      embedded_chunks: Vec::new(),
      expected_chunks: None,
      received_content_chunks: 0,
      has_eof: false,
    }
  }

  pub(crate) fn add_chunk(&mut self, chunk: EmbeddedChunk) {
    match &chunk.chunk {
      PipelineChunk::EndOfFile {
        expected_chunks, ..
      } => {
        self.expected_chunks = Some(*expected_chunks);
      }
      _ => {
        // This is a content chunk
        self.received_content_chunks += 1;
      }
    }
    self.embedded_chunks.push(chunk);
  }

  pub(crate) fn is_complete(&self) -> bool {
    // File is complete when:
    // We have received all expected content chunks
    self.expected_chunks == Some(self.received_content_chunks)
  }
}
