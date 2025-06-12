use async_stream::stream;
use futures_util::StreamExt;

use crate::pipeline::*;
use breeze_chunkers::ProjectChunk;

/// Mock batcher that groups chunks into configurable batch sizes
pub struct MockBatcher {
  batch_size: usize,
  max_tokens: Option<usize>,
}

impl MockBatcher {
  pub fn new(batch_size: usize) -> Self {
    Self {
      batch_size,
      max_tokens: None,
    }
  }

  pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
    self.max_tokens = Some(max_tokens);
    self
  }
}

impl Default for MockBatcher {
  fn default() -> Self {
    Self::new(2)
  }
}

impl Batcher for MockBatcher {
  fn batch(&self, chunks: BoxStream<ProjectChunk>) -> BoxStream<TextBatch> {
    let batch_size = self.batch_size;
    let _max_tokens = self.max_tokens;

    Box::pin(stream! {
        let mut buffer: Vec<ProjectChunk> = Vec::new();

        let mut chunks = chunks;
        while let Some(chunk) = chunks.next().await {
            buffer.push(chunk);

            if buffer.len() >= batch_size {
                yield buffer.drain(..).collect();
            }
        }

        // Yield remaining items
        if !buffer.is_empty() {
            yield buffer;
        }
    })
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::testing::MockPathWalker;
  use futures_util::StreamExt;
  use std::path::Path;

  #[tokio::test]
  async fn test_mock_batcher_by_count() {
    let walker = MockPathWalker::new(2, 5);
    let batcher = MockBatcher::new(3);

    let chunks = walker.walk(Path::new("/test"));
    let mut batches = batcher.batch(chunks);

    let mut batch_count = 0;
    let mut total_texts = 0;

    while let Some(batch) = batches.next().await {
      assert!(batch.len() <= 3);
      batch_count += 1;
      total_texts += batch.len();
    }

    assert_eq!(batch_count, 4); // 10 chunks / 3 per batch = 4 batches
    assert_eq!(total_texts, 10); // 2 files * 5 chunks
  }

  #[tokio::test]
  async fn test_mock_batcher_with_tokens() {
    let walker = MockPathWalker::new(1, 3);
    let batcher = MockBatcher::new(10).with_max_tokens(50);

    let chunks = walker.walk(Path::new("/test"));
    let mut batches = batcher.batch(chunks);

    while let Some(batch) = batches.next().await {
      // For now, just check batch size
      assert!(batch.len() <= 10);
    }
  }
}
