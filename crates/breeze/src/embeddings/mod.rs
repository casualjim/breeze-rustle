use async_trait::async_trait;
use breeze_chunkers::Chunk;
use futures_util::{Stream, StreamExt};

#[async_trait]
trait EmbeddingFunction {
    /// Computes embeddings for the given source code.
    async fn compute(&self, source: &str) -> Result<Vec<u32>, breeze_chunkers::ChunkError>;

    
}


// pub async fn generate_embeddings(
//     chunks: impl Stream<Item = Result<Chunk, breeze_chunkers::ChunkError>> + Unpin,
//     embedder: &impl EmbeddingFunction,
// ) -> Result<Vec<u32>, breeze_chunkers::ChunkError> {
//     let mut embeddings = Vec::new();

//     while let Some(chunk) = chunks.next().await {
//         match chunk {
//             Ok(chunk) => {
//                 // let embedding = embedder.compute_source_embeddings(chunk.content.as_str()).await?;
//                 // embeddings.push(embedding);
//             }
//             Err(e) => {
//                 eprintln!("Error processing chunk: {}", e);
//             }
//         }
//     }

//     Ok(embeddings)
// }
