use futures_util::StreamExt;

use crate::models::CodeDocument;
use crate::pipeline::{BoxStream, DocumentBuilder, ProjectFileWithEmbeddings};

/// Mock document builder that simply concatenates all chunk texts
/// and uses the first chunk's embedding (for testing)
pub struct MockDocumentBuilder;

impl MockDocumentBuilder {
    pub fn new() -> Self {
        Self
    }
}

impl DocumentBuilder for MockDocumentBuilder {
    fn build_documents(&self, files: BoxStream<ProjectFileWithEmbeddings>) -> BoxStream<CodeDocument> {
        let stream = files.then(|file_with_embeddings| {
            let file_path = file_with_embeddings.file_path.clone();
            let metadata = file_with_embeddings.metadata.clone();
            
            async move {
                // Collect all embedded chunks
                let embedded_chunks: Vec<_> = file_with_embeddings.embedded_chunks
                    .filter_map(|result| async move { result.ok() })
                    .collect()
                    .await;
                
                if embedded_chunks.is_empty() {
                    return None;
                }
                
                // Concatenate all chunk texts
                let mut full_content = String::new();
                for embedded_chunk in &embedded_chunks {
                    let text = match &embedded_chunk.chunk {
                        breeze_chunkers::Chunk::Semantic(sc) => &sc.text,
                        breeze_chunkers::Chunk::Text(sc) => &sc.text,
                    };
                    if !full_content.is_empty() {
                        full_content.push('\n');
                    }
                    full_content.push_str(text);
                }
                
                // Create document using first chunk's embedding (for simplicity in tests)
                let mut doc = CodeDocument::new(file_path, full_content);
                doc.file_size = metadata.size;
                doc.last_modified = metadata.modified
                    .duration_since(std::time::UNIX_EPOCH)
                    .ok()
                    .and_then(|d| {
                        let secs = d.as_secs() as i64;
                        let nanos = d.subsec_nanos();
                        chrono::DateTime::from_timestamp(secs, nanos).map(|dt| dt.naive_utc())
                    })
                    .unwrap_or_else(|| chrono::Utc::now().naive_utc());
                
                // Use first chunk's embedding for simplicity
                doc.update_embedding(embedded_chunks[0].embedding.clone());
                
                Some(doc)
            }
        })
        .filter_map(|x| async move { x });
        
        Box::pin(stream)
    }
}