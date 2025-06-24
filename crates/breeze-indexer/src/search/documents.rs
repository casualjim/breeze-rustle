use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::array::Float32Array;
use futures::TryStreamExt;
use lancedb::Table;
use lancedb::query::{ExecutableQuery, QueryBase};
use tokio::sync::RwLock;

use crate::models::{CodeChunk, CodeDocument};
use crate::search::build_hybrid_query;
use crate::{ChunkResult, SearchOptions, SearchResult};

/// Search documents and return results with top chunks
pub(crate) async fn search_documents(
  documents_table: Arc<RwLock<Table>>,
  chunks_table: Arc<RwLock<Table>>,
  query: &str,
  query_vector: &[f32],
  options: &SearchOptions,
  project_id: Option<&str>,
) -> Result<Vec<SearchResult>> {
  let documents_table = documents_table.read().await;

  let mut doc_query = build_hybrid_query(
    &documents_table,
    query,
    query_vector,
    project_id,
    "content_embedding",
  )
  .await?;

  // Add language filter if provided
  if let Some(languages) = &options.languages {
    if !languages.is_empty() {
      let lang_conditions: Vec<String> = languages
        .iter()
        .map(|lang| format!("primary_language = '{}'", lang))
        .collect();
      let lang_filter = lang_conditions.join(" OR ");
      doc_query = doc_query.only_if(lang_filter);
    }
  }

  let mut results = doc_query.limit(options.file_limit).execute().await?;

  // Collect results from RecordBatch stream
  let mut search_results = Vec::new();

  while let Some(batch) = results
    .try_next()
    .await
    .map_err(|e| anyhow!("Error reading results: {}", e))?
  {
    let relevance_scores = batch
      .column_by_name("_relevance_score")
      .and_then(|col| col.as_any().downcast_ref::<Float32Array>())
      .ok_or_else(|| anyhow!("Missing _relevance_score column in results"))?;

    for row in 0..batch.num_rows() {
      let doc = CodeDocument::from_record_batch(&batch, row)
        .map_err(|e| anyhow!("Failed to convert row {}: {}", row, e))?;

      let relevance_score = relevance_scores.value(row);

      search_results.push(SearchResult {
        id: doc.id.to_string(),
        file_path: doc.file_path.clone(),
        relevance_score,
        chunk_count: doc.chunk_count,
        chunks: Vec::new(),
      });
    }
  }

  // Now fetch chunks for each document
  let chunks_table = chunks_table.read().await;

  for result in &mut search_results {
    result.chunks = fetch_chunks_for_document(
      &chunks_table,
      query,
      query_vector,
      &result.id,
      options.chunks_per_file,
    )
    .await?;
  }

  Ok(search_results)
}

/// Fetch top chunks for a specific document
async fn fetch_chunks_for_document(
  chunks_table: &Table,
  query: &str,
  query_vector: &[f32],
  file_id: &str,
  limit: usize,
) -> Result<Vec<ChunkResult>> {
  let chunk_query = build_hybrid_query(chunks_table, query, query_vector, None, "embedding")
    .await?
    .only_if(format!("file_id = '{}'", file_id))
    .limit(limit);

  let mut chunk_results = chunk_query.execute().await?;
  let mut chunks = Vec::new();

  while let Some(batch) = chunk_results
    .try_next()
    .await
    .map_err(|e| anyhow!("Error reading chunk results: {}", e))?
  {
    let relevance_scores = batch
      .column_by_name("_relevance_score")
      .and_then(|col| col.as_any().downcast_ref::<Float32Array>())
      .ok_or_else(|| anyhow!("Missing _relevance_score column in chunk results"))?;

    for row in 0..batch.num_rows() {
      let chunk = CodeChunk::from_record_batch(&batch, row)
        .map_err(|e| anyhow!("Failed to convert chunk row {}: {}", row, e))?;

      let relevance_score = relevance_scores.value(row);

      chunks.push(ChunkResult {
        content: chunk.content,
        start_line: chunk.start_line,
        end_line: chunk.end_line,
        relevance_score,
      });
    }
  }

  Ok(chunks)
}
