use std::sync::Arc;

use anyhow::{Result, anyhow};
use arrow::array::Float32Array;
use futures::TryStreamExt;
use lancedb::Table;
use lancedb::query::{ExecutableQuery, QueryBase};
use tokio::sync::RwLock;

use crate::models::CodeChunk;
use crate::search::build_hybrid_query;
use crate::{ChunkResult, SearchOptions, SearchResult};

/// Apply semantic filters to a chunk query
fn apply_semantic_filters(
  mut query: lancedb::query::VectorQuery,
  options: &SearchOptions,
) -> lancedb::query::VectorQuery {
  // Filter by node types
  if let Some(node_types) = &options.node_types {
    if !node_types.is_empty() {
      let conditions: Vec<String> = node_types
        .iter()
        .map(|nt| format!("node_type = '{}'", nt))
        .collect();
      let filter = conditions.join(" OR ");
      query = query.only_if(format!("({})", filter));
    }
  }

  // Filter by node name pattern
  if let Some(pattern) = &options.node_name_pattern {
    query = query.only_if(format!("node_name = '{}'", pattern));
  }

  // Filter by parent context pattern
  if let Some(pattern) = &options.parent_context_pattern {
    query = query.only_if(format!("parent_context = '{}'", pattern));
  }

  // Filter by scope depth
  if let Some((min_depth, max_depth)) = options.scope_depth {
    query = query.only_if(format!(
      "array_length(scope_path) >= {} AND array_length(scope_path) <= {}",
      min_depth, max_depth
    ));
  }

  // Filter by definitions using array_contains with OR
  if let Some(definitions) = &options.has_definitions {
    if !definitions.is_empty() {
      let conditions: Vec<String> = definitions
        .iter()
        .map(|d| format!("array_contains(definitions, '{}')", d))
        .collect();
      let filter = conditions.join(" OR ");
      query = query.only_if(format!("({})", filter));
    }
  }

  // Filter by references using array_contains with OR
  if let Some(references) = &options.has_references {
    if !references.is_empty() {
      let conditions: Vec<String> = references
        .iter()
        .map(|r| format!("array_contains(references, '{}')", r))
        .collect();
      let filter = conditions.join(" OR ");
      query = query.only_if(format!("({})", filter));
    }
  }

  query
}

/// Search chunks directly and group by file
pub(crate) async fn search_chunks(
  _documents_table: Arc<RwLock<Table>>,
  chunks_table: Arc<RwLock<Table>>,
  query: &str,
  query_vector: &[f32],
  options: &SearchOptions,
  project_id: Option<&str>,
) -> Result<Vec<SearchResult>> {
  let chunks_table = chunks_table.read().await;

  let mut chunk_query =
    build_hybrid_query(&chunks_table, query, query_vector, project_id, "embedding").await?;

  // Add language filter if provided
  if let Some(languages) = &options.languages {
    if !languages.is_empty() {
      // Create case-insensitive language conditions to handle hyperpolyglot's casing
      // (e.g., "Rust" vs "rust", "C++" vs "cpp")
      let lang_conditions: Vec<String> = languages
        .iter()
        .map(|lang| {
          // Use ILIKE for case-insensitive matching
          format!("lower(language) = lower('{}')", lang)
        })
        .collect();
      let lang_filter = lang_conditions.join(" OR ");
      chunk_query = chunk_query.only_if(lang_filter);
    }
  }

  // Apply semantic filters
  chunk_query = apply_semantic_filters(chunk_query, options);

  // Flat chunk ranking: treat file_limit as number of chunks (no grouping by file)
  let mut results = chunk_query.limit(options.file_limit).execute().await?;

  // Collect and rank chunks
  let mut all_chunks = collect_all_chunks(&mut results).await?;
  all_chunks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

  // Take top-N chunks
  let now = chrono::Utc::now().naive_utc();
  let search_results: Vec<SearchResult> = all_chunks
    .into_iter()
    .take(options.file_limit)
    .map(|(chunk, score)| {
      let file_path = chunk.file_path.clone();
      let language = chunk.language.clone();
      let chunk_result = ChunkResult {
        content: chunk.content,
        start_line: chunk.start_line,
        end_line: chunk.end_line,
        start_byte: chunk.start_byte,
        end_byte: chunk.end_byte,
        relevance_score: score,
        node_type: chunk.node_type,
        node_name: chunk.node_name,
        language: chunk.language,
        parent_context: chunk.parent_context,
        scope_path: chunk.scope_path,
        definitions: chunk.definitions,
        references: chunk.references,
      };

      SearchResult {
        id: chunk.file_id.to_string(),
        file_path,
        relevance_score: score,
        chunk_count: 1,
        chunks: vec![chunk_result],
        // Minimal metadata; markdown output doesn't need these, keep simple
        file_size: 0,
        last_modified: now,
        indexed_at: now,
        languages: vec![language.clone()],
        primary_language: Some(language),
      }
    })
    .collect();

  Ok(search_results)
}

/// Collect all chunks from query results
async fn collect_all_chunks<S>(results: &mut S) -> Result<Vec<(CodeChunk, f32)>>
where
  S: futures::stream::TryStream<Ok = arrow::record_batch::RecordBatch> + Unpin,
  S::Error: Into<anyhow::Error>,
{
  let mut all_chunks = Vec::new();

  while let Some(batch) = results.try_next().await.map_err(|e| e.into())? {
    let relevance_scores = batch
      .column_by_name("_relevance_score")
      .and_then(|col| col.as_any().downcast_ref::<Float32Array>())
      .ok_or_else(|| anyhow!("Missing _relevance_score column in chunk results"))?;

    for row in 0..batch.num_rows() {
      let chunk = CodeChunk::from_record_batch(&batch, row)
        .map_err(|e| anyhow!("Failed to convert chunk row {}: {}", row, e))?;

      let relevance_score = relevance_scores.value(row);
      all_chunks.push((chunk, relevance_score));
    }
  }

  Ok(all_chunks)
}
