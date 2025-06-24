use blake3::Hasher;
use lancedb::index::Index;
use lancedb::index::scalar::LabelListIndexBuilder;
use lancedb::index::vector::IvfFlatIndexBuilder;
use tracing::debug;
use typed_builder::TypedBuilder;
use uuid::Uuid;

#[derive(Debug, Clone, TypedBuilder)]
pub struct ChunkMetadataUpdate {
  pub node_type: String,
  pub node_name: Option<String>,
  pub language: String,
  pub parent_context: Option<String>,
  pub scope_path: Vec<String>,
  pub definitions: Vec<String>,
  pub references: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, TypedBuilder)]
pub struct CodeChunk {
  #[builder(default = Uuid::now_v7())]
  pub id: Uuid, // Unique chunk ID (UUID v7)
  pub file_id: Uuid,     // FK to CodeDocument
  pub project_id: Uuid,  // Project this chunk belongs to
  pub file_path: String, // Denormalized for performance
  pub content: String,   // The actual chunk text
  #[builder(default = CodeChunk::compute_hash(&content))]
  pub chunk_hash: [u8; 32], // Blake3 hash of content
  #[builder(default = Vec::new())]
  pub embedding: Vec<f32>, // Chunk embedding vector

  // Position information
  pub start_byte: usize,
  pub end_byte: usize,
  pub start_line: usize,
  pub end_line: usize,

  // Complete semantic metadata from ChunkMetadata
  #[builder(default = String::new())]
  pub node_type: String, // "function", "class", "method", etc.
  #[builder(default = None)]
  pub node_name: Option<String>, // "parse_document", "MyClass", etc.
  #[builder(default = String::new())]
  pub language: String, // Programming language
  #[builder(default = None)]
  pub parent_context: Option<String>, // "class MyClass" for methods
  #[builder(default = Vec::new())]
  pub scope_path: Vec<String>, // ["module", "MyClass", "parse_document"]
  #[builder(default = Vec::new())]
  pub definitions: Vec<String>, // Variable/function names defined
  #[builder(default = Vec::new())]
  pub references: Vec<String>, // Variable/function names referenced

  #[builder(default = chrono::Utc::now().naive_utc())]
  pub indexed_at: chrono::NaiveDateTime,
}

impl CodeChunk {
  /// Create the Arrow schema for CodeChunk with the given embedding dimensions
  pub fn schema(embedding_dim: usize) -> arrow::datatypes::Schema {
    use arrow::datatypes::{DataType, Field, Schema, TimeUnit};

    let fields = vec![
      Field::new("id", DataType::Utf8, false),
      Field::new("file_id", DataType::Utf8, false),
      Field::new("project_id", DataType::Utf8, false), // UUID as string
      Field::new("file_path", DataType::Utf8, false),
      Field::new("content", DataType::Utf8, false),
      Field::new("chunk_hash", DataType::FixedSizeBinary(32), false),
      Field::new(
        "embedding",
        DataType::FixedSizeList(
          std::sync::Arc::new(Field::new("item", DataType::Float32, true)),
          embedding_dim as i32,
        ),
        false,
      ),
      // Position fields
      Field::new("start_byte", DataType::UInt64, false),
      Field::new("end_byte", DataType::UInt64, false),
      Field::new("start_line", DataType::UInt64, false),
      Field::new("end_line", DataType::UInt64, false),
      // Semantic metadata
      Field::new("node_type", DataType::Utf8, false),
      Field::new("node_name", DataType::Utf8, true),
      Field::new("language", DataType::Utf8, false),
      Field::new("parent_context", DataType::Utf8, true),
      Field::new(
        "scope_path",
        DataType::List(std::sync::Arc::new(Field::new(
          "item",
          DataType::Utf8,
          true,
        ))),
        false,
      ),
      Field::new(
        "definitions",
        DataType::List(std::sync::Arc::new(Field::new(
          "item",
          DataType::Utf8,
          true,
        ))),
        false,
      ),
      Field::new(
        "references",
        DataType::List(std::sync::Arc::new(Field::new(
          "item",
          DataType::Utf8,
          true,
        ))),
        false,
      ),
      Field::new(
        "indexed_at",
        DataType::Timestamp(TimeUnit::Microsecond, None),
        false,
      ),
    ];

    Schema::new(fields)
  }

  /// Create a LanceDB table for storing CodeChunks
  pub async fn create_table(
    connection: &lancedb::Connection,
    table_name: &str,
    embedding_dim: usize,
  ) -> lancedb::Result<lancedb::Table> {
    use arrow::array::*;
    use arrow::datatypes::{DataType, Field};
    use arrow::record_batch::{RecordBatch, RecordBatchIterator};

    let schema = std::sync::Arc::new(Self::schema(embedding_dim));

    // Create dummy data - LanceDB requires at least one batch
    let id_array = StringArray::from(vec![Uuid::nil().to_string()]);
    let file_id_array = StringArray::from(vec![Uuid::nil().to_string()]);
    let project_id_array = StringArray::from(vec![Uuid::nil().to_string()]);
    let file_path_array = StringArray::from(vec!["__lancedb_dummy__.txt"]);
    let content_array = StringArray::from(vec![
      "LanceDB requires at least one chunk to create a table with proper schema",
    ]);

    let mut chunk_hash_builder = FixedSizeBinaryBuilder::with_capacity(1, 32);
    chunk_hash_builder.append_value([0u8; 32]).unwrap();
    let chunk_hash_array = chunk_hash_builder.finish();

    // Create dummy embedding
    let embedding_array = Float32Array::from(vec![0.0; embedding_dim]);
    let embedding_field = std::sync::Arc::new(Field::new("item", DataType::Float32, true));
    let embedding_list = FixedSizeListArray::try_new(
      embedding_field,
      embedding_dim as i32,
      std::sync::Arc::new(embedding_array),
      None,
    )
    .map_err(|e| lancedb::Error::Arrow { source: e })?;

    // Position fields
    let start_byte_array = UInt64Array::from(vec![0u64]);
    let end_byte_array = UInt64Array::from(vec![0u64]);
    let start_line_array = UInt64Array::from(vec![0u64]);
    let end_line_array = UInt64Array::from(vec![0u64]);

    // Semantic metadata
    let node_type_array = StringArray::from(vec!["dummy"]);
    let node_name_array = StringArray::from(vec![None as Option<&str>]);
    let language_array = StringArray::from(vec!["text"]);
    let parent_context_array = StringArray::from(vec![None as Option<&str>]);

    // Empty lists for scope_path, definitions, references
    let mut scope_path_builder = ListBuilder::new(StringBuilder::new());
    scope_path_builder.append(true); // Append one empty list
    let scope_path_array = scope_path_builder.finish();

    let mut definitions_builder = ListBuilder::new(StringBuilder::new());
    definitions_builder.append(true); // Append one empty list
    let definitions_array = definitions_builder.finish();

    let mut references_builder = ListBuilder::new(StringBuilder::new());
    references_builder.append(true); // Append one empty list
    let references_array = references_builder.finish();

    let indexed_at_array = arrow::array::TimestampMicrosecondArray::from(vec![0i64]);

    let batch = RecordBatch::try_new(
      schema.clone(),
      vec![
        std::sync::Arc::new(id_array),
        std::sync::Arc::new(file_id_array),
        std::sync::Arc::new(project_id_array),
        std::sync::Arc::new(file_path_array),
        std::sync::Arc::new(content_array),
        std::sync::Arc::new(chunk_hash_array),
        std::sync::Arc::new(embedding_list),
        std::sync::Arc::new(start_byte_array),
        std::sync::Arc::new(end_byte_array),
        std::sync::Arc::new(start_line_array),
        std::sync::Arc::new(end_line_array),
        std::sync::Arc::new(node_type_array),
        std::sync::Arc::new(node_name_array),
        std::sync::Arc::new(language_array),
        std::sync::Arc::new(parent_context_array),
        std::sync::Arc::new(scope_path_array),
        std::sync::Arc::new(definitions_array),
        std::sync::Arc::new(references_array),
        std::sync::Arc::new(indexed_at_array),
      ],
    )
    .map_err(|e| lancedb::Error::Arrow { source: e })?;

    let batch_iter = RecordBatchIterator::new(vec![Ok(batch)].into_iter(), schema);

    // Create table
    let table = connection
      .create_table(table_name, Box::new(batch_iter))
      .execute()
      .await?;

    // Create indices on new table
    Self::ensure_indices(&table).await?;

    Ok(table)
  }

  /// Ensure a table exists - open if it exists, create if it doesn't
  pub async fn ensure_table(
    connection: &lancedb::Connection,
    table_name: &str,
    embedding_dim: usize,
  ) -> lancedb::Result<lancedb::Table> {
    match connection.open_table(table_name).execute().await {
      Ok(table) => {
        Self::ensure_indices(&table).await?;
        Ok(table)
      }
      Err(e) => match &e {
        lancedb::Error::TableNotFound { .. } => {
          Self::create_table(connection, table_name, embedding_dim).await
        }
        _ => Err(e),
      },
    }
  }

  /// Ensure indices exist on all searchable fields
  async fn ensure_indices(table: &lancedb::Table) -> lancedb::Result<()> {
    // Vector index on embeddings
    table
      .create_index(
        &["embedding"],
        Index::IvfFlat(IvfFlatIndexBuilder::default()),
      )
      .execute()
      .await?;
    debug!("Created vector index on embedding field");

    // Basic indices
    table
      .create_index(&["file_id"], Index::Auto)
      .execute()
      .await?;
    debug!("Created index on file_id field");

    table
      .create_index(&["project_id"], Index::Auto)
      .execute()
      .await?;
    debug!("Created index on project_id field");

    table
      .create_index(&["file_path"], Index::Auto)
      .execute()
      .await?;
    debug!("Created index on file_path field");

    // Semantic metadata indices
    table
      .create_index(&["language"], Index::Auto)
      .execute()
      .await?;
    debug!("Created index on language field");

    table
      .create_index(&["node_type"], Index::Auto)
      .execute()
      .await?;
    debug!("Created index on node_type field");

    table
      .create_index(&["node_name"], Index::Auto)
      .execute()
      .await?;
    debug!("Created index on node_name field");

    // Create FTS index on content field for full-text search
    table
      .create_index(&["content"], Index::FTS(Default::default()))
      .execute()
      .await?;
    debug!("Created FTS index on content field");

    // Create LabelList indices for array fields to support array_contains_any queries
    table
      .create_index(
        &["scope_path"],
        Index::LabelList(LabelListIndexBuilder::default()),
      )
      .execute()
      .await?;
    debug!("Created LabelList index on scope_path field");

    table
      .create_index(
        &["definitions"],
        Index::LabelList(LabelListIndexBuilder::default()),
      )
      .execute()
      .await?;
    debug!("Created LabelList index on definitions field");

    table
      .create_index(
        &["references"],
        Index::LabelList(LabelListIndexBuilder::default()),
      )
      .execute()
      .await?;
    debug!("Created LabelList index on references field");

    Ok(())
  }

  pub fn compute_hash(content: &str) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(content.as_bytes());
    let hash = hasher.finalize();
    let mut result = [0u8; 32];
    result.copy_from_slice(hash.as_bytes());
    result
  }

  pub fn update_embedding(&mut self, embedding: Vec<f32>) {
    self.embedding = embedding;
    self.indexed_at = chrono::Utc::now().naive_utc();
  }

  pub fn update_metadata(&mut self, metadata: ChunkMetadataUpdate) {
    self.node_type = metadata.node_type;
    self.node_name = metadata.node_name;
    self.language = metadata.language;
    self.parent_context = metadata.parent_context;
    self.scope_path = metadata.scope_path;
    self.definitions = metadata.definitions;
    self.references = metadata.references;
  }

  /// Convert a RecordBatch row to CodeChunk
  pub fn from_record_batch(
    batch: &arrow::record_batch::RecordBatch,
    row: usize,
  ) -> Result<Self, lancedb::Error> {
    use arrow::array::*;

    if row >= batch.num_rows() {
      return Err(lancedb::Error::Runtime {
        message: format!(
          "Row index {} out of bounds (batch has {} rows)",
          row,
          batch.num_rows()
        ),
      });
    }

    let id_str = batch
      .column_by_name("id")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid id column".to_string(),
      })?
      .value(row);

    let id = Uuid::parse_str(id_str).map_err(|e| lancedb::Error::Runtime {
      message: format!("Invalid UUID in id column: {}", e),
    })?;

    let file_id_str = batch
      .column_by_name("file_id")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid file_id column".to_string(),
      })?
      .value(row);

    let file_id = Uuid::parse_str(file_id_str).map_err(|e| lancedb::Error::Runtime {
      message: format!("Invalid UUID in file_id column: {}", e),
    })?;

    let project_id_str = batch
      .column_by_name("project_id")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid project_id column".to_string(),
      })?
      .value(row);

    let project_id = Uuid::parse_str(project_id_str).map_err(|e| lancedb::Error::Runtime {
      message: format!("Invalid UUID in project_id column: {}", e),
    })?;

    let file_path = batch
      .column_by_name("file_path")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid file_path column".to_string(),
      })?
      .value(row)
      .to_string();

    let content = batch
      .column_by_name("content")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid content column".to_string(),
      })?
      .value(row)
      .to_string();

    let chunk_hash_arr = batch
      .column_by_name("chunk_hash")
      .and_then(|col| col.as_any().downcast_ref::<FixedSizeBinaryArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid chunk_hash column".to_string(),
      })?;

    let mut chunk_hash = [0u8; 32];
    chunk_hash.copy_from_slice(chunk_hash_arr.value(row));

    // Extract embedding
    let embedding_list = batch
      .column_by_name("embedding")
      .and_then(|col| col.as_any().downcast_ref::<FixedSizeListArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid embedding column".to_string(),
      })?;

    let embedding_value = embedding_list.value(row);
    let embedding_values = embedding_value
      .as_any()
      .downcast_ref::<Float32Array>()
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Invalid embedding array type".to_string(),
      })?;

    let embedding: Vec<f32> = (0..embedding_values.len())
      .map(|i| embedding_values.value(i))
      .collect();

    // Position fields
    let start_byte = batch
      .column_by_name("start_byte")
      .and_then(|col| col.as_any().downcast_ref::<UInt64Array>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid start_byte column".to_string(),
      })?
      .value(row) as usize;

    let end_byte = batch
      .column_by_name("end_byte")
      .and_then(|col| col.as_any().downcast_ref::<UInt64Array>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid end_byte column".to_string(),
      })?
      .value(row) as usize;

    let start_line = batch
      .column_by_name("start_line")
      .and_then(|col| col.as_any().downcast_ref::<UInt64Array>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid start_line column".to_string(),
      })?
      .value(row) as usize;

    let end_line = batch
      .column_by_name("end_line")
      .and_then(|col| col.as_any().downcast_ref::<UInt64Array>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid end_line column".to_string(),
      })?
      .value(row) as usize;

    // Semantic metadata
    let node_type = batch
      .column_by_name("node_type")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid node_type column".to_string(),
      })?
      .value(row)
      .to_string();

    let node_name = batch
      .column_by_name("node_name")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid node_name column".to_string(),
      })?;

    let node_name = if node_name.is_null(row) {
      None
    } else {
      Some(node_name.value(row).to_string())
    };

    let language = batch
      .column_by_name("language")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid language column".to_string(),
      })?
      .value(row)
      .to_string();

    let parent_context = batch
      .column_by_name("parent_context")
      .and_then(|col| col.as_any().downcast_ref::<StringArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid parent_context column".to_string(),
      })?;

    let parent_context = if parent_context.is_null(row) {
      None
    } else {
      Some(parent_context.value(row).to_string())
    };

    // Extract arrays
    let scope_path_list = batch
      .column_by_name("scope_path")
      .and_then(|col| col.as_any().downcast_ref::<ListArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid scope_path column".to_string(),
      })?;

    let scope_path: Vec<String> = if scope_path_list.is_null(row) {
      Vec::new()
    } else {
      let values = scope_path_list.value(row);
      let strings = values
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::Runtime {
          message: "Invalid scope_path array type".to_string(),
        })?;

      (0..strings.len())
        .filter(|i| !strings.is_null(*i))
        .map(|i| strings.value(i).to_string())
        .collect()
    };

    let definitions_list = batch
      .column_by_name("definitions")
      .and_then(|col| col.as_any().downcast_ref::<ListArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid definitions column".to_string(),
      })?;

    let definitions: Vec<String> = if definitions_list.is_null(row) {
      Vec::new()
    } else {
      let values = definitions_list.value(row);
      let strings = values
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::Runtime {
          message: "Invalid definitions array type".to_string(),
        })?;

      (0..strings.len())
        .filter(|i| !strings.is_null(*i))
        .map(|i| strings.value(i).to_string())
        .collect()
    };

    let references_list = batch
      .column_by_name("references")
      .and_then(|col| col.as_any().downcast_ref::<ListArray>())
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid references column".to_string(),
      })?;

    let references: Vec<String> = if references_list.is_null(row) {
      Vec::new()
    } else {
      let values = references_list.value(row);
      let strings = values
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| lancedb::Error::Runtime {
          message: "Invalid references array type".to_string(),
        })?;

      (0..strings.len())
        .filter(|i| !strings.is_null(*i))
        .map(|i| strings.value(i).to_string())
        .collect()
    };

    let indexed_at = batch
      .column_by_name("indexed_at")
      .and_then(|col| {
        col
          .as_any()
          .downcast_ref::<arrow::array::TimestampMicrosecondArray>()
      })
      .ok_or_else(|| lancedb::Error::Runtime {
        message: "Missing or invalid indexed_at column".to_string(),
      })?
      .value(row);

    Ok(Self {
      id,
      file_id,
      project_id,
      file_path,
      content,
      chunk_hash,
      embedding,
      start_byte,
      end_byte,
      start_line,
      end_line,
      node_type,
      node_name,
      language,
      parent_context,
      scope_path,
      definitions,
      references,
      indexed_at: chrono::DateTime::from_timestamp_micros(indexed_at)
        .unwrap_or_default()
        .naive_utc(),
    })
  }
}

// Implement IntoArrow for CodeChunk to enable LanceDB persistence
impl lancedb::arrow::IntoArrow for CodeChunk {
  fn into_arrow(self) -> lancedb::Result<Box<dyn arrow::array::RecordBatchReader + Send>> {
    use arrow::array::*;
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    let embedding_dim = self.embedding.len();
    let schema = Arc::new(CodeChunk::schema(embedding_dim));

    // Build arrays for single chunk
    let id_array = StringArray::from(vec![self.id.to_string()]);
    let file_id_array = StringArray::from(vec![self.file_id.to_string()]);
    let project_id_array = StringArray::from(vec![self.project_id.to_string()]);
    let file_path_array = StringArray::from(vec![self.file_path.as_str()]);
    let content_array = StringArray::from(vec![self.content.as_str()]);

    let mut chunk_hash_builder = FixedSizeBinaryBuilder::with_capacity(1, 32);
    chunk_hash_builder.append_value(self.chunk_hash).unwrap();
    let chunk_hash_array = chunk_hash_builder.finish();

    // Create embedding array
    let embedding_array = Float32Array::from(self.embedding);
    let embedding_field = Arc::new(arrow::datatypes::Field::new(
      "item",
      arrow::datatypes::DataType::Float32,
      true,
    ));
    let embedding_list = FixedSizeListArray::try_new(
      embedding_field,
      embedding_dim as i32,
      Arc::new(embedding_array),
      None,
    )
    .map_err(|e| lancedb::Error::Arrow { source: e })?;

    // Position fields
    let start_byte_array = UInt64Array::from(vec![self.start_byte as u64]);
    let end_byte_array = UInt64Array::from(vec![self.end_byte as u64]);
    let start_line_array = UInt64Array::from(vec![self.start_line as u64]);
    let end_line_array = UInt64Array::from(vec![self.end_line as u64]);

    // Semantic metadata
    let node_type_array = StringArray::from(vec![self.node_type.as_str()]);
    let node_name_array = StringArray::from(vec![self.node_name.as_deref()]);
    let language_array = StringArray::from(vec![self.language.as_str()]);
    let parent_context_array = StringArray::from(vec![self.parent_context.as_deref()]);

    // Lists
    let mut scope_path_builder = ListBuilder::new(StringBuilder::new());
    for item in &self.scope_path {
      scope_path_builder.values().append_value(item);
    }
    scope_path_builder.append(true);
    let scope_path_array = scope_path_builder.finish();

    let mut definitions_builder = ListBuilder::new(StringBuilder::new());
    for item in &self.definitions {
      definitions_builder.values().append_value(item);
    }
    definitions_builder.append(true);
    let definitions_array = definitions_builder.finish();

    let mut references_builder = ListBuilder::new(StringBuilder::new());
    for item in &self.references {
      references_builder.values().append_value(item);
    }
    references_builder.append(true);
    let references_array = references_builder.finish();

    let indexed_at_us = self.indexed_at.and_utc().timestamp_micros();
    let indexed_at_array = arrow::array::TimestampMicrosecondArray::from(vec![indexed_at_us]);

    // Create the record batch
    let batch = RecordBatch::try_new(
      schema.clone(),
      vec![
        Arc::new(id_array),
        Arc::new(file_id_array),
        Arc::new(project_id_array),
        Arc::new(file_path_array),
        Arc::new(content_array),
        Arc::new(chunk_hash_array),
        Arc::new(embedding_list),
        Arc::new(start_byte_array),
        Arc::new(end_byte_array),
        Arc::new(start_line_array),
        Arc::new(end_line_array),
        Arc::new(node_type_array),
        Arc::new(node_name_array),
        Arc::new(language_array),
        Arc::new(parent_context_array),
        Arc::new(scope_path_array),
        Arc::new(definitions_array),
        Arc::new(references_array),
        Arc::new(indexed_at_array),
      ],
    )
    .map_err(|e| lancedb::Error::Arrow { source: e })?;

    // Return as RecordBatchReader
    Ok(Box::new(arrow::record_batch::RecordBatchIterator::new(
      vec![Ok(batch)].into_iter(),
      schema,
    )))
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use arrow::array::{FixedSizeListArray, Float32Array, StringArray};
  use futures::stream::TryStreamExt;
  use lancedb::arrow::IntoArrow;
  use lancedb::query::{ExecutableQuery, QueryBase};

  #[test]
  fn test_code_chunk_builder() {
    let id = Uuid::now_v7();
    let file_id = Uuid::now_v7();
    let project_id = Uuid::now_v7();
    let mut chunk = CodeChunk::builder()
      .file_id(file_id)
      .project_id(project_id)
      .file_path("test.rs".to_string())
      .content("fn main() {}".to_string())
      .start_byte(0)
      .end_byte(12)
      .start_line(1)
      .end_line(1)
      .build();
    chunk.id = id; // Set the ID we want to test with

    assert_eq!(chunk.id, id);
    assert_eq!(chunk.file_id, file_id);
    assert_eq!(chunk.project_id, project_id);
    assert_eq!(chunk.file_path, "test.rs");
    assert_eq!(chunk.content, "fn main() {}");
    assert_eq!(chunk.start_byte, 0);
    assert_eq!(chunk.end_byte, 12);
    assert_eq!(chunk.start_line, 1);
    assert_eq!(chunk.end_line, 1);
    assert!(chunk.embedding.is_empty());
    assert_eq!(chunk.node_type, "");
    assert!(chunk.node_name.is_none());
  }

  // Removed test_code_chunk_new_nil_project_id since builder pattern doesn't have validation

  #[test]
  fn test_compute_hash() {
    let content = "fn main() {}";
    let hash1 = CodeChunk::compute_hash(content);
    let hash2 = CodeChunk::compute_hash(content);
    assert_eq!(hash1, hash2, "Hash should be deterministic");

    let different_content = "fn test() {}";
    let hash3 = CodeChunk::compute_hash(different_content);
    assert_ne!(hash1, hash3, "Different content should have different hash");
  }

  #[test]
  fn test_update_embedding() {
    let mut chunk = CodeChunk::builder()
      .file_id(Uuid::now_v7())
      .project_id(Uuid::now_v7())
      .file_path("test.rs".to_string())
      .content("fn main() {}".to_string())
      .start_byte(0)
      .end_byte(12)
      .start_line(1)
      .end_line(1)
      .build();

    let embedding = vec![1.0, 2.0, 3.0];
    chunk.update_embedding(embedding.clone());

    assert_eq!(chunk.embedding, embedding);
    // Just verify it's set, don't test exact time
    assert!(chunk.indexed_at.and_utc().timestamp() > 0);
  }

  #[test]
  fn test_update_metadata() {
    let mut chunk = CodeChunk::builder()
      .file_id(Uuid::now_v7())
      .project_id(Uuid::now_v7())
      .file_path("test.rs".to_string())
      .content("fn main() {}".to_string())
      .start_byte(0)
      .end_byte(12)
      .start_line(1)
      .end_line(1)
      .build();

    chunk.update_metadata(
      ChunkMetadataUpdate::builder()
        .node_type("function".to_string())
        .node_name(Some("main".to_string()))
        .language("rust".to_string())
        .parent_context(None)
        .scope_path(vec!["module".to_string(), "main".to_string()])
        .definitions(vec!["main".to_string()])
        .references(vec!["println".to_string()])
        .build(),
    );

    assert_eq!(chunk.node_type, "function");
    assert_eq!(chunk.node_name, Some("main".to_string()));
    assert_eq!(chunk.language, "rust");
    assert!(chunk.parent_context.is_none());
    assert_eq!(chunk.scope_path, vec!["module", "main"]);
    assert_eq!(chunk.definitions, vec!["main"]);
    assert_eq!(chunk.references, vec!["println"]);
  }

  #[tokio::test]
  async fn test_schema() {
    let schema = CodeChunk::schema(384);
    assert_eq!(schema.fields().len(), 19);

    // Check field names and types
    assert_eq!(schema.field(0).name(), "id");
    assert_eq!(schema.field(1).name(), "file_id");
    assert_eq!(schema.field(2).name(), "project_id");
    assert_eq!(schema.field(3).name(), "file_path");
    assert_eq!(schema.field(4).name(), "content");
    assert_eq!(schema.field(5).name(), "chunk_hash");
    assert_eq!(schema.field(6).name(), "embedding");
    assert_eq!(schema.field(7).name(), "start_byte");
    assert_eq!(schema.field(8).name(), "end_byte");
    assert_eq!(schema.field(9).name(), "start_line");
    assert_eq!(schema.field(10).name(), "end_line");
    assert_eq!(schema.field(11).name(), "node_type");
    assert_eq!(schema.field(12).name(), "node_name");
    assert_eq!(schema.field(13).name(), "language");
    assert_eq!(schema.field(14).name(), "parent_context");
    assert_eq!(schema.field(15).name(), "scope_path");
    assert_eq!(schema.field(16).name(), "definitions");
    assert_eq!(schema.field(17).name(), "references");
    assert_eq!(schema.field(18).name(), "indexed_at");
  }

  #[tokio::test]
  async fn test_create_table() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let db_path = temp_dir.path().to_str().unwrap();
    let conn = lancedb::connect(db_path).execute().await.unwrap();

    let table = CodeChunk::create_table(&conn, "test_chunks", 3)
      .await
      .unwrap();

    // Verify table was created
    let table_names = conn.table_names().execute().await.unwrap();
    assert!(table_names.contains(&"test_chunks".to_string()));

    // Verify we can query the table (should have dummy chunk)
    let count = table.count_rows(None).await.unwrap();
    assert_eq!(count, 1, "Should have dummy chunk");

    // Verify dummy chunk is correct
    let stream = table
      .query()
      .only_if(format!("id = '{}'", Uuid::nil()))
      .execute()
      .await
      .unwrap();

    let batches: Vec<_> = stream.try_collect().await.unwrap();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
  }

  #[tokio::test]
  async fn test_ensure_table() {
    let temp_dir = tempfile::TempDir::new().unwrap();
    let db_path = temp_dir.path().to_str().unwrap();
    let conn = lancedb::connect(db_path).execute().await.unwrap();

    // First call creates table
    let table1 = CodeChunk::ensure_table(&conn, "test_chunks", 3)
      .await
      .unwrap();
    let count1 = table1.count_rows(None).await.unwrap();
    assert_eq!(count1, 1, "Should have dummy chunk");

    // Second call opens existing table
    let table2 = CodeChunk::ensure_table(&conn, "test_chunks", 3)
      .await
      .unwrap();
    let count2 = table2.count_rows(None).await.unwrap();
    assert_eq!(count2, 1, "Should still have only dummy chunk");
  }

  #[test]
  fn test_into_arrow() {
    let mut chunk = CodeChunk::builder()
      .file_id(Uuid::now_v7())
      .project_id(Uuid::now_v7())
      .file_path("test.rs".to_string())
      .content("fn main() {}".to_string())
      .start_byte(0)
      .end_byte(12)
      .start_line(1)
      .end_line(1)
      .build();

    chunk.update_embedding(vec![1.0, 2.0, 3.0]);
    chunk.update_metadata(
      ChunkMetadataUpdate::builder()
        .node_type("function".to_string())
        .node_name(Some("main".to_string()))
        .language("rust".to_string())
        .parent_context(None)
        .scope_path(vec!["main".to_string()])
        .definitions(vec!["main".to_string()])
        .references(vec![])
        .build(),
    );

    let reader = chunk.into_arrow().unwrap();
    let schema = reader.schema();
    assert_eq!(schema.fields().len(), 19);

    // Collect batches
    let batches: Vec<_> = reader.collect();
    assert_eq!(batches.len(), 1);

    let batch = batches[0].as_ref().unwrap();
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(batch.num_columns(), 19);

    // Verify content
    let content_array = batch
      .column(4)
      .as_any()
      .downcast_ref::<StringArray>()
      .unwrap();
    assert_eq!(content_array.value(0), "fn main() {}");

    // Verify embedding
    let embedding_array = batch
      .column(6)
      .as_any()
      .downcast_ref::<FixedSizeListArray>()
      .unwrap();
    let values = embedding_array.value(0);
    let float_array = values.as_any().downcast_ref::<Float32Array>().unwrap();
    assert_eq!(float_array.len(), 3);
    assert_eq!(float_array.value(0), 1.0);
    assert_eq!(float_array.value(1), 2.0);
    assert_eq!(float_array.value(2), 3.0);
  }

  #[test]
  fn test_from_record_batch() {
    let file_id = Uuid::now_v7();
    let project_id = Uuid::now_v7();
    let mut chunk = CodeChunk::builder()
      .file_id(file_id)
      .project_id(project_id)
      .file_path("test.rs".to_string())
      .content("fn main() {}".to_string())
      .start_byte(0)
      .end_byte(12)
      .start_line(1)
      .end_line(1)
      .build();

    chunk.update_embedding(vec![1.0, 2.0, 3.0]);
    chunk.update_metadata(
      ChunkMetadataUpdate::builder()
        .node_type("function".to_string())
        .node_name(Some("main".to_string()))
        .language("rust".to_string())
        .parent_context(Some("module".to_string()))
        .scope_path(vec!["module".to_string(), "main".to_string()])
        .definitions(vec!["main".to_string()])
        .references(vec!["println".to_string()])
        .build(),
    );

    // Convert to arrow and back
    let reader = chunk.clone().into_arrow().unwrap();
    let batches: Vec<_> = reader.collect();
    let batch = batches[0].as_ref().unwrap();

    let reconstructed = CodeChunk::from_record_batch(batch, 0).unwrap();

    assert_eq!(reconstructed.id, chunk.id);
    assert_eq!(reconstructed.file_id, chunk.file_id);
    assert_eq!(reconstructed.project_id, chunk.project_id);
    assert_eq!(reconstructed.file_path, chunk.file_path);
    assert_eq!(reconstructed.content, chunk.content);
    assert_eq!(reconstructed.chunk_hash, chunk.chunk_hash);
    assert_eq!(reconstructed.embedding, chunk.embedding);
    assert_eq!(reconstructed.start_byte, chunk.start_byte);
    assert_eq!(reconstructed.end_byte, chunk.end_byte);
    assert_eq!(reconstructed.start_line, chunk.start_line);
    assert_eq!(reconstructed.end_line, chunk.end_line);
    assert_eq!(reconstructed.node_type, chunk.node_type);
    assert_eq!(reconstructed.node_name, chunk.node_name);
    assert_eq!(reconstructed.language, chunk.language);
    assert_eq!(reconstructed.parent_context, chunk.parent_context);
    assert_eq!(reconstructed.scope_path, chunk.scope_path);
    assert_eq!(reconstructed.definitions, chunk.definitions);
    assert_eq!(reconstructed.references, chunk.references);
    // Don't check indexed_at as it may differ by microseconds
  }

  #[test]
  fn test_from_record_batch_with_nulls() {
    let file_id = Uuid::now_v7();
    let project_id = Uuid::now_v7();
    let mut chunk = CodeChunk::builder()
      .file_id(file_id)
      .project_id(project_id)
      .file_path("test.rs".to_string())
      .content("fn main() {}".to_string())
      .start_byte(0)
      .end_byte(12)
      .start_line(1)
      .end_line(1)
      .build();

    chunk.update_embedding(vec![1.0, 2.0, 3.0]);
    chunk.update_metadata(
      ChunkMetadataUpdate::builder()
        .node_type("function".to_string())
        .node_name(None) // No node name
        .language("rust".to_string())
        .parent_context(None) // No parent context
        .scope_path(vec![]) // Empty scope path
        .definitions(vec![]) // Empty definitions
        .references(vec![]) // Empty references
        .build(),
    );

    // Convert to arrow and back
    let reader = chunk.clone().into_arrow().unwrap();
    let batches: Vec<_> = reader.collect();
    let batch = batches[0].as_ref().unwrap();

    let reconstructed = CodeChunk::from_record_batch(batch, 0).unwrap();

    assert_eq!(reconstructed.node_name, None);
    assert_eq!(reconstructed.parent_context, None);
    assert_eq!(reconstructed.scope_path, Vec::<String>::new());
    assert_eq!(reconstructed.definitions, Vec::<String>::new());
    assert_eq!(reconstructed.references, Vec::<String>::new());
  }

  #[test]
  fn test_from_record_batch_out_of_bounds() {
    let mut chunk = CodeChunk::builder()
      .file_id(Uuid::now_v7())
      .project_id(Uuid::now_v7())
      .file_path("test.rs".to_string())
      .content("fn main() {}".to_string())
      .start_byte(0)
      .end_byte(12)
      .start_line(1)
      .end_line(1)
      .build();

    // Set embedding to avoid arrow conversion error
    chunk.update_embedding(vec![1.0, 2.0, 3.0]);
    chunk.update_metadata(
      ChunkMetadataUpdate::builder()
        .node_type("function".to_string())
        .node_name(None)
        .language("rust".to_string())
        .parent_context(None)
        .scope_path(vec![])
        .definitions(vec![])
        .references(vec![])
        .build(),
    );

    let reader = chunk.into_arrow().unwrap();
    let batches: Vec<_> = reader.collect();
    let batch = batches[0].as_ref().unwrap();

    // Try to read beyond bounds
    let result = CodeChunk::from_record_batch(batch, 1);
    assert!(result.is_err());
    assert!(result.unwrap_err().to_string().contains("out of bounds"));
  }
}
