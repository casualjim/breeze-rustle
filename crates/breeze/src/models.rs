use std::path;

use arrow::array::*;
use arrow::datatypes::DataType;
use arrow_convert::deserialize::{arrow_array_deserialize_iterator, TryIntoCollection};
use arrow_convert::field::ArrowField;
use arrow_convert::serialize::TryIntoArrow;
use blake3::Hasher;
use lancedb::arrow::IntoArrow;
use serde::{Deserialize, Serialize};
use arrow_convert::{ArrowDeserialize, ArrowField, ArrowSerialize};
use tokio::io::AsyncReadExt as _;
use uuid::Uuid;

#[derive(Debug, Clone, PartialEq, ArrowField, ArrowSerialize, ArrowDeserialize)]
pub struct CodeDocument {
    pub id: String,
    pub file_path: String,
    pub content: String,
    pub content_hash: [u8; 32],
    pub content_embedding: Vec<f32>,
    pub file_size: u64,
    pub last_modified: chrono::NaiveDateTime,
    pub indexed_at: chrono::NaiveDateTime,
}

impl CodeDocument {
    pub fn new(file_path: String, content: String) -> Self {
        let id = Uuid::new_v4().to_string();
        let content_hash = Self::compute_hash(content.as_str());
        let file_size = content.len() as u64;
        let last_modified = chrono::Utc::now().naive_utc();
        let indexed_at = last_modified;

        Self {
            id,
            file_path,
            content,
            content_hash,
            content_embedding: Vec::new(),
            file_size,
            last_modified,
            indexed_at,
        }
    }

    pub async fn compute_content_hash(path: &path::Path) -> std::io::Result<blake3::Hash> {
        let mut file = tokio::fs::File::open(path).await?;
        let mut hasher = Hasher::new();
        let mut buffer = vec![0u8; 1024 * 1024]; // 1MB chunks

        loop {
            let n = file.read(&mut buffer).await?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }

        Ok(hasher.finalize())
    }

    pub fn compute_hash(content: &str) -> [u8; 32] {
        let mut hasher = Hasher::new();
        hasher.update(content.as_bytes());
        let hash = hasher.finalize();
        let mut result = [0u8; 32];
        result.copy_from_slice(hash.as_bytes());
        result
    }

    pub fn update(&mut self, content: String) {
        self.content = content;
        self.content_hash = Self::compute_hash(self.content.as_str());
        self.file_size = self.content.len() as u64;
        self.last_modified = chrono::Utc::now().naive_utc();
    }

    pub fn update_embedding(&mut self, embedding: Vec<f32>) {
        self.content_embedding = embedding;
        self.indexed_at = chrono::Utc::now().naive_utc();
    }
}
