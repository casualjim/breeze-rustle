# Advanced Search Implementation Plan

## Overview

This document outlines the implementation of an advanced two-stage search system that supports searching at both document and chunk granularity, with comprehensive metadata filtering capabilities.

## Goals

1. Store individual code chunks with their embeddings alongside aggregated file documents
2. Support flexible search granularity (document-level, chunk-level, or both)
3. Expose all semantic metadata for advanced filtering
4. Provide structured search results with relevant code snippets
5. Enable future semantic queries like "find all error handling functions" or "show methods that reference AWS"

## Architecture

### Data Model Changes

#### New: CodeChunk Model

```rust
pub struct CodeChunk {
    pub id: String,                    // Unique chunk ID (UUID v7)
    pub file_id: String,               // FK to CodeDocument
    pub project_id: Uuid,              // Project this chunk belongs to
    pub file_path: String,             // Denormalized for performance
    pub content: String,               // The actual chunk text
    pub chunk_hash: [u8; 32],          // Blake3 hash of content
    pub embedding: Vec<f32>,           // Chunk embedding vector
    
    // Position information
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
    
    // Complete semantic metadata from ChunkMetadata
    pub node_type: String,             // "function", "class", "method", etc.
    pub node_name: Option<String>,     // "parse_document", "MyClass", etc.
    pub language: String,              // Programming language
    pub parent_context: Option<String>,// "class MyClass" for methods
    pub scope_path: Vec<String>,       // ["module", "MyClass", "parse_document"]
    pub definitions: Vec<String>,      // Variable/function names defined
    pub references: Vec<String>,       // Variable/function names referenced
    
    pub indexed_at: NaiveDateTime,
}
```

#### Updated: CodeDocument Model

```rust
pub struct CodeDocument {
    // Existing fields...
    pub id: String,
    pub project_id: Uuid,
    pub file_path: String,
    pub content: String,
    pub content_hash: [u8; 32],
    pub content_embedding: Vec<f32>,
    pub file_size: u64,
    pub last_modified: NaiveDateTime,
    pub indexed_at: NaiveDateTime,
    
    // New fields
    pub languages: Vec<String>,         // All languages found in file
    pub primary_language: Option<String>, // Most prevalent language by token count
    pub chunk_count: u32,               // Number of chunks in this file
}
```

### Search API

#### Search Options

```rust
// Phase 3 & 4 Implementation (COMPLETED)
pub enum SearchGranularity {
    Document,  // Default - search files, return with top chunks
    Chunk,     // Search chunks directly, return chunks grouped by file
}

pub struct SearchOptions {
    // Basic options
    pub languages: Option<Vec<String>>,     // Filter by languages
    pub file_limit: usize,                  // Number of files to return (default: 5)
    pub chunks_per_file: usize,             // Number of chunks per file (default: 3)
    pub granularity: SearchGranularity,     // Search mode: Document or Chunk
    
    // Semantic filters (mainly for Chunk mode)
    pub node_types: Option<Vec<String>>,       // ["function", "class"]
    pub node_name_pattern: Option<String>,     // Exact match (not regex yet)
    pub parent_context_pattern: Option<String>, // Exact match on parent
    pub scope_depth: Option<(usize, usize)>,   // Min and max nesting level
    pub has_definitions: Option<Vec<String>>,  // Must define these symbols
    pub has_references: Option<Vec<String>>,   // Must reference these symbols
}
```

#### Search Results

```rust
// Phase 3 Implementation (COMPLETED)
pub struct SearchResult {
    pub id: String,
    pub file_path: String,
    pub relevance_score: f32,
    pub chunk_count: u32,
    pub chunks: Vec<ChunkResult>,  // Top chunks from this file
}

pub struct ChunkResult {
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub relevance_score: f32,
}

// Future Phase 4 Implementation
pub enum AdvancedSearchResult {
    Document(Vec<FileSearchResult>),   // For Document granularity
    Chunk(Vec<ChunkGroupResult>),      // For Chunk granularity
    Combined(CombinedSearchResult),    // For Both granularity
}

pub struct FileSearchResult {
    pub id: String,
    pub file_path: String,
    pub content: String,
    pub content_hash: [u8; 32],
    pub relevance_score: f32,
    pub file_size: u64,
    pub last_modified: NaiveDateTime,
    pub indexed_at: NaiveDateTime,
    pub languages: Vec<String>,
    pub primary_language: Option<String>,
    pub chunks: Vec<ChunkSearchResult>,  // Top N chunks from this file
}

pub struct ChunkSearchResult {
    pub id: String,
    pub content: String,
    pub start_line: usize,
    pub end_line: usize,
    pub language: String,
    pub node_type: String,
    pub node_name: Option<String>,
    pub parent_context: Option<String>,
    pub scope_path: Vec<String>,
    pub definitions: Vec<String>,
    pub references: Vec<String>,
    pub relevance_score: f32,
}

pub struct ChunkGroupResult {
    pub file_path: String,
    pub file_id: String,
    pub chunks: Vec<ChunkSearchResult>,
    pub total_relevance: f32,  // Aggregated score for ranking
}
```

### Database Schema

#### LanceDB Tables

1. **code_documents** table (existing, enhanced):
   - Add: languages (array), primary_language (string), chunk_count (uint32)
   - Indices: content_embedding, project_id, file_path, languages, primary_language

2. **code_chunks** table (new):
   - All fields from CodeChunk model
   - Indices:
     - embedding (vector index)
     - file_id (for chunk lookup by file)
     - project_id (for project filtering)
     - file_path (for path prefix matching)
     - language (for language filtering)
     - node_type (for construct type filtering)
     - node_name (for name-based search)
     - definitions (array index for symbol search)
     - references (array index for dependency analysis)
     - Composite: (project_id, language, node_type)

### Search Algorithm

#### Document Granularity (Default)

1. Search code_documents table with filters and query embedding
2. Get top N files by relevance
3. For each file, search code_chunks for top K chunks
4. Return structured results with files and their snippet

#### Chunk Granularity

1. Search code_chunks table directly with all filters
2. Group results by file_id
3. Sort groups by aggregate relevance
4. Return chunks grouped by file

#### Both Granularity

1. Execute both searches in parallel
2. Merge results, preferring document search for diversity
3. Deduplicate chunks
4. Return combined results

## Implementation Phases

### Phase 1: Data Model & Storage ✅ COMPLETED

1. ✅ Created CodeChunk model (`crates/breeze-indexer/src/models/code_chunk.rs`)
2. ✅ Updated CodeDocument model with languages fields
3. ✅ Created LanceDB schemas with proper indices
4. ✅ Updated document builder to return chunks with embeddings

**Key Changes:**

- Fixed ID type inconsistency - all models now use `Uuid`
- Document builder extracts language statistics from chunks
- All tests updated and passing

### Phase 2: Chunk Storage Infrastructure ✅ COMPLETED

1. ✅ Implemented chunk storage in bulk indexer
   - ✅ Created ChunkSink for LanceDB storage (`crates/breeze-indexer/src/sinks/chunk_sink.rs`)
   - ✅ Ensured code_chunks table creation with proper schema
   - ✅ Integrated chunk storage into indexing pipeline
   - ✅ Added comprehensive tests for chunk storage

2. ✅ Testing Infrastructure
   - ✅ Added tests for CodeChunk model (all CRUD operations)
   - ✅ Added tests for FailedEmbedding model
   - ✅ Verified end-to-end chunk storage in pipeline tests
   - ✅ Fixed dummy record handling in queries

**Key Implementation Details:**

- ChunkSink follows same pattern as LanceDbSink with upsert behavior
- Chunks are stored with full metadata during indexing
- Pipeline successfully stores both documents and chunks
- All 132 tests passing with chunk storage enabled

### Phase 3: Basic Search Enhancement ✅ COMPLETED

1. ✅ Updated search to use new document fields
   - ✅ Added language filtering to search API via SearchOptions
   - ✅ Implemented primary_language as a pre-filter (not ranking factor)
   - ✅ Added chunk_count field to search results

2. ✅ Return top chunks with document results
   - ✅ Query chunks by file_id after document search
   - ✅ Sort chunks by relevance within each file
   - ✅ Limit chunks per file based on chunks_per_file parameter

**Key Implementation Details:**

- Created `SearchOptions` struct with language filtering, file_limit, and chunks_per_file parameters
- Modified `hybrid_search` function to accept chunks_table and options
- Implemented two-stage search: documents first, then chunks for each document
- Added `ChunkResult` struct with content, line numbers, and relevance score
- Updated `SearchResult` to include chunks instead of full file content
- Fixed compatibility issues with breeze-server and CLI to use new result structure
- All 199 tests passing across the workspace

**Files Modified:**
- `crates/breeze-indexer/src/search.rs` - Core search implementation
- `crates/breeze-indexer/src/models/code_chunk.rs` - Added from_record_batch method
- `crates/breeze-indexer/src/indexer.rs` - Updated to pass chunks_table to search
- `crates/breeze-indexer/src/lib.rs` - Exported new types
- `crates/breeze-server/src/types.rs` - Updated for new SearchResult structure
- `crates/breeze/src/cli.rs` - Updated to display chunk results

### Phase 4: Advanced Search ✅ COMPLETED

1. ✅ Implemented chunk-level search
   - ✅ Added SearchGranularity enum (Document/Chunk modes)
   - ✅ Direct search on code_chunks table for Chunk mode
   - ✅ Apply semantic metadata filters
   - ✅ Group results by file with aggregate scoring

2. ✅ Added semantic filtering capabilities
   - ✅ Filter by node_types (function, class, method)
   - ✅ Pattern matching on node names (exact match)
   - ✅ Parent context pattern matching
   - ✅ Scope depth filtering (min/max based on scope_path length)
   - ✅ Symbol definition/reference filtering using array_contains

3. ✅ Refactored search architecture
   - ✅ Split search logic into modular components (documents.rs, chunks.rs)
   - ✅ Unified SearchOptions with granularity and semantic filters
   - ✅ Maintained backward compatibility with existing SearchResult structure

**Key Implementation Details:**

- Added SearchGranularity enum to SearchOptions for mode selection
- Refactored hybrid_search into smaller, focused functions:
  - `search_documents()` - Document mode search
  - `search_chunks()` - Chunk mode search with semantic filters
  - Helper functions for chunk aggregation and ranking
- Implemented semantic filters using LanceDB queries:
  - Used `array_contains` with OR conditions (LanceDB doesn't support array_contains_any)
  - Added proper array indices (LabelList) for efficient queries
- Created comprehensive test suite (13 new tests) covering all functionality
- Fixed bugs discovered during testing (array_contains_any → array_contains)
- All 212 tests passing across the workspace

**Files Modified:**
- `crates/breeze-indexer/src/search.rs` - Main search module with granularity support
- `crates/breeze-indexer/src/search/documents.rs` - Document search logic
- `crates/breeze-indexer/src/search/chunks.rs` - Chunk search with semantic filters
- `crates/breeze-indexer/src/search/chunk_tests.rs` - Comprehensive test coverage
- `crates/breeze-indexer/src/models/code_chunk.rs` - Added LabelList indices

**Implementation Notes:**
- The "Both" granularity mode was not implemented as it adds complexity without clear benefits
- Pattern matching is currently exact match, not regex/glob (can be enhanced later)
- All semantic filters are optional and can be combined

### Phase 5: API & Integration ✅ COMPLETED

1. ✅ Updated search API endpoints
   - ✅ Modified indexer.search() to accept SearchOptions instead of just limit
   - ✅ Added all SearchOptions fields to SearchRequest type
   - ✅ Support for granularity selection (Document/Chunk modes)
   - ✅ Exposed all semantic filters in API

2. ✅ Updated MCP tools
   - ✅ Modified search_code tool to accept all SearchOptions parameters
   - ✅ Exposed filter parameters in MCP interface
   - ✅ Returns structured results with chunks

3. ✅ Updated HTTP API
   - ✅ POST /api/v1/search endpoint accepts all advanced search parameters
   - ✅ OpenAPI documentation automatically generated via JsonSchema derives
   - ✅ API docs available at /api/openapi.json and /api/docs (Scalar UI)

**Key Implementation Details:**

- Broke backwards compatibility as per project rules - search() now requires SearchOptions
- All API types use JsonSchema derive for automatic OpenAPI documentation
- Both HTTP and MCP interfaces support identical search capabilities
- All 212 tests passing across the workspace

**Files Modified:**
- `crates/breeze-indexer/src/indexer.rs` - Changed search signature to accept SearchOptions
- `crates/breeze-server/src/types.rs` - Added all SearchOptions fields to SearchRequest
- `crates/breeze-server/src/routes.rs` - Updated search handler to build SearchOptions
- `crates/breeze-server/src/mcp.rs` - Updated MCP search_code to use SearchOptions
- `crates/breeze/src/app.rs` - Updated CLI to use SearchOptions with defaults

## Example API Usage

### HTTP API Examples

#### Basic Document Search (Default)
```bash
curl -X POST http://localhost:3000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "authentication middleware",
    "limit": 10,
    "languages": ["rust"]
  }'
```

#### Chunk-Level Search with Semantic Filters
```bash
curl -X POST http://localhost:3000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "error handling",
    "granularity": "chunk",
    "node_types": ["function", "method"],
    "has_references": ["logger", "log"],
    "parent_context_pattern": "impl ErrorHandler"
  }'
```

#### Find Deeply Nested Code
```bash
curl -X POST http://localhost:3000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "data processing pipeline",
    "granularity": "chunk",
    "scope_depth": [3, 10],
    "node_types": ["function"],
    "languages": ["python"]
  }'
```

## Example Rust Code Usage

### Basic Document Search (Default)

```rust
let options = SearchOptions {
    languages: Some(vec!["rust".to_string()]),
    file_limit: 10,
    chunks_per_file: 3,
    ..Default::default()
};

let results = hybrid_search(
    doc_table,
    chunk_table,
    embedding_provider,
    "authentication middleware",
    options,
    Some(project_id),
).await?;
```

### Semantic Chunk Search

```rust
let options = SearchOptions {
    granularity: SearchGranularity::Chunk,
    node_types: Some(vec!["function".to_string(), "method".to_string()]),
    has_references: Some(vec!["logger".to_string(), "log".to_string()]),
    parent_context_pattern: Some("impl ErrorHandler".to_string()),
    ..Default::default()
};

let results = hybrid_search(
    doc_table,
    chunk_table,
    embedding_provider,
    "error handling",
    options,
    None,
).await?;
```

### Find Deeply Nested Code

```rust
let options = SearchOptions {
    granularity: SearchGranularity::Chunk,
    scope_depth: Some((3, 10)), // Min 3, max 10 levels deep
    node_types: Some(vec!["function".to_string()]),
    languages: Some(vec!["python".to_string()]),
    ..Default::default()
};

let results = hybrid_search(
    doc_table,
    chunk_table,
    embedding_provider,
    "data processing pipeline",
    options,
    None,
).await?;
```

### Find Code with Specific Definitions

```rust
let options = SearchOptions {
    granularity: SearchGranularity::Chunk,
    has_definitions: Some(vec!["Connection".to_string(), "Database".to_string()]),
    node_types: Some(vec!["struct".to_string(), "class".to_string()]),
    ..Default::default()
};

let results = hybrid_search(
    doc_table,
    chunk_table,
    embedding_provider,
    "database connection",
    options,
    None,
).await?;
```

## Implementation Summary

All 5 phases of the advanced search implementation are now complete:

1. ✅ **Phase 1**: Data Model & Storage - Created CodeChunk model and updated CodeDocument
2. ✅ **Phase 2**: Chunk Storage Infrastructure - Implemented ChunkSink and integrated into pipeline
3. ✅ **Phase 3**: Basic Search Enhancement - Added language filtering and chunk results
4. ✅ **Phase 4**: Advanced Search - Implemented chunk-level search with semantic filters
5. ✅ **Phase 5**: API & Integration - Exposed all features through HTTP and MCP APIs

## Benefits

1. **Flexibility**: Search at file or chunk level based on needs
2. **Precision**: Find specific code patterns with semantic filters
3. **Context**: See code snippets in search results
4. **Performance**: Indexed metadata enables fast filtering
5. **Extensibility**: Rich metadata supports future features

## Technical Details

### Code Locations

- **Models**: `crates/breeze-indexer/src/models/`
  - `code_document.rs` - Enhanced with language fields
  - `code_chunk.rs` - New model for chunk storage

- **Pipeline**: `crates/breeze-indexer/src/`
  - `document_builder.rs` - Builds documents and chunks
  - `bulk_indexer.rs` - Needs chunk storage implementation
  - `converter.rs` - Handles data conversion

- **Search**: `crates/breeze-indexer/src/search.rs`
  - Needs enhancement for chunk search
  - Add filter support
  - Implement granularity options

### Performance Considerations

1. **Chunk Storage**: Balance between storage size and query performance
2. **Index Strategy**: Optimize indices for common query patterns
3. **Embedding Storage**: Consider compression for chunk embeddings
4. **Query Optimization**: Use LanceDB's native filtering before vector search

## Future Enhancements

1. **Symbol Search**: Direct search for function/class definitions
2. **Dependency Analysis**: Find all code that uses specific libraries
