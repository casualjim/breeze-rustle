# Work Avoidance Implementation Plan

## Overview

Implement a content hash-based work avoidance system that filters out unchanged files early in the indexing pipeline to improve performance and reduce unnecessary work.

## Key Changes

### 1. Add Database Query for Existing Hashes

- Add method to `BulkIndexer` to query existing file hashes from LanceDB
- Query only `file_path` and `content_hash` columns for efficiency
- Build a `BTreeMap<PathBuf, [u8; 32]>` for fast lookups using proper path types

### 2. Modify WalkOptions Structure

```rust
pub struct WalkOptions {
    pub max_chunk_size: usize,
    pub tokenizer: Tokenizer,
    pub max_parallel: usize,
    pub max_file_size: Option<u64>,
    pub large_file_threads: usize,
    pub existing_hashes: BTreeMap<PathBuf, [u8; 32]>, // Not optional - empty map if no existing files
}
```

### 3. Update Walker Functions

- Modify `collect_files_with_sizes` to also compute hashes
- Add parallel hash computation using blake3's memory-mapped API for large files
- Filter out unchanged files before returning the file list
- Return enriched file entries: `Vec<(PathBuf, u64, [u8; 32])>`
- Compare using `PathBuf` for proper path normalization

### 4. Enhance Process File Function

- Remove duplicate hash computation since it's now done upfront
- Use the pre-computed hash from the walker
- Pass hash through to EOF chunk creation

### 5. Update EOF Chunk Handling

- Ensure EOF chunks always contain content and hash (already done)
- Verify document builder uses EOF chunk data (already done)

### 6. Add Progress Reporting

- Report "X files unchanged, Y files to process" during indexing
- Add metrics for work avoided

## Implementation Steps

1. **Create plan documentation** at `docs/plans/work-avoidance.md`
2. **Add database query method** in `bulk_indexer.rs`:

   ```rust
   async fn get_existing_file_hashes(&self) -> Result<BTreeMap<PathBuf, [u8; 32]>, IndexerError>
   ```

3. **Update WalkOptions** in `walker.rs` with non-optional `existing_hashes` field
4. **Modify file collection** to compute hashes and filter
5. **Update process_file** to use pre-computed hashes
6. **Add tests** for work avoidance scenarios
7. **Update integration points** in `index` and `index_files` methods:
   - Always query existing hashes (empty map if table is empty)
   - Pass to walk functions

## Benefits

- Files are read only once (for hashing)
- Unchanged files never enter the chunking/embedding pipeline
- Better progress tracking and user feedback
- Significant performance improvement for incremental indexing
- Proper path handling with `PathBuf` ensures correct comparisons

## Testing Strategy

- Unit tests for hash comparison logic
- Integration tests with pre-indexed files
- Performance benchmarks comparing with/without work avoidance
- Path normalization tests (relative vs absolute paths)

## Todo List

- [ ] Add `get_existing_file_hashes` method to BulkIndexer
- [ ] Update WalkOptions to include `existing_hashes: BTreeMap<PathBuf, [u8; 32]>`
- [ ] Modify `collect_files_with_sizes` to return `Vec<(PathBuf, u64, [u8; 32])>`
- [ ] Add hash computation during file collection phase
- [ ] Implement file filtering based on hash comparison
- [ ] Update `process_file` to accept pre-computed hash
- [ ] Remove duplicate hash computation from `process_file`
- [ ] Update `walk_project` and `walk_files` to use new signatures
- [ ] Add progress reporting for skipped files
- [ ] Write unit tests for hash comparison
- [ ] Write integration tests for work avoidance
- [ ] Update documentation
