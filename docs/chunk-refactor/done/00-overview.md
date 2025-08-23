# Chunk-first refactor plan (single path)

Goals
- One way to write chunks: per-file ReplaceFileChunks (add new set, then prune stale).
- Stop computing/writing document embeddings; search is chunk-only.
- No blackouts during reindex; no dual paths left after cleanup.
- Maintain quality gates: clippy clean, rustfmt, tests, cargo-audit; no new warnings.
- Async safety: no std::sync::RwLock in async paths.

Scope (what changes now)
- Document builder: emit exactly one ReplaceFileChunks per completed file; no per-chunk streaming.
- Chunk sink: handle ReplaceFileChunks by upserting the batch then deleting stale rows for that file.
- Delete handler: also remove code_chunks rows on file deletions.
- Search: remove document-mode; always query code_chunks (existing chunk search); remove doc search code.
- Documents: keep CodeDocument rows for now but set content_embedding to zeros.
- Cleanup: delete old streaming/buffering code/path for chunks.

Non-goals (deferred)
- Introducing FileMeta type (can come later once CodeDocument is removed entirely).
- Stable chunk IDs or any versioning/history.

Success criteria (global)
- Reindexing the same file leaves exactly and only the newest chunk rows for (project_id, file_path).
- No period exists where a reindexed file disappears from search results.
- Document embeddings are not computed nor read anywhere; search correctness is maintained.
- There is no remaining code path that streams per-chunk writes to the chunk sink.
- CI green: tests, clippy, rustfmt, cargo-audit.

