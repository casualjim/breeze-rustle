# Work item 07: Test plan

Unit tests
- ReplaceFileChunks round-trip: file with N chunks → one message carrying N chunks.
- Sink add-then-prune: after ReplaceFileChunks, exact-id set present; reindex N→N-1 prunes stale id.
- Delete handler for chunks: delete event removes all chunks for (project_id, file_path).
- Document embedding disabled: new docs have zero vector embeddings; no code reads doc embeddings.

Integration tests
- End-to-end indexing: files → ReplaceFileChunks → sink → search returns chunk results.
- Reindex same file; assert no blackout (immediately after ReplaceFileChunks, results still present), and no stale chunks after prune.
- File deletion end-to-end removes both doc and chunk data.

Non-functional
- Clippy clean, rustfmt, cargo-audit clean.
- No std::sync::RwLock in async paths introduced by this refactor.

