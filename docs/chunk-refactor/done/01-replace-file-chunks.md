# Work item 01: ReplaceFileChunks – single write path

Status: complete
Owner: Chunk Refactor
Last updated: 2025-08-22

Summary
- Replace per-chunk streaming with a single per-file write op.
- Document builder emits exactly one message per file: ReplaceFileChunks { project_id, file_path, chunks }.
- No fallback paths and no dual write modes remain.
- Sink semantics (add-then-prune) are handled in Work item 02; this document focuses on the builder and wiring.

Motivation
- Single, atomic write per file eliminates ordering and backpressure risks from per-chunk streaming.
- Simplifies observability (one info log per file), easier to test deterministically.
- Establishes a clean, single path that WI-02 can apply atomically in storage.

Change
- Define new write op: ReplaceFileChunks {
  project_id: Uuid,
  file_path: String,
  chunks: Vec<CodeChunk>,
}.
- Document builder (build_document_from_accumulator):
  - Accumulate all CodeChunk for the file (per-chunk embeddings are already attached upstream).
  - Do NOT send per-chunk messages at any point.
  - When the file is complete (EOF reached and accumulator verified), send exactly one ReplaceFileChunks.
- Remove/replace any channel types carrying individual CodeChunk sends with a channel that carries ReplaceFileChunks.

Scope of this item (WI-01)
- Builder and pipeline message shape only.
- The sink behavior (upsert new set, prune stale) is explicitly out of scope here and covered by WI-02.

Data model & channels
- Message type: ReplaceFileChunks { project_id, file_path, chunks: Vec<CodeChunk> }.
- Channel type: mpsc::Sender<ReplaceFileChunks> / mpsc::Receiver<ReplaceFileChunks> replaces any per-chunk channels.
- No std::sync::RwLock on async paths.

Algorithm in builder
1) Maintain per-file accumulation as today until EOF.
2) On EOF with all expected chunks received, map the accumulated embedded chunks into CodeChunk values.
3) Emit a single ReplaceFileChunks with the full Vec<CodeChunk> for that file.
4) Log once at info level with fields: file_path and chunk_count.
5) Never send per-chunk messages; delete that code path.

Logging
- info: "replace_file_chunks" file="<path>" chunk_count=<N>
- No per-chunk info logs remain (keep any deep debug logs as debug only).

Acceptance criteria
- Builder sends exactly one ReplaceFileChunks per file; no per-chunk sends remain anywhere in the codebase.
- The type of the chunk channel is changed to carry ReplaceFileChunks.
- Logging shows a single per-file op (file path, count of chunks) when a file completes.
- Unit test covers: a file with N chunks yields one ReplaceFileChunks message carrying those N chunks (and a zero-chunk case produces a ReplaceFileChunks with empty chunks).
- No legacy/fallback write path persists; code that streamed chunks is removed.

Test plan (builder-facing)
- document_builder_emits_single_replace_with_n_chunks
  - Arrange a file with N chunks; Assert a single ReplaceFileChunks with chunks.len() == N.
- document_builder_emits_single_replace_with_zero_chunks
  - Arrange a file producing 0 chunks; Assert a single ReplaceFileChunks with chunks.len() == 0.
- compile guard
  - Ensure there are no remaining references to per-chunk send/receive in the builder and indexer wiring.

Out of scope (deferred to WI-02 and others)
- Applying ReplaceFileChunks to storage (add then prune stale) — WI-02.
- Removal of document embeddings — WI-04.
- Removal of document-mode search — WI-05.

Notes
- Keep a single way to write chunks; do not add compatibility shims.
- Ensure async correctness (no std::sync::RwLock on async hot paths).
