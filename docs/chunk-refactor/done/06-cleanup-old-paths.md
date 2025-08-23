# Work item 06: Remove old streaming/buffering paths for chunks

Change
- Remove per-chunk streaming channel and handlers for CodeChunk.
- Remove usage of BufferedRecordBatchConverter for chunk writes.
- Keep LanceDbSink for documents (temporarily) if still used; chunk writes use the new per-file path only.

Acceptance criteria
- No references remain to the old chunk streaming sink or converter for CodeChunk.
- The only way to write chunks is ReplaceFileChunks → sink handler.
- Repo compiles cleanly; clippy/rustfmt ok; cargo-audit clean.

