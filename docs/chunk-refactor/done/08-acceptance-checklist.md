# Work item 08: Acceptance checklist (global)

- One path for chunk writes (ReplaceFileChunks). No legacy per-chunk streaming remains.
- Reindex of a file updates chunks without blackout and without stale rows.
- Document embeddings are not computed and are zero-filled for new docs; search does not use them.
- Search is chunk-only; document search code removed.
- Delete handler removes chunks as well as documents on file delete.
- All tests updated and passing; clippy, rustfmt, cargo-audit clean; no new warnings.
- No std::sync::RwLock usage in async code paths was introduced.

