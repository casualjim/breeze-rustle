# Work item 02: Chunk sink – add then prune stale, gapless

Change
- Chunk sink handles ReplaceFileChunks by:
  1) Building one RecordBatch from the provided chunks.
  2) Upserting the batch with merge_insert on id.
  3) Immediately deleting stale rows for the file:
     delete where project_id = '<pid>' AND file_path = '<path>' AND id NOT IN (<new chunk ids>)

Notes
- We upsert first to avoid any search blackout; the delete removes old/stale chunks milliseconds later.
- No time/size-based buffering for this path; it is invoked per file.

Acceptance criteria
- After a ReplaceFileChunks, querying code_chunks for (project_id, file_path) returns exactly the ids provided.
- No window exists where the set is empty if we poll immediately after ReplaceFileChunks (write-before-delete).
- Unit test: index N chunks, then reindex with N-1 chunks; assert the removed id is gone and remaining ids match exactly.
- Logging shows both phases for the file: upserted M rows, pruned K stale rows.

