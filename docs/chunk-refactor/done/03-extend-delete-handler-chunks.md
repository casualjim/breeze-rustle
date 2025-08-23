# Work item 03: Extend delete handler to remove chunks

Change
- In delete_handler_task (bulk_indexer.rs), when deleting a document row for (project_id, file_path), also delete from code_chunks with the same predicate.

Acceptance criteria
- Deleting a file removes both document row and all associated chunk rows.
- Unit/integration test: simulate a delete event; verify chunk table row count for that (project_id, file_path) becomes zero.

