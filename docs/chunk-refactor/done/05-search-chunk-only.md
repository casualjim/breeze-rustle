# Work item 05: Search is chunk-only

Change
- Remove document-mode search code and switch to chunk search for all cases.
  - Delete search/documents.rs.
  - Update search.rs to route only to the chunk path.
- Enrich results from existing document/file metadata as needed (no embeddings required).

Acceptance criteria
- search/documents.rs is removed; no references to CodeDocument::from_record_batch remain in search.
- All SearchOptions and response types remain compatible; server/CLI unchanged in wire format.
- All search tests pass using chunk-only retrieval and ranking.

