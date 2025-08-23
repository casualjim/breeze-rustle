# Work item 04: Stop computing/writing document embeddings

Change
- In document_builder.rs build_document_from_accumulator:
  - Remove weighted-average embedding calculation and doc.update_embedding call.
  - Set doc.content_embedding = vec![0.0; embedding_dim] to satisfy schema.
- Do not read document embeddings anywhere in search.

Acceptance criteria
- No code path computes an aggregated document embedding.
- CodeDocument.content_embedding is a zero vector of the configured dimension for newly written docs.
- Search compiles and runs without referencing CodeDocument.content_embedding.
- Tests that previously assumed document embeddings are updated to reflect chunk-only search.

