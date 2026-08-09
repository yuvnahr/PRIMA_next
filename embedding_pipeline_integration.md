# Phase 11 — Production Embedding Integration

All production memory and query embeddings route through `CanonicalEmbeddingPipeline.embed_memory()` or `.embed_query()`.

The pipeline applies deterministic identity normalization, serializes the structured semantic representation, and sends only that representation to the configured backend. Original user content remains in `MemoryNote.content`; the serialized representation and embedding metadata are stored separately.

Stored metadata includes backend, model, dimension, representation version, identity version, and a SHA-256 backend fingerprint. Chroma repository startup rejects mismatches by default; `PRIMA_EMBEDDING_MISMATCH=readonly` permits inspection and `rebuild` marks the repository for rebuilding. `python -m memory.embedding_rebuild` supports dry-run, force, worker, batch-size, and progress options.

Retrieval scoring, ranking, fusion, reranking, query expansion, thresholds, metrics, and benchmark code were not changed.
