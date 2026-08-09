# Dense Retrieval Root Cause Report

## Was The Memory Ever Created?

Memory creation failures: `0`. Stored-memory failures: `0`. The benchmark ingestion created one episodic `MemoryNote` per LoCoMo turn and embedded the stored `speaker: text` content.

## What Was Embedded?

The embedded representation is the single-turn `MemoryNote.content`. No summary is inserted in this benchmark path, so representation compression is mostly absent; `memory_representation_analysis.json` records the raw turn, stored note text, embedded text, preservation fields, and information-loss tokens.

## Did The Correct Memory Enter Dense Top30?

Candidate generation success rate: `0.046053`. Miss rate: `0.953947`. Average expected-memory rank: `213.276316`. Average similarity margin: `0.374287`.

## Dense Failure Taxonomy

| Category | Count | Share |
|---|---:|---:|
| memory_not_created | 0 | 0.000000 |
| memory_filtered | 0 | 0.000000 |
| representation_loss | 0 | 0.000000 |
| embedding_mismatch | 57 | 0.393103 |
| chunking_failure | 88 | 0.606897 |
| candidate_generation_failure | 0 | 0.000000 |
| ranking_failure | 0 | 0.000000 |
| storage_failure | 0 | 0.000000 |
| unknown | 0 | 0.000000 |

## Chunking

Average supporting turns: `1.551020`. Split-memory frequency: `0.377551`. Missing-context frequency: `0.377551`.

## Conclusion

The dominant dense failure category is `chunking_failure`. This phase does not change retrieval behavior; it identifies whether the bottleneck is creation, representation, embedding mismatch, chunking, candidate generation, or ranking.

## Highest Expected-Impact Repair

The highest-impact change would be changing memory span or adding evidence-preserving multi-turn representation, because questions require context split across one-turn memories.
