# Phase 8 Dense Investigation Summary

Dataset: `benchmarks\locomo\outputs\processed\locomo_debug_100.json`
Queries evaluated: `98`

| Metric | Phase 4.2 | Retrieval V2 | Delta |
|---|---:|---:|---:|
| recall_at_1 | 0.279352 | 0.170918 | -0.108434 |
| recall_at_5 | 0.544352 | 0.333333 | -0.211019 |
| mrr | 0.393704 | 0.261565 | -0.132139 |
| ndcg_at_5 | 0.424999 | 0.267301 | -0.157698 |

## Dense Findings

- Candidate generation success rate: `0.046053`
- Candidate generation miss rate: `0.953947`
- Dominant failure category: `chunking_failure`
- Memory creation failures: `0`
- Stored memory failures: `0`
- Split-memory frequency: `0.377551`
- Average similarity margin: `0.374287`

## Evidence-Backed Next Step

The highest-impact change would be changing memory span or adding evidence-preserving multi-turn representation, because questions require context split across one-turn memories.
