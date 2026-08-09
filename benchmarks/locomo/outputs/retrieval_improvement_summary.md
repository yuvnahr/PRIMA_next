# Retrieval V2 Scientific Validation

Dataset: `benchmarks\locomo\outputs\processed\locomo_debug_100.json`
Dataset available: `True`
Queries evaluated: 98

## Phase 4.2 Baseline vs Retrieval V2

| Metric | Phase 4.2 baseline | Retrieval V2 | Gain |
|---|---:|---:|---:|
| recall_at_1 | 0.279352 | 0.170918 | -0.108434 |
| recall_at_5 | 0.544352 | 0.333333 | -0.211019 |
| mrr | 0.393704 | 0.261565 | -0.132139 |
| ndcg_at_5 | 0.424999 | 0.267301 | -0.157698 |

## Ablation Summary

| Configuration | Recall@1 | Recall@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|
| dense | 0.010204 | 0.010204 | 0.010204 | 0.010204 |
| dense_query_analysis | 0.010204 | 0.010204 | 0.010204 | 0.010204 |
| dense_expansion | 0.010204 | 0.010204 | 0.010204 | 0.010204 |
| dense_sparse | 0.110544 | 0.244048 | 0.176190 | 0.183209 |
| dense_sparse_reranker | 0.170918 | 0.333333 | 0.261565 | 0.267301 |
| full_retrieval_v2 | 0.170918 | 0.333333 | 0.261565 | 0.267301 |
