# Retrieval Phase 7 vs Phase 4.2

| Metric | Phase 4.2 | Retrieval V2 | Delta |
|---|---:|---:|---:|
| recall_at_1 | 0.279352 | 0.170918 | -0.108434 |
| recall_at_5 | 0.544352 | 0.333333 | -0.211019 |
| mrr | 0.393704 | 0.261565 | -0.132139 |
| ndcg_at_5 | 0.424999 | 0.267301 | -0.157698 |

## Explanation

Retrieval V2 remains below the stored Phase 4.2 aggregate baseline on this run. The largest localized failure bucket is `dense_failure`.
Confidence is not reliable when `confidence_reliable` is `False` with Pearson `-0.106271`, Spearman `-0.118451`, ECE `0.140957`, and Brier `0.264104`.

No additional retrieval fix is claimed here unless the metric deltas above improve. The artifact is intended to explain the regression and guide the next small repair.
