# Retrieval Validation Artifacts

## Reproduce

```bash
python -m compileall memory/retrieval benchmarks/locomo runtime
python -m benchmarks.locomo.retrieval_validation --dataset-path benchmarks/locomo/outputs/processed/locomo_debug_100.json --output-dir benchmarks/locomo/outputs --top-k 5
python -m benchmarks.locomo.retrieval_validation --dataset-path benchmarks/locomo/external/data/locomo10.json --output-dir benchmarks/locomo/outputs/debug_full --top-k 5
```

## Artifacts

- Output directory: `benchmarks\locomo\outputs`
- `retrieval_trace.json`: per-query dense, sparse, fused, reranked, and final candidates.
- `retrieval_ablation_v2.json` and `retrieval_ablation.csv`: Recall@1/5, MRR, nDCG@5, latency, and retrieved count by config.
- `retrieval_category.csv`: per-category retrieval metrics.
- `retrieval_failure_breakdown.json`: failure classes with representative examples.
- `retrieval_confidence_validation.json` and `confidence_curve.csv`: confidence calibration against Recall@5 and MRR.
- `retrieval_improvement_summary.md`: compact baseline comparison and ablation summary.

## Current Run

- Dataset: `benchmarks\locomo\outputs\processed\locomo_debug_100.json`
- Queries evaluated: `98`
- Dataset available: `True`
- Phase 4.2 Recall@5: `0.544352`
- Retrieval V2 Recall@5: `0.333333`
- Recall@5 gain: `-0.211019`
- Confidence vs Recall@5 correlation: `-0.151454`

This run is diagnostic rather than merge-ready when the baseline gains are negative. The trace should be used to drive the next repair before claiming Phase 7.2 improvement.

## Failure Focus

- `entity_resolution_failure`: 34
- `temporal_failure`: 16
- `relation_failure`: 10
- `retrieval_miss`: 0
- `preference_failure`: 0

Use `retrieval_trace.json` to inspect where each expected memory drops out. The most useful stage keys are `dense_top30`, `sparse_top30`, `fused_top30`, and `reranked_top30` under each trace record's `diagnostics` field.
