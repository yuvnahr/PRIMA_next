# Retrieval V2 Debug Artifacts

## Reproduce

```bash
git checkout retrieval-v2-debug
python -m compileall memory/retrieval benchmarks/locomo runtime
python -m benchmarks.locomo.retrieval_validation --dataset-path benchmarks/locomo/outputs/processed/locomo_debug_100.json --output-dir benchmarks/locomo/outputs --top-k 5
python -m benchmarks.locomo.retrieval_validation --dataset-path benchmarks/locomo/external/data/locomo10.json --output-dir benchmarks/locomo/outputs/debug_full --top-k 5
```

## Artifacts

- Output directory: `evaluation\results`
- `retrieval_failure_localization.json`: failed-query stage forensics with rank history.
- `retrieval_stage_statistics.json`: aggregate stage-loss counts and percentages.
- `retrieval_ablation_v2.json` and `retrieval_ablation.csv`: Recall@1/5, MRR, nDCG@5, latency, and retrieved count by config.
- `retrieval_ablation_stage.csv`: Recall@1/5, MRR, and nDCG@5 by ablation and stage.
- `retrieval_category.csv`: per-category retrieval metrics.
- `retrieval_confidence_calibration.json`: confidence bins, Pearson, Spearman, ECE, and Brier score.
- `candidate_drift.json`: Retrieval V2 stage presence plus Phase 4.2 trace availability notes.
- `context_analysis.json`: token and duplicate-context accounting.
- `retrieval_phase7_vs_phase42.md` and `retrieval_root_cause_report.md`: scientific summaries.
- `retrieval_trace.json`: optional full raw trace only when `PRIMA_WRITE_FULL_RETRIEVAL_TRACE=1`.

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

Use `retrieval_failure_localization.json` to inspect where each expected memory drops out. Full raw traces are opt-in because they can become too large for version control.
