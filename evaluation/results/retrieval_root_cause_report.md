# Retrieval Root Cause Report

## Where Correct Memories Are First Lost

The dominant localized stage is `dense_failure`. Stage counts are:

| Stage | Count | Share |
|---|---:|---:|
| dense_failure | 77 | 0.785714 |
| sparse_failure | 3 | 0.030612 |
| fusion_failure | 6 | 0.061224 |
| reranker_failure | 0 | 0.000000 |
| selection_failure | 12 | 0.122449 |
| context_truncation | 0 | 0.000000 |
| generation_failure | 0 | 0.000000 |

## Why They Are Lost

The localization artifact records per-expected-memory rank histories across dense, sparse, fusion, reranker, and final selection. A `selection_failure` means the correct memory survives into reranker top30 but falls below final top5. A `fusion_failure` means candidate generation found it but fusion removed it from the tracked top30. A `dense_failure` here means neither initial dense nor sparse top30 contained the expected memory.

## Component Most Responsible

On this run, `dense_failure` contributes the largest share of localized failures. See `retrieval_failure_localization.json` for the query-level evidence.

## Confidence Calibration

Confidence reliability is `False`. Pearson is `-0.106271`, Spearman is `-0.118451`, ECE is `0.140957`, and Brier score is `0.264104`. When this remains weak, confidence should not be used as a success proxy.

## Candidate Drift

Stored Phase 4.2 results in evaluation/results/retrieval_optimization_results.json contain aggregate metrics only.
Exact Phase 4.2 candidate drift cannot be proven from the stored aggregate baseline alone. The current report records Retrieval V2 stage presence and explicitly marks Phase 4.2 candidate traces as unavailable.

## Context Analysis

Average context tokens: `205.989796`. Average unused context tokens: `1394.010204`. Average tokens per retrieved memory: `99.095918`. Duplicate final memory count: `0`.

## Fixes Measurably Improved Retrieval

No new algorithmic fix is introduced by this forensic pass. The instruction prohibits blind changes, so this run only localizes failure and measures calibration/context behavior.

## Rejected Hypotheses

Context truncation is rejected for this retrieval-only run unless `context_analysis.json` shows exhausted token budgets. Phase 4.2 candidate drift is unproven until a baseline candidate trace exists.

## Future Work

Regenerate Phase 4.2 with the same per-stage trace schema, then apply one small fix to the dominant failure stage and rerun the 100-query benchmark before considering full LoCoMo validation.
