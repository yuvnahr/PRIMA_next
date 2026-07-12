# Campaign III — Identity Normalization Ablation

## Methodology

Nomic and raw representation were fixed from the preceding subset gates. Only identity normalization changed. Both rows used the same 197-query subset and production pipeline.

## Results

| Identity normalization | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@5 | Candidate success |
|---|---:|---:|---:|---:|---:|---:|
| Off | 0.207276 | 0.415821 | 0.513536 | 0.329488 | 0.324730 | 0.746193 |
| On | 0.210660 | 0.403130 | 0.503384 | 0.323311 | 0.317139 | 0.715736 |

## Observation

Identity normalization slightly improved Recall@1 (`+0.003384`) but reduced Recall@5, Recall@10, MRR, nDCG@5, and candidate-generation success on this subset. The measured impact is mixed, not a uniformly positive improvement.

## Limitations

The entity-resolution metric requires a larger labeled identity subset for a reliable estimate; this run reports no inferred positive entity-resolution result. Full-dataset validation is still required.
