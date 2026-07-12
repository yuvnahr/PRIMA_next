# Campaign II — Representation Ablation

## Methodology

The best learned backend from Campaign I, Nomic, was fixed. Raw, semantic, and event modes were evaluated on the same one-conversation, 197-query subset. Identity normalization remained enabled for every row; only representation mode changed.

## Results

| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@5 | Candidate success |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 0.210660 | 0.403130 | 0.503384 | 0.323311 | 0.317139 | 0.715736 |
| Semantic | 0.145939 | 0.329103 | 0.419205 | 0.254356 | 0.246090 | 0.695431 |
| Event | 0.158629 | 0.355753 | 0.443739 | 0.268545 | 0.266548 | 0.654822 |

## Observation

Raw representation was highest on every listed retrieval metric in this subset. Structured semantic representation did not improve retrieval here; event representation also did not provide a measured benefit.

## Limitations

This is one conversation and is preliminary validation, not a full-dataset conclusion. Event mode is included only because it executed through the production pipeline.
