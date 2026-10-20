# LoCoMo Benchmark Integration

This folder contains PRIMA's architecture-only integration layer for the LoCoMo benchmark.
The official LoCoMo repository is kept isolated under `external/` as a git submodule.

## Folder Purpose

- `config.py`: benchmark-local configuration.
- `loader.py`: locates and loads raw LoCoMo JSON.
- `adapter.py`: converts raw LoCoMo records into shared `Conversation` objects.
- `runner.py`: reuses the benchmark-agnostic runner contract.
- `evaluate.py`: computes metrics from runner outputs only.
- `outputs/`: benchmark-local raw, processed, metric, plot, and log artifacts.

## Execution Flow

```text
LoCoMo JSON
-> loader.py
-> adapter.py
-> Conversation
-> generic agent runtime
-> runner.py outputs
-> evaluate.py metrics
```

The LoCoMo integration does not import or call PRIMA memory internals. It only prepares normalized
conversation objects and consumes generic runner outputs.

## Expected Outputs

Benchmark outputs should be written under `outputs/`:

- `raw/`: raw runtime outputs.
- `processed/`: normalized intermediate artifacts.
- `metrics/`: evaluation summaries.
- `plots/`: benchmark-specific figures.
- `logs/`: benchmark logs.

## Evidence and scoring contract

Canonical experiment question/metric schema 2.0 separates all-hop candidate,
integrator-accepted, actually serialized prompt, and model-cited evidence. Runtime
`output_data.evidence_audit` version 1.0 is available without full diagnostic tracing;
the legacy `retrieval` view explicitly covers only the last retrieval call. Missing
old-record observations remain unavailable rather than zero recall.

The evaluator reports legacy normalized EM/SQuAD-style F1 and smoothed sentence BLEU
alongside the pinned official LoCoMo core score and scorer identity. Structured
category-5 abstention and the official raw-text refusal score are separate metrics.
Correctness and annotation-based evidence flags are independent; recall denominators
separate answerable/adversarial and unannotated populations.

The adapted official scorer retains snap-research/LoCoMo attribution and its
[CC BY-NC 4.0 license](official_scoring_LICENSE.txt). Core scoring requires no
BERTScore import or model download. See the
[implementation handoff](../../docs/locomo-review-remediation.md) for exact versions,
compatibility aliases, fixtures, resume rules and validation limits.
