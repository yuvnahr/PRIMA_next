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
