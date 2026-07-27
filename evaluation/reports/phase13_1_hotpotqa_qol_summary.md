# Phase 13.1 — HotpotQA benchmark UX and local evaluation QoL

## Files changed

Added `benchmarks/hotpotqa/data_sources.py`, `benchmarks/hotpotqa/console.py`, and `tests/benchmarks/test_hotpotqa_qol.py`. Updated `config.py`, `convert_to_json.py`, `checkpoint.py`, `evaluate.py`, `experiment.py`, and `README.md`. Production runtime, reasoning, retrieval, reflection, prompting, answer extraction, and supporting-fact projection were not changed.

## Design and behavior

The immutable `HotpotQADataSource` registry contains exactly `distractor` and `fullwiki`, with their human labels, Hugging Face validation Parquet URLs, official local filenames, normal modes, and context-source labels. Train sources are absent.

`convert_validation_set()` lazily imports `datasets` and `pyarrow`, preserves the existing Hotpot fields, validates through `HotpotQADataset`, and atomically replaces the output. Valid prepared JSON is reused; `--force` or experiment `--refresh-data` regenerates it. Missing optional dependencies produce the exact install command, and PyArrow 19.0.0 remains rejected.

The experiment accepts either `--dataset-set` or the backward-compatible `--dataset-path`. A selected set derives its normal mode and prepares missing data. Oracle remains explicit. Incompatible normal mode/set pairs are rejected unless an explicit custom path is intentionally supplied.

Random selection is the default. A new run without `--seed` uses `secrets.randbits(63)` and a local `random.Random(resolved_seed)`; global random state is untouched. Fixed seeds reproduce selection. Sequential mode retains source order. Both apply offset before limiting. The manifest records configured and resolved seeds, strategy, offset, requested limit, and exact selected IDs. Resume reuses the stored resolved seed when omitted and validates those selection fields.

`HotpotQATerminalReporter` uses only `shutil`, `textwrap`, and `collections`. It prints a width-aware plain-text header, one block per newly completed question when `--progress` is enabled, runtime failures, aggregate official metrics, reasoning counters, run identity, and artifact paths. It emits no colour escapes, so PowerShell, redirected output, and `NO_COLOR` terminals require no special fallback. `--quiet` suppresses human output; `--json-summary` preserves machine-readable output.

`score_hotpot_record()` is the sole per-record metric path used by both the reporter and `HotpotQAEvaluator`; saved metrics retain 0–1 values.

## Example terminal output

```text
============================================================
HotpotQA Result 1/5
============================================================
Sample ID       : 5ac...
Type            : comparison
Level           : hard

Question
Sam Mendes and Ruby Yang share what profession?

Prediction
filmmaker

Scores
Answer EM       : 100.00%
Answer F1       : 100.00%
Supporting EM   : 0.00%
Supporting F1   : 66.67%
Joint EM        : 0.00%
Joint F1        : 66.67%

Execution
Runtime status  : completed
Stop reason     : sufficient
Reasoning hops  : 1
Retrieval calls : 1
Reflections     : 0
Evidence items  : 10
Latency         : 19.67 seconds
============================================================
```

## Validation

Executed:

```powershell
python -m py_compile benchmarks\hotpotqa\data_sources.py benchmarks\hotpotqa\convert_to_json.py benchmarks\hotpotqa\console.py benchmarks\hotpotqa\evaluate.py benchmarks\hotpotqa\experiment.py benchmarks\hotpotqa\checkpoint.py benchmarks\hotpotqa\config.py
python -m pytest -q tests\benchmarks\test_hotpotqa_qol.py tests\benchmarks\test_hotpotqa.py
```

Mocked validation covers both source choices, train exclusion, output paths, reuse/force/dependency behavior, menu behavior, dataset/mode resolution, random and sequential sampling, offset/limit behavior, generated seed persistence, exact resume selection, success/partial/failure terminal reports, wrapping, plain-text output, quiet output, JSON summary, and metric consistency. Result: `17 passed, 1 skipped` with one existing Pydantic deprecation warning. A prepared distractor sample was also exercised through the fake runtime without network or Ollama (`completed=1`, `scored=1`).

## Commands

```powershell
py -m benchmarks.hotpotqa.experiment --dataset-set distractor --max-samples 5 --top-k 10 --provider ollama --model "qwen3.5:4b" --progress
py -m benchmarks.hotpotqa.experiment --dataset-set fullwiki --max-samples 5 --provider ollama --model "qwen3.5:4b" --progress
py -m benchmarks.hotpotqa.experiment --dataset-set distractor --max-samples 20 --seed 42 --provider ollama --model "qwen3.5:4b" --progress
py -m benchmarks.hotpotqa.experiment --mode oracle --dataset-set distractor --max-samples 5 --provider ollama --model "qwen3.5:4b" --progress
py -m benchmarks.hotpotqa.convert_to_json --dataset-set fullwiki
```

## Limitations and deviations

Parallel workers still report completed records in selected input order because the existing ordered executor map is retained. `checkpoint_every` remains accepted for compatibility, while every record continues to flush immediately for interruption safety. No `rich` dependency or colour layer was added; plain text fully covers the requested terminals. No material specification deviation was introduced.
