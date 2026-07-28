# HotpotQA benchmark

This package runs supplied-context HotpotQA through `PrimaRuntimeAdapter`, `PrimaRuntime`, and the unchanged production `ReasoningController`. It does not implement benchmark-specific retrieval, reasoning, or Wikipedia access.

## Evaluation data

Only the two official validation/evaluation sets are offered:

- `distractor` writes `benchmarks\hotpotqa\data\hotpot_dev_distractor_v1.json` and uses all supplied distractor context.
- `fullwiki` writes `benchmarks\hotpotqa\data\hotpot_dev_fullwiki_v1.json` and is labelled `official_retrieved_context`. This is the context returned by the benchmark authors' retriever, not a complete Wikipedia corpus.

Train sets are intentionally excluded. Missing JSON is prepared automatically from the selected Hugging Face validation Parquet file. Existing valid JSON is reused unless `--refresh-data` is passed. Preparation requires:

```powershell
py -m pip install datasets pyarrow==19.0.1
```

PyArrow 19.0.0 is rejected because it cannot reliably read these files. Optional conversion imports are loaded only when preparation is needed.

Prepare data without running the benchmark:

```powershell
py -m benchmarks.hotpotqa.convert_to_json --dataset-set distractor
py -m benchmarks.hotpotqa.convert_to_json --dataset-set fullwiki
py -m benchmarks.hotpotqa.convert_to_json --dataset-set distractor --output-path C:\data\hotpot.json --force
```

In an interactive terminal, omitting `--dataset-set` shows a two-choice evaluation menu. Scripts and redirected commands must provide `--dataset-set` explicitly.

## Running locally

Fresh runs use a newly generated random seed, printed in the run header and stored in the manifest. Supply `--seed` for reproducible selection. `--sampling sequential` preserves file order; both strategies apply `--offset` before `--max-samples`.

Distractor:

```powershell
py -m benchmarks.hotpotqa.experiment `
  --dataset-set distractor `
  --max-samples 5 `
  --top-k 10 `
  --provider ollama `
  --model "qwen3.5:4b" `
  --progress
```

Fullwiki:

```powershell
py -m benchmarks.hotpotqa.experiment `
  --dataset-set fullwiki `
  --max-samples 5 `
  --provider ollama `
  --model "qwen3.5:4b" `
  --progress
```

Reproducible selection:

```powershell
py -m benchmarks.hotpotqa.experiment `
  --dataset-set distractor `
  --max-samples 20 `
  --seed 42 `
  --provider ollama `
  --model "qwen3.5:4b" `
  --progress
```

Oracle diagnostic:

```powershell
py -m benchmarks.hotpotqa.experiment `
  --mode oracle `
  --dataset-set distractor `
  --max-samples 5 `
  --provider ollama `
  --model "qwen3.5:4b" `
  --progress
```

Sequential selection:

```powershell
py -m benchmarks.hotpotqa.experiment --dataset-set distractor --sampling sequential --offset 10 --max-samples 5 --progress
```

The original explicit-path form remains supported:

```powershell
py -m benchmarks.hotpotqa.experiment --mode distractor --dataset-path C:\data\hotpot_dev_distractor_v1.json --max-samples 5 --progress
```

Use `--refresh-data` to regenerate the selected validation JSON. Use `--quiet` to suppress the run header, per-question blocks, and aggregate summary while still writing artifacts. Use `--json-summary` for a machine-readable final result. `--progress` enables one readable result block per newly completed question; it never prints retrieved documents, raw metadata, or hidden reasoning.

## Resume

```powershell
py -m benchmarks.hotpotqa.experiment --dataset-set distractor --max-samples 20 --resume --progress
```

When no seed is supplied on resume, the stored resolved seed is reused. Resume validates the dataset fingerprint and set, mode, provider/model, reasoning settings, sampling strategy, offset, requested sample count, resolved seed, and exact selected sample IDs. Per-question output covers only work completed by the current command; the final summary includes all compatible checkpoint records.

## Architecture, metrics, and artifacts

`loader.py` validates official-style JSON. `adapter.py` creates one isolated conversation per sample and preserves exact title and sentence provenance without exposing gold labels during inference. Oracle alone filters gold context and writes to its own output directory. Answers use the existing production structured-output path; supporting facts are projected only from retained production evidence.

The evaluator uses official HotpotQA normalization and reports answer, supporting-fact, and joint EM/F1/precision/recall. The same scoring helper powers per-question terminal scores and saved aggregate metrics; terminal values are percentages while JSON remains in the 0–1 range.

Each mode writes under `benchmarks\hotpotqa\outputs\<mode>\` (or the selected `--output-path`) with:

- `raw\hotpot_results.jsonl`, `hotpot_predictions.json`, `hotpot_failures.json`, retrieval diagnostics, and reasoning diagnostics.
- `metrics\hotpot_metrics.json`, `supporting_fact_metrics.json`, `hotpot_summary.json`, and `run_manifest.json`.
- `logs\hotpotqa.log` and `prima_runtime_adapter.log` where produced by the existing runtime.

Do not run the obsolete external baseline preprocessing, scrape Wikipedia, or modify `external\`.
