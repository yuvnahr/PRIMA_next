# HotpotQA benchmark

This package runs supplied-context HotpotQA samples through the shared benchmark models, `PrimaRuntimeAdapter`, `PrimaRuntime`, and the unchanged production `ReasoningController`. It contains no benchmark-specific retriever, reasoning loop, LLM client, or network lookup.

## Data and modes

Download the official files into a location of your choice and pass the path explicitly:

- `hotpot_dev_distractor_v1.json`: `distractor`, using every supplied context sentence.
- `hotpot_dev_fullwiki_v1.json`: `fullwiki`, labelled `official_retrieved_context`. This file contains paragraphs returned by the benchmark authors' retriever; it is not a complete Wikipedia corpus or a custom full-Wikipedia retrieval run.
- Either development file: `oracle`, labelled `oracle_gold_context`. Oracle is a diagnostic reasoning ceiling and must not be reported as the primary result.

Do not run the obsolete baseline preprocessing, download GloVe, scrape Wikipedia, or modify `external/`.

## Architecture and leakage safeguards

`loader.py` validates official JSON. `adapter.py` creates one isolated conversation per sample. Context is ingested at sentence granularity through the runtime's generic document-ingestion API, preserving exact title, paragraph index, sentence ID, text, and source label. Sentence granularity creates more retrieval units than paragraph ingestion, but makes supporting-fact projection exact: only evidence retained by production reasoning becomes a predicted `[title, sentence_id]`.

Answers and supporting facts remain evaluator-only fields. The shared runner constructs a clean inference question and attaches evaluator metadata only after the runtime returns. Distractor/fullwiki adaptation never filters, sorts, or labels context using gold facts. Oracle filters before ingestion and writes to a separate mode directory.

## Ollama

Install Ollama, pull `qwen3.5:4b`, and ensure the service is available. Configuration respects `OLLAMA_URL`, `PRIMA_LLM_PROVIDER`, and `PRIMA_LLM_MODEL`; CLI flags override provider/model. No credentials are hard-coded.

## Progressive commands

```powershell
python -m benchmarks.hotpotqa.experiment --mode oracle --dataset-path PATH\hotpot_dev_distractor_v1.json --max-samples 1 --provider ollama --model qwen3.5:4b --progress
python -m benchmarks.hotpotqa.experiment --mode distractor --dataset-path PATH\hotpot_dev_distractor_v1.json --max-samples 1 --provider ollama --model qwen3.5:4b --progress
python -m benchmarks.hotpotqa.experiment --mode fullwiki --dataset-path PATH\hotpot_dev_fullwiki_v1.json --max-samples 1 --provider ollama --model qwen3.5:4b --progress
python -m benchmarks.hotpotqa.experiment --mode distractor --dataset-path PATH\hotpot_dev_distractor_v1.json --max-samples 5 --provider ollama --model qwen3.5:4b --progress
```

Scale only after inspecting those outputs. Defaults are adaptive reasoning, top-k 5, max-hops 3, and one worker. Use `--offset`, `--seed`, `--reasoning-mode`, `--top-k`, `--max-hops`, `--parallel-workers`, and `--output-path` as needed.

Resume an interrupted compatible run:

```powershell
python -m benchmarks.hotpotqa.experiment --mode distractor --dataset-path PATH\hotpot_dev_distractor_v1.json --max-samples 5 --resume --checkpoint-every 1 --progress
```

Resume rejects changes to dataset fingerprint, mode, provider, model, reasoning mode, top-k, max-hops, or seed. Each completion, including failures, is flushed to `raw/hotpot_results.jsonl`; final artifacts are rebuilt from it.

## Outputs and metrics

Each mode has its own `raw/`, `processed/`, `metrics/`, and `logs/` directory. Outputs include official-format `hotpot_predictions.json`, failures, retrieval/reasoning diagnostics, run manifest, summary, and metrics. The evaluator reproduces official lowercase, punctuation/article removal, whitespace normalization, yes/no/noanswer handling, answer EM/F1/precision/recall, supporting-fact EM/F1/precision/recall, and joint metrics. Full reasoning traces are not enabled by the benchmark; production trace exposure remains governed by the reasoning mode.
