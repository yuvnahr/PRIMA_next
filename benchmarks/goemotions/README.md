# GoEmotions

`qwen_zero_shot` is the label-names-only baseline, `qwen_definitions` adds the official definitions, and `qwen_schema` also constrains Ollama output. `qwen_with_prima_telemetry` preserves Qwen labels while recording an independent PRIMA update. `prima_qwen` scores a documented PRIMA augmentation of the same schema-constrained Qwen labels. Every Reddit comment receives fresh affect state.

Learned backends are opt-in via `PRIMA_AFFECT_BACKEND=goemotions_pretrained` or `goemotions_deberta`; otherwise PRIMA keeps `LegacyAffectClassifier`.

Install learned inference/training dependencies with `pip install -r requirements-goemotions.txt`.

```powershell
.venv\Scripts\python.exe -m benchmarks.goemotions.experiment --system qwen_schema --provider ollama --model qwen3.5:4b --seed 13
.venv\Scripts\python.exe -m benchmarks.goemotions.training.cli --limit 4 --device cuda
```

Create the fixed diagnostic development manifest and run a 20-example smoke:

```powershell
.venv\Scripts\python.exe -m benchmarks.goemotions.smoke
.venv\Scripts\python.exe -m benchmarks.goemotions.experiment --system prima_qwen --provider ollama --model qwen3.5:4b --split dev --sample-manifest evaluation/goemotions/smoke/dev_diagnostic_manifest.json --output-path evaluation/goemotions/smoke/repaired_dev20_prima --seed 13 --parallel-workers 1
```

This manifest deliberately covers neutral combinations, multilabel rows, rare labels, and commonly confused labels. It is for interface validation, not model selection.

`prima_qwen` reports its same-response Qwen baseline beside the final PRIMA metrics. Use that comparison for PRIMA attribution: separate Ollama runs can vary slightly despite a fixed seed and temperature zero. The current PRIMA augmentation is deliberately narrow (lexical evidence for six core emotions); validate any expansion on the full development split before evaluating the untouched test split.

Full development validation:

```powershell
.venv\Scripts\python.exe -m benchmarks.goemotions.experiment --system qwen_schema --provider ollama --model qwen3.5:4b --split dev --output-path evaluation/goemotions/dev/qwen_schema --seed 13 --parallel-workers 1
.venv\Scripts\python.exe -m benchmarks.goemotions.experiment --system prima_qwen --provider ollama --model qwen3.5:4b --split dev --output-path evaluation/goemotions/dev/prima_qwen --seed 13 --parallel-workers 1
```

The locally frozen TF-IDF PRIMA candidate is ignored by Git. Build it once, then run its full test split with:

```powershell
.venv\Scripts\python.exe -m benchmarks.goemotions.training.classical --train --data-dir benchmarks/goemotions/external/goemotions/data --output-dir evaluation/goemotions/models/final_candidate
.venv\Scripts\python.exe -m benchmarks.goemotions.training.classical --model evaluation/goemotions/models/final_candidate/model.joblib --thresholds evaluation/goemotions/models/final_candidate/thresholds.json --data-dir benchmarks/goemotions/external/goemotions/data --split test --output-dir evaluation/goemotions/final/prima_tfidf
```
