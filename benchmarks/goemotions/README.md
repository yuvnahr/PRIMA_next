# GoEmotions

The default `qwen_zero_shot` system is the frozen Qwen label-generation baseline. `qwen_schema` uses Ollama's JSON-schema `format` field through the shared LLM layer. `prima_qwen` runs the same Qwen labels through PRIMA's stateful affect engine while preserving the original fine-grained labels for scoring. These Qwen systems are sequential by default and do not expose calibrated probabilities.

Learned backends are opt-in via `PRIMA_AFFECT_BACKEND=goemotions_pretrained` or `goemotions_deberta`; otherwise PRIMA keeps `LegacyAffectClassifier`.

Install learned inference/training dependencies with `pip install -r requirements-goemotions.txt`.

```powershell
venv\Scripts\python.exe -m benchmarks.goemotions.experiment --system qwen_schema --provider ollama --model qwen3.5:4b --seed 13
venv\Scripts\python.exe -m benchmarks.goemotions.training.cli --limit 4 --device cuda
```

The locally frozen TF-IDF PRIMA candidate is ignored by Git. Run its full test split with:

```powershell
venv\Scripts\python.exe -m benchmarks.goemotions.training.classical --model evaluation/goemotions/models/final_candidate/model.joblib --thresholds evaluation/goemotions/models/final_candidate/thresholds.json --data-dir benchmarks/goemotions/external/goemotions/data --split test --output-dir evaluation/goemotions/final/prima_tfidf
```
