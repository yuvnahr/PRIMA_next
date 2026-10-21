# GoEmotions

The default is `affect_telemetry_preserve_labels`: the existing schema-constrained `AffectTelemetrySystem` runs the affect-only classification route and preserves the model's parsed labels, including neutral-plus-emotion combinations. `model_only_zero_shot` is the label-names-only baseline; `schema_constrained_model_only` adds taxonomy definitions and schema constraints. Neither uses a semantic correction policy.

`qwen_zero_shot`, `qwen_definitions`, `qwen_schema` and `qwen_with_prima_telemetry` remain compatibility aliases. `prima_qwen` / `bounded_prima_affect_decision` are deprecated entry points: new runs explicitly record `label_policy_version=preserve_labels_v1`, `legacy_policy_version=bounded_lexical_v0` and `legacy_policy_enabled=false`. The historical unversioned bounded experiment added lexical/hash-derived labels; old records retain that meaning. New runs suppress that heuristic, including literal-word and negation-based changes. Do not pool old/new policies by system name alone.

Each response retains its original text/hash, baseline/final labels, parse error, correction decision and evidence source. `label_correction` is the canonical audit; `affect_decision` and `prima_decision` remain compatibility views, not runtime reflection. A neutral fallback may keep runtime telemetry operating after malformed output, but it is explicitly `runtime_only_not_scored`: parsing still fails, scored labels remain empty and the record is excluded from the valid same-response headline. QA, retrieval, planning, tools and runtime reflection are outside this route.

Reports use schema 2.0. Label-only confidence is unavailable (`confidence_available=false`, `confidence=null`); raw sigmoid scores also require verified dev calibration, model/revision and artifact identities before probability metrics are eligible. Every selected case must have a valid full 28-label vector. The generic runtime prediction type is unchanged.

Use `performance.wall_throughput_examples_per_second` for measured active wall throughput. Summed-latency diagnostics, provider inference-window rates, cache hits, logical tokens and observed uncached tokens are separate. Deprecated keys retain their explicitly documented historical meanings. `attempt_timings.jsonl` separates resumed active work from calendar downtime; an unclosed attempt makes wall throughput unavailable. Native reports include per-label support/prediction counts, distinct execution/parse/label failures and partial-F1 correction gains/harms. Campaign `report_semantics` consumes these values and keeps LoCoMo replay coverage separate from QA coverage. See the handoff for exact scopes, aliases and the measured bootstrap comparison.

Learned backends are opt-in via `PRIMA_AFFECT_BACKEND=goemotions_pretrained` or `goemotions_deberta`; otherwise PRIMA keeps `LegacyAffectClassifier`.

The learned inference/training dependencies are included in the repository's `requirements.txt`.

```powershell
.venv\Scripts\python.exe -m benchmarks.goemotions.experiment --system qwen_schema --provider ollama --model qwen3.5:4b --seed 13
.venv\Scripts\python.exe -m benchmarks.goemotions.training.cli --limit 4 --device cuda
```

Create the fixed diagnostic development manifest and run a 20-example smoke:

```powershell
.venv\Scripts\python.exe -m benchmarks.goemotions.smoke
.venv\Scripts\python.exe -m benchmarks.goemotions.experiment --system affect_telemetry_preserve_labels --provider ollama --model qwen3.5:4b --split dev --sample-manifest evaluation/goemotions/smoke/dev_diagnostic_manifest.json --output-path evaluation/goemotions/smoke/preserve_dev20 --seed 13 --parallel-workers 1
```

This manifest deliberately covers neutral combinations, multilabel rows, rare labels, and commonly confused labels. It is for interface validation, not model selection.

The preserve-label default reports its same-response baseline beside the final metrics; its correction gain is zero. Separate provider runs can vary despite a fixed seed and temperature zero. No replacement policy is enabled. A future policy requires pinned trained per-label scores, dev-only calibration/threshold identities, predeclared allowed edits/margins, paired benefit with acceptable per-label harm and integrated cost. The test split has prior review exposure and must not tune the policy. See `docs/goemotions-kaggle-remediation.md` for the Goal 1 evidence and later prerequisites.

Full development validation:

```powershell
.venv\Scripts\python.exe -m benchmarks.goemotions.experiment --system qwen_schema --provider ollama --model qwen3.5:4b --split dev --output-path evaluation/goemotions/dev/qwen_schema --seed 13 --parallel-workers 1
.venv\Scripts\python.exe -m benchmarks.goemotions.experiment --system affect_telemetry_preserve_labels --provider ollama --model qwen3.5:4b --split dev --output-path evaluation/goemotions/dev/preserve_labels --seed 13 --parallel-workers 1
```

The locally frozen TF-IDF PRIMA candidate is ignored by Git. Build it once, then run its full test split with:

```powershell
.venv\Scripts\python.exe -m benchmarks.goemotions.training.classical --train --data-dir benchmarks/goemotions/external/goemotions/data --output-dir evaluation/goemotions/models/final_candidate
.venv\Scripts\python.exe -m benchmarks.goemotions.training.classical --model evaluation/goemotions/models/final_candidate/model.joblib --thresholds evaluation/goemotions/models/final_candidate/thresholds.json --data-dir benchmarks/goemotions/external/goemotions/data --split test --output-dir evaluation/goemotions/final/prima_tfidf
```
