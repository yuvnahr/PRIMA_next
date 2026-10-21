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

Goal 3 development selection is frozen in `docs/goemotions-development/selection.json`
and `dev_manifest.json`: official split/label hashes, 512 stratified dev IDs,
exactly three templates (`definitions_schema`, `expressed_evidence`,
`taxonomy_distinctions`) and preregistered promotion rules. The test set was
previously exposed; it is never used for tuning. Duplicate texts remain distinct
examples. These manifests contain evaluation metadata; do not pass them into
classification decisions.

`--prompt-variant` selects the template; it is recorded in the experiment identity
and invalidates incompatible resumes. Exact prompt/model/decoding/schema/text
changes invalidate provider caching. Changed prompts require separate requests;
same-response correction comparisons require identical raw response hashes.
`compare_development()` checks the frozen IDs, texts/gold, request identities and
comparison mode, reports paired uncertainty and always leaves production
promotion disabled. Matched Qwen comparisons use the frozen GenerationConfig
through the Python/campaign API (512 output tokens), context 8,192, one worker,
and the exact expected digest. The small-model CLI examples above are smoke
examples, not matched Qwen quality evidence.

The CPU TF-IDF/logistic system is a separately attributed **trained baseline**.
Train on train and fit thresholds on dev, using the preregistered word variant:

```powershell
.venv\Scripts\python.exe -m benchmarks.goemotions.training.classical --help
.venv\Scripts\python.exe -m benchmarks.goemotions.training.classical --train --features word --unbalanced --seed 13 --data-dir benchmarks/goemotions/external/goemotions/data --output-dir evaluation/goemotions/models/word_dev
.venv\Scripts\python.exe -m benchmarks.goemotions.training.classical --model evaluation/goemotions/models/word_dev/model.joblib --thresholds evaluation/goemotions/models/word_dev/thresholds.json --data-dir benchmarks/goemotions/external/goemotions/data --split dev --output-dir evaluation/goemotions/smoke/word_dev_reload
```

Use a new empty training/freeze destination. `freeze_tfidf_candidate()` now copies
the exact fitted model beside its matching dev thresholds and rejects settings
that differ from the fit; it no longer silently builds combined features.
`artifact_manifest.json` pins dependencies, feature settings, seed, label order,
model/threshold hashes and all split identities. Pin its fingerprint with
`--expected-fingerprint` at handoff. Loading checks hashes before unpickling;
use trusted local artifacts only. Legacy unbound models/thresholds must be
rebuilt through this path. `evaluate_frozen_tfidf()` retains its test-scoring API
but loads an existing frozen fit instead of retraining. Test inference remains
for reviewed integration and a later admitted final evaluation.

The measured full-dev CPU result, explicit optional-encoder/correction
non-admission, selected safe configuration and pending Qwen/cost gates are in
`docs/goemotions-kaggle-remediation.md`. Dev threshold-fit gains are not
independent validation, Qwen improvements or PRIMA correction gains.
