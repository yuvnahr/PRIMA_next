# Phase 13 — GoEmotions Canonical Runtime Migration

## Outcome

GoEmotions now evaluates affect and multilabel-classification behavior through the canonical public runtime. Each prediction uses a typed `PrimaRequest` with `TaskKind.EMOTION_CLASSIFICATION` and `ExecutionProfile.AFFECT_ONLY`; the route does not activate retrieval, QA reasoning, world simulation, model execution, or tools.

This is deliberately a component/task benchmark. Reports do not describe GoEmotions as a full PRIMA-wrapper or QA-architecture evaluation.

## Truthful systems and configuration

The runner exposes five system identities:

- `model_only_zero_shot` scores the parsed labels from an unconstrained zero-shot model response;
- `schema_constrained_model_only` scores labels from a provider schema-constrained response;
- `affect_telemetry_preserve_labels` updates bounded affect telemetry without changing parsed model labels;
- `bounded_prima_affect_decision` applies the bounded lexical decision to the same model response inside the classifier invoked by the public runtime;
- `trained_encoder` is reported as a separate trained baseline, never as a wrapper improvement.

Historical command names remain accepted as compatibility aliases, but manifests and reports emit the truthful canonical names. The TF-IDF/logistic workflow is likewise labeled `trained_tfidf_logistic` with `wrapper_attribution=false`, and its scoring path uses the canonical emotion-classification route.

Generation settings use immutable `GenerationConfig`. Encoder device, batch size, thresholds, and calibration paths use immutable `ClassifierSettings`. The runner no longer mutates process-global generation environment variables, and the removed hybrid path no longer reads classifier or adjudication settings from the environment.

## Split validation and resumable artifacts

Every campaign validates and SHA-256 hashes `train.tsv`, `dev.tsv`, and `test.tsv`, rejects duplicate IDs within a split, and rejects ID overlap across splits. Training remains train-only, threshold/calibration selection remains development-only, and test scoring consumes frozen settings. Manifests identify `full_split` versus `sample` execution explicitly.

GoEmotions now uses the Phase 10 `BenchmarkArtifactStore`. Each completed or failed example receives an append-only checkpoint immediately. Resume validates the dataset, selected IDs, source, model, generation settings, classifier settings, route, prompt, dependencies, Python, and hardware before suppressing completed IDs. Each initial or resumed campaign receives a unique run ID and preserves resume lineage. Completion, failure, and selected totals are reconciled before finalization.

## Metrics and attribution

Metrics are computed from unrounded floating-point values and rounded only when serialized. The runner reports:

- exact-set accuracy;
- macro, micro, and support-weighted precision, recall, and F1;
- sample precision, recall, and F1;
- Hamming loss, Jaccard, gold/predicted cardinality, and cardinality error;
- per-class support and label/prediction distributions;
- parse failures, empty outputs, neutral combinations, and multilabel rates;
- latency, provider token usage, throughput, and token throughput;
- probability diagnostics only for systems that provide complete probability vectors.

PR-AUC now evaluates all examples tied at one score as one threshold, removing input-order sensitivity. The old multilabel “confusion matrix” key and filename are replaced with label co-occurrence.

For systems that transform one underlying model response, baseline validity and affect outcomes are separate reconciled partitions. Parse recovery, unrecovered parse failure, label correction, regression, other changes, and unchanged predictions are visible counts. The headline paired affect comparison excludes baseline parse failures, so parse recovery cannot inflate its improvement claim. A deterministic paired bootstrap reports the sample-F1 delta and percentile confidence interval on the valid-baseline subset.

## Validation coverage

Focused tests cover malformed JSON, unknown labels, duplicate labels, empty label output, mixed neutral combinations, tie-sensitive PR-AUC, paired outcome reconciliation, canonical bounded-route metadata, TF-IDF/logistic route usage, interruption, resume, and unique run IDs.

## Focused verification

```text
venv\Scripts\python.exe --version
Python 3.10.11

venv\Scripts\python.exe -m pytest tests/test_goemotions_benchmark.py tests/test_goemotions_upgrade.py tests/benchmarks/test_common_artifacts.py tests/benchmarks/test_preflight.py -q
35 passed in 3.76s.

venv\Scripts\python.exe -m compileall -q benchmarks/goemotions tests/test_goemotions_benchmark.py tests/test_goemotions_upgrade.py
Exit 0; no output.

venv\Scripts\python.exe -m ruff check <Phase 13 files>
All checks passed.

venv\Scripts\python.exe -m mypy benchmarks/goemotions/experiment.py benchmarks/goemotions/systems.py benchmarks/goemotions/metrics.py benchmarks/goemotions/schemas.py benchmarks/goemotions/dataset.py benchmarks/goemotions/training/data.py
Success: no issues found in 6 source files; one non-failing unused optional-module override note.

venv\Scripts\python.exe -m bandit -q -c bandit.yaml -r benchmarks/goemotions
Exit 0; no output.
```

## Final quality gates

The repository-wide CI matrix was run once after implementation and documentation completed:

```text
venv\Scripts\python.exe -m benchmarks.preflight --json
Exit 0; schema 1.0; rouge_l, bertscore, and goemotions_encoder available.

venv\Scripts\python.exe -m compileall -q -x "(^|[\\/])(\.git|\.venv|venv|external)([\\/]|$)" .
Exit 0; no output.

venv\Scripts\python.exe -m pytest -q
302 passed, 1 skipped in 140.32s.

venv\Scripts\python.exe -m ruff check .
All checks passed.

venv\Scripts\python.exe -m mypy .
Success: no issues found in 221 source files.

venv\Scripts\python.exe -m bandit -q -c bandit.yaml -r .
Exit 0; only existing nosec-without-failed-test warnings for bounded Git probes in HotpotQA and LoCoMo.

venv\Scripts\python.exe scripts\run_xenon.py
Exit 0; production source roots only.
```

## Changed files

- `README.md`
- `benchmarks/goemotions/{dataset.py,experiment.py,metrics.py,schemas.py,systems.py}`
- `benchmarks/goemotions/training/{classical.py,data.py}`
- `tests/{test_goemotions_benchmark.py,test_goemotions_upgrade.py}`
- `docs/implementation/{STATUS.md,phase_13_goemotions_canonical_runtime.md}`

## Risks and deferrals

- No live model or full GoEmotions score regeneration is performed in this implementation phase.
- Encoder and TF-IDF/logistic improvements remain learned-baseline results; they are not attributed to a PRIMA wrapper.
- Bootstrap intervals quantify the bounded same-response transformation only and do not establish general model superiority.
