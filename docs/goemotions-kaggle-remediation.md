# GoEmotions and Kaggle remediation handoff

## Goal 1: safe, auditable preserve-label classification

Delivery branch: `dev-p2`. Actual implementation base/predecessor:
`c25a178d6308bc48b3a5e44d9519316a5179c499`, matching the planning snapshot.
Status: **Goal 1 implementation acceptance complete**. Later checkpoint status
is recorded in the corresponding sections below; measured release acceptance is separate.
Goal 1 is the first checkpoint; its own commit SHA belongs in the completion
message. Work was isolated in the attached `goemotions-b1` worktree because
the user's checkout contained unrelated untracked documents/directories.
No Person 1 files, external submodules, notebook, dependencies or CI were edited.
No push, PR, merge to `dev`, training or production campaign is authorized by this checkpoint.

The four-goal workflow supersedes the source plan's intermediate PR/merge order.
This section records the B1 checkpoint. Later work includes final integration
against Person 1's completed pinned checkpoint. Person 1 completion was supplied
by the user; this checkpoint does not independently certify it.

## Requirements, implementation and evidence

| B1 requirement | Implementation | Deterministic coverage |
|---|---|---|
| Safe existing default | `benchmarks/goemotions/experiment.py`: API/CLI default selects existing `AffectTelemetrySystem`; `_system()` factory uses it | Default/factory/CLI and full tiny campaign tests |
| No hash/literal/negation-driven edits | `systems.py`: bounded source audits legacy proposals but always returns baseline prediction; telemetry preserves parsed labels | Synthetic hash, literal disagreement, negation, sarcasm and mixed-neutral cases |
| Evidence provenance without changing legacy output | `affect/emotion_profile.py`, `affect/emotion_classifier.py`: defaulted source field; lexical vs hash path tagged | Scores, keywords, cache behavior, negation compatibility, external-vector unknown source |
| Raw response, hash and labels retained | `systems.py`, `experiment.py`: original response preserved, SHA-256 verified, parsed baseline cross-checked, preserve-policy edits rejected | Raw byte/text preservation, hash/baseline/final mismatch rejection |
| Parse recovery distinct from correction | Runtime-only neutral fallback, empty scored set, original parse failure and headline exclusion | Empty, malformed, duplicate/unknown labels; mixed valid/invalid populations |
| Compatible names and historical meaning | Legacy constructors and decision aliases retained; explicit current/legacy policy versions | Constructor and historical unversioned-record tests |
| Affect-only route | Existing public runtime route, now explicitly rejects reflection and all retrieval stages as well as planning/tools/generic generation | Real fake-provider route diagnostics in regression tests; no runtime source edits |
| IDs and checkpoint audits preserved | Single record owns example ID and original response hash; metadata serialized in checkpoint diagnostics | Tiny campaign serializes and resumes with identical records and zero additional requests |
| Historical replay when available | Checksum-verified archive replay through current telemetry and serializer | 5,427 unique IDs, exact prompt/hash/baseline matching and metrics below |

Synthetic input fixtures: `tests/fixtures/goemotions_review/preserve_labels.json`.
New tests: `tests/test_goemotions_review_regressions.py`. Existing correction
expectations in `tests/test_goemotions_upgrade.py` now assert explicit suppression.
Unrelated legacy classifier scoring/modifier behavior is retained.

## Versioned consumer contract

- API and CLI default: `affect_telemetry_preserve_labels`. Explicit pure model
  baselines remain available. Model/context/decoding defaults were not changed.
- `label_audit_schema_version="1.0"`, `label_policy_version="preserve_labels_v1"`.
  These are emitted on label-only LLM records; the manifest also fingerprints
  schema/policy versions and `legacy_correction_enabled=false`.
- `raw_model_response` and returned `raw_response` retain original provider text
  for new LLM records. `raw_model_response_hash` / serialized `response_hash` are
  SHA-256 of that text's UTF-8 bytes. Metadata baseline labels must match a fresh
  parse of that same response; changed labels under the preserve policy fail.
- `baseline_labels`, `baseline_parse_error`, `baseline_empty_output` and
  `final_scored_labels` remain available. Serialized `predicted_labels` is the
  scored label set. Valid baseline and final sets are exactly equal.
- `label_correction` is canonical: `policy_version`, `evidence_source`,
  `accepted`, `added_labels`, `removed_labels`, `reason`, `preserved_baseline`.
  No edit is accepted in this checkpoint. Default telemetry uses unavailable
  correction evidence; deprecated bounded requests also expose legacy proposals
  as `affect_derived_labels` without applying them.
- Evidence sources: `literal_span` is a lexicon match, `hash_neighbour` is a
  deterministic hash-vector lookup, `trained_probability` describes the existing
  encoder's scores, and `unavailable` is the default/unknown source. These names
  describe provenance, not calibrated reliability or authorization to correct.
  External vectors without validated model provenance remain unavailable.
- `parse_recovery={applied, labels, scope:"runtime_only_not_scored"}` explains
  the internal neutral fallback used by the runtime's nonempty prediction type.
  It never repairs the scored result invisibly. Malformed output remains a parse
  failure, with no scored labels and `same_response_comparison_eligible=false`.
  A completed classification checkpoint can therefore contain a parse failure;
  execution failure and parse failure are distinct.
- `affect_decision` and `prima_decision` are compatibility views of the label
  decision, never runtime reflection. Their familiar telemetry reason strings
  remain readable. New bounded requests say
  `unvalidated_legacy_correction_suppressed`, with
  `legacy_policy_version="bounded_lexical_v0"`, `legacy_policy_enabled=false`.
- Historical `bounded_prima_affect_decision` / `prima_qwen` records lacking policy
  versions retain their old lexical/hash-edit interpretation. New calls through
  those names are explicitly versioned compatibility requests with edits disabled;
  the historical heuristic is not retained as an executable correction policy.
  Do not infer old records' policy or evidence source from the new defaults.
- The valid same-response headline now covers default telemetry too and reports
  `sample_count` plus `excluded_count`. Invalid/unpaired records are excluded.
  Existing outcome aliases and all old records remain readable.
- New manifest fields change resume identity. Existing new-format checkpoints
  round-trip their complete audits. An old campaign is not resumable into the new
  policy merely because it uses the same system name; start a new output identity.
  Reading/scoring old artifacts does not rewrite their predictions.

Example new malformed-output record (abridged):

```json
{
  "id": "synthetic-invalid",
  "raw_response": "not JSON",
  "raw_model_response": "not JSON",
  "baseline_labels": [],
  "predicted_labels": [],
  "baseline_parse_error": "<original parser error>",
  "parse_error": "<original parser error>",
  "label_policy_version": "preserve_labels_v1",
  "parse_recovery": {"applied": true, "labels": ["neutral"], "scope": "runtime_only_not_scored"},
  "same_response_comparison_eligible": false
}
```

An old record may instead contain `prima_decision.reason=add_prima_lexical_emotion`
with differing baseline/final labels and no policy fields. Readers preserve that
historical observation; they do not reinterpret it as a v1 preserve-label run.

## Measured offline historical replay

Input archive (unchanged):
`C:\Users\galad\Downloads\PRIMA-output\PRIMA_goemotions_qwen3.8-27b_20261006T152423Z-71202fc0.zip`.
SHA-256: `3d2df340286f55d1f597ff670e7cd4b90822c923cf67b4bca85160ba54e9bec4`.
The current replay checked this hash and ZIP CRC, read the canonical root
`predictions.jsonl` records' `diagnostics.goemotions_record`, and fed each saved
`raw_model_response` into the current `AffectTelemetrySystem` using a mocked
`LLMClient.chat`. Gold labels were used only by the serializer/evaluator after
classification, never by the prediction source.

All 5,427 unique IDs, prompts, original response hashes and parsed baseline sets
matched. All final sets equaled those baselines; zero parse failures. Current
`_classify_response()` and `evaluate()` produced:

| Metric | Preserve-label replay | Historical corrected labels |
|---|---:|---:|
| Exact matches | 1,188 | 811 |
| Micro F1 | 0.373806275580 | 0.358666579686 |
| Macro F1 | 0.356061201815 | 0.343855899175 |
| Sample F1 | 0.381911869225 | 0.358860459607 |

The replay took 26.628 seconds CPU wall time in this environment including hash,
CRC, reading, classification-route replay and metric checks. This is not a new
Qwen timing or quality experiment, not a bootstrap optimization measurement, and
not a prediction of Kaggle throughput. The 377 exact-match difference refers
only to these saved responses. No new model inference, GPU allocation or download
was used. A small local summary is at
`C:\Users\galad\OneDrive\Desktop\Capstone\code\PRIMA_next\evaluation\goemotions\smoke\b1_historical_replay.json`
(ignored output); the table/hash above are the committed evidence.

## Validation ledger

Project interpreter: `C:\Users\galad\OneDrive\Desktop\Capstone\code\PRIMA_next\.venv\Scripts\python.exe`
(Python 3.14.8); existing installed dependencies, no upgrades. Use that interpreter
for every `python` command below.

- Before edits: focused benchmark/upgrade/affect suite, **35 passed in 8.51 s**.
- Initial sandbox test collection could not create worktree log/cache directories;
  rerunning with authorized worktree filesystem access resolved the environment
  error. Initial new-test fixture setup errors were corrected before acceptance.
- Final required focused suite: **66 passed in 5.35 s**, no skips.
- Broader affected suite: **222 passed, 1 failed in 102.35 s**. The only failure is
  `test_notebook_smoke_stages_native_artifacts_and_partial_packages[goemotions]`,
  with `Canary child identities do not match`. Reproduced separately on a clean,
  unchanged worktree at base `c25a178d6308bc48b3a5e44d9519316a5179c499`:
  **1 failed in 9.22 s**, same error. Existing GoEmotions `_git_commit()` reads
  `.git/HEAD` directly and returns `None` in a worktree, while the notebook's
  LoCoMo Git helper returns the actual commit. Both the original base helper and
  current helper returned `None` in the implementation worktree. This predates
  B1; no task-caused test failure remains. Resolve source-identity integration
  in Goal 4 rather than changing the notebook or unrelated identity code here.
- Scoped Ruff: **all checks passed**. `git diff --check`: **passed**.
- Notebook full-run default remains `RUN_FULL_BENCHMARK=False`; notebook and
  Person 1 producer/config paths have no diff.

Commands actually run (from the implementation worktree):

```powershell
$py = 'C:\Users\galad\OneDrive\Desktop\Capstone\code\PRIMA_next\.venv\Scripts\python.exe'
& $py -m pytest -q tests/test_goemotions_benchmark.py tests/test_goemotions_upgrade.py tests/test_goemotions_review_regressions.py tests/affect --tb=short
& $py -m pytest -q tests/test_goemotions_benchmark.py tests/test_goemotions_upgrade.py tests/test_goemotions_review_regressions.py tests/affect tests/benchmarks tests/integration --tb=short
& $py -m ruff check affect benchmarks/goemotions tests/test_goemotions_benchmark.py tests/test_goemotions_upgrade.py tests/test_goemotions_review_regressions.py tests/affect
git diff --check
```

The single notebook test was also run with `pytest -q` and `--tb=short` in the
separate unchanged baseline worktree. Wider CI-equivalent mypy, compileall,
Bandit, Xenon and full-repository pytest are reserved for Goal 4, not claimed here.

No applicable `AGENTS.md` was found in the parent path or owned source/test paths.
The main checkout's unrelated untracked files were preserved. All three external
submodule status checks encountered Git metadata permissions; their cleanliness
cannot be certified. None was edited or initialized by this goal.

## Next prerequisites and limits

Goal 2 starts only from this successful checkpoint and its tested audit contract.
It owns bootstrap optimization/profiling, confidence/probability eligibility,
additional correction metrics and truthful provider/throughput/cache reports.
The 0/1 runtime prediction vector remains an internal label representation;
this B1 checkpoint does not claim calibrated confidence. Notebook defaults and
shared report migration stay assigned to Goal 4.

Goal 3 may consider a replacement policy only with actually trained pinned
per-label scores, dev-only calibration and threshold artifact/split identities,
predeclared allowed edits and margins, positive paired development sample-F1
change with its 95% lower confidence bound above zero, acceptable per-label harm
and integrated resource/time cost. Classifier disagreement, hash neighbours,
literal keywords and broad negation alone never validate an edit. No such
replacement policy was implemented or enabled here.

Goal 4 still requires Person 1's pinned final producer checkpoint, exact proposed
Person 2 candidate snapshot and separate integration checkout evidence. No
producer changes or hand-written substitute contracts were introduced in B1.
Keep full production execution disabled pending reviewed integration and fresh
resource/time/identity gates. No GPU, Kaggle quota, fresh provider run or full CI
acceptance is inferred from these offline checks.

Rollback: revert this one scoped checkpoint or select a previously reviewed
explicit configuration using a new campaign identity. Never reset `dev`, rewrite
the historical archive, or resume across incompatible policy fingerprints.

## Goal 2: metrics, confidence and report semantics

Predecessor and actual base: `e0dd105484a2f78fc7a13811c83ef4e04aaad440` (Goal 1),
verified as `dev-p2` HEAD before work. The separate `goemotions-b2` worktree
preserves the original checkout's unrelated files. Status: **Goal 2 implementation
acceptance complete**; the successful checkpoint's own SHA belongs in
the completion response. Only B2 is in scope, with final notebook presentation
and actual combined producer/consumer integration reserved for Goal 4.

### Requirement and caller coverage

| B2 requirement | Producer/consumer | Evidence |
|---|---|---|
| Same paired sample-F1 statistic and sampling | `benchmarks/goemotions/metrics.py` precomputes per-case differences; identical seeded `randrange` calls and percentile indices | Frozen old-reference tests, empty/zero/negative/single/wrong/mixed/swapped cases; full historical profiling below |
| Confidence requires eligible scores | `systems.py`, `experiment.py`, `metrics.py`; benchmark dictionaries suppress label-indicator confidence | Tests for label-only, uncalibrated, partial/nonfinite/out-of-range/bool scores and identity mismatch |
| Actual calibration identities | Encoder receives generation revision; calibration writer/fitter/CLI retain model revision and dev hash | Synthetic artifact/hash/model/temperature checks; missing old revision remains unavailable |
| Complete selected probability population | Metric eligibility checks every selected case, including failed checkpoints, one calibration identity and full official 28-label vector | Positive calibrated fixture, missing/failed/mixed-identity and empty populations |
| Multilabel metrics and counts | Existing evaluator plus prediction counts; native report adds weighted/precision/recall/Jaccard/Hamming/cardinality and per-label table | Metric/reference and report assertions; low-support context retained |
| Partial correction gains/harms | Valid same-response populations only; original hashes/parses rechecked | Partial-F1 gain without exact gain, regression and added-label precision fixtures |
| Correct provider/cache/token accounting | Existing bounded provider adds thread-local request measurements; default standalone systems reuse it | Actual fake-provider cache/retry tests and a tiny native campaign; failures have unavailable billed tokens |
| Monotonic active time and resume | Shared `AttemptTimer`, native experiment and campaign orchestrator | Checkpointed active attempts, interrupted/open-tail handling and synthetic calendar downtime |
| Old/new report consumers | `benchmarks/common/reporting.py`, campaign aggregation | Exact committed Person 1 old/new fixtures; missing fields stay unavailable; legacy reflection/throughput aliases documented |

Caller searches included bootstrap/evaluation functions, `_metrics`, native
serialization/reporting, all system constructors and source conversions, shared
provider cache/retry paths, campaign aggregation and calibration CLI/helper calls.
No Person 1 source/config/runtime files, notebook cells, dependency manifests,
external data or submodules were changed. Existing classification labels and B1
correction safety remain unchanged.

### Frozen report contract for Goal 3 and Goal 4

`report_schema_version="2.0"` is additive in metrics/manifests and changes exact
resume identity. It does not reinterpret historical stored predictions. A B1
campaign needs an explicitly new identity for B2 execution; old archives remain
readable for offline evaluation with unavailable observations.

- Label-only output exposes `confidence_available=false`, `confidence=null` and
  a reason, including the serialized benchmark `prediction` dictionary. The
  generic runtime `EmotionPrediction` stays unchanged; its internal indicator
  vector cannot become probabilistic confidence through report conversion.
- `probability_provenance` requires trained scores, matched actual model/revision,
  `scalar_temperature`, dev split/hash and calibration artifact SHA-256. Encoder
  temperature must match the loaded calibration. The fitter/CLI forwards the
  existing `--revision`. Old artifacts with no revision can still load, but do
  not authorize confidence or calibrated metrics. No new calibration fit or
  training was performed in B2.
- Aggregate probability metrics require one verified identity and a valid full
  28-label vector for every selected case. Missing scores, parse/execution
  failures, mismatched calibration or a partial population yields unavailable
  metrics and confidence, with eligible/selected counts. Raw sigmoid scores are
  insufficient. Calibration identity is checked against the selected dataset's
  actual development/validation split hash.
  Eligible confidence is the maximum calibrated label score (aggregate: its
  mean), not a calibrated probability that the whole predicted set is correct.
- `populations` and `primary_outcome_counts` disclose the legacy metric
  denominator (execution-success records, with parse failures scored as their
  observed prediction), execution errors, original parse failures, valid label
  errors and exact successes. Primary counts reconcile to all selected cases.
  An old neutral fallback never conceals an original baseline parse failure;
  its historical prediction is retained rather than rewritten.
- `label_correction` adds partial-F1 gain/harm/unchanged counts, mean change,
  correct/total added labels and added-label precision (null for no additions).
  Only verified valid same-response pairs enter it; exclusions are counted.
  An empty eligible population is unavailable, with null mean change, rather
  than evidence of zero correction gain.
  `improved_after_label_correction` / `regressed_after_label_correction` are
  canonical. Old reflection-named aliases are explicitly deprecated label
  outcomes. Runtime reflection is separately not applicable / zero executed in
  affect-only classification. No inference about reflection benefit is made.
- `classification_provider` is request-scoped and includes actual send attempts,
  successes, retries, cache hits, known uncached tokens and inference-window
  endpoints. Logical classification calls count even when generic runtime
  `MODEL_EXECUTOR` is not applicable. Cached usage is logical; it is not charged
  again as actual inference. Failed provider requests can consume unknown tokens,
  so total actual tokens are unavailable when such usage cannot be observed.
  Uninstrumented old/external clients also report actual counts unavailable.
- `performance.wall_throughput_examples_per_second` uses completed checkpoints
  (including recorded execution failures) / measured active wall seconds.
  `examples_per_summed_latency_second` is a distinct diagnostic. Provider rate
  is successful uncached sends / observed inference-window duration, grouped by
  active attempt; overlapping latencies are never summed as wall time. Logical
  and actual token totals/rates are separate, with unknown usage unavailable.
  Deprecated `throughput_examples_per_second`, `total_tokens` and
  `tokens_per_second` retain their old summed-latency/logical meanings, with an
  explicit alias map and observed partial sums where old usage was incomplete.
  Canonical full totals/rates remain unavailable when their coverage is missing.
  Notebook readers must use the new canonical fields.
- `attempt_timings.jsonl` records cumulative monotonic elapsed time durably on
  checkpoint and phase boundaries. Summaries add active attempts, never calendar
  downtime. An unclosed hard-killed attempt is a measured lower bound and makes
  wall throughput unavailable. A reporting snapshot explicitly excludes its
  final statistics/summary refresh; it is not claimed as an all-inclusive
  session or model timing. Calendar seconds are reported separately.
- Campaign aggregates expose `report_semantics`. QA component coverage and
  conversation replay/admission/maintenance coverage have separate populations;
  missing replay records are unavailable. Old last-hop recall never becomes
  all-hop recall. Candidate/accumulated/supplied/cited stages stay distinct;
  evidence flags are non-additive and new primary failures remain disjoint.
  Official/legacy `answer_scores` and authoritative reflection audits are
  preserved from producers rather than reconstructed.

Person 1 fixture source: `dev-p1` pinned at
`4d4b76b1462f18d70450d51a4ee1c2714d558dd9`. Exact committed bytes of
`report_contract.json` and `goal4_producer_contract.json` were copied into the
owned `tests/fixtures/campaign/person1_*.json` paths; producer files were not
modified or merged. Their old/new expectations are exercised now. Actual
producer execution against final Person 2 notebook/consumers remains the
explicit Goal 4 combined-checkout acceptance gate, not a substitute handwritten
schema or an unimplemented GoEmotions B2 dependency.

### Measurement and validation ledger

The historical archive and expected SHA-256 are the same verified input as B1.
The full old/new 1,000-resample CPU comparison is run serially using the exact
pre-B2 metric source from predecessor `e0dd105`, identical 5,427 saved cases,
seed 13, and the existing Python 3.14.8 environment. An initial profiling probe
was deliberately stopped when a competing test process was noticed; it is not
used as performance evidence. No test jobs overlap the retained comparison.
Metrics, figures, report serialization and five-artifact ZIP packaging are
measured separately. The retained serial comparison completed successfully:

| Measured CPU phase | Seconds |
|---|---:|
| Original bootstrap, 1,000 resamples | 1367.622843 |
| Optimized bootstrap, same resamples/seed/cases | 3.025457 |
| One multilabel evaluation | 0.975321 |
| Complete current metric producer, including bootstrap | 5.529434 |
| Three figures, including matplotlib setup | 3.418314 |
| Native report and metric JSON serialization | 0.012327 |
| Five-artifact ZIP write and CRC validation | 0.143370 |

This is an observed 452.04x bootstrap speedup for this one local calculation,
not an end-to-end campaign speedup. Packaging covers only three figures, the
native report and metrics JSON, not a complete campaign bundle. Reference
source SHA-256: `319ce5887533519e3f2cbf99d7a423c1233790a49ed9c1b73c7b060550c1aee0`.
Environment: Python 3.14.8, Windows 11 build 26300, unchanged project dependencies.

The old result was delta `-0.02305140961857377`, interval
`[-0.025403844972667566, -0.020748111295375016]`; new result was delta
`-0.023051409618573798`, interval
`[-0.025403844972667528, -0.020748111295374978]`. All differences passed 1e-12
absolute-tolerance checks (small fixtures use 1e-14). This reproduces the
historical negative correction result on saved responses; it does not measure a
new policy or infer the cause of all 840 historical postprocessing seconds.
The serial final-code recheck (after metadata/report compatibility hardening)
matched the same complete historical statistic again. Its bootstrap was
1.319162 s, one evaluation 0.589854 s, complete producer metrics 5.503815 s,
figures 2.781660 s, report serialization 0.004723 s and packaging 0.092156 s.
These repeat timings vary; the 452.04x figure above belongs to the first paired
observation, not a universal latency claim or a cherry-picked comparison using
the faster repeat. All ten modified production Python files matched the
normalized source hashes recorded by this final recheck.

The archive's original corrected predictions independently yielded 9 partial-F1
gains, 403 partial-F1 harms, 5,015 unchanged sample-F1 scores, 669 added labels
with 9 correct additions (precision 0.013452914798), 2 exact-set improvements and
379 exact-set regressions. Counts reconcile over 5,427 valid pairs. There were
811 exact matches and 4,616 valid label errors, with zero execution/parse errors.
This is historical correction analysis; the safe new preserve-label policy has
no semantic edits and is not being credited with a trained-model improvement.

Local evidence is preserved under
`C:\Users\galad\OneDrive\Desktop\Capstone\code\PRIMA_next\evaluation\goemotions\smoke\b2_profile`
(ignored output, excluded from the commit): drivers, summaries, native reports,
metrics, figures and both CRC-validated five-artifact ZIP probes.
`summary.json` SHA-256 is
`a517da996faca17eccf88d36faa81f32a4d84131543e213dd58f762852589d21`;
`final_recheck.json` SHA-256 is
`925b41d7731f9a107bdde33f9c62370171ec9d4d71535787e7653d80d34aac8a`.
The committed results above remain the durable evidence even without those
ignored outputs. The two exact pinned Person 1 fixture copies were byte-checked
against their Git blobs before staging.

Final validation, existing Python 3.14.8 environment without dependency changes:

- Focused safety/affect/metrics/report suite: **120 passed in 5.83 s**. The frozen
  scalar reference is independent of current scoring helpers; absolute tolerance
  is 1e-14 with relative tolerance disabled.
- Full affected GoEmotions/affect/campaign/common/integration run: **276 passed,
  1 failed in 72.22 s**. The sole failure is the unchanged worktree notebook
  child-identity issue already reproduced on the clean base in B1.
- Final broader recheck: **276 passed, 1 explicitly deselected in 66.66 s**;
  only that already-confirmed notebook test was excluded. Its failure remains
  recorded rather than presented as a passing check.
- Scoped Ruff over affect, GoEmotions, common, campaign and affected tests: **all
  checks passed**. `git diff --check --ignore-submodules=all`: **passed**.
- Scoped strict-package mypy: **success, no issues in 5 entry files**. Its initial
  split-validation `object.get` finding was fixed with an explicit dictionary/
  string boundary; no task-caused check failure remains.
- Notebook and Person 1 producer/config paths have no diff. The B1 checkpoint
  remains the predecessor; no unrelated tracked changes were included.

Commands actually executed (use the same `$py` interpreter defined above):

```powershell
& $py -m evaluation.goemotions.smoke.b2_profile.profile
& $py -m evaluation.goemotions.smoke.b2_profile.final_recheck
& $py -m pytest -q tests/test_goemotions_benchmark.py tests/test_goemotions_upgrade.py tests/test_goemotions_review_regressions.py tests/affect tests/benchmarks/test_goemotions_metrics_reporting.py --tb=short
& $py -m pytest -q tests/test_goemotions_benchmark.py tests/test_goemotions_upgrade.py tests/test_goemotions_review_regressions.py tests/affect tests/benchmarks tests/integration --tb=short
& $py -m pytest -q tests/test_goemotions_benchmark.py tests/test_goemotions_upgrade.py tests/test_goemotions_review_regressions.py tests/affect tests/benchmarks tests/integration --deselect 'tests/benchmarks/test_kaggle_smoke.py::test_notebook_smoke_stages_native_artifacts_and_partial_packages[goemotions]' --tb=short
& $py -m ruff check affect benchmarks/goemotions benchmarks/common benchmarks/campaign tests/test_goemotions_benchmark.py tests/test_goemotions_upgrade.py tests/test_goemotions_review_regressions.py tests/affect tests/benchmarks/test_goemotions_metrics_reporting.py tests/fixtures/goemotions_review/bootstrap_reference.py
& $py -m mypy benchmarks/common/reporting.py benchmarks/common/timing.py benchmarks/campaign/provider_session.py benchmarks/campaign/orchestrator.py benchmarks/campaign/aggregation.py
git diff --check --ignore-submodules=all
```

Goal 3's prerequisite is this successful checkpoint with the frozen statistic,
eligible-population rules, raw metric precision and policy/report identities.
No prompt, threshold or correction-policy selection was performed here. Final
notebook consumers and actual combined execution with Person 1's pinned final
head remain Goal 4 acceptance work; the available B2 fixture consumers are done.

The known worktree-only notebook child-identity issue from B1 remains assigned
to Goal 4. Full CI/Kaggle Python, GPU resource/time gates, account quota, fresh
inference, trained-model quality and final notebook migration are not inferred
from local fake-provider tests or offline metric profiling. No push, PR, merge
to `dev`, Goal 3 tuning or full production run is part of B2.

## Goal 3: frozen development comparisons and CPU baseline

Delivery branch: `dev-p2`; successful predecessors are Goal 1
`e0dd105484a2f78fc7a13811c83ef4e04aaad440` and Goal 2
`def8f76dbe25699e63fd10cedcb9a6afe311ac93`. B3 was implemented in the
isolated `goemotions-b3` worktree from the exact Goal 2 head. The historical
planning base remains `c25a178d6308bc48b3a5e44d9519316a5179c499`.
Status: **Goal 3 implementation acceptance complete; candidates unpromoted**.
This is the third checkpoint, not Goal 4 or measured release acceptance.
Its final commit SHA belongs in the completion response.

### Requirement and evidence map

| B3 requirement | Implementation / evidence |
|---|---|
| Official labels, split identities, repeated texts | `dataset.py` enforces exact official order and valid nonnegative indices; `training/data.py` validates unique/disjoint IDs and records raw hashes, counts, label-file hash, repeated-text rows and cross-split exact-text overlap |
| Fixed dev population before scores | `development.freeze_development()` freezes rarest-label proportional-deficit stratification, seed 13, immutable 512 IDs/text hashes/gold, dataset identity and criteria in `docs/goemotions-development/` |
| Three bounded prompts | `prompts.py` has definitions/schema baseline, expressed-evidence and taxonomy-distinction variants; mixed neutral remains valid; no few-shot examples or test tuning |
| Exact request/cache and comparison identity | Existing bounded-provider cache already keys exact prompt/system/schema/GenerationConfig; tests verify separate calls for changed variants/text/model/decoding/schema and reuse only exact requests |
| Separate inference versus same-response policy | `compare_development()` checks frozen dev IDs/order/text/gold, raw-response hashes, prompt/model/schema/generation identities and request/cache provenance; same-response mode rejects changed prompts/raw hashes and parse failures; trained comparisons have a separately attributed frozen artifact identity |
| Actual CPU baseline and freeze | Existing TF-IDF/logistic pipeline fits train only and thresholds on dev; word/unbalanced/seed13 is the sole preregistered real trial; all 5,426 dev rows were scored and frozen reload metrics matched exactly |
| Fix incorrect frozen feature declaration | `freeze_tfidf_candidate()` copies the exact fitted artifact instead of always refitting combined features; mismatching features/balance/seed are rejected; word/char/combined synthetic paths are covered |
| Freeze/load integrity | Artifact manifest binds complete split/label identities, feature settings, logistic parameters, Python/package versions, model and threshold hashes; checks run before joblib unpickling; dev/model/config threshold binding and externally pinned fingerprint are supported |
| Optional encoder | Explicitly not admitted: no matching live T4 allocation, account quota, sequence smoke or validation/export pilot; existing pinned revision/config recorded; no encoder training/calibration/load-integrity claim |
| Optional correction | Explicitly not admitted; no allowed edits; no calibrated per-label evidence or integrated cost; preserve-label gain is the tested zero-gain invariant, not a fresh Qwen measurement |
| Gates, ledger and Goal 4 handoff | Frozen criteria require higher micro F1, nondecreasing macro F1 and zero parse-rate tolerance; paired sample-F1 uncertainty, exact set, precision/recall/cardinality and separate correction populations are reported; compute ledger and exact safe candidate identity remain unpromoted |

All new deterministic regression coverage is in
`tests/benchmarks/test_goemotions_development.py`. No Person 1 producer/config,
external submodule/data, canonical notebook, dependency or CI file was changed.
The original checkout's unrelated documents/directories were preserved.

### Immutable data and preregistration

The four supplied plan documents were read; the source plan's B3 section is
implemented under the newer four-checkpoint operating contract. Existing CLI
help was inspected for experiment, classical training and optional training
before experiments. Existing training/freezing/threshold helpers were reused;
no new dependency, encoder framework, GPU training or Cartesian search was added.

Available data was read unchanged from the preserved local backup:
`C:\Users\galad\OneDrive\Desktop\Capstone\code\PRIMA_next\.local-sync-backup-20260831\benchmarks\goemotions\external\goemotions\data`.
The backup was neither modified nor copied into tracked source. Counts are
train **43,410**, dev **5,426**, test **5,427**; all IDs are unique within and
disjoint between splits. Test was read for identity/count/overlap validation,
never fitted, scored or used for candidate selection. Prior test exposure is
explicit in the manifest.

| Input | SHA-256 |
|---|---|
| train.tsv | `a8336fed4b2f651b11c6a6c65e4dab122fa9da28f49b200a892c71094eeba109` |
| dev.tsv | `8f629aa53c1b1d6dd023b168e3762a09f4d1f4e575fcd9035c02bd74170ef56b` |
| test.tsv | `dc3aab545d76079817c3002a77239d53accc11ec8f16313aa19d8e1a8f2e412e` |
| emotions.txt | `932218ae8b67b25341eec30f785580207e952fa8eb2c917807ad5eb439a8f901` |

Exact-text overlaps count distinct shared text strings: train/dev **41**,
train/test **32**, dev/test **10**. Repeated rows beyond unique texts are
train **183**, dev **3**, test **6**. None were removed or merged.

`dev_manifest.json` fingerprint:
`a52f38cb539216e2e70f51c9a2b23ff7cac27ffd2a428af3ab555099af7be172`.
`selection.json` fingerprint:
`e650c545e01e57cdf452ed55ec342c41b356b3bd9058aa9624326592598cacf2`.
The latter was written before CPU scoring; the driver start record pins it.
The frozen 512 IDs are the future prompt population. CPU comparisons also
expanded to all dev, retaining the same predeclared criteria. Raw dev prediction
and ID-sequence hashes are retained in `cpu_evidence.json`.

The safe candidate is existing `affect_telemetry_preserve_labels`, policy
`preserve_labels_v1`, prompt `definitions_schema`, matched `qwen3.8:27b` digest
`aaee06c39dcf2437cde036998d960e1fc1494b8191be7cc9657d01e509097813`,
context 8,192, output 512, seed13, temperature0, schema output, one worker and
full GPU residency required. Production/test execution remains disabled.
`candidate_status.json` pins its exact configuration and source files (UTF-8/LF normalization, matching
Git text blobs so Windows/Linux checkout line endings do not alter identity);
its fingerprint is
`b1cedd430c8fd341208fefb1500d271ce95cb65c746856d0358828cb677f2013`.
The final source pin includes a protocol annotation added after scoring; that
annotation changes no prompt, prediction, model, threshold or selection gate.

### Actual CPU measurements and decisions

Environment: existing Python **3.14.8**, package versions frozen in
`classical_artifact_manifest.json`; OMP/OpenBLAS/MKL thread limits were each 1.
Real trial: word TF-IDF, word unigrams/bigrams, min_df2, max_features80,000,
sublinear TF, unbalanced one-vs-rest liblinear logistic regression, max_iter200,
seed13. All features/coefficients are fitted from train; thresholds use dev
per-label F1 on the existing 0.10–0.90 grid in steps of 0.05, minimum support5.
No alternate real feature/balance trial was added after observing scores.

| Full dev metric | Global 0.5 thresholds | Dev-selected thresholds |
|---|---:|---:|
| Micro precision | 0.4791931674 | 0.5063643596 |
| Micro recall | 0.4133228840 | 0.5985893417 |
| Micro F1 | 0.4438273163 | 0.5486280707 |
| Macro F1 | 0.2062097772 | 0.4193352866 |
| Sample F1 | 0.4570708932 | 0.5633299984 |
| Exact-set accuracy | 0.4122742352 | 0.3903427940 |
| Prediction cardinality | 1.0141909325 | 1.3899741983 |

Paired full-dev sample-F1 delta **0.1062591052**, 1,000-resample seeded interval
**[0.0951362926, 0.1178988293]**. The frozen 512-case comparison also passes the
metric gate: delta **0.0962890625**, interval **[0.0575520833, 0.1311848958]**.
There are 2,641 changed predictions: 527 exact-set improvements, **646 exact-set
regressions**, and 2,785 unchanged. The exact-set loss is retained, not hidden.
Parse failures are inapplicable to this non-text trained predictor; no parser
improvement is claimed. Full metrics/per-label distributions remain in the
ignored result and dev records.

These are **dev threshold-fit gains**, not independent validation or evidence
that the classifier beats Qwen. Bootstrap resampling of the same threshold-fit
dev rows does not account for model/threshold selection optimism. The trained
candidate is frozen but **unpromoted**; it provides no calibrated correction
scores. No fresh Qwen prompt result, helpful PRIMA correction, encoder result or
production-default improvement is claimed. Exact-set loss is a review concern.

Fit plus initial dev scoring/threshold analysis took **93.406 s** before the
batch-loop repair. A subsequent reload and concurrently running broader tests
stalled with live processes, flat CPU, no artifact progress and waiting loopback
sockets. Only those task processes were stopped; the completed fit and hashes
were retained. The exact cause of the Windows socket stall was not established.
Those incomplete checks are not reported as successes or measured full timings.

The existing CPU helper created one asyncio loop per row. It now uses one loop
per batch, retaining a separate runtime/affect state per independent example.
New tests prove both the single-loop behavior and absence of emotional carryover.
The repaired frozen reload scored **all 5,426 dev IDs**, reproduced every full
metric exactly, and took **34.0458719000 s**. Sampled process RSS peaked at
**405,929,984 bytes**, cadence target0.1s. This is current-process RSS only;
fit peak memory is unavailable because its monitor was interrupted. Neither is
a system/cgroup/Kaggle memory gate. No speedup ratio or GPU-feasibility claim is
made from these differently scoped timings.

### Frozen artifacts, compatibility and optional decisions

The trained artifact fingerprint is
`99ab7a9a3afee350e3ef192a8be05b930605613c6b135da079db42688c8a8376`.
Model SHA-256:
`00622b00061d4a3dbabc9b883e3c3b0ab1d92923b9df7bf5596740880e15f752`;
threshold SHA-256:
`857e9a6e99a819eb0962ad847c159d6566e9a8f07435fee41573e8a05b36c5d5`.
The exact fit and frozen model files are byte-identical. Weights (~15.4 MB each),
full outputs and diagnostics are ignored, not committed. Durable compact evidence,
selection, ledger, artifact manifest and candidate status are committed under
`docs/goemotions-development/`.

`freeze_tfidf_candidate()` retains its parameters with optional `source_model`,
but now requires a fitted companion model and model-bound dev thresholds.
Legacy unbound artifacts must be rebuilt; no unverifiable legacy model is silently
accepted. `load_frozen_model()` adds optional dataset/threshold/fingerprint
checks. `run_frozen_tfidf()` and CLI support `expected_fingerprint` /
`--expected-fingerprint`; pin it when moving the candidate. Checksums do not make
untrusted pickle files safe: these are trusted local artifacts.
`evaluate_frozen_tfidf()` retains its public test-scoring API and settings but
loads the matching frozen fit instead of refitting. It was not invoked on real
test data in B3.

The prompt default is byte-identical to Goal 2 (eight old/new combinations checked
against its committed source and a frozen regression hash). New variants are
keyword/default-compatible and recorded in runtime metadata/experiment identity;
changed variants reject stale resumes. New sealed dev manifests are verified
before any request. Legacy sample manifests remain readable under their earlier
validation semantics. Classifier decisions receive only text/taxonomy, never gold
or split fields; dev metadata is attached for evaluation afterward.

Local GPU inspection observed **RTX 3050 Laptop, 4,096 MiB**, not the required
T4 allocation; localhost Ollama tags timed out. No model was loaded or downloaded.
Live Kaggle quota/session information is unavailable. DeBERTa-v3-small remains
not admitted with existing pinned revision
`a59be8aa63396e73dbb45a1487e4cde4be98bfa4`, max_length128, ceiling3 epochs.
A future admission requires a documented need after the cheap baseline, Qwen
unloaded, actual sequence-length smoke, memory/step/validation/export timing and
an independent bounded session. No optional training, temperature/threshold
calibration or encoder load-integrity validation was performed or claimed.
The existing encoder utilities remain unchanged; this checkpoint does not certify
them for an admitted encoder campaign.

Replacement correction remains not admitted, with **no allowed edits**. Actual
calibrated scores, immutable training/calibration/revision identities, predeclared
label edits/margins, neutral-only/mixed/other paired analyses, positive mean and
95% lower bound above zero, acceptable per-label harm and integrated cost are all
required before a later proposal. A trained classifier or dev threshold win alone
cannot authorize it. Preserve-label correction gain remains zero by invariant.

### Validation and next checkpoint

Completed commands use the existing root `.venv\\Scripts\\python.exe` from the
isolated worktree; no dependency changes or downloads:

```powershell
& $py -m benchmarks.goemotions.experiment --help
& $py -m benchmarks.goemotions.training.classical --help
& $py -m benchmarks.goemotions.training.cli --help
& $py -m evaluation.goemotions.smoke.b3_cpu.driver
& $py -m evaluation.goemotions.smoke.b3_cpu.reload
& $py -m pytest -q tests/benchmarks/test_goemotions_development.py tests/test_goemotions_benchmark.py tests/test_goemotions_upgrade.py tests/test_goemotions_review_regressions.py tests/affect tests/benchmarks/test_goemotions_metrics_reporting.py --tb=short
& $py -m pytest -q tests/test_goemotions_benchmark.py tests/test_goemotions_upgrade.py tests/test_goemotions_review_regressions.py tests/affect tests/benchmarks tests/integration --deselect 'tests/benchmarks/test_kaggle_smoke.py::test_notebook_smoke_stages_native_artifacts_and_partial_packages[goemotions]' -o faulthandler_timeout=45 --tb=short
& $py -m pytest -q tests/benchmarks/test_goemotions_development.py --tb=short
& $py -m pytest -q tests/benchmarks/test_goemotions_development.py tests/test_goemotions_upgrade.py --tb=short
& $py -m ruff check affect benchmarks/goemotions benchmarks/common benchmarks/campaign tests/test_goemotions_benchmark.py tests/test_goemotions_upgrade.py tests/test_goemotions_review_regressions.py tests/affect tests/benchmarks/test_goemotions_development.py
& $py -m compileall -q benchmarks/goemotions
git diff --check --ignore-submodules=all
```

The CPU driver command produced a completed train/dev fit, then was interrupted
during reload; the second command is the successful verified reload. Focused
safety/affect/report regressions passed **139 tests in 20.65 s** before the final
boundary additions. The final broader affected suite passed **307 tests with
1 explicitly deselected in 87.57 s**. All then-current 31 B3 tests were included.
After adding delivery-manifest and baseline-byte checks, **33 B3 tests passed in
19.15 s**. The final protocol-only/source-pin adjustment was rechecked separately
below. Ruff, compileall and diff whitespace checks passed.

The sole broader exclusion remains the pre-existing worktree notebook child-Git
identity test documented/reproduced in B1/B2; it is assigned to Goal 4, not claimed
fixed here. No full-repository CI, Kaggle Python, full test inference or real GPU
resource/latency validation is claimed. Person 1 completion remains the user's
status; its actual producer-consumer integration is a Goal 4 requirement.

Goal 4 must consume the exact safe unpromoted identity and frozen evidence,
finish notebook/campaign/report consumers and safeguards, and validate its entire
proposed snapshot with Person 1's pinned final checkpoint in a separate checkout.
It must resolve the known notebook identity issue, complete required wider checks,
and keep full execution disabled pending reviewed merged source, fresh matching
resource/time pilots and live quota. Qwen dev comparisons and independent full
5,427-ID test evaluation remain pending; all future production sessions must keep
the original digest/context/coverage and hard resource/reserve limits.
No push, PR, merge to `dev`, Goal 4 work or full production run belongs to B3.
Rollback is a scoped revert/new explicit run identity; do not reuse incompatible
manifests, reset the user's checkout, or discard negative evidence.

Final post-annotation consumer/source checks: **50 passed in 19.26 s**.
Two final identity/baseline checks passed in **1.01 s** after making source hashes
explicitly UTF-8/LF-normalized to match Git text blobs across platforms. Selection
criteria, data identities, prompts and model/threshold bytes were unchanged.
The exact original checkpoint/ledger files were validated again before staging.
Local ignored model/evidence directories are preserved in the original checkout:
`evaluation/goemotions/models/b3_word_unbalanced`,
`evaluation/goemotions/models/b3_word_frozen`, and
`evaluation/goemotions/smoke/b3_cpu`. No existing destination was overwritten.
