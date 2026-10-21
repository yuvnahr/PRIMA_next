# GoEmotions and Kaggle remediation handoff

## Goal 1: safe, auditable preserve-label classification

Delivery branch: `dev-p2`. Actual implementation base/predecessor:
`c25a178d6308bc48b3a5e44d9519316a5179c499`, matching the planning snapshot.
Status: **Goal 1 implementation acceptance complete**; Goals 2–4 and measured
benchmark/release acceptance remain pending.
Goal 1 is the first checkpoint; its own commit SHA belongs in the completion
message. Work was isolated in the attached `goemotions-b1` worktree because
the user's checkout contained unrelated untracked documents/directories.
No Person 1 files, external submodules, notebook, dependencies or CI were edited.
No push, PR, merge to `dev`, training or production campaign is authorized by this checkpoint.

The four-goal workflow supersedes the source plan's intermediate PR/merge order.
Only B1 is implemented here. Goals 2–4 remain pending, including final integration
against Person 1's completed pinned checkpoint. Person 1 completion is supplied
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
