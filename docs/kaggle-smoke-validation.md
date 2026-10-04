# Three-benchmark Kaggle smoke validation

`PRIMA_Kaggle_Benchmark.ipynb` defaults to `SMOKE_MODE=True`,
`SMOKE_ITEMS=2`, and `RUN_FULL_BENCHMARK=False`. Each benchmark uses two
unique examples, evaluated again by the pilot. This means four example
evaluations across the two stages, with only two unique IDs. LoCoMo uses
one question from each of two seed-selected conversations and preserves
their entire history replay. Selecting three smoke examples is the hard
upper bound. No training is launched.

## Exact Kaggle steps

1. Upload the edited notebook to a Kaggle notebook with Internet enabled and
   **two T4 GPUs**. Add these dataset inputs:
   [LoCoMo](https://www.kaggle.com/datasets/thegifman/prima-locomo),
   [HotpotQA](https://www.kaggle.com/datasets/thegifman/prima-hotpot-qa), and
   [GoEmotions](https://www.kaggle.com/datasets/thegifman/prima-goemotions).
2. The edited repository changes are published to `3.10_to_3.14`. Setup clones only
   that branch and records its fetched commit for the session; running an
   uploaded notebook against the unchanged remote branch will omit the
   required campaign helpers. Do not change an existing full-run checkout.
3. Provide an **already available** Ollama cache containing the configured
   `qwen3.8:27b` model. Set `OLLAMA_MODELS_PATH` to that cache and verify
   `EXPECTED_MODEL_DIGEST`. If the configured default has no cache, the
   notebook can discover a single standard cache under `/root/.ollama`,
   `/kaggle/working`, or the attached `/kaggle/input` inputs. Multiple caches
   require an explicit path. Smoke never calls `ollama pull`; missing models
   or required ancillary Hugging Face resources block the run. The smoke
   process sets HF offline flags. Attached caches use a writable metadata
   mirror with links to immutable blobs, avoiding a full weight copy.
   Native default dependencies come from `requirements-kaggle.txt`; optional
   model backends/BERTScore explicitly use the full requirements list.
   The default model digest is the available 2026-10-05 27B revision. Set the
   old expected digest explicitly when using the original cached snapshot;
   never mix the two model revisions in one reported experiment.
4. Keep `SMOKE_MODE=True`, `SMOKE_ITEMS=2`, `RUN_FULL_BENCHMARK=False`,
   `RUN_PAIRED_BASELINE=False`. Select `BENCHMARK_SELECTION=1`, `2`, or `3`.
   Dataset discovery searches the supplied slug in the actual mount, including
   both `/kaggle/input/<slug>` and `/kaggle/input/datasets/thegifman/<slug>`;
   nested data files work, and ambiguous candidates fail closed.
5. `EXPECTED_DATASET_SHA256[benchmark]` can enforce a separately supplied
   expected digest. LoCoMo's existing pin remains required and enforced.
   HotpotQA/GoEmotions instead freeze the validated selected Kaggle input's
   SHA before the canary when the optional expected digest is absent. The
   report records this snapshot policy; every stage must match its original
   source IDs/count/hash and GoEmotions split hashes. Unpinned external files
   cannot be used for full runs. Never copy LoCoMo's digest into another benchmark.
6. Run sections **1 through 9** in order (setup, hardware inspection, model
   preflight, dataset validation, canary and tiny pilot). **Stop after 9.**
   Section 10 rejects smoke even if the full-run flag is enabled. Use a new
   kernel/session for each selector so cache/offline settings and clocks are
   unambiguous. Reuse the cached model between sessions.
7. Inspect `smoke_results.json` and both `*-PARTIAL-NONREPORTABLE.zip` files
   under `/kaggle/working/prima_outputs/smoke/<benchmark>-<session>/`.
   Download these via Kaggle's output browser. Any exception is a failed or
   blocked stage, never a pass. A failed pilot is recorded in the results
   file and must be investigated before accepting concurrency.

LoCoMo's pilot uses the existing isolated two-conversation worker path.
It must preserve native predictions/evaluation decisions and actually show
two overlapping provider requests. HotpotQA and GoEmotions now exercise
their existing isolated native two-worker paths too. HotpotQA compares
answers, supporting facts and reasoning decisions; GoEmotions compares raw
model output, baseline/final labels and bounded affect decisions. The real
pilot must preserve those results and pass resource/overlap checks before
concurrency can be accepted; the implementation alone is not validation.

## Validation and measurements

Campaign smoke configuration rejects unbounded or greater-than-three
selections, extra baseline modes, and resume. Smoke root manifests remain
`partial` even when all selected child items finish; the native child
`complete` status means only its small selection finished. Aggregate/report
markers, validation JSON, archive names and isolated paths identify the
package as non-reportable. Full selected-source coverage and the notebook's
full-run guards prevent smoke packages from qualifying as full results.

Each stage records setup, Ollama model-load duration, inference wall time,
elapsed campaign time, per-GPU sampled peaks, system RAM and process-tree
RSS peaks, maximum active requests, selected IDs and artifact status.
Model load/warmup also has resource sampling. Peaks are observations from
polls at most two seconds apart; brief peaks between polls can be missed.
The canary/pilot monitor retains the existing GPU placement/context/digest,
12 GiB/GPU, 19 GB RAM planning cap and 3 GB reserve checks. Complete LoCoMo
history replay can still dominate smoke runtime despite the tiny QA count.

The package verifies native manifests, summary/checkpoint/prediction/failure
schemas, metrics, native diagnostic outputs and GoEmotions figures, reads
every JSON/JSONL artifact strictly, scans for secrets, and checks ZIP
integrity and required members. Packaging does not disable scoring or
replace inference/evaluation. The actual Kaggle run always uses Ollama.

Full-run artifact validation now compares **every mode** against independently
selected source IDs, full dataset count and hash, complete checkpoint
statuses, repository/model identity and GoEmotions validation/test hashes.
Missing, duplicate, unexpected, truncated or unsuccessful coverage fails
closed. Existing metric, post-evaluation, diagnostic, resource and archive
checks remain. The representative timing/full-run path is selected only
with `SMOKE_MODE=False` in a new session.

## Full-run readiness for each benchmark

Use **one separate Kaggle session per benchmark**, as requested. The 12-hour
session limit applies independently to each run; the notebook selects one
benchmark per session. A smoke pass does not provide production timing evidence.

For each selector, start a new session with `SMOKE_MODE=False`,
`RUN_FULL_BENCHMARK=False`, the pinned model digest and selected Kaggle inputs. The fetched
`3.10_to_3.14` branch commit is recorded and a fresh `FULL_RUN_ID` is automatic.
Run sections 1–9 to obtain representative fresh timing:

| Selector | Representative canary | Full selection | Scheduling |
|---|---|---|---|
| 1: LoCoMo | Three conversations, five questions each; complete histories | All source conversations and questions; 1,986 required | Sequential or validated faster two-worker pilot |
| 2: HotpotQA | 20 seed-sampled questions | Every question in the pinned source | Sequential or validated faster two-worker pilot |
| 3: GoEmotions | 50 seed-sampled test examples | Every test example in the pinned source; validation split checked separately | Sequential or validated faster two-worker pilot |

Review `timing-<session>.json` and `budget-<session>.json`. Section 10 requires
the measured estimate to fit 10.5 campaign hours and the entire session,
including elapsed setup/pilots and artifact reserve, to fit 12 hours. It
rechecks clean repository identity, dataset/model digests, unchanged measured
configuration, full source IDs and live resources before launching. Only after
these checks pass, enable `CONFIG['RUN_FULL_BENCHMARK'] = True` immediately
before section 10. Do not rerun configuration or reset the clock.

Polling and budget limits are validated before model warmup. Live monitoring
stops a full campaign before its campaign/session limit, retaining at least
900 seconds of artifact headroom. Invalid/nonfinite RAM or GPU samples fail
closed. An unsafe final resource sample also stops Ollama and saves the
failure. These guards prevent an unsafe continuation; they do not prove that
the full workload will finish in time.

## Execution evidence from this workspace

Inspection started on branch `3.10_to_3.14`, HEAD
`3a8698294c11d2c9e7870216f4436422686f63c2`. The pre-existing dirty
`benchmarks/locomo-v2/external` submodule was preserved. This is a local
Windows workspace; `/kaggle/input` is absent and no actual Kaggle runtime
or supplied mounted datasets/cache are available to this task.

| Benchmark | Actual Kaggle stages | Real examples | Actual artifact checks | Elapsed / GPU / RAM | Actual concurrency | Full-run decision |
|---|---|---:|---|---|---|---|
| LoCoMo | Setup/preflight/canary/pilot blocked | 0 | Not run | Unmeasured | Unmeasured | **NO-GO** |
| HotpotQA | Setup/preflight/canary/pilot blocked | 0 | Not run | Unmeasured | Unmeasured; two-worker candidate configured | **NO-GO** |
| GoEmotions | Setup/preflight/canary/pilot blocked | 0 | Not run | Unmeasured | Unmeasured; two-worker candidate configured | **NO-GO** |

Local checks use tiny fixtures with fake inference and synthetic hardware;
they validate native selection, scoring, checkpoints, packaging and guards,
and cannot establish real model accuracy, latency, memory use or concurrency.
The one-turn LoCoMo fixture can legitimately abstain without calling the
provider; it is no evidence of a successful concurrent model pilot.

The final combined focused run passed **56 checks in 32.76 seconds**, exercising
all three native artifact paths, full-data configuration/launch guards,
incomplete coverage rejection and resource/budget limits. The two Windows
process-signal/resume checks that initially timed out before their first
checkpoint subsequently passed unchanged in 31.66 seconds. Ruff passed on the
modified campaign code/tests. Mypy passed on all six modified campaign source
files with `--follow-imports silent`; native selector import aliases were fixed
after type checking found an incompatible reused local name.
The added final-sample cleanup assertion passed in a focused follow-up
(one test, 6.69 seconds). All 19 notebook code cells parse with no saved outputs,
and `git diff --check` passes.

Reproduce the focused checks with the existing repository environment:

```powershell
.venv\Scripts\python.exe -m pytest tests/benchmarks/test_kaggle.py tests/benchmarks/test_kaggle_smoke.py -q -p no:cacheprovider
.venv\Scripts\python.exe -m ruff check benchmarks/campaign tests/benchmarks/test_kaggle.py tests/benchmarks/test_kaggle_smoke.py
```

No full campaign, model/dataset download, or heavy training run was launched.
A tiny smoke pass alone cannot establish that any full benchmark fits within
12 hours. Production still requires pinned identities, representative fresh
measurements, the existing resource/budget gates, and full artifact validation.
