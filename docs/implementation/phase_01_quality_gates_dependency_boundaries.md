# Phase 01 — Quality Gates and Dependency Boundaries

## Outcome

Phase 01 makes repository checks trustworthy without starting the runtime architecture refactor.

- Python, Ruff, mypy and CI now target Python 3.12.
- Mypy no longer uses `ignore_errors`; orchestration/runtime and new modules are strict, with documented legacy/external boundaries.
- Dependencies are split into core runtime, development, benchmark core, optional semantic metrics and encoder/training groups.
- LoCoMo uses an internal deterministic ROUGE-L F-score.
- BERTScore is disabled by default, enabled only by `--bertscore`, and reports `disabled`, `available` or `unavailable: <module>`.
- `python -m benchmarks.preflight [--json]` reports optional capabilities without imports, installs or downloads.
- CI runs Python 3.12 compileall, pytest, Ruff, mypy, Bandit and the existing Xenon check.
- Cache ignore/tracking behavior is executable through a repository hygiene test.

No task routing, workflow ownership, retrieval, model execution or benchmark gold data changed.

## Contracts

### Dependencies

| Group | File | Default install |
|---|---|---|
| Core runtime | `requirements-core.txt` | Yes |
| Development | `requirements-dev.txt` | Yes |
| Benchmark core | `requirements-benchmark.txt` | Yes |
| Optional semantic metrics | `requirements-semantic-metrics.txt` | No |
| Encoder/training extras | `requirements-encoder.txt` | No |

`requirements.txt` composes the three default groups. `requirements-goemotions.txt` remains a compatibility alias for the encoder group.

### Typing

Mypy uses `pyproject.toml` as its only configuration source. Strict modules are `action`, `events`, `llm`, `planning`, `reasoning`, `runtime`, `security`, `state`, `tools`, `uncertainty`, `workflow`, `world`, and the new benchmark preflight module.

Verified legacy benchmark/evaluation/test and standalone validation trees are excluded from this staged gate. Affect, memory and reflection retain narrowly documented legacy options for NumPy, Chroma and plugin boundaries. The only disabled error codes are scoped to the dynamic Chroma payload and optional Pydantic base-class modules. No `ignore_errors`, blanket type ignore or skipped test was added.

### Metrics and preflight

ROUGE-L uses normalized tokens and a two-row longest-common-subsequence algorithm, returning the harmonic F-score. It requires no optional package.

LoCoMo’s default metric payload now contains:

```json
{"bertscore": null, "bertscore_status": "disabled"}
```

When requested but unavailable, it contains `"bertscore_status": "unavailable: bert_score"` instead of raising an import error. Preflight JSON has schema version `1.0`.

## Files changed

- `.github/workflows/ci.yml`
- `README.md`
- `benchmarks/preflight.py`
- `benchmarks/locomo/evaluate.py`
- `benchmarks/locomo/experiment.py`
- `memory/__init__.py`
- `memory/embedding_backend.py`
- `pyproject.toml`
- `reasoning/config.py`
- `reasoning/stopping_policy.py`
- `requirements.txt`
- `requirements-core.txt`
- `requirements-dev.txt`
- `requirements-benchmark.txt`
- `requirements-semantic-metrics.txt`
- `requirements-encoder.txt`
- `requirements-goemotions.txt`
- `runtime/prima_runtime.py`
- `security/input_sanitizer.py`
- `security/output_validator.py`
- `tests/test_locomo_qa_correctness.py`
- `tests/benchmarks/test_preflight.py`
- `tests/test_repository_hygiene.py`
- `docs/implementation/STATUS.md`
- `docs/implementation/phase_01_quality_gates_dependency_boundaries.md`
- Removed duplicate `mypy.ini`.

The existing `codex_rules.txt`, Phase 00 documents and dirty `benchmarks/locomo-v2/external` submodule predate Phase 01 and remain preserved.

## Commands and exact results

```text
venv\Scripts\python.exe -m pytest -q tests\test_locomo_qa_correctness.py tests\benchmarks\test_preflight.py tests\test_repository_hygiene.py
Initial characterization: collection failed with ModuleNotFoundError: benchmarks.preflight.
After implementation: 8 passed in 63.10s.

C:\Users\Yuv Nahar\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe --version
Python 3.12.13

C:\Users\Yuv Nahar\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m compileall -q -x '(^|[\\/])(\.git|\.venv|venv|external)([\\/]|$)' .
Exit 0; no output.

venv\Scripts\python.exe -m pytest -q
215 passed in 76.72s (0:01:16).

venv\Scripts\python.exe -m ruff check .
All checks passed.

venv\Scripts\python.exe -m mypy .
Success: no issues found in 202 source files.

venv\Scripts\python.exe -m bandit -c bandit.yaml -r .
No issues identified; 22,708 lines scanned; 14 disabled checks; 0 files skipped.

venv\Scripts\python.exe scripts\run_xenon.py
Exit 0.

venv\Scripts\python.exe -m benchmarks.preflight --json
Exit 0; schema_version 1.0; rouge_l, bertscore and goemotions_encoder available in the current environment.

venv\Scripts\python.exe -m pytest -q tests\reasoning tests\test_locomo_qa_correctness.py
20 passed in 6.39s.
```

## Risks and deferred work

- The complete pytest suite ran locally on the existing Python 3.10.11 venv; Python 3.12.13 compiled all owned sources. The new CI job is the authoritative full Python 3.12 dependency/test run.
- BERTScore may download model assets when a user explicitly installs its group and passes `--bertscore`; smoke tests never enable it.
- Strict typing for legacy affect, memory, reflection, benchmark, evaluation and test boundaries remains staged and documented rather than hidden by `ignore_errors`.
- Ruff’s `UP017`, `UP042` and `UP046` migrations are deferred only for exact legacy files because they require behavior or local-runtime compatibility review. Five broader exemptions were removed/narrowed in this phase.
- Stop here. Runtime boundary refactoring remains `ARCH-001` and was not started.
