# PRIMA-NEXT

PRIMA-NEXT is an experimental Python runtime for building AI agents that can remember, reason, plan, reflect, and act through controlled tools.

In simple terms: a request enters one runtime, the runtime chooses the needed cognitive components, and it returns a typed answer with diagnostics. The project is designed for local research, reproducible benchmarks, and safe extension.

## What it includes

- Long- and short-term memory with dense, sparse, temporal, and graph retrieval.
- Emotion-aware state and retrieval signals.
- Planning, uncertainty checks, world simulation, and bounded reflection.
- Provider-backed language-model calls with structured output.
- Policy-gated tools with validation, timeouts, and audit records.
- HotpotQA, LoCoMo, and GoEmotions benchmark integrations.

PRIMA-NEXT is research software. It is not yet a hosted assistant or a production service.

## Setup

Requirements:

- Windows PowerShell
- Python 3.10 or newer
- Git

From the repository root, run one command:

```powershell
.\setup.ps1
```

The script creates `.venv`, installs the development and benchmark dependencies, initializes Git submodules, and creates a local `.env` from `.env.example` when needed. It never overwrites an existing `.env`.

If PowerShell blocks local scripts, use:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

## Quick start

Activate the environment and run the small affect demo:

```powershell
. .\.venv\Scripts\Activate.ps1
python main.py
```

Use the canonical runtime from Python:

```python
import asyncio

from runtime import ExecutionOptions, ExecutionProfile, PrimaRequest, PrimaRuntime, TaskKind


async def main() -> None:
    response = await PrimaRuntime().execute(
        PrimaRequest(
            task_kind=TaskKind.FACTUAL_QA,
            profile=ExecutionProfile.SIMPLE_RAG,
            input_text="What should I remember?",
            options=ExecutionOptions(max_retrieval_calls=1),
        )
    )
    print(response.output_text)


asyncio.run(main())
```

Configuration values and provider examples are documented in `.env.example`. The runtime has deterministic local fallbacks, so the test suite does not require a model download or API key.

## Development

Run the same quality checks used by CI:

```powershell
python -m benchmarks.preflight --json
python -m compileall -q -x '(^|[\\/])(\.git|\.venv|venv|external)([\\/]|$)' .
python -m pytest -q
python -m ruff check .
python scripts/run_xenon.py
python -m mypy .
python -m bandit -q -c bandit.yaml -r .
```

Optional encoder/training dependencies live in `requirements-encoder.txt`; optional semantic metrics live in `requirements-semantic-metrics.txt`. Benchmark instructions are in [benchmarks/README.md](benchmarks/README.md), and operational details are in [docs/benchmarks/RUNBOOK.md](docs/benchmarks/RUNBOOK.md).

## Project structure

- `runtime/` provides the public request/response boundary.
- `workflow/` coordinates the selected cognitive components.
- `memory/`, `affect/`, `reasoning/`, `planning/`, and `reflection/` contain the core subsystems.
- `action/` and `tools/` enforce controlled execution.
- `benchmarks/` and `evaluation/` contain research and validation code.
- `tests/` contains deterministic unit, integration, architecture, and release checks.

## Contact and license

[Yuv Nahar](https://github.com/yuvnahr) is the project lead and primary contact. Everyone represented in the Git history is listed in [CONTRIBUTORS.md](CONTRIBUTORS.md).

PRIMA-NEXT is licensed under the [Apache License 2.0](LICENSE). Third-party dependencies, datasets, and submodules remain subject to their own licenses.
