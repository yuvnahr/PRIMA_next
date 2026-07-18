"""Run Xenon against PRIMA-NEXT's first-party Python source only."""

from __future__ import annotations

import subprocess  # nosec B404
import sys
from pathlib import Path


SOURCE_ROOTS = (
    "main.py",
    "action",
    "affect",
    "benchmarks/common",
    "benchmarks/locomo",
    "config",
    "events",
    "evaluation",
    "llm",
    "memory",
    "monitoring",
    "planning",
    "reflection",
    "runtime",
    "security",
    "state",
    "tools",
    "uncertainty",
    "workflow",
    "world",
)

# Xenon receives directory names, so these filters also cover nested generated
# artifacts and the external benchmark submodules.
IGNORED_DIRECTORIES = (
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".codex_pycache",
    "cache",
    "outputs",
    "results",
    "generated",
    "qa",
    "qa_diagnostics",
    "qa_validation",
    "locomo-v2",
    "reports",
    "metrics",
    "external",
)

IGNORED_FILES = "*generated*.py,*.pyc"


def build_command(root: Path) -> list[str]:
    paths = [str(root / relative) for relative in SOURCE_ROOTS if (root / relative).exists()]
    if not paths:
        raise SystemExit("No first-party source paths were found.")
    return [
        sys.executable,
        "-m",
        "xenon",
        "--max-average",
        "D",
        "--max-modules",
        "E",
        "--max-absolute",
        "F",
        "--paths-in-front",
        "-i",
        ",".join(IGNORED_DIRECTORIES),
        "-e",
        IGNORED_FILES,
        *paths,
    ]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    command = build_command(root)
    print("Running Xenon on first-party source roots only:")
    print(" ".join(command))
    return subprocess.run(command, cwd=root, check=False).returncode  # nosec B603


if __name__ == "__main__":
    raise SystemExit(main())
