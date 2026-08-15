"""Run Xenon against PRIMA-NEXT production Python source only."""

from __future__ import annotations

import subprocess  # nosec B404
import sys
from pathlib import Path

SOURCE_ROOTS = (
    "main.py",
    "action",
    "affect",
    "config",
    "events",
    "llm",
    "memory",
    "monitoring",
    "planning",
    "reasoning",
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
XENON_TIMEOUT_SECONDS = 30


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
    print("Running Xenon on PRIMA production source roots only:")
    print(" ".join(command))
    try:
        return subprocess.run(  # noqa: S603  # nosec B603
            command,
            cwd=root,
            check=False,
            timeout=XENON_TIMEOUT_SECONDS,
        ).returncode
    except subprocess.TimeoutExpired:
        print(f"Xenon exceeded {XENON_TIMEOUT_SECONDS}s and was terminated.", file=sys.stderr)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
