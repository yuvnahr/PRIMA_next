"""Regression tests for the bounded Xenon quality gate."""

from pathlib import Path

from scripts.run_xenon import SOURCE_ROOTS, build_command


def test_xenon_targets_production_source_only(tmp_path: Path) -> None:
    """The command must not descend into benchmarks, evaluation, tests or environments."""
    for relative in (*SOURCE_ROOTS, "benchmarks", "evaluation", "tests", "venv"):
        path = tmp_path / relative
        path.touch() if path.suffix == ".py" else path.mkdir(parents=True, exist_ok=True)

    command = build_command(tmp_path)
    analyzed_paths = command[command.index("-e") + 2 :]

    assert analyzed_paths == [str(tmp_path / relative) for relative in SOURCE_ROOTS]
    assert not any(part in path for path in analyzed_paths for part in ("benchmarks", "evaluation", "tests", "venv"))
