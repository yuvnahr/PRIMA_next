"""Regression tests for the bounded Xenon quality gate."""

import subprocess
from pathlib import Path

import scripts.run_xenon as run_xenon
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


def test_xenon_timeout_is_bounded(monkeypatch, capsys) -> None:
    def time_out(*_args, **_kwargs):
        raise subprocess.TimeoutExpired("xenon", run_xenon.XENON_TIMEOUT_SECONDS)

    monkeypatch.setattr(run_xenon.subprocess, "run", time_out)

    assert run_xenon.main() == 124
    assert "was terminated" in capsys.readouterr().err
