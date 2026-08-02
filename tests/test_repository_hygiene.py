from __future__ import annotations

import re
import shutil
import subprocess

CACHE_PATH = re.compile(r"(^|/)(__pycache__|\.pytest_cache|\.mypy_cache|\.ruff_cache)(/|$)|\.py[co]$")


def test_generated_caches_are_ignored_and_untracked() -> None:
    git = shutil.which("git")
    assert git is not None
    tracked = subprocess.run(  # noqa: S603 - fixed executable and arguments
        [git, "ls-files"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert not [path for path in tracked if CACHE_PATH.search(path)]

    for probe in ("__pycache__/probe.pyc", ".pytest_cache/probe", ".mypy_cache/probe", ".ruff_cache/probe"):
        ignored = subprocess.run(  # noqa: S603 - fixed executable and arguments
            [git, "check-ignore", "-q", "--", probe],
            check=False,
        )
        assert ignored.returncode == 0, probe
