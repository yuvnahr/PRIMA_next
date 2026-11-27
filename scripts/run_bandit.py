"""Run the existing Bandit rules on first-party, tracked source roots."""

from __future__ import annotations

import shutil
import subprocess  # nosec B404
import sys
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    git = shutil.which("git")
    if git is None:
        raise RuntimeError("Git is required to enumerate first-party source roots")
    tracked = subprocess.check_output(  # noqa: S603  # nosec B603
        [git, "ls-files", "-z", "--", "*.py"], cwd=ROOT, encoding="utf-8",
    )
    roots = sorted({PurePosixPath(path).parts[0] for path in tracked.split("\0") if path})
    if not roots:
        raise RuntimeError("No first-party Python sources found")
    return subprocess.call(  # noqa: S603  # nosec B603
        [sys.executable, "-m", "bandit", "-q", "-c", "bandit.yaml", "-r", "--", *roots], cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
