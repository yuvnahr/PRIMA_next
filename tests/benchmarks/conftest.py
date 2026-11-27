"""Isolated Git identities for campaign subprocess and notebook fixtures."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def reviewed_benchmark_checkout(tmp_path):
    checkout = tmp_path / "reviewed"
    checkout.mkdir()
    git_path = shutil.which("git")
    assert git_path

    def git(*args):
        subprocess.run([git_path, *args], cwd=checkout, check=True, capture_output=True)  # noqa: S603  # nosec B603

    notebook = Path(__file__).parents[2] / "PRIMA_Kaggle_Benchmark.ipynb"
    shutil.copyfile(notebook, checkout / notebook.name)
    git("init", "-q")
    git("add", notebook.name)
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "-c", "commit.gpgsign=false", "commit", "-qm", "reviewed fixture")
    return checkout
