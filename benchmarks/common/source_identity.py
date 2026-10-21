"""Git-aware identity of the source actually present in a checkout."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess  # nosec B404
from pathlib import Path
from typing import Any

SOURCE_SUFFIXES = {".py", ".yaml", ".yml", ".toml", ".ipynb", ".ini", ".cfg", ".json"}


def git_commit(root: Path | None = None) -> str | None:
    try:
        return _git(root or Path(__file__).resolve().parents[2], "rev-parse", "HEAD").strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git(root: Path, *args: str) -> str:
    executable = shutil.which("git")
    if executable is None:
        raise FileNotFoundError("Git is required to identify executed source")
    return subprocess.check_output(  # noqa: S603  # nosec B603 - fixed internal Git arguments, no shell
        [executable, "-C", str(root), *args], text=True, encoding="utf-8", stderr=subprocess.PIPE)


def source_identity(root: Path) -> dict[str, Any]:
    """Hash tracked code/config/notebook text; generated output and secrets are excluded."""
    root = root.resolve()
    head = _git(root, "rev-parse", "HEAD").strip()
    entries = _git(root, "ls-files", "--stage", "-z").split("\0")
    files, submodules = {}, {}
    for entry in filter(None, entries):
        metadata, name = entry.split("\t", 1)
        mode, object_id, stage = metadata.split()
        if stage != "0":
            raise ValueError("Unmerged source cannot be identified")
        if mode == "160000":
            submodules[name] = object_id
            continue
        path = Path(name)
        if path.suffix not in SOURCE_SUFFIXES and not (path.name.startswith("requirements") and path.suffix == ".txt"):
            continue
        if path.parts[0] == "docs":
            continue  # Review prose/manifests are provenance, not executed source.
        if any(part in {"outputs", "results", "models", "cache", ".git", ".codex"} for part in path.parts):
            continue
        content = (root / path).read_text(encoding="utf-8").encode("utf-8")
        files[name] = hashlib.sha256(content).hexdigest()
    if not files or "PRIMA_Kaggle_Benchmark.ipynb" not in files:
        raise ValueError("Tracked executed source/notebook is incomplete")
    dirty = _git(root, "status", "--porcelain", "--untracked-files=normal", "--ignore-submodules=all").splitlines()
    content_identity = {"files": files, "submodule_pins": submodules, "normalization": "utf8_lf"}
    digest = hashlib.sha256(json.dumps(content_identity, sort_keys=True).encode()).hexdigest()
    return {"git_commit": head, "source_fingerprint": digest, "dirty": bool(dirty), "status": dirty,
            "notebook_sha256": files["PRIMA_Kaggle_Benchmark.ipynb"], **content_identity}


def require_reviewed_source(identity: dict[str, Any], expected_sha: str, expected_fingerprint: str | None = None) -> None:
    if identity.get("dirty") or identity.get("git_commit") != expected_sha or not identity.get("source_fingerprint"):
        raise ValueError("Production requires clean, pinned, reproducibly identified reviewed source")
    if expected_fingerprint is not None and identity["source_fingerprint"] != expected_fingerprint:
        raise ValueError("Executed source fingerprint changed after measurement")


def executed_notebook_identity(identity: dict[str, Any], executed: Path, tracked: Path) -> dict[str, Any]:
    """Allow literal CONFIG edits only; pin the exact input notebook independently."""
    import ast

    original = json.loads(tracked.read_text(encoding="utf-8"))
    actual = json.loads(executed.read_text(encoding="utf-8"))
    left = [cell["source"] for cell in original["cells"] if cell["cell_type"] == "code"]
    right = [cell["source"] for cell in actual["cells"] if cell["cell_type"] == "code"]
    if len(left) != len(right):
        raise ValueError("Executed notebook has a different code-cell population")
    def normalized(source: str | list[str]) -> str:
        tree = ast.parse("".join(source))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "CONFIG" for target in node.targets):
                if not isinstance(ast.literal_eval(node.value), dict):
                    raise ValueError("Notebook CONFIG must be a literal mapping")
                node.value = ast.Constant(None)
        return ast.dump(tree)
    if any(normalized(before) != normalized(after) for before, after in zip(left, right, strict=True)):
        raise ValueError("Executed notebook code differs from pinned tracked source")
    notebook_hash = hashlib.sha256(executed.read_bytes()).hexdigest()
    combined = hashlib.sha256(json.dumps({"tracked_source_fingerprint": identity["source_fingerprint"],
                                          "executed_notebook_sha256": notebook_hash}, sort_keys=True).encode()).hexdigest()
    return {**identity, "tracked_source_fingerprint": identity["source_fingerprint"],
            "executed_notebook_sha256": notebook_hash, "source_fingerprint": combined}
