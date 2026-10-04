"""Tiny synthetic checks; these are not evidence of Kaggle/model smoke success."""
from __future__ import annotations

import copy
import hashlib
import json
import random
import subprocess
import sys
import time
import types
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.campaign.config import CampaignConfig, load_campaign_config
from benchmarks.campaign.kaggle import (
    DATASET_FILES,
    CheckpointProgress,
    budget_gate,
    discover_dataset,
    persist_sample,
    prepare_run,
    require_production_identity,
    resource_guard,
    selected_case_ids,
    validate_coverage,
)
from benchmarks.campaign.orchestrator import run_campaign
from benchmarks.common.artifacts import read_jsonl
from tests.benchmarks.test_campaign import FIXTURES, _payload
from tests.benchmarks.test_kaggle import REPO, notebook_namespace, sample


@pytest.mark.parametrize("benchmark", ["locomo", "hotpotqa", "goemotions"])
def test_full_notebook_launch_requires_budget_resources_and_complete_source_selection(tmp_path, benchmark):
    """Synthetic launch checks only: never execute inference or a full campaign."""
    ns = notebook_namespace()
    if benchmark == "goemotions":
        dataset = tmp_path / "goemotions_test.json"
        dataset.write_text(json.dumps([{"id": f"test-{i}", "text": "happy", "labels": [17]} for i in range(5)]))
        source_ids = [f"test-{i}" for i in range(5)]
    elif benchmark == "hotpotqa":
        dataset = FIXTURES / "campaign" / "hotpot.json"
        source_ids = [row["_id"] for row in json.loads(dataset.read_text())]
    else:
        from benchmarks.locomo.experiment import question_case_id
        from benchmarks.locomo.loader import LoCoMoDataset

        dataset = FIXTURES / "campaign" / "locomo.json"
        ns["conversations"] = list(LoCoMoDataset(dataset).conversations())
        source_ids = [question_case_id(c.id, q.question_id) for c in ns["conversations"] for q in c.questions]
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    ns.update(json=json, re=__import__("re"), time=time, REPOSITORY_DIR=REPO, OUTPUT_ROOT=tmp_path,
              SELECTED_BENCHMARK=benchmark, SELECTED_DATASET=dataset, SESSION_STAMP="synthetic",
              OLLAMA_ENDPOINT="http://unused", STATE={
                  "repository": {"url": "https://example.invalid", "requested_ref": "test", "commit_sha": "a" * 40},
                  "model_identity": {"name": "synthetic", "digest": "b" * 64},
                  "preflight": {"actual_context_length": 8192}, "selected_workers": 1,
              }, dataset_summary={"items": len(source_ids), "expected_full_run_count": len(source_ids)},
              require_production_identity=require_production_identity, budget_gate=budget_gate)
    ns["CONFIG"].update(RUN_FULL_BENCHMARK=True, FULL_RUN_ID="synthetic-full", PINNED_COMMIT_SHA="a" * 40,
                        EXPECTED_MODEL_DIGEST="b" * 64)
    ns["CONFIG"]["EXPECTED_DATASET_SHA256"][benchmark] = digest
    full, config_path, root, mode_id = ns["build_campaign_config"]("full", full=True)
    mode = full["benchmarks"][0]
    assert full["run_kind"] == "benchmark" and mode["max_items"] == 0
    assert mode["profile"] == ns["CONFIG"]["BENCHMARK_PROFILE"][benchmark]
    if benchmark == "locomo":
        assert mode["options"]["full_dataset"] and mode["options"]["max_conversations"] == 0
    assert ns["expected_ids"](full)[mode_id] == source_ids
    measured = copy.deepcopy(full)
    measured["output_root"] = str(tmp_path / "measured")
    measured["benchmarks"][0]["max_items"] = 1
    if benchmark == "locomo":
        measured["benchmarks"][0]["options"].update(full_dataset=False, max_conversations=1)
    measured_root = tmp_path / "measured"
    measured_root.mkdir()
    (measured_root / "run_identity.json").write_text(json.dumps({"effective_config": measured}))
    estimate = {"fresh": True, "resources_validated": True, "projected_campaign_seconds": 100,
                "projected_remaining_seconds": 100, "projected_session_seconds": 100,
                "identity": {"repository_sha": "a" * 40, "model_digest": "b" * 64,
                             "dataset_sha256": digest, "session_stamp": "synthetic"}}
    ns["STATE"].update(timing_canary=estimate, pilot_results={"1": {"run_root": str(measured_root)}})
    ns["run_checked"] = lambda cmd, **kwargs: types.SimpleNamespace(stdout="" if "status" in cmd else "a" * 40)
    ns["inspect_model_identity"] = lambda: {"digest": "b" * 64}
    ns["sha256_file"] = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    ns["safety_sample"] = lambda _: sample()
    ns["check_safety"] = resource_guard
    launched = []
    ns["execute_campaign"] = lambda *args, **kwargs: launched.append((args, kwargs)) or ([], tmp_path / "log", 1)
    notebook = json.loads((REPO / "PRIMA_Kaggle_Benchmark.ipynb").read_text(encoding="utf-8"))
    cell = "".join(notebook["cells"][20]["source"])
    estimate["projected_campaign_seconds"] = 11 * 3600
    with pytest.raises(ValueError, match="shortfall"):
        exec(cell, ns)  # noqa: S102 - checked-in launch guard with synthetic inputs
    assert not launched
    estimate["projected_campaign_seconds"] = 100
    unsafe = sample()
    unsafe["gpus"][0]["memory_used_mib"] = 13000
    ns["safety_sample"] = lambda _: unsafe
    with pytest.raises(ValueError, match="VRAM"):
        exec(cell, ns)  # noqa: S102
    assert not launched
    ns["safety_sample"] = lambda _: sample()
    exec(cell, ns)  # noqa: S102
    assert launched == [((config_path, root, len(source_ids)), {"fresh": False})]


@pytest.mark.parametrize("benchmark", DATASET_FILES)
@pytest.mark.parametrize("nested", [False, True])
def test_slug_discovery_and_ambiguity(tmp_path, benchmark, nested):
    slug, filename = DATASET_FILES[benchmark]
    root = tmp_path / "input"
    mount = root / "datasets" / "thegifman" / slug if nested else root / slug
    path = mount / "nested" / filename
    path.parent.mkdir(parents=True)
    path.write_text("[]")
    assert discover_dataset(benchmark, root / "missing", root) == path.resolve()
    duplicate = mount / filename
    duplicate.write_text("[]")
    with pytest.raises(ValueError, match="ambiguous"):
        discover_dataset(benchmark, root / "missing", root)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "extra", "truncated-selection", "hash", "failed"])
def test_coverage_rejects_corrupt_full_outputs(mutation):
    ids = ["a", "b"]
    manifest = {"selected_ids": ids[:], "dataset_hash": "source"}
    rows = [{"case_id": case_id, "status": "complete", "prediction": {"case_id": case_id}} for case_id in ids]
    validate_coverage(manifest, rows, ids, expected_count=2, dataset_sha256="source")
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows[1] = copy.deepcopy(rows[0])
    elif mutation == "extra":
        rows[1]["case_id"] = "unexpected"
    elif mutation == "truncated-selection":
        manifest["selected_ids"].pop()
        rows.pop()
    elif mutation == "hash":
        manifest["dataset_hash"] = "changed"
    else:
        rows[1]["status"] = "failed"
    with pytest.raises(ValueError):
        validate_coverage(manifest, rows, ids, expected_count=2, dataset_sha256="source")


def test_smoke_limits_and_resume_rejected_before_execution(tmp_path):
    payload = _payload(tmp_path / "smoke")
    payload.update(run_kind="smoke", benchmarks=[payload["benchmarks"][1]])
    config = CampaignConfig.model_validate(payload)
    with pytest.raises(ValueError, match="fresh"):
        run_campaign(config, resume=True)
    for count in (0, 4):
        changed = copy.deepcopy(payload)
        changed["benchmarks"][0]["max_items"] = count
        with pytest.raises(ValidationError, match="three"):
            CampaignConfig.model_validate(changed)
    locomo = _payload(tmp_path)["benchmarks"][2]
    locomo.update(max_items=2, options={"max_conversations": 2})
    payload["benchmarks"] = [locomo]
    with pytest.raises(ValidationError, match="three"):
        CampaignConfig.model_validate(payload)


@pytest.mark.parametrize("benchmark", ["locomo", "hotpotqa", "goemotions"])
def test_notebook_smoke_stages_native_artifacts_and_partial_packages(tmp_path, monkeypatch, benchmark):
    """Use real native runners/scorers with fake inference and synthetic hardware only."""
    from benchmarks.campaign import orchestrator
    from benchmarks.locomo.experiment import git_commit
    from benchmarks.locomo.loader import LoCoMoDataset

    # Hardware and package inventories are synthetic in this unit check; keep it tiny.
    monkeypatch.setattr(orchestrator, "collect_host_telemetry", lambda _: {})

    ns = notebook_namespace(smoke=True)
    ns.update(json=json, subprocess=subprocess, time=time, REPOSITORY_DIR=REPO, OUTPUT_ROOT=tmp_path,
              SELECTED_BENCHMARK=benchmark, SESSION_STAMP="synthetic", OLLAMA_ENDPOINT="http://unused",
              prepare_run=prepare_run, CheckpointProgress=CheckpointProgress, persist_sample=persist_sample,
              sys=sys, client_env={}, datetime=datetime, timezone=types.SimpleNamespace(utc=UTC),
              shutil=__import__("shutil"), re=__import__("re"))
    ns["CONFIG"]["RUN_THROUGHPUT_PILOT"] = False
    if benchmark == "goemotions":
        dataset = tmp_path / "goemotions_test.json"
        rows = [{"id": f"test-{i}", "text": f"I am happy {i}", "labels": [17]} for i in range(4)]
        dataset.write_text(json.dumps(rows))
        dataset.with_name("goemotions_val.json").write_text(json.dumps([{"id": "val-0", "text": "sad", "labels": [25]}]))
    else:
        fixture_path = FIXTURES / "campaign" / f"{'locomo' if benchmark == 'locomo' else 'hotpot'}.json"
        rows = json.loads(fixture_path.read_text())
        source = [copy.deepcopy(rows[i % len(rows)]) for i in range(3)]
        for i, row in enumerate(source):
            row["sample_id" if benchmark == "locomo" else "_id"] = f"case-{i}"
        dataset = tmp_path / f"{benchmark}.json"
        dataset.write_text(json.dumps(source))
    digest = hashlib.sha256(dataset.read_bytes()).hexdigest()
    ns.update(SELECTED_DATASET=dataset, sha256_file=lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest(),
              dataset_summary={"sha256": digest, "expected_digest_pinned": False}, STATE={
                  "repository": {"url": "https://example.invalid", "requested_ref": "test", "commit_sha": git_commit()},
                  "model_identity": {"name": "fake", "digest": "b" * 64},
                  "preflight": {"actual_context_length": 8192}, "mode_id": f"{benchmark}-prima",
                  "ollama_parallel_slots": 1, "model_load_seconds": 1, "smoke_setup_seconds": 2,
              })
    if benchmark == "locomo":
        ns["conversations"] = list(LoCoMoDataset(dataset).conversations())
    for name in ("model_identity.json", "model_gpu_preflight.json", "repository_metadata.json", "pip-check.json", "environment_metadata.json"):
        (tmp_path / name).write_text("{}")
    (tmp_path / "pip-freeze.txt").write_text("synthetic")
    persist_sample(tmp_path / "model_load_resources.jsonl", sample())
    (tmp_path / "session-synthetic-timing.jsonl").write_text('{"phase":"synthetic","seconds":1}\n')
    ns["record_phase"] = lambda *args: None
    ns["configure_workers"] = lambda workers: ns["STATE"].update(ollama_parallel_slots=workers)

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            config = load_campaign_config(command[command.index("--config") + 1])
            config = config.model_copy(update={"provider": config.provider.model_copy(update={"kind": "fake", "fake_delay_seconds": 0.01})})
            result = run_campaign(config)
            assert result["status"] == "partial" and result["smoke_execution_complete"]

        def poll(self):
            return 0

        def wait(self):
            return 0

    ns["subprocess"] = types.SimpleNamespace(Popen=FakeProcess, STDOUT=subprocess.STDOUT,
                                            TimeoutExpired=subprocess.TimeoutExpired)
    ns["safety_sample"] = lambda *args: {**sample(), "timestamp": "synthetic", "elapsed_seconds": 1}
    ns["check_safety"] = resource_guard
    ns["run_smoke"]()
    assert ns["STATE"]["budget_gate"]["allowed"] is False
    results = ns["STATE"]["smoke_measurements"]
    assert len(results) == 2
    for result in results.values():
        assert result["examples"] == 2 and result["validation"] == "pass"
        assert result["status"] == "partial" and not result["reportable"]
        # The tiny LoCoMo fixture legitimately abstains without a provider call.
        assert result["max_active_requests"] == (0 if benchmark == "locomo" else 1)
        assert not result["concurrent_requests_observed"]
        with zipfile.ZipFile(result["archive"]) as archive:
            marker = json.loads(archive.read("smoke_validation.json"))
            assert not marker["full_dataset_complete"] and marker["runtime_estimate"] is None
    stage_ids = [r["selected_ids"] for r in results.values()]
    assert stage_ids[0] == stage_ids[1]
    with pytest.raises(RuntimeError, match="full benchmark"):
        ns["build_campaign_config"]("full", full=True)
    full_mode = copy.deepcopy(json.loads(next(tmp_path.glob("campaign-*.json")).read_text())["benchmarks"][0])
    full_mode["max_items"] = 0
    if benchmark == "locomo":
        full_mode["options"].update(full_dataset=True, max_conversations=0)
    full_ids = selected_case_ids(full_mode)
    assert len(full_ids) > 2
    if benchmark == "goemotions":
        assert stage_ids[0] == [rows[i]["id"] for i in random.Random(13).sample(range(4), 2)]  # noqa: S311
    root = Path(next(iter(results.values()))["archive"]).with_name(next(iter(results)))
    manifest = json.loads((root / "manifest.json").read_text())
    child = root / manifest["modes"][f"{benchmark}-prima"]["child_manifest"]
    with pytest.raises(ValueError, match="coverage"):
        validate_coverage(json.loads(child.read_text()), read_jsonl(child.parent / "checkpoints/records.jsonl"),
                          full_ids, expected_count=len(full_ids), dataset_sha256=digest)
