"""B4 measured admission, source identity and canonical/legacy artifact boundaries."""

import copy
import json
import shutil
import zipfile

import pytest

from benchmarks.campaign.config import CampaignConfig
from benchmarks.campaign.kaggle import (
    budget_gate,
    choose_pilot_workers,
    observed_backend_slots,
    prepare_run,
    resource_cadence,
    validate_coverage,
    validate_pilot,
)
from benchmarks.campaign.orchestrator import run_campaign
from benchmarks.common.artifacts import read_json, read_jsonl
from benchmarks.common.bundle import MANIFEST, create_bundle, validate_bundle
from benchmarks.common.source_identity import _git, require_reviewed_source, source_identity
from tests.benchmarks.test_campaign import _payload


def pilot_pair():
    record = {"case_id": "a", "prediction": "Alpha", "execution_failed": False}
    reference = {"workers": 1, "identity": {"source": "s", "model": "m", "prompt": "p"},
                 "selected_ids": ["a"], "records": [record], "resources_validated": True,
                 "wall_seconds_repeats": [100, 101]}
    candidate = {**copy.deepcopy(reference), "workers": 2, "wall_seconds_repeats": [80, 81],
                 "observed_backend_slots": 2, "max_outstanding_requests": 2, "inference_topology": "shared",
                 "estimate": {"fresh": True, "resources_validated": True, "projected_campaign_seconds": 1000,
                              "projected_remaining_seconds": 1000, "projected_session_seconds": 1100,
                              "final_artifact_reserve_seconds": 900, "safety_margin": .2}}
    return reference, candidate


@pytest.mark.parametrize("bad", ["slower", "noisy", "one_slot", "unknown_slots", "predictions", "identity", "budget", "resources", "unreplicated", "replicas"])
def test_two_worker_optimistic_projection_cannot_override_measured_rejection(bad):
    reference, candidate = pilot_pair()
    if bad == "slower":
        candidate["wall_seconds_repeats"] = [102, 103]
    elif bad == "noisy":
        candidate["wall_seconds_repeats"] = [80, 99]
    elif bad in {"one_slot", "unknown_slots"}:
        candidate["observed_backend_slots"] = 1 if bad == "one_slot" else None
    elif bad == "predictions":
        candidate["records"][0]["prediction"] = "changed"
    elif bad == "identity":
        candidate["identity"]["prompt"] = "changed"
    elif bad == "budget":
        candidate["estimate"]["projected_campaign_seconds"] = 40000
    elif bad == "resources":
        candidate["resources_validated"] = False
    elif bad == "unreplicated":
        candidate["wall_seconds_repeats"] = [80]
    else:
        candidate["inference_topology"] = "replicated"
    result = choose_pilot_workers(reference, candidate, benchmark="locomo", elapsed_seconds=100)
    assert result["workers"] == 1 and result["actual_inference_overlap"] is None


def test_cpu_pipeline_is_explicit_and_restores_serial_inference_semantics():
    reference, candidate = pilot_pair()
    candidate.update(observed_backend_slots=1, cpu_pipeline=True, max_outstanding_requests=1)
    selected = choose_pilot_workers(reference, candidate, benchmark="locomo", elapsed_seconds=100)
    assert selected["workers"] == 2 and selected["scheduling"] == "CPU overlap with serialized inference"
    assert selected["actual_inference_overlap"] is None
    assert choose_pilot_workers(reference, candidate, benchmark="hotpotqa", elapsed_seconds=100)["workers"] == 1
    assert observed_backend_slots('OLLAMA_NUM_PARALLEL=2') is None
    assert observed_backend_slots('runner n_slots = 1') == 1
    assert observed_backend_slots('n_slots = 2\nrunner restarted n_slots = 1') == 1


@pytest.mark.parametrize("field,value", [("remaining_quota_seconds", None), ("remaining_quota_seconds", 100),
                                        ("pilot_age_seconds", 901), ("quota_age_seconds", float("nan"))])
def test_production_budget_needs_live_quota_and_fresh_measurements(field, value):
    _, candidate = pilot_pair()
    kwargs = {"remaining_quota_seconds": 2000, "quota_age_seconds": 1, "pilot_age_seconds": 1}
    assert budget_gate(candidate["estimate"], elapsed_seconds=100, require_live=True, **kwargs)["allowed"]
    kwargs[field] = value
    with pytest.raises(ValueError):
        budget_gate(candidate["estimate"], elapsed_seconds=100, require_live=True, **kwargs)


def test_cadence_uses_actual_gaps_and_distinguishes_missed_deadlines():
    result = resource_cadence([{"elapsed_seconds": value} for value in (0, 1, 2.1, 5)], 1, 2)
    assert result["max_gap_seconds"] == 2.9
    assert result["missed_deadlines"] == 1 and not result["complete"]
    with pytest.raises(ValueError):
        resource_cadence([{"elapsed_seconds": 0}, {"elapsed_seconds": float("nan")}], 1, 2)
    with pytest.raises(ValueError):
        resource_cadence([{"elapsed_seconds": 1}, {"elapsed_seconds": 0}], 1, 2)


def test_full_coverage_includes_auditable_failures_but_canaries_require_success():
    manifest = {"selected_ids": ["a"], "dataset_hash": "digest"}
    rows = [{"case_id": "a", "status": "failed", "failure": {"reason": "observed failure"}}]
    validate_coverage(manifest, rows, ["a"], expected_count=1, dataset_sha256="digest", require_success=False)
    with pytest.raises(ValueError, match="unsuccessful"):
        validate_coverage(manifest, rows, ["a"], expected_count=1, dataset_sha256="digest")


def test_source_fingerprint_supports_worktrees_and_rejects_dirty_or_changed_source(tmp_path):
    def git(*args):
        return _git(tmp_path, *args)
    git("init")
    git("config", "user.name", "Synthetic")
    git("config", "user.email", "synthetic@example.invalid")
    (tmp_path / "PRIMA_Kaggle_Benchmark.ipynb").write_text("{}")
    code = tmp_path / "main.py"
    code.write_text("print('fixed')\n")
    git("add", "main.py", "PRIMA_Kaggle_Benchmark.ipynb")
    git("commit", "-m", "Synthetic source fixture")
    identity = source_identity(tmp_path)
    require_reviewed_source(identity, identity["git_commit"], identity["source_fingerprint"])
    worktree = tmp_path / "other"
    git("worktree", "add", "--detach", str(worktree), "HEAD")
    assert source_identity(worktree)["source_fingerprint"] == identity["source_fingerprint"]
    code.write_text("print('changed')\n")
    changed = source_identity(tmp_path)
    assert changed["source_fingerprint"] != identity["source_fingerprint"]
    with pytest.raises(ValueError):
        require_reviewed_source(changed, identity["git_commit"])


@pytest.fixture
def tiny_campaign(tmp_path):
    root = tmp_path / "run"
    payload = _payload(root)
    payload["benchmarks"] = [payload["benchmarks"][0]]
    prepare_run(root, {"synthetic": True}, resume=False, fresh=True)
    run_campaign(CampaignConfig.model_validate(payload))
    child = root / read_json(root / "manifest.json")["modes"]["go"]["child_manifest"]
    shutil.copy2(child.parent / "predictions/records.jsonl", root / "predictions.jsonl")
    return root, child


def test_bundle_reconstructs_all_views_prompts_and_legacy_reads(tiny_campaign, tmp_path):
    root, child = tiny_campaign
    archive = tmp_path / "result.zip"
    result = create_bundle(root, archive)
    assert result["valid"]
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as bundle:
        manifest = json.loads(bundle.read(MANIFEST))
        assert manifest["artifacts"]["predictions.jsonl"]["view"] == "predictions"
        assert manifest["prompt_templates"]
        for name in bundle.namelist():
            target = extracted / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.read(name))
    for original in root.rglob("*"):
        if original.suffix in {".json", ".jsonl"}:
            target = extracted / original.relative_to(root)
            read = read_jsonl if original.suffix == ".jsonl" else read_json
            assert read(target) == read(original)
    child_after = extracted / child.relative_to(root)
    assert read_json(child_after)["selected_ids"] == read_json(child)["selected_ids"]
    legacy = tmp_path / "legacy.zip"
    shutil.make_archive(str(legacy.with_suffix("")), "zip", root_dir=root)
    assert validate_bundle(legacy)["schema_version"] == "legacy"


@pytest.mark.parametrize("bad", ["unsafe", "duplicate", "hash"])
def test_bundle_rejects_unsafe_duplicate_and_tampered_paths(tiny_campaign, tmp_path, bad):
    root, _ = tiny_campaign
    original = tmp_path / "good.zip"
    create_bundle(root, original)
    broken = tmp_path / "broken.zip"
    with zipfile.ZipFile(original) as source, zipfile.ZipFile(broken, "w") as target:
        for name in source.namelist():
            content = source.read(name)
            if bad == "hash" and name == "manifest.json":
                content += b" "
            target.writestr(name, content)
        if bad == "unsafe":
            target.writestr("../outside.json", "{}")
        if bad == "duplicate":
            target.writestr("manifest.json", "{}")
    with pytest.raises(ValueError):
        validate_bundle(broken)


def test_bundle_rejects_secrets_malformed_records_and_missing_selected_ids(tiny_campaign, tmp_path):
    root, child = tiny_campaign
    secret = root / "credentials.log"
    secret.write_text("api_key=" + "0" * 32)
    with pytest.raises(ValueError, match="secret"):
        create_bundle(root, tmp_path / "secret.zip")
    secret.unlink()
    malformed = root / "broken.jsonl"
    malformed.write_text("[]\n")
    with pytest.raises(ValueError, match="objects"):
        create_bundle(root, tmp_path / "malformed.zip")
    malformed.unlink()
    manifest = read_json(child)
    manifest["selected_ids"].append("missing")
    child.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        create_bundle(root, tmp_path / "missing.zip")


def test_executed_notebook_allows_only_literal_config_changes(tmp_path):
    from benchmarks.common.source_identity import executed_notebook_identity
    tracked, executed = tmp_path / "tracked.ipynb", tmp_path / "executed.ipynb"
    original = {"cells": [{"cell_type": "code", "source": "CONFIG = {'model': 'fixed'}\nprint(CONFIG)\n"}]}
    tracked.write_text(json.dumps(original))
    actual = copy.deepcopy(original)
    actual["cells"][0]["source"] = "CONFIG = {'model': 'explicit'}\nprint(CONFIG)\n"
    executed.write_text(json.dumps(actual))
    source = {"source_fingerprint": "f" * 64, "git_commit": "a" * 40, "dirty": False}
    identity = executed_notebook_identity(source, executed, tracked)
    assert identity["executed_notebook_sha256"]
    assert identity["source_fingerprint"] != source["source_fingerprint"]
    actual["cells"][0]["source"] += "print('untracked implementation')\n"
    executed.write_text(json.dumps(actual))
    with pytest.raises(ValueError, match="code differs"):
        executed_notebook_identity(source, executed, tracked)


def test_export_is_immutable_and_materialization_preserves_resume_inputs(tiny_campaign, tmp_path):
    from benchmarks.common.artifacts import atomic_write_json
    from benchmarks.common.bundle import materialize_bundle
    root, _ = tiny_campaign
    archive = tmp_path / "compact.zip"
    create_bundle(root, archive)
    extracted = tmp_path / "immutable"
    with zipfile.ZipFile(archive) as source:
        for name in source.namelist():
            target = extracted / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source.read(name))
    with pytest.raises(ValueError, match="immutable"):
        atomic_write_json(extracted / "new.json", {"unsafe_mutation": True})
    writable = tmp_path / "recovery"
    materialize_bundle(archive, writable)
    for path in root.rglob("*.jsonl"):
        assert read_jsonl(path) == read_jsonl(writable / path.relative_to(root))
    atomic_write_json(writable / "new.json", {"writable": True})
    assert read_json(writable / "new.json")["writable"]


def test_failed_backend_restoration_blocks_notebook_production():
    from tests.benchmarks.test_kaggle import notebook_namespace
    ns = notebook_namespace()
    calls = []
    ns.update(SELECTED_BENCHMARK="locomo", time=__import__("time"), STATE={},
              start_ollama=lambda *args, **kwargs: calls.append("start"),
              warm_model=lambda: ([], [], {}, 0.1),
              placement_snapshot=lambda *args: {"actual_context_length": 4096},
              stop_ollama=lambda: calls.append("stop"))
    with pytest.raises(RuntimeError, match="restoration placement"):
        ns["configure_workers"](1)
    assert calls == ["start", "stop"]
    assert not ns["STATE"]["backend_restored"]


def test_independent_projection_uses_measured_overlap_not_requested_workers():
    from benchmarks.campaign.kaggle import estimate_independent
    rows = [{"latency_ms": 1000}, {"latency_ms": 1000}]
    kwargs = {"full_items": 100, "campaign_wall": 2.1, "packaging_seconds": 1, "setup_seconds": 3, "safety_margin": .2}
    sequential = estimate_independent(rows, workers=1, **kwargs)
    queued = estimate_independent(rows, workers=2, **kwargs)
    assert sequential["projected_remaining_seconds"] == queued["projected_remaining_seconds"]
    assert queued["measured_schedule_factor"] == 1


@pytest.mark.parametrize("mutation", ["dirty", "commit", "fingerprint"])
def test_campaign_preflight_rejects_unreviewed_source_before_provider_work(tmp_path, monkeypatch, mutation):
    from benchmarks.campaign import preflight
    from benchmarks.campaign.provider_session import SharedProviderSession
    identity = {"git_commit": "a" * 40, "dirty": False, "source_fingerprint": "f" * 64}
    payload = _payload(tmp_path / "never-started")
    payload["repository"] = {"requested_ref": "dev", "commit_sha": "a" * 40,
        "source_fingerprint": "f" * 64, "require_clean_pinned_source": True}
    changed = {**identity, {"dirty": "dirty", "commit": "git_commit", "fingerprint": "source_fingerprint"}[mutation]:
               True if mutation == "dirty" else "changed"}
    monkeypatch.setattr(preflight, "source_identity", lambda root: changed)
    config = CampaignConfig.model_validate(payload)
    session = SharedProviderSession(config.provider, max_active_requests=1)
    with pytest.raises(ValueError, match="source"):
        preflight.run_preflight(config, session)
    assert session.telemetry()["request_attempts"] == 0
    assert not config.output_root.exists()


def test_source_change_invalidates_resume(tiny_campaign, monkeypatch):
    from benchmarks.campaign import preflight
    root, _ = tiny_campaign
    saved = read_json(root / "manifest.json")["preflight"]["source_identity"]
    monkeypatch.setattr(preflight, "source_identity", lambda root: {**saved, "source_fingerprint": "changed"})
    payload = _payload(root)
    payload["benchmarks"] = [payload["benchmarks"][0]]
    with pytest.raises(ValueError, match="source"):
        run_campaign(CampaignConfig.model_validate(payload), resume=True)


def test_refused_partial_pilot_retains_evidence_and_restores_reference(tmp_path):
    import ast
    import time
    from pathlib import Path

    from tests.benchmarks.test_kaggle import REPO, notebook_namespace
    ns = notebook_namespace()
    ns["CONFIG"].update(SMOKE_MODE=False, RUN_THROUGHPUT_PILOT=True)
    calls = []
    record = {"case_id": "a", "prediction": "Alpha", "execution_failed": False}
    def configure(workers):
        calls.append(workers)
        ns["STATE"].update(backend_restored=True, ollama_parallel_slots=workers, observed_backend_slots=workers)
    def build(label, **kwargs):
        root = tmp_path / label
        config = tmp_path / (label + ".json")
        payload = {"output_root": str(root), "scheduler": {}, "benchmarks": [{"options": {}}]}
        config.write_text(json.dumps(payload))
        return payload, config, root, "mode"
    def execute(config, root, count, **kwargs):
        root.mkdir()
        (root / "checkpoint-evidence.json").write_text('{"recoverable":true}')
        if root.name.startswith("canary-2"):
            raise RuntimeError("synthetic resource stop")
        return [], None, 100
    ns.update(json=json, shutil=shutil, time=time, OUTPUT_ROOT=tmp_path, SESSION_STAMP="synthetic",
        SESSION_STARTED=time.monotonic(), SELECTED_BENCHMARK="locomo", conversations=[],
        dataset_summary={"items": 1}, STATE={"mode_id": "mode", "backend_restored": True,
            "ollama_parallel_slots": 1, "preflight": {}}, configure_workers=configure,
        build_campaign_config=build, expected_ids=lambda payload: {"mode": ["a"]}, execute_campaign=execute,
        validate_canary=lambda *args: ({"telemetry": {"max_active_requests": 1, "successful_requests": 1}}, [record], []),
        measure_canary_packaging=lambda *args: 1, record_phase=lambda *args: None,
        budget_gate=budget_gate, validate_pilot=validate_pilot, gate_identity=lambda: {"unchanged": True}, pilot_identity=lambda payload: {"unchanged": True},
        live_quota=lambda: {"remaining_quota_seconds": 100000, "quota_age_seconds": 1},
        estimate_locomo=lambda *args, **kwargs: {"fresh": True, "projected_campaign_seconds": 100,
            "projected_remaining_seconds": 100, "projected_session_seconds": 100, "safety_margin": .2,
            **dict.fromkeys(("full_items", "measured_items", "historical_replay_seconds", "projected_qa_work_seconds",
                            "projected_execution_seconds", "projected_finalization_seconds", "projected_packaging_seconds"), 1)})
    book = json.loads((REPO / "PRIMA_Kaggle_Benchmark.ipynb").read_text(encoding="utf-8"))
    tree = ast.parse("".join(book["cells"][18]["source"]))
    tree.body = [tree.body[-1]]
    exec(compile(tree, "pilot-admission", "exec"), ns)  # noqa: S102 - trusted notebook with offline stubs
    refused = ns["STATE"]["pilot_results"]["2"]
    assert refused["status"] == "refused" and len(refused["run_roots"]) == 1
    partial = Path(refused["run_roots"][0])
    assert read_json(partial / "checkpoint-evidence.json")["recoverable"]
    assert read_json(partial / "pilot_refusal.json")["reason"] == "synthetic resource stop"
    assert read_json(partial / "effective_config.json")["output_root"] == str(partial)
    assert calls == [2, 1] and ns["STATE"]["selected_workers"] == 1


def test_bundle_reconciles_terminal_status_counts(tiny_campaign, tmp_path):
    root, child = tiny_campaign
    summary_path = child.parent / "summary.json"
    summary = read_json(summary_path)
    summary.update(completed_cases=0, failed_cases=1)
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="terminal status counts"):
        create_bundle(root, tmp_path / "wrong-counts.zip")



def test_notebook_failure_diagnostics_separate_adversarial_and_answer_errors(tmp_path):
    import ast

    from tests.benchmarks.test_kaggle import REPO
    book = json.loads((REPO / "PRIMA_Kaggle_Benchmark.ipynb").read_text(encoding="utf-8"))
    rows = [{"case_id": "correct", "category": 1, "answer_correct": True, "supplied_evidence_recall": 0},
            {"case_id": "wrong", "category": 1, "answer_correct": False},
            {"case_id": "adversarial", "category": 5, "answer_correct": True, "outcome": "abstained", "supplied_evidence_recall": 0}]
    ns = {"records": rows, "SELECTED_BENCHMARK": "locomo", "RUN_ROOT": tmp_path, "json": json}
    exec("".join(book["cells"][28]["source"]), ns)  # noqa: S102 - trusted notebook on synthetic rows
    result = ns["failure_analysis"]
    assert [row["id"] for row in result["wrong_predictions"]] == ["wrong"]
    assert [row["id"] for row in result["retrieval_misses"]] == ["correct"]
    assert [row["id"] for row in result["abstentions"]] == ["adversarial"]
    tree = ast.parse("".join(book["cells"][26]["source"]))
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(tree, "percentile-definitions", "exec"), ns)  # noqa: S102 - definitions only
    assert ns["percentile"]([], 50) is None
