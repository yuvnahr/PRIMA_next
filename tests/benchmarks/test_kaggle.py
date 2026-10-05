from __future__ import annotations

import ast
import copy
import json
import shutil
import subprocess
import time
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.campaign.config import CampaignConfig
from benchmarks.campaign.kaggle import (
    CheckpointProgress,
    IncrementalJSONL,
    budget_gate,
    estimate_locomo,
    persist_sample,
    prepare_run,
    read_cgroup_memory,
    require_production_identity,
    resource_guard,
    validate_pilot,
)
from benchmarks.campaign.manifest import CampaignManifestStore
from benchmarks.common.artifacts import BenchmarkArtifactStore, read_jsonl
from benchmarks.locomo.experiment import run_locomo_experiment
from tests.test_locomo_qa_correctness import FakeRuntime, fixture, reset_fake

REPO = Path(__file__).parents[2]


def payload(root):
    return {"output_root": str(root), "provider": {"kind": "fake", "model": "fixture", "context_window": 8192},
            "benchmarks": [{"id": "locomo", "benchmark": "locomo", "dataset_path": "unused.json",
                            "profile": "prima_full", "variant": "normal_prima_admission", "context_budget": 6144,
                            "options": {"parallel_workers": 2}}],
            "scheduler": {"max_gpu_requests": 2, "cpu_workers": 2, "ollama_parallel_slots": 2}}


def events():
    return [{"phase": "conversation", "conversation_id": cid, "history_seconds": 10,
             "qa_seconds": 4, "worker_seconds": 14, "question_count": 2, "historical_turn_count": 10}
            for cid in ("a", "b")] + [
                {"phase": "execution", "seconds": 20, "parallel_workers": 2},
                {"phase": "finalization", "seconds": 1},
            ]


def estimate():
    return estimate_locomo(events(), {"a": 10, "b": 10}, workers=2, campaign_wall=23,
                           setup_seconds=30, packaging_seconds=1, safety_margin=0.2)


def test_canary_always_fresh_and_partial_resume_exact(tmp_path):
    root = tmp_path / "canary"
    identity = {"repo": "a", "model": "b", "dataset": "c", "ids": ["1"], "config": {"workers": 1}}
    assert prepare_run(root, identity, resume=False, fresh=True) is False
    with pytest.raises(ValueError, match="Fresh"):
        prepare_run(root, identity, resume=True, fresh=True)
    (root / "manifest.json").write_text('{"status":"partial"}')
    assert prepare_run(root, identity, resume=True, fresh=False)
    for key in identity:
        changed = {**identity, key: "changed"}
        with pytest.raises(ValueError, match="mismatch"):
            prepare_run(root, changed, resume=True, fresh=False)
    (root / "manifest.json").write_text('{"status":"complete"}')
    with pytest.raises(ValueError, match="Completed"):
        prepare_run(root, identity, resume=True, fresh=False)


def test_canary_rejects_orphan_checkpoints(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints" / "records.jsonl").write_text('{}\n')
    with pytest.raises(ValueError, match="Fresh"):
        prepare_run(tmp_path, {}, resume=False, fresh=True)


def test_campaign_resume_checks_live_dataset_even_if_mode_marked_complete(tmp_path):
    config = CampaignConfig.model_validate(payload(tmp_path))
    store = CampaignManifestStore(tmp_path)
    store.initialize(config, {"datasets": {"locomo": {"hashes": {"data": "old"}}}}, resume=False)
    store.update_mode("locomo", status="complete")
    with pytest.raises(ValueError, match="dataset identity"):
        store.initialize(config, {"datasets": {"locomo": {"hashes": {"data": "new"}}}}, resume=True)


def test_estimate_deduplicates_replay_and_uses_measured_efficiency():
    value = estimate()
    assert value["historical_replay_seconds"] == 20  # two histories, not twenty questions
    assert value["projected_qa_work_seconds"] == 40
    assert value["measured_schedule_factor"] == pytest.approx(20 / 28)  # observed, not assumed 0.5
    assert value["projected_execution_seconds"] == pytest.approx(60 * 20 / 28)
    assert value["projected_finalization_seconds"] == 5
    assert value["projected_packaging_seconds"] == 5
    assert value["projected_remaining_seconds"] == pytest.approx((2 + 60 * 20 / 28 + 5 + 5) * 1.2)
    assert value["projected_session_seconds"] == value["projected_remaining_seconds"] + 30


def test_projection_of_unsampled_conversation_uses_slowest_measured_rates():
    rows = events()
    rows.insert(2, {**rows[1], "conversation_id": "c", "history_seconds": 20, "qa_seconds": 6,
                    "worker_seconds": 26})
    value = estimate_locomo(rows, {"a": 10, "b": 10, "c": 10, "d": 20}, workers=2,
        campaign_wall=23, setup_seconds=0, packaging_seconds=1, safety_margin=0.2,
        full_turn_counts={"a": 10, "b": 10, "c": 10, "d": 50})
    assert value["historical_replay_seconds"] == 140  # 10+10+20 + 50*2, once per history
    assert value["projected_qa_work_seconds"] == 130  # 20+20+30 + 20*3
    assert value["measured_conversations"] == 3 and value["full_conversations"] == 4
    rows[0]["historical_turn_count"] = 9
    with pytest.raises(ValueError, match="full history"):
        estimate_locomo(rows, {"a": 10, "b": 10, "c": 10, "d": 20}, workers=2,
            campaign_wall=23, setup_seconds=0, packaging_seconds=1, safety_margin=0.2,
            full_turn_counts={"a": 10, "b": 10, "c": 10, "d": 50})


def test_checkpoint_append_reuses_disk_index_but_still_refuses_duplicate_ids(tmp_path, monkeypatch):
    from benchmarks.common import artifacts
    from benchmarks.common.contracts import CheckpointRecord, RunStatus
    original = artifacts.read_jsonl
    reads = []
    monkeypatch.setattr(artifacts, "read_jsonl", lambda path: reads.append(path) or original(path))
    store = BenchmarkArtifactStore(tmp_path)
    for i in range(10):
        assert store.append_checkpoint(CheckpointRecord(case_id=str(i), status=RunStatus.CANCELLED))
    assert not store.append_checkpoint(CheckpointRecord(case_id="0", status=RunStatus.CANCELLED))
    assert len(reads) == 1
    assert len(original(store.layout.checkpoints)) == 10


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "no-finalization", "nan", "zero-qa"])
def test_estimate_rejects_incomplete_or_stale_timing(mutation):
    rows = events()
    if mutation == "duplicate":
        rows.append(rows[0])
    elif mutation == "missing":
        rows.pop(0)
    elif mutation == "no-finalization":
        rows.pop()
    elif mutation == "nan":
        rows[0]["qa_seconds"] = float("nan")
    else:
        rows[0]["qa_seconds"] = 0
    with pytest.raises(ValueError):
        estimate_locomo(rows, {"a": 10, "b": 10}, workers=2, campaign_wall=23,
                        setup_seconds=30, packaging_seconds=1, safety_margin=0.2)


def test_budget_fail_closed_includes_elapsed_setup_and_packaging():
    value = estimate()
    with pytest.raises(ValueError, match="resource validation"):
        budget_gate(value, elapsed_seconds=100)
    value["resources_validated"] = True
    assert budget_gate(value, elapsed_seconds=100)["allowed"]
    with pytest.raises(ValueError, match="shortfall="):
        budget_gate(value, elapsed_seconds=12 * 3600)
    value["projected_campaign_seconds"] = 10.5 * 3600 + 1
    with pytest.raises(ValueError, match="shortfall=1.0s"):
        budget_gate(value, elapsed_seconds=100)
    value["projected_campaign_seconds"] = float("nan")
    with pytest.raises(ValueError, match="invalid"):
        budget_gate(value, elapsed_seconds=100)


def test_incremental_parser_reads_only_appends_and_retains_partial_utf8(tmp_path):
    path = tmp_path / "rows.jsonl"
    first = '{"id":"é"}\n'.encode()
    path.write_bytes(first[:-4])
    reader = IncrementalJSONL()
    assert reader.poll([path]) == []
    with path.open("ab") as stream:
        stream.write(first[-4:])
    assert reader.poll([path]) == [{"id": "é"}]
    assert reader.poll([path]) == []
    assert reader.bytes_read == len(first)
    path.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="truncated"):
        reader.poll([path])


def test_progress_detects_duplicates_and_malformed_tails(tmp_path):
    path = tmp_path / "records.jsonl"
    row = {"case_id": "a", "status": "complete", "prediction": {
        "timing": {"total_ms": 42}, "tokens": {"total_tokens": 3}}}
    persist_sample(path, row)
    progress = CheckpointProgress()
    assert progress.poll([path]) == 1
    assert progress.poll([path]) == 0
    assert progress.tokens == 3 and progress.latency_sum == 42
    persist_sample(path, row)
    with pytest.raises(ValueError, match="duplicate"):
        progress.poll([path])
    path.write_bytes(b'{"case_id":')
    with pytest.raises(ValueError, match="Incomplete"):
        IncrementalJSONL().poll([path], final=True)
    path.write_bytes(b"bad\n")
    with pytest.raises(ValueError):
        IncrementalJSONL().poll([path])


def sample():
    return {"gpus": [{"index": i, "memory_used_mib": 12000, "utilization_percent": 50} for i in range(2)],
            "ram": {"total_bytes": 19 * 10**9, "available_bytes": 4 * 10**9,
                    "process_rss_bytes": {"123": 1000}}}


def test_telemetry_survives_failure_before_exit(tmp_path, monkeypatch):
    from benchmarks.common import artifacts
    calls = []
    monkeypatch.setattr(artifacts.os, "fsync", lambda fd: calls.append(fd))
    path = tmp_path / "samples.jsonl"
    value = sample()
    persist_sample(path, value)
    value["gpus"][0]["memory_used_mib"] = 12289
    persist_sample(path, value)
    with pytest.raises(ValueError, match="VRAM"):
        resource_guard(value)
    rows = read_jsonl(path)
    assert len(rows) == len(calls) == 2
    assert rows[0]["ram"]["process_rss_bytes"] == {"123": 1000}
    assert rows[1]["gpus"][0]["memory_used_mib"] == 12289


@pytest.mark.parametrize("case", ["vram", "ram", "cgroup", "missing", "relax", "gpu-nan", "ram-nan", "cgroup-nan"])
def test_resource_limits_fail_closed(case):
    value = sample()
    kwargs = {}
    if case == "vram":
        value["gpus"][1]["memory_used_mib"] = 12289
    elif case == "ram":
        value["ram"]["available_bytes"] = 2 * 10**9
    elif case == "cgroup":
        value["ram"].update(cgroup_current_bytes=8 * 10**9, cgroup_limit_bytes=10 * 10**9)
    elif case == "missing":
        value["ram"] = {}
    elif case == "gpu-nan":
        value["gpus"][0]["memory_used_mib"] = float("nan")
    elif case == "ram-nan":
        value["ram"]["available_bytes"] = float("nan")
    elif case == "cgroup-nan":
        value["ram"].update(cgroup_current_bytes=float("nan"), cgroup_limit_bytes=10 * 10**9)
    else:
        kwargs["vram_gib"] = 13
    with pytest.raises(ValueError):
        resource_guard(value, **kwargs)


@pytest.mark.parametrize("version", [1, 2])
def test_cgroup_disk_cache_is_recorded_and_does_not_trigger_false_reserve_failure(tmp_path, version):
    directory = tmp_path if version == 2 else tmp_path / "memory"
    directory.mkdir(exist_ok=True)
    current, limit = ("memory.current", "memory.max") if version == 2 else (
        "memory.usage_in_bytes", "memory.limit_in_bytes")
    (directory / current).write_text(str(24 * 10**9))
    (directory / limit).write_text(str(25 * 10**9))
    stat = {"file": 12 * 10**9, "inactive_file": 7 * 10**9, "active_file": 4 * 10**9, "shmem": 10**9,
            "file_dirty": 400 * 10**6, "file_writeback": 300 * 10**6, "unevictable": 300 * 10**6}
    aliases = {"file": "cache", "file_dirty": "dirty", "file_writeback": "writeback"}
    (directory / "memory.stat").write_text("\n".join(
        f"{key if version == 2 else 'total_' + aliases.get(key, key)} {value}"
        for key, value in stat.items()))
    value = sample()
    value["ram"].update(read_cgroup_memory(tmp_path))
    assert value["ram"]["cgroup_stat_bytes"] == stat
    assert value["ram"]["cgroup_reclaimable_cache_bytes"] == 10 * 10**9
    assert value["ram"]["cgroup_working_set_bytes"] == 14 * 10**9
    assert value["ram"]["cgroup_available_bytes"] == 11 * 10**9
    resource_guard(value)
    # Model loading promotes the same clean pages; it must not inflate working RAM.
    stat["active_file"] += stat["inactive_file"]
    stat["inactive_file"] = 0
    (directory / "memory.stat").write_text("\n".join(
        f"{key if version == 2 else 'total_' + aliases.get(key, key)} {value}"
        for key, value in stat.items()))
    value["ram"].update(read_cgroup_memory(tmp_path))
    assert value["ram"]["cgroup_working_set_bytes"] == 14 * 10**9
    resource_guard(value)
    (directory / "memory.stat").unlink()
    value["ram"].update(read_cgroup_memory(tmp_path))
    with pytest.raises(ValueError, match="raw=24.000 GB.*clean disk cache=0.000 GB"):
        resource_guard(value)


@pytest.mark.parametrize("mutation", ["no-file-lru", "dirty", "writeback", "shmem", "locked", "missing", "negative",
                                       "active-missing", "active-negative"])
def test_cgroup_cache_credit_cannot_hide_unreclaimable_memory(mutation):
    stat = {"file": 5 * 10**9, "inactive_file": 5 * 10**9, "active_file": 0, "shmem": 0,
            "file_dirty": 0, "file_writeback": 0, "unevictable": 0}
    key = {"no-file-lru": "inactive_file", "dirty": "file_dirty", "writeback": "file_writeback",
           "shmem": "shmem", "locked": "unevictable", "negative": "file_dirty",
           "active-negative": "active_file"}.get(mutation)
    if mutation in ("missing", "active-missing"):
        stat.pop("active_file" if mutation == "active-missing" else "file_writeback")
    else:
        stat[key] = 0 if mutation == "no-file-lru" else -1 if "negative" in mutation else 5 * 10**9
    value = sample()
    value["ram"].update(cgroup_current_bytes=18 * 10**9, cgroup_limit_bytes=19 * 10**9,
                        cgroup_stat_bytes=stat, cgroup_reclaimable_cache_bytes=18 * 10**9)
    with pytest.raises(ValueError, match="Cgroup RAM reserve violated"):
        resource_guard(value)


def test_cgroup_working_set_still_obeys_the_19_gb_plan():
    value = sample()
    value["ram"].update(cgroup_current_bytes=17 * 10**9, cgroup_limit_bytes=32 * 10**9)
    with pytest.raises(ValueError, match="working=17.000 GB"):
        resource_guard(value)


def test_only_explicit_two_slot_isolated_locomo_concurrency_allowed(tmp_path):
    source = payload(tmp_path)
    source["provider"].update(kind="ollama", endpoint="http://localhost:11434")
    assert CampaignConfig.model_validate(source).scheduler.max_gpu_requests == 2
    for field, value in (("max_gpu_requests", 3), ("ollama_parallel_slots", 1), ("cpu_workers", 1), ("mode", "interleaved")):
        changed = copy.deepcopy(source)
        changed["scheduler"][field] = value
        with pytest.raises(ValidationError):
            CampaignConfig.model_validate(changed)
    changed = copy.deepcopy(source)
    changed["benchmarks"][0]["options"]["parallel_workers"] = 1
    with pytest.raises(ValidationError):
        CampaignConfig.model_validate(changed)


def test_pilot_prediction_change_or_nonoverlap_rejects_concurrency():
    row = {"case_id": "a", "prediction": "Alpha", "execution_failed": False}
    validate_pilot([row], [row], ["a"], resources_validated=True, max_active_requests=2)
    for changed, active, safe in (({**row, "prediction": "Beta"}, 2, True), (row, 1, True), (row, 2, False)):
        with pytest.raises(ValueError):
            validate_pilot([row], [changed], ["a"], resources_validated=safe, max_active_requests=active)


def test_fake_provider_measures_real_overlap_at_two_request_limit(tmp_path):
    from benchmarks.campaign.provider_session import SharedProviderSession
    from llm.generation_config import GenerationConfig
    from llm.llm_types import LLMRequest
    source = payload(tmp_path)
    source["provider"]["fake_delay_seconds"] = 0.02
    config = CampaignConfig.model_validate(source)
    session = SharedProviderSession(config.provider, max_active_requests=2)
    requests = [LLMRequest(prompt=f"unique {i}", generation=GenerationConfig(model="fixture", provider="fake"))
                for i in range(4)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        responses = list(executor.map(session.provider.send, requests))
    telemetry = session.telemetry()
    assert len(responses) == telemetry["successful_requests"] == 4
    assert telemetry["max_active_requests"] == 2
    assert telemetry["response_cache_hits"] == 0
    assert telemetry["inference_wall_time_seconds"] > 0


def test_explicit_production_identities_required():
    require_production_identity("a" * 40, "b" * 64, "c" * 64)
    with pytest.raises(ValueError, match="repository"):
        require_production_identity(None, "b" * 64, "c" * 64)


def test_two_workers_keep_full_history_question_order_isolated_memory_and_artifacts(tmp_path):
    reset_fake()
    source = fixture()
    source += [{**copy.deepcopy(source[0]), "sample_id": "conversation-2"}]
    dataset = tmp_path / "locomo.json"
    dataset.write_text(json.dumps(source))

    class IsolatedRuntime(FakeRuntime):
        instances = []

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.ordered = []
            self.instances.append(self)

        async def execute(self, request):
            self.ordered.append((request.session_id, request.task_kind.value, request.input_text))
            # Synthetic provider delay only; not a hardware speed claim.
            time.sleep(0.002)
            return await super().execute(request)

    outputs = []
    for workers in (1, 2):
        result = run_locomo_experiment(
            dataset_path=str(dataset), output_path=str(tmp_path / str(workers)), runtime_profile="prima_full",
            ingestion_policy="normal_prima_admission", maintenance_mode="flush_before_question",
            max_conversations=2, max_questions=2, parallel_workers=workers, provider="test", model="fixture",
            runtime_factory=IsolatedRuntime,
        )
        assert result["questions"] == 4  # max_questions is per conversation
        root = Path(result["artifacts"]["manifest"]).parent
        store = BenchmarkArtifactStore(root)
        checkpoints = store.checkpoints()
        records = [row.prediction.diagnostics["locomo_record"] for row in checkpoints]
        events_actual = read_jsonl(root / "timing.jsonl")
        convo = [e for e in events_actual if e["phase"] == "conversation"]
        assert len(convo) == 2 and all(e["historical_turn_count"] == 2 for e in convo)
        assert len([e for e in events_actual if e["phase"] == "history_replay"]) == 2
        assert any(e["phase"] == "finalization" for e in events_actual)
        assert store.read_manifest().status.value == "complete"
        outputs.append(records)
    validate_pilot(outputs[0], outputs[1], [r["case_id"] for r in outputs[0]],
                   resources_validated=True, max_active_requests=2)
    for instance in IsolatedRuntime.instances:
        assert len({r[0] for r in instance.ordered}) == 1
        assert [r[1] for r in instance.ordered] == ["historical_replay", "historical_replay", "factual_qa", "factual_qa"]
        assert [r[2] for r in instance.ordered[-2:]] == ["What happened?", "Unknown?"]
    assert len(IsolatedRuntime.instances) == 4


def test_notebook_contract_and_full_gate_no_launch_without_measurement():
    notebook = json.loads((REPO / "PRIMA_Kaggle_Benchmark.ipynb").read_text(encoding="utf-8"))
    code = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
    for cell in code:
        ast.parse(cell)
    ns = {}
    exec(code[0], ns)  # noqa: S102 - execute checked-in notebook config in an isolated namespace
    config = ns["CONFIG"]
    assert config["RUN_FULL_BENCHMARK"] is False
    assert config["SMOKE_MODE"] is True and config["SMOKE_ITEMS"] == 2
    assert config["GPU_MEMORY_CEILING_GIB"] == 12
    assert config["MAINTENANCE_MODE"] == "flush_before_question"
    assert config["CONTEXT_LENGTH"] == 8192 and config["MAX_OUTPUT_TOKENS"] == 512
    assert config["PROVIDER_RETRIES"] == 1 and config["RUN_PAIRED_BASELINE"] is False
    assert config["OPTIONAL_METRICS"]["locomo"] == []
    assert config["EXPECTED_LOCOMO_ITEMS"] == 1986
    assert "REDUCED_CONTEXT_LENGTH" not in config
    full_cell = "".join(notebook["cells"][20]["source"])
    with pytest.raises(RuntimeError, match="Smoke mode"):
        exec(full_cell, ns)  # noqa: S102
    config["SMOKE_MODE"] = False
    with pytest.raises(RuntimeError, match="disabled"):
        exec(full_cell, ns)  # noqa: S102 - disabled gate must stop before any launch
    assert full_cell.index("budget_gate(") < full_cell.index("execute_campaign(")
    assert "fresh=True" in "".join(notebook["cells"][18]["source"])


@pytest.mark.parametrize("cell_index,message", [(8, "tracked edits"), (20, "tracked changes")])
def test_notebook_persisted_checkout_ignores_missing_external_dirs_but_rejects_source_edits(tmp_path, cell_index, message):
    git_path = shutil.which("git")
    assert git_path

    def git(*args):
        return subprocess.run([git_path, *args], cwd=tmp_path, check=True, capture_output=True, text=True)  # noqa: S603  # nosec B603 - fixed Git fixture commands

    git("init", "-q")
    source = tmp_path / "owned.py"
    source.write_text("original\n")
    git("add", "owned.py")
    commit_args = ("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                   "commit", "-qm", "fixture")
    git(*commit_args)
    revision = git("rev-parse", "HEAD").stdout.strip()
    for benchmark in ("hotpotqa", "locomo", "locomo-v2"):
        git("update-index", "--add", "--cacheinfo", f"160000,{revision},benchmarks/{benchmark}/external")
    git(*commit_args)
    assert "external" in git("status", "--porcelain", "--ignore-submodules=dirty").stdout
    notebook = json.loads((REPO / "PRIMA_Kaggle_Benchmark.ipynb").read_text(encoding="utf-8"))
    tree = ast.parse(notebook["cells"][cell_index]["source"])
    if cell_index == 8:
        checkout = next(node for node in tree.body if isinstance(node, ast.If) and node.orelse)
        checks = checkout.orelse[:2]
    else:
        index = next(i for i, node in enumerate(tree.body)
                     if isinstance(node, ast.Assign) and node.targets[0].id == "tracked_changes")
        checks = tree.body[index:index + 2]
    guard = compile(ast.Module(body=checks, type_ignores=[]), "notebook-checkout-guard", "exec")
    ns = {"REPOSITORY_DIR": tmp_path, "run_checked": lambda args, **kwargs: git(*args[1:])}
    exec(guard, ns)  # noqa: S102 - checked-in guard only; no clone, fetch or dependency installation
    source.write_text("modified\n")
    with pytest.raises(RuntimeError, match=message):
        exec(guard, ns)  # noqa: S102


def test_notebook_updates_persisted_shallow_branch_without_discarding_local_commits(tmp_path):
    git_path = shutil.which("git")
    assert git_path
    remote, checkout = tmp_path / "origin", tmp_path / "checkout"
    remote.mkdir()

    def git(*args, cwd=remote):
        return subprocess.run([git_path, *args], cwd=cwd, check=True, capture_output=True, text=True)  # noqa: S603  # nosec B603 - fixed local Git fixture commands

    branch = "3.10_to_3.14"
    git("init", "-q", "-b", branch)
    commit_args = ("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false",
                   "commit", "-qam", "fixture")
    source = remote / "owned.py"
    source.write_text("original\n")
    git("add", "owned.py")
    git(*commit_args)
    notebook = json.loads((REPO / "PRIMA_Kaggle_Benchmark.ipynb").read_text(encoding="utf-8"))
    tree = ast.parse(notebook["cells"][8]["source"])
    tree.body = tree.body[:next(i for i, node in enumerate(tree.body) if isinstance(node, ast.FunctionDef))]
    setup = compile(tree, "notebook-branch-setup", "exec")
    ns = {"CONFIG": {"GIT_REF": branch, "REPOSITORY_URL": remote.as_uri()}, "REPOSITORY_DIR": checkout,
          "subprocess": subprocess, "time": time,
          "run_checked": lambda args, cwd=None, **kwargs: git(*args[1:], cwd=cwd or remote)}
    exec(setup, ns)  # noqa: S102 - checked-in Git setup only, local fixture origin
    assert git("rev-parse", "--is-shallow-repository", cwd=checkout).stdout.strip() == "true"
    source.write_text("upstream update\n")
    git(*commit_args)
    # Reproduce the old notebook's two disconnected grafted tips.
    git("fetch", "--depth", "1", "origin", f"+refs/heads/{branch}:refs/remotes/origin/{branch}", cwd=checkout)
    with pytest.raises(subprocess.CalledProcessError):
        git("merge", "--ff-only", f"origin/{branch}", cwd=checkout)
    exec(setup, ns)  # noqa: S102
    assert (checkout / "owned.py").read_text() == "upstream update\n"
    assert git("rev-parse", "--is-shallow-repository", cwd=checkout).stdout.strip() == "false"
    source.write_text("another upstream update\n")
    git(*commit_args)
    exec(setup, ns)  # noqa: S102 - subsequent non-shallow update must also work
    assert (checkout / "owned.py").read_text() == "another upstream update\n"
    (checkout / "owned.py").write_text("local committed work\n")
    git(*commit_args, cwd=checkout)
    local_head = git("rev-parse", "HEAD", cwd=checkout).stdout.strip()
    source.write_text("divergent upstream update\n")
    git(*commit_args)
    with pytest.raises(subprocess.CalledProcessError):
        exec(setup, ns)  # noqa: S102 - retain fast-forward-only refusal for local commits
    assert git("rev-parse", "HEAD", cwd=checkout).stdout.strip() == local_head
    assert (checkout / "owned.py").read_text() == "local committed work\n"


def notebook_namespace(*, smoke=False):
    """Load only definitions from checked-in cells; never run setup or inference cells."""
    notebook = json.loads((REPO / "PRIMA_Kaggle_Benchmark.ipynb").read_text(encoding="utf-8"))
    ns = {}
    exec("".join(notebook["cells"][2]["source"]), ns)  # noqa: S102 - trusted notebook definitions
    ns["CONFIG"]["SMOKE_MODE"] = smoke
    for index in (16, 18):
        tree = ast.parse("".join(notebook["cells"][index]["source"]))
        tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
        exec(compile(tree, f"notebook-cell-{index}", "exec"), ns)  # noqa: S102 - imports/functions only
    return ns


@pytest.mark.parametrize("field,value", [
    ("RESOURCE_SAMPLE_SECONDS", 3), ("RESOURCE_SAMPLE_SECONDS", float("nan")),
    ("SESSION_BUDGET_HOURS", 13), ("CAMPAIGN_BUDGET_HOURS", 11),
    ("RUNTIME_SAFETY_MARGIN", 0.19), ("FINALIZATION_RESERVE_SECONDS", 899),
])
def test_notebook_rejects_relaxed_guards_before_model_setup(field, value):
    notebook = json.loads((REPO / "PRIMA_Kaggle_Benchmark.ipynb").read_text(encoding="utf-8"))
    ns = notebook_namespace()
    ns["CONFIG"][field] = value
    with pytest.raises(ValueError):
        exec("".join(notebook["cells"][4]["source"]), ns)  # noqa: S102 - guard before any setup


def test_notebook_ram_sampling_tolerates_a_process_exiting_during_poll(monkeypatch):
    class ProcessGone(Exception):
        pass

    class Process:
        pid = 123

        def __init__(self, pid):
            self.pid = pid

        def children(self, recursive):
            raise ProcessGone()

    monkeypatch.setitem(__import__("sys").modules, "psutil", types.SimpleNamespace(
        Process=Process, NoSuchProcess=ProcessGone, AccessDenied=PermissionError))
    ns = notebook_namespace()
    ns.update(os=__import__("os"), OLLAMA_PROCESS=None, system_ram=lambda: sample()["ram"])
    ram = ns["measured_ram"]()
    assert ram["available_bytes"] == 4 * 10**9
    assert ram["process_tree_rss_bytes"] == 0


def test_notebook_fresh_canary_monitor_validation_and_packaging_with_fake_campaign(tmp_path, monkeypatch):
    from benchmarks.campaign import provider_session
    from benchmarks.campaign.config import load_campaign_config
    from benchmarks.campaign.orchestrator import run_campaign
    from benchmarks.locomo.loader import LoCoMoDataset

    ns = notebook_namespace()
    ns.update(json=json, subprocess=subprocess, time=time, REPOSITORY_DIR=REPO, OUTPUT_ROOT=tmp_path,
              SELECTED_BENCHMARK="locomo", SESSION_STAMP="synthetic-test", OLLAMA_ENDPOINT="http://unused",
              prepare_run=prepare_run, CheckpointProgress=CheckpointProgress, persist_sample=persist_sample)
    source = fixture() + [{**copy.deepcopy(fixture()[0]), "sample_id": "conversation-2"}]
    dataset = tmp_path / "dataset.json"
    dataset.write_text(json.dumps(source))
    from datetime import datetime, timezone

    from benchmarks.locomo.experiment import fingerprint, git_commit
    ns.update(sys=__import__("sys"), client_env={}, datetime=datetime, timezone=timezone,
              shutil=__import__("shutil"), re=__import__("re"))
    ns["sha256_file"] = lambda path: fingerprint(Path(path))
    ns["CONFIG"]["FULL_RUN_ID"] = "synthetic-full"
    ns.update(SELECTED_DATASET=dataset, conversations=list(LoCoMoDataset(dataset).conversations()),
              dataset_summary={"items": 6, "sha256": fingerprint(dataset)}, STATE={
                  "repository": {"url": "https://example.invalid", "requested_ref": "test", "commit_sha": git_commit()},
                  "model_identity": {"name": "fake", "digest": "b" * 64},
                  "preflight": {"actual_context_length": 8192}, "mode_id": "locomo-prima",
                  "ollama_parallel_slots": 1, "timing_started": time.monotonic(),
              })
    monkeypatch.setattr(provider_session.SharedProviderSession, "runtime_factory", lambda self, **kw: FakeRuntime(**kw))

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs):
            config = load_campaign_config(command[command.index("--config") + 1])
            config = config.model_copy(update={"provider": config.provider.model_copy(update={"kind": "fake"})})
            run_campaign(config)
            self.first_poll = True

        def poll(self):
            if self.first_poll:
                self.first_poll = False
                return None
            return 0

        def wait(self):
            return 0

    ns["subprocess"] = types.SimpleNamespace(Popen=FakeProcess, STDOUT=subprocess.STDOUT,
                                            TimeoutExpired=subprocess.TimeoutExpired)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    ns["safety_sample"] = lambda started, process=None: {**sample(), "timestamp": "synthetic", "elapsed_seconds": 1}
    ns["check_safety"] = resource_guard
    config, config_path, root, _ = ns["build_campaign_config"]("canary-1", full=False, max_items=2)
    samples, log, wall = ns["execute_campaign"](config_path, root, 4, fresh=True)
    manifest, records, timing = ns["validate_canary"](config, root)
    assert len(records) == ns["STATE"]["canary_measured_items"] == 4
    assert samples and log.is_file() and wall > 0
    assert ns["measure_canary_packaging"](root, manifest) > 0
    assert (root.parent / f"{root.name}-validated.zip").is_file()
    assert read_jsonl(root / "resource_samples.jsonl")
    assert read_jsonl(root / "progress.jsonl")[-1]["event"] == "exit"
    estimate_locomo(timing, {"conversation-1": 3, "conversation-2": 3}, workers=1,
                    campaign_wall=wall, setup_seconds=0, packaging_seconds=1, safety_margin=0.2)
    with pytest.raises(ValueError, match="Fresh"):
        ns["execute_campaign"](config_path, root, 4, fresh=True)
    full_config, _, _, _ = ns["build_campaign_config"]("full", full=True)
    assert full_config["benchmarks"][0]["max_items"] == 0
    assert len(ns["expected_ids"](full_config)["locomo-prima"]) == 6
    bad_sample = sample()
    bad_sample["gpus"][0]["memory_used_mib"] = 13000
    ns["safety_sample"] = lambda started, process=None: {
        **bad_sample, "timestamp": "synthetic", "elapsed_seconds": 1}
    stopped = []
    ns["stop_ollama"] = lambda: stopped.append(True)
    _, unsafe_path, unsafe_root, _ = ns["build_campaign_config"]("unsafe", full=False, max_items=2)
    with pytest.raises(ValueError, match="VRAM"):
        ns["execute_campaign"](unsafe_path, unsafe_root, 4, fresh=True)
    assert stopped
    assert read_jsonl(unsafe_root / "resource_samples.jsonl")[0]["gpus"][0]["memory_used_mib"] == 13000
    assert read_jsonl(unsafe_root / "progress.jsonl")[-1]["event"] == "resource_failure"
    # A violation after the child exits must also release the model and persist evidence.
    stopped.clear()
    final_samples = iter([sample(), bad_sample])
    ns["safety_sample"] = lambda started, process=None: {
        **next(final_samples), "timestamp": "synthetic", "elapsed_seconds": 1}
    _, final_path, final_root, _ = ns["build_campaign_config"]("unsafe-final", full=False, max_items=2)
    with pytest.raises(ValueError, match="VRAM"):
        ns["execute_campaign"](final_path, final_root, 4, fresh=True)
    assert stopped == [True]
    assert read_jsonl(final_root / "resource_samples.jsonl")[-1]["gpus"][0]["memory_used_mib"] == 13000
    assert read_jsonl(final_root / "progress.jsonl")[-1]["event"] == "resource_failure"
