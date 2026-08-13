from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.campaign.config import CampaignConfig, load_campaign_config
from benchmarks.campaign.manifest import CampaignManifestStore
from benchmarks.campaign.orchestrator import run_campaign
from benchmarks.campaign.provider_session import SharedProviderSession
from benchmarks.common.artifacts import read_jsonl
from llm.generation_config import GenerationConfig
from llm.llm_types import LLMRequest, LLMResponse

REPOSITORY = Path(__file__).parents[2]
FIXTURES = REPOSITORY / "tests" / "fixtures"


@pytest.fixture(autouse=True)
def _disable_locomo_file_log(monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmarks.locomo import loader

    monkeypatch.setattr(loader.logger, "disabled", True)


def _payload(output: Path) -> dict:
    return {
        "schema_version": "1.0",
        "output_root": str(output),
        "provider": {
            "kind": "fake",
            "model": "fake-campaign-model",
            "revision": "fixture-v1",
            "context_window": 4096,
        },
        "scheduler": {"max_gpu_requests": 1, "cpu_workers": 2, "min_free_disk_gb": 0},
        "benchmarks": [
            {
                "id": "go",
                "benchmark": "goemotions",
                "dataset_path": str(FIXTURES / "goemotions" / "test.tsv"),
                "profile": "affect_only",
                "variant": "model_only_zero_shot",
                "context_budget": 512,
                "max_items": 1,
                "options": {"split": "test", "bootstrap_samples": 10},
            },
            {
                "id": "hotpot",
                "benchmark": "hotpotqa",
                "dataset_path": str(FIXTURES / "campaign" / "hotpot.json"),
                "profile": "model_only",
                "variant": "distractor",
                "context_budget": 2048,
                "max_items": 2,
            },
            {
                "id": "locomo",
                "benchmark": "locomo",
                "dataset_path": str(FIXTURES / "campaign" / "locomo.json"),
                "profile": "model_only",
                "variant": "controlled_document_ingestion",
                "context_budget": 2048,
                "max_items": 1,
                "options": {"max_conversations": 1},
            },
        ],
    }


def test_smoke_campaign_runs_three_canonical_benchmarks_and_resumes(tmp_path: Path) -> None:
    config = CampaignConfig.model_validate(_payload(tmp_path / "campaign"))
    first = run_campaign(config)
    assert first["status"] == "complete"
    assert set(first["modes"]) == {"go", "hotpot", "locomo"}
    assert first["telemetry"]["max_active_requests"] <= 1
    assert "gpu_peak_observed_memory_bytes" not in first["telemetry"]
    assert first["telemetry"]["gpu_boundary_snapshots"]["measurement"].endswith("not peak usage")

    store = CampaignManifestStore(config.output_root)
    manifest = store.read()
    assert len(manifest.preflight["runtime_config_fingerprint"]) == 64
    assert manifest.preflight["effective_generation"]["hotpot"]["structured_output"] == "json_schema"
    assert all(
        state.child_manifest and not Path(state.child_manifest).is_absolute()
        for state in manifest.modes.values()
    )
    artifact_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in config.output_root.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md"}
    )
    assert str(REPOSITORY) not in artifact_text
    assert str(tmp_path) not in artifact_text
    store.update_mode("hotpot", status="partial")
    second = run_campaign(config, resume=True)
    assert second["campaign_id"] == manifest.campaign_id
    checkpoint = config.output_root / "modes" / "hotpot" / "distractor" / "model_only" / "checkpoints" / "records.jsonl"
    case_ids = [row["case_id"] for row in read_jsonl(checkpoint)]
    assert len(case_ids) == len(set(case_ids)) == 2


def test_comparison_with_different_seed_is_refused_during_validation(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    right = dict(payload["benchmarks"][1])
    right.update(id="hotpot-other", seed=99)
    payload["benchmarks"].append(right)
    payload["comparisons"] = [{"id": "invalid", "left": "hotpot", "right": "hotpot-other", "metric": "metrics.f1"}]
    with pytest.raises(ValidationError, match="same seed"):
        CampaignConfig.model_validate(payload)


def test_comparison_with_different_selected_order_is_refused(tmp_path: Path) -> None:
    payload = _payload(tmp_path / "campaign")
    left = dict(payload["benchmarks"][1])
    left.update(id="hotpot-left", max_items=1)
    right = dict(left)
    right.update(id="hotpot-right", options={"offset": 1})
    payload["benchmarks"] = [left, right]
    payload["comparisons"] = [
        {"id": "selected-ids", "left": "hotpot-left", "right": "hotpot-right", "metric": "metrics.f1"}
    ]
    result = run_campaign(CampaignConfig.model_validate(payload))
    assert result["status"] == "partial"
    assert result["comparisons"]["selected-ids"]["status"] == "refused"
    assert "selected_ids_and_order" in result["comparisons"]["selected-ids"]["error"]


def test_full_example_refuses_unresolved_environment_fields() -> None:
    with pytest.raises(ValueError, match="Unresolved campaign environment placeholder"):
        load_campaign_config(REPOSITORY / "benchmarks" / "campaign" / "full_gpu.example.yaml", environment={})


def test_shared_provider_enforces_active_request_bound() -> None:
    session = SharedProviderSession(
        CampaignConfig.model_validate(_payload(Path("unused"))).provider,
        max_active_requests=1,
    )
    lock = threading.Lock()
    active = maximum = 0

    class SlowProvider:
        name = "slow"
        capabilities = session.provider.capabilities

        def validate(self, request: LLMRequest) -> None:
            return None

        def send(self, request: LLMRequest) -> LLMResponse:
            nonlocal active, maximum
            with lock:
                active += 1
                maximum = max(maximum, active)
            time.sleep(0.01)
            with lock:
                active -= 1
            return LLMResponse(text="Alpha", raw={}, provider="slow")

    session.provider.provider = SlowProvider()
    request = LLMRequest(prompt="test", generation=GenerationConfig(model="fake", provider="fake"))
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(session.provider.send, [request] * 8))
    assert maximum == 1
    telemetry = session.telemetry()
    assert telemetry["max_active_requests"] == 1
    assert telemetry["throughput_requests_per_second"] > 0
    assert "provider_session_initialization_ms" in telemetry
    assert "model_load_time" not in telemetry


def test_local_gpu_concurrency_above_one_is_refused_even_with_device_names(tmp_path: Path) -> None:
    payload = _payload(tmp_path)
    payload["provider"].update(kind="local", endpoint="http://localhost:8000")
    payload["scheduler"].update(max_gpu_requests=2, gpu_devices=["0", "1"])
    with pytest.raises(ValidationError, match="explicit endpoint routing"):
        CampaignConfig.model_validate(payload)


def test_isolated_mode_failure_does_not_corrupt_other_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmarks.campaign import orchestrator

    original = orchestrator.execute_mode

    def fail_hotpot(config, mode, session, output_root, *, resume, cpu_workers=None):
        if mode.id == "hotpot":
            raise RuntimeError("isolated fixture failure")
        return original(config, mode, session, output_root, resume=resume, cpu_workers=cpu_workers)

    monkeypatch.setattr(orchestrator, "execute_mode", fail_hotpot)
    result = run_campaign(CampaignConfig.model_validate(_payload(tmp_path / "campaign")))
    assert result["status"] == "partial"
    assert result["modes"]["hotpot"]["status"] == "failed"
    assert result["modes"]["go"]["status"] == "complete"
    assert result["modes"]["locomo"]["status"] == "complete"


def test_preflight_uses_effective_qa_schema_and_full_window_budget(tmp_path: Path) -> None:
    payload = _payload(tmp_path / "schema")
    payload["benchmarks"] = [payload["benchmarks"][1]]
    payload["provider"]["structured_output"] = False
    with pytest.raises(ValueError, match="require provider JSON-schema"):
        run_campaign(CampaignConfig.model_validate(payload))

    payload = _payload(tmp_path / "window")
    payload["benchmarks"] = [payload["benchmarks"][1]]
    payload["benchmarks"][0]["context_budget"] = 3800
    with pytest.raises(ValueError, match="context, output, and safety overhead"):
        run_campaign(CampaignConfig.model_validate(payload))

@pytest.mark.parametrize("interrupt", ["sigint", "sigterm"])
def test_campaign_process_signal_resume_has_one_record_per_case(tmp_path: Path, interrupt: str) -> None:
    payload = _payload(tmp_path / interrupt)
    payload["provider"]["fake_delay_seconds"] = 0.4
    payload["benchmarks"] = [payload["benchmarks"][1]]
    source_rows = json.loads((FIXTURES / "campaign" / "hotpot.json").read_text(encoding="utf-8"))
    interrupt_rows = []
    for index in range(6):
        row = dict(source_rows[index % len(source_rows)])
        row.update(_id=f"interrupt-hotpot-{index}", question=f"{row['question']} Case {index}")
        interrupt_rows.append(row)
    dataset_path = tmp_path / f"{interrupt}-hotpot.json"
    dataset_path.write_text(json.dumps(interrupt_rows), encoding="utf-8")
    payload["benchmarks"][0].update(dataset_path=str(dataset_path), max_items=len(interrupt_rows))
    config_path = tmp_path / f"{interrupt}.json"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(  # noqa: S603  # nosec B603 - fixed interpreter and module
        [sys.executable, "-m", "benchmarks.campaign.cli", "run", "--config", str(config_path)],
        cwd=REPOSITORY,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    checkpoint = (
        Path(payload["output_root"])
        / "modes"
        / "hotpot"
        / "distractor"
        / "model_only"
        / "checkpoints"
        / "records.jsonl"
    )
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and len(read_jsonl(checkpoint)) < 1:
        if process.poll() is not None:
            pytest.fail(f"campaign exited before interruption with {process.returncode}")
        time.sleep(0.02)
    interrupted_count = len(read_jsonl(checkpoint))
    assert 0 < interrupted_count < len(interrupt_rows)
    if interrupt == "sigint":
        process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGINT)
    else:
        process.send_signal(signal.SIGTERM)
    process.wait(timeout=10)
    assert process.returncode != 0

    config = load_campaign_config(config_path)
    result = run_campaign(config, resume=True)
    assert result["status"] == "complete"
    checkpoints = read_jsonl(checkpoint)
    predictions = read_jsonl(checkpoint.parents[1] / "predictions" / "records.jsonl")
    assert len(checkpoints) == len({row["case_id"] for row in checkpoints}) == len(interrupt_rows)
    assert len(predictions) == len({row["case_id"] for row in predictions}) == len(interrupt_rows)
