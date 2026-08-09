from __future__ import annotations

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

    store = CampaignManifestStore(config.output_root)
    manifest = store.read()
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
    assert session.telemetry()["max_active_requests"] == 1


def test_isolated_mode_failure_does_not_corrupt_other_modes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from benchmarks.campaign import orchestrator

    original = orchestrator.execute_mode

    def fail_hotpot(config, mode, session, output_root, *, resume):
        if mode.id == "hotpot":
            raise RuntimeError("isolated fixture failure")
        return original(config, mode, session, output_root, resume=resume)

    monkeypatch.setattr(orchestrator, "execute_mode", fail_hotpot)
    result = run_campaign(CampaignConfig.model_validate(_payload(tmp_path / "campaign")))
    assert result["status"] == "partial"
    assert result["modes"]["hotpot"]["status"] == "failed"
    assert result["modes"]["go"]["status"] == "complete"
    assert result["modes"]["locomo"]["status"] == "complete"
