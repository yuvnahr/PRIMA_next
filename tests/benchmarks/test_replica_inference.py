"""No-hardware checks for explicit replica routing and residency guards."""

import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest
from pydantic import ValidationError

from benchmarks.campaign.config import ProviderConfig
from benchmarks.campaign.kaggle import validate_replica_models
from benchmarks.campaign.provider_session import FakeProvider, _ReplicaProvider
from llm.generation_config import GenerationConfig
from llm.llm_types import LLMRequest

ENDPOINTS = ("http://127.0.0.1:11434", "http://127.0.0.1:11435")
DIGEST = "a" * 64


@pytest.mark.parametrize("endpoints", [
    (ENDPOINTS[0], "http://localhost:11434"),
    (ENDPOINTS[0], "https://127.0.0.1:11435"),
    (ENDPOINTS[0], "http://example.com:11435"),
    (ENDPOINTS[0], "http://127.0.0.1:11435/api"),
])
def test_replica_config_rejects_ambiguous_or_nonlocal_origins(endpoints):
    with pytest.raises(ValidationError):
        ProviderConfig(kind="ollama", model="pinned", revision=DIGEST,
                       endpoint=ENDPOINTS[0], replica_endpoints=endpoints, context_window=8192)


def test_replica_workers_are_sticky_and_really_overlap():
    barrier = threading.Barrier(2)
    calls = [[], []]

    class RecordingProvider(FakeProvider):
        def __init__(self, index):
            super().__init__()
            self.index = index

        def send(self, request):
            calls[self.index].append(threading.get_ident())
            barrier.wait(timeout=5)
            return super().send(request)

    router = _ReplicaProvider([RecordingProvider(0), RecordingProvider(1)], ENDPOINTS)
    request = LLMRequest(prompt="hello", generation=GenerationConfig(model="pinned", provider="ollama"))

    def worker():
        for _ in range(2):
            router.send(request)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker) for _ in range(2)]
        for future in futures:
            future.result(timeout=10)
    assert all(len(call) == 2 and len(set(call)) == 1 for call in calls)
    assert calls[0][0] != calls[1][0]
    assert router.telemetry()["max_active_requests"] == 2
    assert [row["request_attempts"] for row in router.telemetry()["endpoints"]] == [2, 2]


@pytest.mark.parametrize("field,value", [
    ("digest", "b" * 64), ("context_length", 4096), ("size_vram", 99),
    ("size", -1), ("size", True), ("name", "different"),
])
def test_second_replica_cannot_bypass_identity_or_residency(field, value):
    placements = [{"endpoint": endpoint, "model": {
        "name": "pinned", "digest": DIGEST, "context_length": 8192,
        "size": 100, "size_vram": 100,
    }} for endpoint in ENDPOINTS]
    validate_replica_models(placements, list(ENDPOINTS), "pinned", DIGEST, 8192)
    corrupted = deepcopy(placements)
    corrupted[1]["model"][field] = value
    with pytest.raises(ValueError, match="identity/context/residency"):
        validate_replica_models(corrupted, list(ENDPOINTS), "pinned", DIGEST, 8192)


@pytest.mark.parametrize("wrong_gpu", [False, True])
def test_notebook_warms_replicas_on_separate_gpus(tmp_path, monkeypatch, wrong_gpu):
    from tests.benchmarks.test_kaggle import notebook_namespace

    class ImmediateExecutor:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def submit(self, function, *args, **kwargs):
            result = function(*args, **kwargs)
            return types.SimpleNamespace(done=lambda: True, result=lambda: result)

    monkeypatch.setattr("concurrent.futures.ThreadPoolExecutor", ImmediateExecutor)
    ns = notebook_namespace()
    ns["CONFIG"]["INFERENCE_TOPOLOGY"] = "replicated"
    snapshots = iter([[0, 0], [0, 1000] if wrong_gpu else [1000, 0],
                      [0, 1000] if wrong_gpu else [1000, 0],
                      [1000, 0], [1000, 1000], [1000, 1000]])
    origins, stopped = [], []

    def request(path, payload, timeout, *, endpoint):
        origins.append(endpoint)
        return {"response": "READY", "load_duration": 100}

    ns.update(time=time, OUTPUT_ROOT=tmp_path, STATE={"ollama_context_length": 8192},
              nvidia_rows=lambda: [{"memory_used_mib": value} for value in next(snapshots)],
              system_ram=lambda: {}, read_cgroup_memory=lambda: {},
              persist_sample=lambda *args: None, resource_guard=lambda *args, **kwargs: None,
              url_json=request, stop_ollama=lambda: stopped.append(True))
    if wrong_gpu:
        with pytest.raises(RuntimeError, match="assigned GPU"):
            ns["warm_model"]()
        assert stopped == [True]
    else:
        before, after, warmup, _ = ns["warm_model"]()
        assert origins == list(ENDPOINTS)
        assert warmup["load_duration"] == 200
        assert [row["gpu_device"] for row in warmup["replicas"]] == ["0", "1"]
        assert before == [{"memory_used_mib": 0}] * 2
        assert after == [{"memory_used_mib": 1000}] * 2
