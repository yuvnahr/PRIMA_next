from __future__ import annotations

import asyncio

from llm.generation_config import FallbackPolicy, GenerationConfig
from llm.llm_types import LLMResponse
from llm.provider import ProviderError
from memory.memory_repository import InMemoryMemoryRepository
from runtime import DiagnosticMode, ExecutionOutcome, ExecutionProfile, PrimaRequest, PrimaRuntime, TaskKind
from security.redaction import redact


class RecordingClient:
    provider_name = "test"
    capabilities = {"system_prompts": True, "token_accounting": True}

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls: list[dict[str, object]] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise ProviderError("provider unavailable")
        return LLMResponse("generated", usage={"prompt_tokens": 5, "completion_tokens": 2})


def _runtime(tmp_path, client: RecordingClient, fallback: FallbackPolicy = FallbackPolicy.FAIL):
    return PrimaRuntime(
        llm_client=client,
        generation_config=GenerationConfig(
            model="fixture-model",
            provider="test",
            fallback_policy=fallback,
        ),
        memory_repository=InMemoryMemoryRepository(),
        log_path=tmp_path / "runtime.log",
    )


def test_runtime_reuses_client_and_reports_real_generation_diagnostics(tmp_path) -> None:
    client = RecordingClient()
    runtime = _runtime(tmp_path, client)
    responses = [
        asyncio.run(
            runtime.execute(
                PrimaRequest(
                    task_kind=TaskKind.CONVERSATION,
                    profile=ExecutionProfile.MODEL_ONLY,
                    input_text=f"turn {index}",
                    diagnostic_mode=DiagnosticMode.DIAGNOSTIC,
                )
            )
        )
        for index in range(2)
    ]

    assert runtime.llm_client is client
    assert len(client.calls) == 2
    assert all(response.diagnostics.model_call_count == 1 for response in responses)
    assert responses[0].diagnostics.model_usage == {"prompt_tokens": 5, "completion_tokens": 2}
    assert responses[0].diagnostics.latency_ms > 0
    assert responses[0].diagnostics.trace_event_count == len(responses[0].diagnostics.trace_events)
    assert responses[0].diagnostics.trace_event_count > 1


def test_standard_mode_retains_trace_count_without_full_trace(tmp_path) -> None:
    response = asyncio.run(
        _runtime(tmp_path, RecordingClient()).execute(
            PrimaRequest(
                task_kind=TaskKind.CONVERSATION,
                profile=ExecutionProfile.MODEL_ONLY,
                input_text="hello",
            )
        )
    )
    assert response.diagnostics.trace_event_count > 1
    assert response.diagnostics.trace_events == ()


def test_configured_abstention_is_typed_and_classified(tmp_path) -> None:
    response = asyncio.run(
        _runtime(tmp_path, RecordingClient(fail=True), FallbackPolicy.ABSTAIN).execute(
            PrimaRequest(
                task_kind=TaskKind.CONVERSATION,
                profile=ExecutionProfile.MODEL_ONLY,
                input_text="hello",
            )
        )
    )
    assert response.outcome is ExecutionOutcome.ABSTAINED
    assert response.output_text == "I do not have sufficient evidence to answer this question."
    assert not response.output_text.startswith("{")
    assert response.diagnostics.provider["fallback_policy"] == "abstain"
    assert response.diagnostics.provider["fallback_used"] is True


def test_diagnostics_redaction_covers_headers_urls_and_prompts() -> None:
    result = redact(
        {
            "Authorization": "Bearer secret",
            "endpoint": "https://example.test/path?token=abc&safe=yes",
            "raw_prompt": "sensitive input",
            "nested": {"api_key": "key"},
        }
    )
    assert result["Authorization"] == "[REDACTED]"
    assert "abc" not in result["endpoint"] and "safe=yes" in result["endpoint"]
    assert result["raw_prompt"] == "[REDACTED]"
    assert result["nested"]["api_key"] == "[REDACTED]"
