"""Own one canonical runtime and its process-local conversation lifetime."""

from __future__ import annotations

from uuid import uuid4

from config.runtime_config import RuntimeConfig
from config.runtime_mode import RuntimeMode
from config.settings import Settings, get_settings
from demo_cli.config import PROFILES, DemoConfig
from demo_cli.diagnostics import (
    DemoTurnDiagnostics,
    adapt_response,
    memory_rows,
    plan_rows,
    state_rows,
    world_rows,
)
from llm.llm_client import LLMClient
from llm.llm_types import LLMRequest
from llm.provider import ProviderError
from memory.memory_repository import InMemoryMemoryRepository
from monitoring.tracing import Tracing, configure_tracing
from runtime import DiagnosticMode, ExecutionProfile, PrimaRequest, PrimaResponse, PrimaRuntime, TaskKind
from state.state_manager import InMemoryStateManager


class DemoSession:
    def __init__(
        self, config: DemoConfig, *, settings: Settings | None = None,
        llm_client: LLMClient | None = None, runtime_config: RuntimeConfig | None = None,
        observability: Tracing | None = None,
    ) -> None:
        self.config = config
        self.current_profile = config.profile
        self._session_id = config.session_id
        self.trace_mode = config.trace_mode
        self.latest_response: PrimaResponse | None = None
        self.latest_diagnostics: DemoTurnDiagnostics | None = None
        self._closed = False
        client = llm_client or LLMClient(provider_name=config.generation.provider, settings=settings or get_settings())
        client.provider.validate(LLMRequest("", config.generation, system_prompt="Respond to the user."))
        self.observability = observability or configure_tracing(settings or get_settings(), enabled=config.langfuse)
        self.runtime = self._create_runtime(client, runtime_config)

    def _create_runtime(self, client: LLMClient, runtime_config: RuntimeConfig | None) -> PrimaRuntime:
        return PrimaRuntime(
            llm_client=client, generation_config=self.config.generation,
            mode=RuntimeMode.TEST, memory_repository=InMemoryMemoryRepository(),
            state_manager=InMemoryStateManager(), runtime_config=runtime_config,
            observability=self.observability,
        )

    @property
    def session_id(self) -> str:
        return self._session_id

    def status_rows(self) -> tuple[tuple[str, str], ...]:
        return (
            ("Provider", self.config.generation.provider), ("Model", self.config.generation.model),
            ("Profile", self.current_profile.value), ("Session", self.session_id),
            ("Memory", self.runtime.repository_selection.backend), ("Trace", self.trace_mode.value),
            ("Langfuse", self.observability.status),
        )

    async def start(self) -> None:
        provider = self.runtime.llm_client.provider
        if provider.name in {"groq", "openai", "anthropic"} and not (getattr(provider, "api_key", None) or "").strip():
            raise ProviderError(f"{provider.name} API key is not configured. Set {provider.name.upper()}_API_KEY.")
        await self.runtime.start_maintenance()

    async def send(self, text: str) -> PrimaResponse:
        self.latest_response = None
        self.latest_diagnostics = None
        previous_version = self.runtime.state_manager.load(self.session_id).version
        response = await self.runtime.execute(PrimaRequest(
            task_kind=TaskKind.CONVERSATION, profile=self.current_profile,
            session_id=self.session_id, input_text=text, diagnostic_mode=DiagnosticMode.DIAGNOSTIC,
        ))
        self.latest_response = response
        self.latest_diagnostics = adapt_response(response, previous_state_version=previous_version)
        return response

    def set_profile(self, name: str) -> ExecutionProfile:
        if name not in PROFILES:
            raise ValueError(f"Unknown profile '{name[:80]}'. Available: {', '.join(PROFILES)}")
        previous = self.current_profile
        self.current_profile = PROFILES[name]
        return previous

    def current_state(self) -> tuple[tuple[str, str], ...]:
        return state_rows(self.runtime.state_manager.load(self.session_id), self.session_id)

    def stored_memories(self) -> tuple[tuple[str, str], ...]:
        # Untagged derived notes belong to this runtime's dedicated repository.
        notes = [note for note in self.runtime.memory_repository.list()
                 if note.context.get("source_session_id") in (None, "", self.session_id)]
        return memory_rows(notes)

    def latest_plan(self) -> tuple[tuple[str, str], ...]:
        return plan_rows(self.latest_response)

    def latest_world(self) -> tuple[tuple[str, str], ...]:
        return world_rows(self.latest_response)

    async def reset(self) -> tuple[str, str]:
        # Runtime.reset retains affect history; replace only this demo's owned runtime.
        replacement = self._create_runtime(self.runtime.llm_client, self.runtime.runtime_config)
        await self.close()
        previous = self.session_id
        self.runtime = replacement
        self._closed = False
        self._session_id = f"demo-{uuid4().hex[:12]}"
        self.latest_response = None
        self.latest_diagnostics = None
        await replacement.start_maintenance()
        return previous, self.session_id

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self.runtime.stop_maintenance(graceful=True)
