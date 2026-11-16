"""Own one canonical runtime and its optional session-scoped persistent stores."""

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
from demo_cli.live_trace import LiveTraceObserver
from demo_cli.renderer import Renderer
from demo_cli.storage import PersistenceError, clear_session, create_stores, display_data_root
from llm.llm_client import LLMClient
from llm.llm_types import LLMRequest
from llm.provider import ProviderError
from monitoring.tracing import Tracing, configure_tracing
from runtime import DiagnosticMode, ExecutionProfile, PrimaRequest, PrimaResponse, PrimaRuntime, TaskKind


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
        self.live_trace = config.live_trace
        self._live_observer: LiveTraceObserver | None = None
        self.latest_response: PrimaResponse | None = None
        self.latest_diagnostics: DemoTurnDiagnostics | None = None
        self._closed = False
        client = llm_client or LLMClient(provider_name=config.generation.provider, settings=settings or get_settings())
        client.provider.validate(LLMRequest("", config.generation, system_prompt="Respond to the user."))
        self.observability = observability or configure_tracing(settings or get_settings(), enabled=config.langfuse)
        self.runtime = self._create_runtime(client, runtime_config, self.session_id)

    def _create_runtime(
        self, client: LLMClient, runtime_config: RuntimeConfig | None, session_id: str,
    ) -> PrimaRuntime:
        runtime_config = runtime_config or RuntimeConfig.from_environment()
        state, repository = create_stores(self.config, session_id, runtime_config)
        try:
            return PrimaRuntime(
                llm_client=client, generation_config=self.config.generation,
                mode=RuntimeMode.TEST, memory_repository=repository,
                state_manager=state, runtime_config=runtime_config,
                observability=self.observability,
            )
        except BaseException:
            repository.close()
            raise

    @property
    def session_id(self) -> str:
        return self._session_id

    def status_rows(self) -> tuple[tuple[str, str], ...]:
        return (
            ("Provider", self.config.generation.provider), ("Model", self.config.generation.model),
            ("Profile", self.current_profile.value), ("Session", self.session_id),
            ("Memory", self.runtime.repository_selection.backend),
            ("Persistence", "enabled" if self.config.persist else "disabled"),
            *((("Data root", display_data_root(self.config)),) if self.config.persist else ()),
            ("Trace", self.trace_mode.value),
            ("Live trace", "on" if self.live_trace else "off"),
            ("Langfuse", self.observability.status),
        )

    def observe_workflow(self, renderer: Renderer) -> None:
        if self._live_observer is None or self._live_observer.closed:
            self._live_observer = LiveTraceObserver(self.runtime.workflow.engine.event_bus.event_bus, renderer)

    def persistence_messages(self) -> tuple[str, ...]:
        if not self.config.persist:
            return ()
        version = self.runtime.state_manager.load(self.session_id).version
        count = self.runtime.memory_repository.count()
        resumed = (f"Resumed persisted session: state version {version}; memories {count}."
                   if version or count else "Started new persistent session.")
        # PowerShell single-quoted literals preserve arbitrary IDs and directory spaces.
        session = self.session_id.replace("'", "''")
        directory = display_data_root(self.config).replace("'", "''")
        return resumed, f"To resume: .\\demo.ps1 chat --persist --session '{session}' --data-dir '{directory}'"

    async def start(self) -> None:
        if self.config.persist and self._closed:
            raise PersistenceError("The persistent session is closed. Reopen it before sending another request.")
        provider = self.runtime.llm_client.provider
        if provider.name in {"groq", "openai", "anthropic"} and not (getattr(provider, "api_key", None) or "").strip():
            raise ProviderError(f"{provider.name} API key is not configured. Set {provider.name.upper()}_API_KEY.")
        await self.runtime.start_maintenance()

    async def send(self, text: str) -> PrimaResponse:
        if self.config.persist and self._closed:
            raise PersistenceError("The persistent session is closed. Reopen it before sending another request.")
        self.latest_response = None
        self.latest_diagnostics = None
        previous_version = self.runtime.state_manager.load(self.session_id).version
        if self._live_observer is not None:
            self._live_observer.begin_turn(enabled=self.live_trace)
        try:
            response = await self.runtime.execute(PrimaRequest(
                task_kind=TaskKind.CONVERSATION, profile=self.current_profile,
                session_id=self.session_id, input_text=text, diagnostic_mode=DiagnosticMode.DIAGNOSTIC,
            ))
        finally:
            if self._live_observer is not None:
                self._live_observer.end_turn()
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
        if self.config.persist and self._closed:
            raise PersistenceError("The persistent session is closed. Reopen it before resetting it.")
        new_id = f"demo-{uuid4().hex[:12]}"
        renderer = self._live_observer.renderer if self._live_observer is not None else None
        try:
            if self.config.persist:
                await self.close(clear=True)
            replacement = self._create_runtime(self.runtime.llm_client, self.runtime.runtime_config, new_id)
            if not self.config.persist:
                await self.close()
            previous = self.session_id
            self.runtime = replacement
            self._closed = False
            self._session_id = new_id
            self.latest_response = None
            self.latest_diagnostics = None
            if renderer is not None:
                self.observe_workflow(renderer)
            await replacement.start_maintenance()
            return previous, self.session_id
        except Exception:
            if not self.config.persist:
                raise
            try:
                await self.close()
            finally:
                raise PersistenceError("Persistent reset did not complete. The session is closed; check storage before reopening it. Some active-session data may have been cleared; other sessions were not touched.") from None

    async def close(self, *, clear: bool = False) -> None:
        if self._live_observer is not None:
            self._live_observer.close()
        if self._closed:
            return
        self._closed = True
        try:
            await self.runtime.stop_maintenance(graceful=True)
            if clear and self.config.persist:
                clear_session(self.config, self.session_id, self.runtime)
        finally:
            if self.config.persist:
                self.runtime.memory_index.close()
