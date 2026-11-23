"""Shared optional OpenTelemetry tracing; telemetry never owns runtime outcomes."""

from __future__ import annotations

import atexit
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any
from urllib.parse import urlsplit

from config.settings import Settings, get_settings
from monitoring.trace_content import sanitize_content

try:
    from opentelemetry import trace
    from opentelemetry.context import Context
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
except Exception:  # An incompatible optional OTel cohort must also degrade safely.
    trace = None  # type: ignore[assignment]

SCOPE = "prima.runtime"
_LOGGER = logging.getLogger(__name__)
_LOCK = threading.RLock()
_CLIENTS: dict[Any, Tracing] = {}
_CONSOLE_PROVIDERS: set[Any] = set()
_OWNED_PROVIDERS: set[Any] = set()
_CURRENT: ContextVar[Tracing | None] = ContextVar("prima_tracing", default=None)
_ROOT: ContextVar[SafeSpan | None] = ContextVar("prima_root", default=None)
_CAPTURE: ContextVar[bool] = ContextVar("prima_trace_content", default=False)
_UNSET = object()


class _ExportLogFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        # SDK errors can contain headers, URLs and server response bodies.
        record.msg = "Observability SDK/export diagnostic; conversation execution is unaffected."
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def _provider(provider: Any = None, *, create: bool = False, service_name: str = "prima-service") -> Any:
    if provider is not None or trace is None:
        return provider
    selected = trace.get_tracer_provider()
    if create and isinstance(selected, trace.ProxyTracerProvider):
        selected = TracerProvider(resource=Resource.create({"service.name": service_name}), shutdown_on_exit=False)
        trace.set_tracer_provider(selected)
        _OWNED_PROVIDERS.add(selected)
    return selected


def init_tracing(service_name: str = "prima-service", *, provider: Any = None) -> Any:
    """Opt into console diagnostics once without replacing an installed provider."""
    with _LOCK:
        selected = _provider(provider, create=True, service_name=service_name)
        if selected is not None and selected not in _CONSOLE_PROVIDERS:
            selected.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
            _CONSOLE_PROVIDERS.add(selected)
        return selected


class _NoopTracer:
    @contextmanager
    def start_as_current_span(self, *args: Any, **kwargs: Any) -> Iterator[None]:
        yield None


def get_tracer(name: str | None = None) -> Any:
    return trace.get_tracer(name or "prima") if trace is not None else _NoopTracer()


class SafeSpan:
    def __init__(self, raw: Any = None, owner: Tracing | None = None) -> None:
        self.raw = raw
        self.owner = owner
        self.generation: Any = None
        self.observation: Any = None
        self.capture_enabled = bool(_CAPTURE.get() and raw is not None and owner and owner.client and not owner.closed)

    @property
    def trace_id(self) -> str | None:
        try:
            context = self.raw.get_span_context() if self.raw is not None else None
            return f"{context.trace_id:032x}" if context and context.is_valid else None
        except Exception:
            return None

    def content(self, *, input: Any = _UNSET, output: Any = _UNSET) -> None:
        owner = self.owner
        if not self.capture_enabled or owner is None:
            return
        try:
            wrapper = self.generation or self.observation
            if wrapper is None:
                wrapper = owner.span_type(otel_span=self.raw, langfuse_client=owner.client)
                self.observation = wrapper
            values = {key: sanitize_content(value, owner.secrets)
                      for key, value in (("input", input), ("output", output)) if value is not _UNSET}
            wrapper.update(**values)
        except Exception as exc:
            owner.warn("content capture", exc)

    def set_attributes(self, values: dict[str, Any]) -> None:
        if self.raw is None:
            return
        try:
            clean: dict[str, Any] = {}
            for key, value in values.items():
                if isinstance(value, str):
                    for secret in self.owner.secrets if self.owner else ():
                        value = value.replace(secret, "[redacted]")
                    clean[f"prima.{key}"] = value[:200]
                elif isinstance(value, bool | int | float):
                    clean[f"prima.{key}"] = value
                elif isinstance(value, tuple | list) and all(isinstance(item, str) for item in value):
                    clean[f"prima.{key}"] = tuple(item[:80] for item in value[:8])
            self.raw.set_attributes(clean)
        except Exception as exc:
            if self.owner:
                self.owner.warn("attributes", exc)

    def fail(self, error_type: str | None = None) -> None:
        self.set_attributes({"error_type": error_type})
        try:
            if self.raw is not None and trace is not None:
                self.raw.set_status(trace.StatusCode.ERROR)
        except Exception as exc:
            if self.owner:
                self.owner.warn("status", exc)

    def usage(self, usage: dict[str, Any] | None) -> None:
        if not isinstance(usage, dict):
            return
        counts = {key: usage[key] for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                  if type(usage.get(key)) is int and usage[key] >= 0}
        self.set_attributes(counts)
        try:
            if self.generation is not None and counts:
                names = {"prompt_tokens": "input", "completion_tokens": "output", "total_tokens": "total"}
                self.generation.update(usage_details={names[key]: value for key, value in counts.items()})
        except Exception as exc:
            if self.owner:
                self.owner.warn("usage", exc)


class Tracing:
    def __init__(self, provider: Any = None, *, status: str = "disabled", settings: Settings | None = None) -> None:
        self.provider = provider
        self.status = status
        self.client: Any = None
        self.propagate: Any = None
        self.generation_type: Any = None
        self.span_type: Any = None
        self.closed = False
        self.warning: str | None = None
        self.exporting = threading.Event()
        self._warned: set[str] = set()
        self.secrets = tuple(value for name in (
            "langfuse_public_key", "langfuse_secret_key", "groq_api_key", "openai_api_key", "anthropic_api_key",
        ) if isinstance(value := getattr(settings, name, None), str) and value)

    def warn(self, operation: str, exc: BaseException) -> None:
        if operation not in self._warned:
            self._warned.add(operation)
            _LOGGER.warning("Observability %s failed (%s); continuing without it.", operation, type(exc).__name__)

    @contextmanager
    def span(self, name: str, *, root: bool = False) -> Iterator[SafeSpan]:
        manager = None
        raw = None
        started = time.perf_counter()
        try:
            if self.provider is not None:
                candidate = self.provider.get_tracer(SCOPE).start_as_current_span(
                    name, context=Context() if root else None,
                    attributes={"prima.owned": True, "prima.langfuse_enabled": bool(self.client and not self.closed)},
                    record_exception=False, set_status_on_exception=False,
                )
                raw = candidate.__enter__()
                manager = candidate
        except Exception as exc:
            self.warn("span start", exc)
        observation = SafeSpan(raw, self)
        try:
            yield observation
        except BaseException as exc:
            observation.fail(type(exc).__name__)
            raise
        finally:
            observation.set_attributes({"latency_ms": round((time.perf_counter() - started) * 1000, 3)})
            if manager is not None:
                try:
                    manager.__exit__(None, None, None)
                except Exception as exc:
                    self.warn("span end", exc)

    @contextmanager
    def turn(self, request: Any, *, capture_content: bool = False, name: str = "prima.turn") -> Iterator[SafeSpan]:
        token = _CURRENT.set(self)
        capture_token = _CAPTURE.set(capture_content)
        try:
            with self.span(name, root=True) as observation:
                root_token = _ROOT.set(observation)
                propagation = None
                observation.set_attributes({"request_id": request.request_id, "session_id": request.session_id,
                                            "task_kind": request.task_kind.value, "execution_profile": request.profile.value})
                correlation = {key: request.metadata[key] for key in
                               ("comparison_id", "comparison_turn", "comparison_branch") if key in request.metadata}
                observation.set_attributes(correlation)
                observation.content(input={"user_input": request.input_text, "task": request.task_kind.value,
                                           "profile": request.profile.value})
                try:
                    if self.propagate is not None:
                        session = request.session_id
                        if session and (not session.isascii() or len(session) > 200 or any(s in session for s in self.secrets)):
                            raise ValueError("Session ID cannot be propagated safely")
                        safe_metadata = sanitize_content(correlation, self.secrets)
                        candidate = self.propagate(session_id=session, metadata={key: str(value) for key, value in safe_metadata.items()})
                        candidate.__enter__()
                        propagation = candidate
                except Exception as exc:
                    self.warn("session propagation", exc)
                try:
                    yield observation
                finally:
                    if propagation is not None:
                        try:
                            propagation.__exit__(None, None, None)
                        except Exception as exc:
                            self.warn("session propagation exit", exc)
                    _ROOT.reset(root_token)
        finally:
            _CAPTURE.reset(capture_token)
            _CURRENT.reset(token)

    def shutdown(self, timeout: float = 3.0) -> None:
        if self.closed:
            return
        self.closed = True
        self.exporting.clear()
        if self.client is None:
            return

        def close() -> None:
            try:
                self.client.shutdown()  # SDK shutdown flushes its batched exports.
            except Exception as exc:
                self.warn("shutdown", exc)
            finally:
                if self.provider in _OWNED_PROVIDERS:
                    try:
                        self.provider.shutdown()
                    except Exception as exc:
                        self.warn("provider shutdown", exc)

        worker = threading.Thread(target=close, name="prima-tracing-shutdown", daemon=True)
        worker.start()
        worker.join(timeout)
        if worker.is_alive():
            self.warn("shutdown timeout", TimeoutError())


def should_export_span(observation: Any) -> bool:
    scope = getattr(observation, "instrumentation_scope", None)
    attrs = observation.attributes or {}
    return bool(getattr(scope, "name", None) == SCOPE and attrs.get("prima.owned") and attrs.get("prima.langfuse_enabled"))


def langfuse_readiness(settings: Settings, *, enabled: bool) -> tuple[str, str | None]:
    """Check configuration and the modern public SDK without starting exporters."""
    if not enabled:
        return "disabled", None
    try:
        url = urlsplit(settings.langfuse_base_url or "")
        if (not (settings.langfuse_public_key or "").strip() or not (settings.langfuse_secret_key or "").strip() or
                url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment):
            raise ValueError("Incomplete Langfuse configuration")
    except ValueError:
        return "misconfigured", "Set LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY and LANGFUSE_BASE_URL. Continuing without Langfuse tracing."
    try:
        from langfuse import Langfuse, LangfuseGeneration, propagate_attributes

        if trace is None or not all(callable(item) for item in (Langfuse, LangfuseGeneration, propagate_attributes)):
            raise ImportError("Observability dependencies are incompatible")
    except Exception:
        return "unavailable", "Observability dependencies are unavailable/incompatible. Install: python -m pip install -r requirements-observability.txt. Continuing without Langfuse tracing."
    return "ready", None


def configure_tracing(settings: Settings | None = None, *, enabled: bool | None = None, provider: Any = None) -> Tracing:
    settings = settings or get_settings()
    active = settings.langfuse_enabled if enabled is None else enabled
    if not active:
        return Tracing(_provider(provider), settings=settings)
    status, hint = langfuse_readiness(settings, enabled=True)
    if status != "ready":
        handle = Tracing(_provider(provider), status="misconfigured", settings=settings)
        handle.warning = hint
        return handle
    try:
        import langfuse
        from langfuse import Langfuse, LangfuseGeneration, propagate_attributes

        if trace is None:
            raise ImportError("OpenTelemetry is unavailable")
        with _LOCK:
            selected = _provider(provider, create=True)
            if selected in _CLIENTS:
                return _CLIENTS[selected]
            handle = Tracing(selected, status="enabled", settings=settings)
            for name in ("langfuse", "opentelemetry.exporter.otlp.proto.http.trace_exporter",
                         "opentelemetry.sdk._shared_internal", "opentelemetry.sdk.trace.export"):
                logger = logging.getLogger(name)
                if not any(isinstance(item, _ExportLogFilter) for item in logger.filters):
                    logger.addFilter(_ExportLogFilter())
            handle.exporting.set()
            handle.client = Langfuse(
                public_key=settings.langfuse_public_key, secret_key=settings.langfuse_secret_key,
                base_url=settings.langfuse_base_url, environment=settings.langfuse_tracing_environment,
                tracer_provider=selected, should_export_span=lambda value: handle.exporting.is_set() and should_export_span(value),
                debug=False, timeout=2, flush_interval=1,
            )
            handle.propagate = propagate_attributes
            handle.generation_type = LangfuseGeneration
            handle.span_type = getattr(langfuse, "LangfuseSpan", None)
            _CLIENTS[selected] = handle
            return handle
    except Exception as exc:
        _LOGGER.warning("Langfuse misconfigured or unavailable (%s); tracing disabled. Check settings and requirements-observability.txt.", type(exc).__name__)
        handle = Tracing(_provider(provider), status="misconfigured", settings=settings)
        handle.warning = "Langfuse initialization failed (optional). Continuing without Langfuse tracing."
        return handle


@contextmanager
def span(name: str) -> Iterator[SafeSpan]:
    owner = _CURRENT.get()
    if owner is None:
        yield SafeSpan()
    else:
        with owner.span(name) as observation:
            yield observation


@contextmanager
def generation_span(config: Any) -> Iterator[SafeSpan]:
    with span("llm.generation") as observation:
        observation.set_attributes({"provider": config.provider, "model": config.model, "temperature": config.temperature,
                                    "max_tokens": config.max_output_tokens})
        owner = observation.owner
        if owner is not None and owner.generation_type is not None and observation.raw is not None:
            try:
                observation.generation = owner.generation_type(
                    otel_span=observation.raw, langfuse_client=owner.client, model=config.model,
                    model_parameters={"temperature": config.temperature, "max_tokens": config.max_output_tokens},
                )
            except Exception as exc:
                owner.warn("generation metadata", exc)
        yield observation


def root_attributes(values: dict[str, Any]) -> None:
    root = _ROOT.get()
    if root is not None:
        root.set_attributes(values)


def shutdown_tracing() -> None:
    for handle in tuple(_CLIENTS.values()):
        handle.shutdown()


atexit.register(shutdown_tracing)
