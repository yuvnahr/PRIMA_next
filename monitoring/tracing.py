"""Tracing helpers using OpenTelemetry with safe fallback.

This file provides `init_tracing()` to configure the tracer and `get_tracer()`
to obtain a tracer for instrumentation.
"""

try:
    from opentelemetry import trace as _trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor
    _OTEL_AVAILABLE = True
except Exception:
    _OTEL_AVAILABLE = False


def init_tracing(service_name: str = "prima-service") -> None:
    if not _OTEL_AVAILABLE:
        return
    provider = TracerProvider()
    exporter = ConsoleSpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    _trace.set_tracer_provider(provider)


def get_tracer(name: str | None = None):
    if not _OTEL_AVAILABLE:
        class _Noop:
            def start_as_current_span(self, *args, **kwargs):
                class _Ctx:
                    def __enter__(self):
                        return None

                    def __exit__(self, exc_type, exc, tb):
                        return False

                return _Ctx()

        return _Noop()
    return _trace.get_tracer(name or "prima")
