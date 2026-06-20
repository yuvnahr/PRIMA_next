"""Telemetry initialization helper that wires tracing and metrics."""
from .metrics import create_counter
from .tracing import init_tracing


def init_telemetry(service_name: str = "prima-service") -> None:
    init_tracing(service_name)
    # create a few common counters as examples; real apps should declare
    # metrics centrally and reuse them.
    create_counter("llm.requests", description="Number of LLM requests")
    create_counter("llm.errors", description="Number of failed LLM requests")
