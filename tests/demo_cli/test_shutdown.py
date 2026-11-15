from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from demo_cli import cli
from monitoring.tracing import Tracing
from tests.demo_cli.test_session_cli import _session, _settings


@pytest.mark.parametrize("ending", ["quit", "eof", "prompt_interrupt", "request_interrupt", "request_cancel", "reset"])
@pytest.mark.parametrize("telemetry", ["success", "failure", "timeout"])
def test_cleanup_once_per_runtime_and_bounded_telemetry(monkeypatch, capsys, caplog, ending, telemetry):
    session, _ = _session(["--profile", "model"])
    stopped = []
    stop = type(session.runtime).stop_maintenance
    async def counted_stop(self, **kwargs):
        stopped.append(self)
        await stop(self, **kwargs)
    monkeypatch.setattr(type(session.runtime), "stop_maintenance", counted_stop)
    release = threading.Event()
    calls = []
    def flush():
        calls.append(True)
        if telemetry == "failure":
            raise RuntimeError("private-secret")
        if telemetry == "timeout":
            release.wait(1)
    handle = Tracing()
    handle.client = SimpleNamespace(shutdown=flush)
    inputs = iter(("/reset", "/quit") if ending == "reset" else ("hello",) if ending.startswith("request") else ("/quit",))
    def read_input(prompt):
        if ending == "eof":
            raise EOFError
        if ending == "prompt_interrupt":
            raise KeyboardInterrupt
        return next(inputs)
    async def interrupted(text):
        raise KeyboardInterrupt if ending == "request_interrupt" else asyncio.CancelledError
    if ending.startswith("request"):
        monkeypatch.setattr(session, "send", interrupted)
    monkeypatch.setattr(cli, "get_settings", _settings)
    monkeypatch.setattr(cli, "DemoSession", lambda *a, **kw: session)
    monkeypatch.setattr(cli, "shutdown_tracing", lambda: handle.shutdown(timeout=0.02))
    monkeypatch.setattr("builtins.input", read_input)
    try:
        assert cli.main(["chat", "--no-color"]) == 0
        asyncio.run(session.close())
        assert len(stopped) == len({id(runtime) for runtime in stopped}) == (2 if ending == "reset" else 1)
        assert all(not runtime.maintenance_supervisor.running for runtime in stopped)
        assert calls == [True] and handle.closed
        output = capsys.readouterr().out
        assert "private-secret" not in output + caplog.text and "Traceback" not in output
        if telemetry != "success":
            assert "Observability shutdown" in caplog.text
        if ending.startswith("request"):
            assert "may run until its timeout" in output
    finally:
        release.set()


def test_unexpected_telemetry_shutdown_failure_does_not_fail_exit(monkeypatch, capsys):
    session, _ = _session()
    monkeypatch.setattr(cli, "get_settings", _settings)
    monkeypatch.setattr(cli, "DemoSession", lambda *a, **kw: session)
    monkeypatch.setattr("builtins.input", lambda prompt: "/quit")
    def broken():
        raise RuntimeError("private-secret")
    monkeypatch.setattr(cli, "shutdown_tracing", broken)
    assert cli.main(["chat"]) == 0
    output = capsys.readouterr().out
    assert "Warning: telemetry shutdown failed" in output and "private-secret" not in output
