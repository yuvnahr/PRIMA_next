"""Whitelist public response fields into bounded terminal diagnostic rows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from memory.memory_note import MemoryNote
from runtime.contracts import ExecutionProfile, PrimaResponse, RuntimeComponent
from security.redaction import redact
from state.cognitive_state import CognitiveState

PREVIEW_LENGTH = 240
MEMORY_LIMIT = 5
INSPECTION_LIMIT = 10
_PHASES = {
    RuntimeComponent.AFFECT_ENGINE: "affect",
    RuntimeComponent.DENSE_RETRIEVAL: "evidence_acquisition",
    RuntimeComponent.PLANNER: "planning",
    RuntimeComponent.WORLD_MODEL: "world_simulation",
    RuntimeComponent.UNCERTAINTY_ESTIMATOR: "uncertainty_estimation",
    RuntimeComponent.REFLECTION: "reflection",
    RuntimeComponent.MODEL_EXECUTOR: "answer_generation",
    RuntimeComponent.MEMORY_COMMIT: "memory_commit",
}


@dataclass(frozen=True, slots=True)
class DemoTurnDiagnostics:
    summary: tuple[tuple[str, str], ...]
    details: tuple[tuple[str, str], ...]
    events: tuple[str, ...]
    event_count: int | None
    errors: tuple[str, ...]
    failed_stage: str | None


def _value(value: Any) -> str:
    if isinstance(value, str) and value:
        return value if len(value) <= PREVIEW_LENGTH else value[:PREVIEW_LENGTH - 3] + "..."
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return "n/a"


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    return value if isinstance(value, dict) else {}


def component_status(response: PrimaResponse, component: RuntimeComponent) -> str:
    """Distinguish profile skips, conditional skips, failures and absent diagnostics."""
    diagnostics = response.diagnostics
    phase = _PHASES.get(component, component.value)
    last_event = next((event for event in reversed(diagnostics.trace_events)
                       if event.get("phase") == phase and event.get("event_type") in {"phase_failed", "phase_completed"}), {})
    details = diagnostics.component_details.get(component.value, {})
    if last_event.get("event_type") == "phase_failed" or details.get("status") == "failed":
        return "failed"
    if component in diagnostics.skipped_components or details.get("status") == "disabled":
        return "skipped by profile"
    if component is RuntimeComponent.MODEL_EXECUTOR and _section(response.output_data, "generation").get("outcome") == "failed":
        return "failed"
    if component is RuntimeComponent.PLANNER and _section(response.output_data, "plan").get("status") == "failed":
        return "failed"
    if component is RuntimeComponent.DENSE_RETRIEVAL and response.output_data.get("sufficiency_status") == "error":
        return "failed"
    if component in diagnostics.executed_components or details.get("status") == "executed":
        return "executed"
    if component in diagnostics.not_executed_components or details.get("status") == "enabled_not_executed":
        return "not executed"
    return "unavailable"


def _mapping_text(value: Any) -> str:
    if not isinstance(value, Mapping):
        return "n/a"
    fields = []
    for key, item in value.items():
        if isinstance(key, str) and (item is None or isinstance(item, str | int | float | bool)):
            fields.append(f"{key[:40]}={_value(redact({key: item})[key])}")
            if len(fields) == 5:
                break
    return _value("; ".join(fields)) if fields else "n/a"


def state_rows(state: CognitiveState, session_id: str) -> tuple[tuple[str, str], ...]:
    rows = [
        ("Session", _value(session_id)), ("Version", str(state.version)),
        ("Goal", _mapping_text(state.goal_state)), ("Task", _mapping_text(state.task_state)),
        ("Confidence", _mapping_text(state.confidence_state)),
    ]
    emotion = getattr(state, "emotional_state", None)
    pad = getattr(emotion, "current_pad", None)
    rows.extend((label, _value(getattr(source, field, None))) for label, source, field in (
        ("Emotion", emotion, "dominant_emotion"), ("Pleasure (PAD)", pad, "pleasure"),
        ("Arousal", pad, "arousal"), ("Dominance", pad, "dominance"),
        ("Affect confidence", emotion, "confidence"), ("Stability", emotion, "emotional_stability"),
        ("Volatility", emotion, "emotional_volatility"),
    ))
    return tuple(rows)


def memory_rows(notes: Sequence[MemoryNote]) -> tuple[tuple[str, str], ...]:
    rows = [("Stored memories", str(len(notes)))]
    for index, note in enumerate(sorted(notes, key=lambda item: item.timestamp, reverse=True)[:MEMORY_LIMIT], 1):
        rows.extend((
            (f"[{index}] {note.memory_type.value}", _value(note.content)), ("ID", _value(note.id)),
            ("Salience", _value(note.salience_score)), ("Timestamp", note.timestamp.isoformat()),
        ))
    if len(notes) > MEMORY_LIMIT:
        rows.append(("More", f"... {len(notes) - MEMORY_LIMIT} more"))
    return tuple(rows)


def plan_rows(response: PrimaResponse | None) -> tuple[tuple[str, str], ...]:
    if response is None:
        return (("Status", "No turn response is available yet."),)
    status = component_status(response, RuntimeComponent.PLANNER)
    rows = [("Last turn profile", response.profile.value), ("Component", status)]
    if status in {"skipped by profile", "not executed"}:
        return (*rows, ("Status", "Planning was not executed for the last turn."))
    plan = _section(response.output_data, "plan")
    if not plan:
        return (*rows, ("Plan", "No plan available for the last turn."))
    rows.extend((label, _value(plan.get(field))) for label, field in (
        ("Goal", "goal"), ("Confidence", "confidence"), ("Revision", "revision"), ("Plan status", "status"),
    ))
    for label, field, count in (("Action", "actions", "action_count"), ("Constraint", "constraints", "constraint_count")):
        items = plan.get(field)
        if not isinstance(items, list):
            continue
        for index, item in enumerate(items[:INSPECTION_LIMIT], 1):
            rows.append((f"{label} {index}", _value(item.get("description")) if isinstance(item, dict) else "n/a"))
        total = plan.get(count, len(items))
        if isinstance(total, int) and total > INSPECTION_LIMIT:
            rows.append((label, f"... {total - INSPECTION_LIMIT} more"))
    return tuple(rows)


def world_rows(response: PrimaResponse | None) -> tuple[tuple[str, str], ...]:
    if response is None:
        return (("Status", "No turn response is available yet."),)
    status = component_status(response, RuntimeComponent.WORLD_MODEL)
    rows = [("Last turn profile", response.profile.value), ("Component", status)]
    if status in {"skipped by profile", "not executed"}:
        return (*rows, ("Status", "World simulation was not executed for the last turn."))
    world = _section(response.output_data, "world_prediction")
    if not world:
        return (*rows, ("World model", "No world-model output available for the last turn."))
    predictions = world.get("action_predictions")
    constraints = world.get("constraint_predictions")
    rows.extend((
        ("Actions simulated", str(len(predictions)) if isinstance(predictions, list) else "n/a"),
        ("Risk score", _value(world.get("risk_score"))),
        ("Failure probability", _value(world.get("failure_probability"))),
        ("Constraint predictions", str(len(constraints)) if isinstance(constraints, list) else "n/a"),
    ))
    transitions = world.get("state_transitions")
    if isinstance(transitions, list):
        for index, transition in enumerate(transitions[:INSPECTION_LIMIT], 1):
            delta = _section(transition, "delta") if isinstance(transition, dict) else {}
            effects = "; ".join(f"{key}: {_mapping_text(delta[key])}" for key in (
                "goal_state", "task_state", "confidence_state", "emotional_state", "environment_state", "execution_state"
            ) if key in delta)
            rows.append((f"Predicted effect {index}", _value(effects)))
        if len(transitions) > INSPECTION_LIMIT:
            rows.append(("Predicted effects", f"... {len(transitions) - INSPECTION_LIMIT} more"))
    warnings = world.get("warnings")
    if isinstance(warnings, list):
        rows.extend((f"Warning {index}", _value(warning)) for index, warning in enumerate(warnings[:INSPECTION_LIMIT], 1))
        if not warnings:
            rows.append(("Warnings", "none"))
        if len(warnings) > INSPECTION_LIMIT:
            rows.append(("Warnings", f"... {len(warnings) - INSPECTION_LIMIT} more"))
    return tuple(rows)


def adapt_response(response: PrimaResponse, *, previous_state_version: int | None = None) -> DemoTurnDiagnostics:
    """Format reported values without rerunning or interpreting cognitive stages."""
    data, diagnostics = response.output_data, response.diagnostics
    skipped = set(diagnostics.skipped_components)

    def module(component: RuntimeComponent, value: str) -> str:
        status = component_status(response, component)
        if status == "skipped by profile":
            return "skipped"
        if status in {"failed", "not executed"}:
            return status
        return value

    affect = _section(data, "affect_state")
    retrieval = _section(data, "retrieval")
    plan = _section(data, "plan")
    world = _section(data, "world_prediction")
    uncertainty = _section(data, "uncertainty")
    decision = _section(data, "execution_decision")
    admission = _section(data, "memory_admission")
    reflected = data.get("reflection_triggered")
    reflection = "triggered" if reflected is True else "skipped" if reflected is False else "n/a"
    fields = diagnostics.model_fields_set
    current_version = diagnostics.state_version if "state_version" in fields else response.state_delta.changes.get("version")
    state_version = f"v{current_version}" if isinstance(current_version, int) else "n/a"
    if previous_state_version is not None and isinstance(current_version, int):
        state_version = (f"unchanged (v{previous_state_version})" if previous_state_version == current_version
                         else f"v{previous_state_version} -> v{current_version}")
    model_calls = f"{diagnostics.model_call_count} model call(s)" if "model_call_count" in fields else "n/a"
    latency = diagnostics.latency_ms if "latency_ms" in fields else data.get("latency_ms")
    architecture = (
        ("Affect", module(RuntimeComponent.AFFECT_ENGINE, (f"{_value(affect.get('dominant_emotion'))} / valence {_value(affect.get('valence'))}" + (f" / arousal {_value(affect['arousal'])}" if "arousal" in affect else "")) if affect else "n/a")),
        ("Retrieval", module(RuntimeComponent.DENSE_RETRIEVAL, f"{_value(retrieval.get('selected_count'))} selected / {_value(retrieval.get('candidate_count'))} candidates / {_value(retrieval.get('final_context_count'))} in context" if retrieval else "n/a")),
        ("Planning", module(RuntimeComponent.PLANNER, f"{_value(plan.get('action_count'))} actions / confidence {_value(plan.get('confidence'))}" if plan else "n/a")),
        ("World model", module(RuntimeComponent.WORLD_MODEL, f"risk {_value(world.get('risk_score'))} / failure {_value(world.get('failure_probability'))}" if world else "n/a")),
        ("Uncertainty", module(RuntimeComponent.UNCERTAINTY_ESTIMATOR, f"{_value(uncertainty.get('uncertainty_band'))} / {_value(uncertainty.get('uncertainty'))}" if uncertainty else "n/a")),
        ("Decision", _value(decision.get("decision", diagnostics.execution_decision))),
        ("Reflection", module(RuntimeComponent.REFLECTION, reflection)),
    )
    if response.profile is ExecutionProfile.MODEL_ONLY:
        architecture = (("Generation", module(RuntimeComponent.MODEL_EXECUTOR, model_calls)),)
    summary = (
        ("Profile", response.profile.value),
        *architecture,
        ("Commit", module(RuntimeComponent.MEMORY_COMMIT, f"memory admitted {_value(admission.get('stored'))}" if admission else "n/a")),
        ("State", state_version),
        ("Latency", f"{latency:.0f} ms" if isinstance(latency, int | float) else "n/a"),
    )
    warnings = world.get("warnings")
    warning_text = ("; ".join(item[:240] for item in warnings[:5] if isinstance(item, str)) or "none") if isinstance(warnings, list | tuple) else "n/a"
    details = (
        ("Request", response.request_id[:240]),
        ("Outcome", response.outcome.value if response.outcome is not None else response.status.value),
        ("Provider", _value(diagnostics.provider.get("name"))),
        ("Model", _value(diagnostics.provider.get("model"))),
        ("Executed", (", ".join(item.value for item in diagnostics.executed_components[:40]) + (f" ({len(diagnostics.executed_components)} total)" if len(diagnostics.executed_components) > 40 else "")) or "none"),
        ("Skipped", ", ".join(sorted(item.value for item in skipped)) or "none"),
        ("Not executed", ", ".join(item.value for item in diagnostics.not_executed_components[:40]) or "none"),
        ("Components", "; ".join(f"{component.value}={component_status(response, component)}" for component in _PHASES)),
        ("Retrieval pool", _value(retrieval.get("candidate_count"))),
        ("Retrieval confidence", _value(retrieval.get("retrieval_confidence"))),
        ("Plan confidence", _value(plan.get("confidence"))),
        ("Plan revision", _value(plan.get("revision"))),
        ("World warnings", warning_text),
        ("Uncertainty score", _value(uncertainty.get("uncertainty"))),
        ("Confidence", _value(uncertainty.get("confidence"))),
        ("Decision rule", _value(decision.get("rule", diagnostics.decision_rule))),
        ("Corrections", f"{len(diagnostics.correction_attempts)} attempts / {diagnostics.accepted_correction_count} accepted"
         if {"correction_attempts", "accepted_correction_count"} <= fields else "n/a"),
        ("Model calls", str(diagnostics.model_call_count) if "model_call_count" in fields else "n/a"),
    )
    events = tuple(
        " / ".join(_value(event.get(key)) for key in ("event_type", "phase", "status"))
        for event in diagnostics.trace_events[:30]
    )
    failed_stage = next((
        _value(event.get("phase")) for event in reversed(diagnostics.trace_events)
        if event.get("event_type") == "phase_failed"
    ), None)
    return DemoTurnDiagnostics(summary, details, events, diagnostics.trace_event_count if "trace_event_count" in fields else None, tuple(error[:500] for error in response.errors[:5]), failed_stage)
