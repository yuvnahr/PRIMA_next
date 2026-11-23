"""Render selected facts from a recorded turn without rerunning its phases."""

from __future__ import annotations

from typing import Any

from demo_cli.diagnostics import INSPECTION_LIMIT, MEMORY_LIMIT, _section, _value
from monitoring.trace_content import sanitize_content
from runtime.contracts import PrimaResponse, RuntimeComponent

Rows = list[tuple[str, Any]]


def _records(value: Any) -> list[dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, (list, tuple)) else []


def _fields(data: dict[str, Any], *fields: tuple[str, str]) -> Rows:
    return [(label, data[key]) for label, key in fields if data.get(key) is not None]


def _scores(prefix: str, data: dict[str, Any]) -> Rows:
    scores = [(name, score) for name, score in data.items() if type(score) in (float, int)]
    return [(f"{prefix}{name}", score) for name, score in sorted(scores, key=lambda pair: (-pair[1], pair[0]))]


def _phase(response: PrimaResponse, phase: str, component: RuntimeComponent | None = None) -> str:
    for event in reversed(response.diagnostics.trace_events):
        if event.get("phase") != phase:
            continue
        match event.get("event_type"):
            case "phase_completed":
                return "executed"
            case "phase_failed":
                return "failed"
            case "phase_started" | "phase_retry":
                return "started without a completed result"
    if component in response.diagnostics.skipped_components:
        return "skipped by profile"
    if component in response.diagnostics.executed_components:
        return "executed"
    return "not executed"


def _affect(response: PrimaResponse, detail: bool) -> Rows:
    data = response.output_data
    rows: Rows = [("Affect phase", _phase(response, "affect", RuntimeComponent.AFFECT_ENGINE))]
    profile = _section(data, "affect_state")
    if not profile:
        return rows
    scores = _scores("Score: ", _section(profile, "emotions"))
    rows += _fields(profile, ("Dominant score label", "dominant_emotion"), ("Heuristic confidence", "confidence"))
    rows += scores or [("Emotion scores", "No labels reported")]
    rows += _fields(profile, ("Turn valence", "valence"), ("Turn arousal", "arousal"), ("Turn dominance", "dominance"))
    changes = response.state_delta.changes
    emotional = _section(changes, "emotional_state")
    if emotional:
        rows.append(("State update", "committed" if changes.get("committed") else "provisional"))
        rows += _fields(_section(emotional, "current_pad"), ("Updated pleasure", "pleasure"),
                        ("Updated arousal", "arousal"), ("Updated dominance", "dominance"))
        rows += _fields(emotional, ("Stability", "emotional_stability"), ("Volatility", "emotional_volatility"))
    if detail:
        extra = _section(_section(data, "inspection"), "affect")
        rows += _fields(extra, ("Salience", "salience"), ("Dissonance", "dissonance"))
        rows += _scores("Evolution: ", _section(extra, "evolution"))
        rows += _scores("Momentum: ", _section(emotional, "emotional_momentum"))
        rows += _scores("Retrieval prior: ", _section(extra, "retrieval_priors"))
        for signal in _records(extra.get("reflection_signals"))[:INSPECTION_LIMIT]:
            rows.append((f"Reflection cue: {_value(signal.get('type'))}", signal.get("strength")))
        cues = profile.get("emotional_keywords")
        if isinstance(cues, (list, tuple)):
            rows += [("Classifier cue", cue) for cue in cues[:INSPECTION_LIMIT] if isinstance(cue, str)]
    return rows


def _decision(response: PrimaResponse, detail: bool) -> Rows:
    data = response.output_data
    rows: Rows = [("Gate phase", _phase(response, "execution_decision")),
                  ("Estimator phase", _phase(response, "uncertainty_estimation", RuntimeComponent.UNCERTAINTY_ESTIMATOR))]
    gate = _section(data, "execution_decision")
    history = _records(data.get("execution_decision_history")) or ([gate] if gate else [])
    for index, step in enumerate(history, 1):
        rows.append((f"Gate {index}", f"{_value(step.get('decision'))} [{_value(step.get('rule'))}]"))
        rows += _fields(step, ("Gate reason", "rationale"))
        if step.get("rule") == "retrieval_retry_threshold":
            rows += _fields(_section(step, "thresholds"), ("Retry threshold", "retry_retrieval_uncertainty"))
    estimate = _section(data, "uncertainty")
    if not estimate:
        return rows
    rows += _fields(data, ("Gate retrieval retries", "retrieval_retry_count"))
    rows += _fields(estimate, ("Final recommendation", "recommended_decision"), ("Final confidence", "confidence"),
                    ("Final uncertainty", "uncertainty"), ("Uncertainty band", "uncertainty_band"))
    for option in _records(estimate.get("decision_probabilities")):
        probability = option.get("probability")
        if type(probability) in (float, int):
            rows.append((f"Estimator: {_value(option.get('decision_type'))}", f"{probability:.2%}"))
            if detail:
                rows += _fields(option, ("Estimator rationale", "rationale"))
    thresholds = _section(gate, "thresholds")
    rows += _fields(thresholds, ("Reflection threshold", "reflect_uncertainty"), ("Risk denial threshold", "deny_world_risk"))
    rows += _fields(_section(data, "world_prediction"), ("Final symbolic risk", "risk_score"))
    if detail:
        interval = estimate.get("confidence_interval")
        if isinstance(interval, (list, tuple)) and len(interval) == 2:
            rows.append(("Confidence interval", f"{_value(interval[0])} to {_value(interval[1])}"))
        for signal in _records(estimate.get("signals"))[:INSPECTION_LIMIT]:
            rows.append((f"Signal: {_value(signal.get('source'))}",
                         f"confidence {_value(signal.get('confidence'))}, uncertainty {_value(signal.get('uncertainty'))}, "
                         f"effective weight {_value(signal.get('effective_weight'))}"))
        rows += _fields(thresholds, ("Retry threshold", "retry_retrieval_uncertainty"),
                        ("Max gate retries", "max_retrieval_retries"),
                        ("Clarify uncertainty", "clarification_uncertainty"), ("Clarify confidence", "clarification_confidence"))
    return rows


def _evidence(response: PrimaResponse) -> Rows:
    data = response.output_data
    phase = _phase(response, "evidence_acquisition", RuntimeComponent.DENSE_RETRIEVAL)
    rows: Rows = [("Evidence phase", phase)]
    retrieval = _section(data, "retrieval")
    if phase in {"skipped by profile", "not executed"}:
        return rows
    rows += _fields(retrieval, ("Latest candidates", "candidate_count"), ("Latest selected", "selected_count"),
                    ("Final context items", "final_context_count"), ("Retrieval confidence", "retrieval_confidence"))
    rows += _fields(data, ("Evidence stop reason", "stop_reason"))
    if retrieval.get("final_context_count") and not _section(data, "generation").get("llm_called"):
        rows.append(("Generation", "Evidence assembled without a model attempt"))
    compression = _section(data, "context_compression")
    rows += _fields(compression, ("Compression enabled", "enabled"), ("Context input words", "original_tokens"),
                    ("Context retained words", "compressed_tokens"), ("Context word budget", "token_budget"))
    for label, field, count_key in (("Selected", "retrieved_memories", "selected_count"),
                                    ("In context", "final_context_memories", "final_context_count")):
        items = _records(retrieval.get(field))
        for index, item in enumerate(items[:MEMORY_LIMIT], 1):
            heading = " / ".join(_value(item.get(key)) for key in ("id", "memory_type", "score"))
            rows.append((f"{label} {index}", heading))
            if field == "final_context_memories":
                rows += _fields(item, ("Excerpt", "preview"))
        count = retrieval.get(count_key)
        if type(count) is int and count > MEMORY_LIMIT:
            rows.append((label, f"{count - MEMORY_LIMIT} more"))
    if retrieval.get("final_context_count") == 0:
        rows.append(("Final evidence", "No memory evidence in final context"))
    return rows


def _admission(response: PrimaResponse) -> Rows:
    phase = _phase(response, "memory_commit", RuntimeComponent.MEMORY_COMMIT)
    rows: Rows = [("Memory commit", phase)]
    if phase != "executed":
        return rows
    admission = _section(response.output_data, "memory_admission")
    rows += _fields(admission, ("Stored", "stored"), ("Policy", "policy"), ("Reason", "reason"),
                    ("Importance score", "total_score"), ("Threshold", "threshold"), ("Novelty", "novelty"),
                    ("Emotional salience", "emotion"), ("Relevance", "relevance"), ("Recurrence", "recurrence"),
                    ("Reflection score", "reflection"))
    ids = response.output_data.get("memory_ids_created")
    if isinstance(ids, (list, tuple)):
        rows.append(("Notes created", len(ids)))
        rows += [("Created note", note_id) for note_id in ids[:MEMORY_LIMIT]]
    return rows


def _action(response: PrimaResponse) -> Rows:
    rows: Rows = [("Action phase", _phase(response, "action"))]
    action = _section(_section(response.output_data, "inspection"), "action")
    rows += _fields(action, ("Recorded result", "status"), ("Intent", "intent"),
                    ("Tool submissions", "tool_invocations"), ("External submissions", "external_actions"),
                    ("Steps recorded", "step_count"))
    for index, step in enumerate(_records(action.get("steps"))[:INSPECTION_LIMIT], 1):
        rows.append((f"Step {index}", f"{_value(step.get('status'))} [{_value(step.get('id'))}]"))
        rows += _fields(step, ("Tool", "tool_name"), ("Tool status", "tool_status"), ("Step result", "message"))
    count = action.get("step_count")
    if type(count) is int and count > INSPECTION_LIMIT:
        rows.append(("Steps", f"{count - INSPECTION_LIMIT} more"))
    return rows


def _review(response: PrimaResponse) -> Rows:
    data = response.output_data
    inspection = _section(data, "inspection")
    rows: Rows = [("Reflection phase", _phase(response, "reflection", RuntimeComponent.REFLECTION)),
                  ("Reasoning reviews", response.diagnostics.reasoning_reflection_count)]
    rows += _fields(_section(inspection, "reflection"), ("Reflection triggered", "triggered"),
                    ("Trigger score", "trigger_score"), ("Advice", "advice"))
    for reason in _records(data.get("reflection_reasons"))[:INSPECTION_LIMIT]:
        if reason.get("triggered"):
            rows += _fields(reason, ("Triggered by", "reason"))
    attempts = _records(data.get("correction_attempts"))
    rows.append(("Workflow corrections", len(attempts)))
    for index, attempt in enumerate(attempts[:INSPECTION_LIMIT], 1):
        verdict = "accepted" if attempt.get("accepted") is True else "rejected"
        advice = _section(attempt, "advice")
        rows.append((f"Correction {index}", f"{_value(attempt.get('stage'))} / {_value(advice.get('action'))} / {verdict}"))
        rows += _fields(advice, ("Advice reason", "reason_code"))
        rows += _fields(attempt, ("Correction reason", "decision_reason"))
        if attempt.get("accepted") is True:
            rows += _fields(attempt, ("Confidence change", "confidence_delta"))
    if len(attempts) > INSPECTION_LIMIT:
        rows.append(("Corrections", f"{len(attempts) - INSPECTION_LIMIT} more"))
    rows.append(("Validation phase", _phase(response, "output_validation", RuntimeComponent.OUTPUT_VALIDATOR)))
    validation = _section(inspection, "validation")
    if validation:
        rows.append(("Technical verdict", "passed" if validation.get("valid") is True else "failed"))
        signals = validation.get("signals")
        if isinstance(signals, list):
            rows += [("Validation signal", signal) for signal in signals[:INSPECTION_LIMIT]] or [("Validation signals", "none")]
        passes = sum(event.get("phase") == "output_validation" and event.get("event_type") == "phase_completed"
                     for event in response.diagnostics.trace_events)
        if passes > 1:
            rows.append(("Validation passes", passes))
    return rows


def turn_rows(
    response: PrimaResponse | None, session_id: str, command: str, *, detail: bool = False, secrets: tuple[str, ...] = (),
) -> tuple[tuple[str, str], ...]:
    rows: Rows = [("Session", session_id)]
    if response is None:
        rows.append(("Status", "No turn response is available yet."))
    else:
        rows.extend((("Last turn profile", response.profile.value), ("Turn status", response.status.value)))
        if command == "/affect":
            rows += _affect(response, detail)
        elif command == "/decision":
            rows += _decision(response, detail)
        else:
            rows += {"/evidence": _evidence, "/memory": _admission, "/action": _action, "/review": _review}[command](response)
    return tuple((_value(sanitize_content(label, secrets)), _value(sanitize_content(value, secrets))) for label, value in rows)
