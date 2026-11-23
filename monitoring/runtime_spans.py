"""Aggregate metadata and opt-in projections of canonical phase inputs/results."""

from __future__ import annotations

from typing import Any

from monitoring.tracing import SafeSpan, root_attributes


def _record(value: Any) -> Any:
    return value.to_dict() if value is not None and hasattr(value, "to_dict") else value


def _state(value: Any) -> dict[str, Any]:
    return {"version": value.version, "goal": dict(value.goal), "task": dict(value.task),
            "confidence": dict(value.confidence), "environment": dict(value.environment),
            "emotional": value.emotional.to_dict()}


def _retrieval(result: Any) -> Any:
    if result is None:
        return None
    return {"confidence": result.confidence.to_dict(), "selected_count": len(result.results),
            "selected": [{"id": item.note.id, "text": item.note.content, "score": item.score,
                          "memory_type": item.note.memory_type.value, "provenance": item.note.context}
                         for item in result.results]}


def _evidence(result: Any) -> list[dict[str, Any]]:
    return [{"id": item.evidence_id, "source_id": item.source_id, "text": item.text,
             "query": item.query, "hop": item.hop, "provenance": item.provenance}
            for item in getattr(result, "evidence_references", ())]


def record_phase_input(observation: SafeSpan, phase: str, context: Any) -> None:
    """Snapshot only the relevant inputs, before controller-owned mutations."""
    if not observation.capture_enabled:
        return
    try:
        fields = {
            "affect": ("cognitive_state",),
            "planning": ("cognitive_state", "retrieval_response", "affect_update", "plan"),
            "world_simulation": ("cognitive_state", "plan"),
            "uncertainty_estimation": ("cognitive_state", "retrieval_response", "affect_update", "plan", "reflection_result"),
            "execution_decision": ("uncertainty", "world_prediction"),
            "reflection": ("cognitive_state", "retrieval_response", "affect_update"),
            "action": ("cognitive_state", "plan", "world_prediction", "uncertainty"),
            "answer_generation": ("cognitive_state", "plan", "affect_update", "reasoning_result"),
            "output_validation": ("generation_result", "action_result", "reasoning_result", "plan"),
            "output": ("generation_result", "action_result", "execution_decision", "output_validation"),
            "state_commit": ("cognitive_state", "output"),
            "memory_commit": ("output", "affect_update", "reflection_result"),
            "maintenance_enqueue": ("memory_admission",),
        }
        values: dict[str, Any] = {"phase": phase}
        if phase in {"affect", "planning", "reflection", "answer_generation", "memory_commit", "document_ingestion"}:
            values["user_input"] = context.user_input
        for name in fields.get(phase, ()):
            value = getattr(context, name)
            values[name] = (_state(value) if name == "cognitive_state" else
                            _retrieval(value) if name == "retrieval_response" else
                            {"selected_evidence": _evidence(value)} if name == "reasoning_result" else
                            _record(value))
        metadata_fields = {
            "state_load": ("state_session_id",),
            "evidence_acquisition": ("top_k", "reasoning_mode", "max_hops", "max_retrieval_calls", "max_context_tokens", "profile"),
            "planning": ("replan_reason", "reflection_signals"),
            "world_simulation": ("execution_policy",),
            "uncertainty_estimation": ("execution_policy",),
            "execution_decision": ("retrieval_retry_count", "clarification_required"),
            "reflection": ("reflection_history",),
            "action": ("execution_policy",),
            "answer_generation": ("tool_results",),
            "output": ("correction_terminal_action",),
        }
        for name in metadata_fields.get(phase, ()):
            if name in context.metadata:
                values[name] = _record(context.metadata[name])
        if phase == "evidence_acquisition":
            values["query"] = context.metadata.get("retrieval_query_override", context.user_input)
        if phase == "maintenance_enqueue":
            values["memory_ids"] = [note.id for note in context.memory_notes_created]
        observation.content(input=values)
    except Exception as exc:
        if observation.owner:
            observation.owner.warn("phase input", exc)


def _phase_output(phase: str, result: Any) -> Any:
    if phase == "affect":
        return {**result.to_dict(), "confidence_interpretation": "heuristic estimate, not a calibrated probability"}
    if phase == "state_load":
        return _state(result)
    if phase == "evidence_acquisition":
        answer = result.answer_result
        return {"selected_evidence": _evidence(answer), "retrieval": _retrieval(result.latest_retrieval),
                "confidence": answer.confidence, "stop_reason": answer.stop_reason, "hop_count": answer.hop_count}
    if phase == "state_commit":
        return {"committed": result["committed"], "state": _state(result["state"])}
    if phase == "memory_commit":
        return {"admission": result.get("admission"),
                "notes": [{"id": note.id, "text": note.content, "memory_type": note.memory_type.value}
                          for note in result.get("notes", ())]}
    if phase == "action":
        return _record(result.get("execution_result"))
    if phase == "output_validation":
        return {**result.to_dict(), "validation_scope": "technical; answer quality not assessed"}
    if phase == "document_ingestion":
        return {"memory_id": result.memory_id, "memory_type": result.memory_type.value}
    return _record(result)


def record_retrieval(observation: SafeSpan, result: Any) -> None:
    try:
        diagnostics = result.diagnostics
        observation.set_attributes({
            "candidate_count": len(diagnostics["hybrid_candidates"]) if "hybrid_candidates" in diagnostics else None,
            "selected_count": len(result.results), "retrieval_confidence": result.confidence.confidence,
            "memory_types": sorted({item.note.memory_type.value for item in result.results}),
        })
        if observation.capture_enabled:
            observation.content(output=_retrieval(result))
    except Exception as exc:
        if observation.owner:
            observation.owner.warn("retrieval metadata", exc)


def record_phase(observation: SafeSpan, phase: str, result: Any, context: Any) -> None:
    if observation.capture_enabled:
        try:
            observation.content(output=_phase_output(phase, result))
        except Exception as exc:
            if observation.owner:
                observation.owner.warn("phase output", exc)
    try:
        values: dict[str, Any] = {}
        if phase == "state_load":
            values["state_version"] = result.version
            root_attributes({"state_version_before": result.version})
        elif phase == "affect":
            values = {name: getattr(result.profile, name, None) for name in (
                "dominant_emotion", "valence", "arousal", "dominance", "confidence",
            )}
            values.update(salience=result.salience_score, stability=result.evolution.get("stability"),
                          volatility=result.evolution.get("volatility"))
        elif phase == "evidence_acquisition":
            if result.latest_retrieval is not None:
                record_retrieval(observation, result.latest_retrieval)
            values["final_context_count"] = len(result.answer_result.evidence_references)
        elif phase == "planning":
            values = {"action_count": len(result.actions), "constraint_count": len(result.constraints),
                      "revision": result.revision, "plan_confidence": getattr(result.evaluation, "confidence", None)}
        elif phase == "world_simulation":
            values = {"predicted_action_count": len(result.action_predictions), "risk_score": result.risk_score,
                      "failure_probability": result.failure_probability, "warning_count": len(result.warnings),
                      "constraint_prediction_count": len(result.constraint_predictions)}
        elif phase == "uncertainty_estimation":
            values = {"uncertainty_score": result.uncertainty, "uncertainty_level": result.uncertainty_band.value}
        elif phase == "execution_decision":
            values = {"execution_decision": result.decision.value}
        elif phase == "reflection":
            advice = result.advice
            values = {"triggered": result.should_reflect, "trigger_score": result.trigger_score,
                      "correction_applied": result.correction_applied, "prior_correction_count": len(context.correction_attempts),
                      "correction_type": advice.action.value if advice else None,
                      "reason": advice.reason_code if advice else None}
        elif phase == "answer_generation":
            values = {"outcome": result.outcome.value}
            if result.outcome.value == "failed":
                observation.fail()
        elif phase == "output_validation":
            values = {"valid": result.valid, "signal_count": len(result.signals)}
            if not result.valid:
                observation.fail()
        elif phase == "state_commit":
            values = {"committed": result["committed"], "state_version": result["state"].version}
            root_attributes({"state_version_after": result["state"].version})
        elif phase == "memory_commit":
            admission = result.get("admission") or {}
            values = {"created_count": len(result.get("notes", ())), "admitted": admission.get("stored")}
        elif phase == "maintenance_enqueue":
            values = {"queued_count": result.get("queued_count"), "queue_size": result.get("queue_size")}
        observation.set_attributes(values)
    except Exception as exc:
        if observation.owner:
            observation.owner.warn("phase metadata", exc)
