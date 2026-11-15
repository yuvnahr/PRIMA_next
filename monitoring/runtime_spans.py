"""Bounded aggregate metadata from canonical phase results, never prompt content."""

from __future__ import annotations

from typing import Any

from monitoring.tracing import SafeSpan, root_attributes


def record_retrieval(observation: SafeSpan, result: Any) -> None:
    try:
        diagnostics = result.diagnostics
        observation.set_attributes({
            "candidate_count": len(diagnostics["hybrid_candidates"]) if "hybrid_candidates" in diagnostics else None,
            "selected_count": len(result.results), "retrieval_confidence": result.confidence.confidence,
            "memory_types": sorted({item.note.memory_type.value for item in result.results}),
        })
    except Exception as exc:
        if observation.owner:
            observation.owner.warn("retrieval metadata", exc)


def record_phase(observation: SafeSpan, phase: str, result: Any, context: Any) -> None:
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
