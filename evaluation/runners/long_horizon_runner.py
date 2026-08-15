"""Long-Horizon Runtime — Deliverable 7.

Loads synthetic users, simulates conversations through PrimaRuntime,
stores runtime traces, and collects evaluation metrics.

Outputs (evaluation/results/):
    long_horizon_results.json   — per-user per-turn records
    long_horizon_trace.json     — lightweight turn trace
    long_horizon_summary.json   — cohort-level aggregate summary
    memory_retention_analysis.json
    memory_evolution_analysis.json
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from evaluation.metrics.long_horizon_metrics import (
    average_latency,
    build_long_horizon_summary,
    emotion_continuity_curve,
    emotion_continuity_score,
    fact_retention_curve,
    identity_consistency_curve,
    identity_consistency_score,
    latency_vs_memory,
    memory_evolution_analysis,
    memory_growth_curve,
    preference_retention_at_turn,
    preference_retention_curve,
    reflection_stability,
    retrieval_stability,
)
from evaluation.synthetic.synthetic_user import SyntheticUser
from evaluation.synthetic.user_generator import UserGenerator
from runtime.prima_runtime import PrimaRuntime
from runtime.runtime_context import RuntimeContext

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("evaluation/results")
DEFAULT_N_USERS = 100
DEFAULT_MIN_TURNS = 500
DEFAULT_MAX_TURNS = 1000


# ---------------------------------------------------------------------------
# Probe evaluation helpers (no LLM — heuristic keyword matching)
# ---------------------------------------------------------------------------

def _check_preference_retrieved(
    retrieved_memories: tuple[Any, ...],
    ground_truth: str,
) -> str:
    """Return the first retrieved preference value that matches any known preference."""
    if not ground_truth:
        return ""
    for item in retrieved_memories:
        content = ""
        if hasattr(item, "note") and hasattr(item.note, "content"):
            content = item.note.content
        elif hasattr(item, "content"):
            content = item.content
        else:
            content = str(item)
        if ground_truth.lower() in content.lower():
            return ground_truth
    return ""


def _check_identity_field(
    retrieved_memories: tuple[Any, ...],
    field_value: str,
) -> bool:
    """Return True if any retrieved memory contains the expected field value."""
    if not field_value:
        return False
    for item in retrieved_memories:
        content = ""
        if hasattr(item, "note") and hasattr(item.note, "content"):
            content = item.note.content
        elif hasattr(item, "content"):
            content = item.content
        else:
            content = str(item)
        if field_value.lower() in content.lower():
            return True
    return False


def _get_dominant_emotion(affect_state: dict[str, Any]) -> str:
    """Extract dominant emotion string from affect_state dict."""
    return str(affect_state.get("dominant_emotion", "neutral"))


# ---------------------------------------------------------------------------
# Per-user simulation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class UserSimulationResult:
    """Aggregated results for a single simulated user."""

    user_id: str
    name: str
    seed: int
    total_turns: int
    turn_records: list[dict[str, Any]] = field(default_factory=list)
    identity_probe_results: list[dict[str, Any]] = field(default_factory=list)
    emotion_records: list[dict[str, Any]] = field(default_factory=list)
    fact_probe_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        # Compute metrics
        pref_ret = preference_retention_at_turn(self.turn_records, self.total_turns)
        pref_curve = preference_retention_curve(self.turn_records)
        id_score = identity_consistency_score(self.identity_probe_results)
        id_curve = identity_consistency_curve(self.identity_probe_results)
        emo_score = emotion_continuity_score(self.emotion_records)
        emo_curve = emotion_continuity_curve(self.emotion_records)
        fact_curve = fact_retention_curve(self.fact_probe_records)
        mem_curve = memory_growth_curve(self.turn_records)
        refl = reflection_stability(self.turn_records)
        ret_stab = retrieval_stability(self.turn_records)
        avg_lat = average_latency(self.turn_records)
        evo = memory_evolution_analysis(self.turn_records)

        total_created = sum(int(r.get("memory_notes_created", 0)) for r in self.turn_records)

        return {
            "user_id": self.user_id,
            "name": self.name,
            "seed": self.seed,
            "total_turns": self.total_turns,
            "total_errors": len(self.errors),
            # Core metrics
            "preference_retention": pref_ret,
            "preference_retention_curve": {str(k): v for k, v in pref_curve.items()},
            "identity_consistency": id_score,
            "identity_consistency_curve": {str(k): v for k, v in id_curve.items()},
            "emotion_continuity": emo_score,
            "emotion_continuity_curve": {str(k): v for k, v in emo_curve.items()},
            "fact_retention_curve": {str(k): v for k, v in fact_curve.items()},
            "total_memories_created": total_created,
            "memory_growth_curve": mem_curve[-20:] if len(mem_curve) > 20 else mem_curve,
            "reflection_frequency": refl["reflection_frequency"],
            "correction_frequency": refl["correction_frequency"],
            "mean_reflection_utility": refl["mean_utility"],
            "retrieval_degradation_ratio": ret_stab["degradation_ratio"],
            "retrieval_first_third": ret_stab["first_third"],
            "retrieval_final_third": ret_stab["final_third"],
            "average_latency_ms": avg_lat,
            "retrieval_hit_rate": ret_stab.get("hit_rate", 0.0),
            "memory_evolution": evo,
            "errors": list(self.errors),
        }


def _simulate_user(
    user: SyntheticUser,
    max_turns: int | None = None,
    runtime_factory: Callable[..., PrimaRuntime] = PrimaRuntime,
) -> UserSimulationResult:
    """Run a full conversation for one synthetic user through PrimaRuntime."""
    runtime = runtime_factory(log_path=f"logs/long_horizon_{user.user_id}.log")
    ctx = RuntimeContext(session_id=f"lh_{user.user_id}")

    result = UserSimulationResult(
        user_id=user.user_id,
        name=user.name,
        seed=user.seed,
        total_turns=user.total_turns,
    )

    turns = user.conversation_turns
    if max_turns is not None:
        turns = turns[:max_turns]

    for turn in turns:
        t_idx = turn.turn_index

        try:
            rt_result = runtime.process(turn.utterance, context=ctx)
        except Exception as exc:  # noqa: BLE001
            err = f"Turn {t_idx}: {exc}"
            result.errors.append(err)
            logger.warning("User %s %s", user.user_id, err)
            continue

        # Preference probe
        retrieved_pref = _check_preference_retrieved(
            rt_result.retrieved_memories, turn.ground_truth_preference
        )

        # Dominant emotion
        dom_emotion = _get_dominant_emotion(rt_result.affect_state)

        # Build turn record
        record: dict[str, Any] = {
            "turn_index": t_idx,
            "turn_type": turn.turn_type,
            "utterance": turn.utterance,
            "memory_notes_created": len(rt_result.memory_notes_created),
            "retrieval_count": len(rt_result.retrieved_memories),
            "reflection_triggered": rt_result.reflection_triggered,
            "correction_count": rt_result.correction_count,
            "reflection_utility_score": rt_result.reflection_utility_score,
            "confidence_score": rt_result.confidence_score,
            "latency_ms": rt_result.latency_ms,
            "dominant_emotion": dom_emotion,
            "retrieved_preference": retrieved_pref,
            "ground_truth_preference": turn.ground_truth_preference,
            "expected_emotion": turn.expected_emotion,
            "retrieved_emotion": dom_emotion,
            "temporal_week": turn.temporal_week,
        }
        result.turn_records.append(record)

        # Emotion record
        result.emotion_records.append({
            "turn_index": t_idx,
            "retrieved_emotion": dom_emotion,
            "expected_emotion": turn.expected_emotion,
        })

        # Identity and fact probes at key milestones
        if t_idx in user.identity_probes:
            expected = user.identity_probes[t_idx]
            for field_name, field_val in expected.items():
                consistent = _check_identity_field(rt_result.retrieved_memories, field_val)
                result.identity_probe_results.append({
                    "probe_turn": t_idx,
                    "field": field_name,
                    "ground_truth": field_val,
                    "consistent": consistent,
                })

        if t_idx in user.fact_probes:
            expected_facts = user.fact_probes[t_idx]
            for fact_key, fact_val in expected_facts.items():
                correct = _check_identity_field(rt_result.retrieved_memories, fact_val)
                result.fact_probe_records.append({
                    "probe_turn": t_idx,
                    "fact_key": fact_key,
                    "fact_value": fact_val,
                    "correct": correct,
                })

    return result


# ---------------------------------------------------------------------------
# Main runner class
# ---------------------------------------------------------------------------

@dataclass
class LongHorizonRunner:
    """Orchestrate long-horizon evaluation over a cohort of synthetic users."""

    n_users: int = DEFAULT_N_USERS
    min_turns: int = DEFAULT_MIN_TURNS
    max_turns: int = DEFAULT_MAX_TURNS
    seed_base: int = UserGenerator.DEFAULT_SEED_BASE
    results_dir: Path = field(default_factory=lambda: DEFAULT_RESULTS_DIR)
    max_turns_override: int | None = None
    runtime_factory: Callable[..., PrimaRuntime] = PrimaRuntime

    def __post_init__(self) -> None:
        self.results_dir = Path(self.results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "plots").mkdir(parents=True, exist_ok=True)

    def run(self) -> dict[str, Any]:
        """Run the full long-horizon evaluation and persist all output files."""
        logger.info(
            "Starting long-horizon evaluation: %d users, %d–%d turns",
            self.n_users, self.min_turns, self.max_turns,
        )
        start_wall = time.perf_counter()

        generator = UserGenerator(
            n_users=self.n_users,
            min_turns=self.min_turns,
            max_turns=self.max_turns,
            seed_base=self.seed_base,
        )
        users = generator.generate_all()
        logger.info("Generated %d synthetic users", len(users))

        all_results: list[dict[str, Any]] = []
        all_turn_records: list[dict[str, Any]] = []  # lightweight trace
        forgotten_records: list[dict[str, Any]] = []
        all_latency_memory: list[dict[str, Any]] = []

        for idx, user in enumerate(users):
            logger.info("Simulating user %d/%d: %s", idx + 1, len(users), user.user_id)
            sim = _simulate_user(
                user,
                max_turns=self.max_turns_override,
                runtime_factory=self.runtime_factory,
            )
            user_dict = sim.to_dict()
            all_results.append(user_dict)

            # Build lightweight trace (subset of fields)
            for record in sim.turn_records:
                trace_record = {
                    "user_id": user.user_id,
                    "turn_index": record["turn_index"],
                    "turn_type": record["turn_type"],
                    "memory_notes_created": record["memory_notes_created"],
                    "retrieval_count": record["retrieval_count"],
                    "reflection_triggered": record["reflection_triggered"],
                    "latency_ms": record["latency_ms"],
                    "dominant_emotion": record["dominant_emotion"],
                }
                all_turn_records.append(trace_record)

            # Latency vs memory
            lvm = latency_vs_memory(sim.turn_records)
            for entry in lvm:
                entry["user_id"] = user.user_id
            all_latency_memory.extend(lvm)

        # Cohort-level summary
        summary = build_long_horizon_summary(all_results)
        summary["total_wall_time_seconds"] = round(time.perf_counter() - start_wall, 3)

        # Memory retention analysis
        retention_analysis = {
            "total_users": len(users),
            "forgetting_records": forgotten_records,
            "per_user_forgetting": [
                {"user_id": r["user_id"], "total_memories": r.get("total_memories_created", 0)}
                for r in all_results
            ],
        }

        # Memory evolution analysis
        evolution_analysis = {
            "total_users": len(users),
            "per_user_evolution": [
                {"user_id": r["user_id"], **r.get("memory_evolution", {})}
                for r in all_results
            ],
            "cohort_means": {
                "total_memories_created": round(
                    sum(r.get("total_memories_created", 0) for r in all_results) / max(1, len(all_results)), 2
                ),
                "average_latency_ms": round(
                    sum(r.get("average_latency_ms", 0.0) for r in all_results) / max(1, len(all_results)), 3
                ),
            },
        }

        # Persist outputs
        self._write_json(all_results, "long_horizon_results.json")
        self._write_json(all_turn_records, "long_horizon_trace.json")
        self._write_json(summary, "long_horizon_summary.json")
        self._write_json(retention_analysis, "memory_retention_analysis.json")
        self._write_json(evolution_analysis, "memory_evolution_analysis.json")

        logger.info(
            "Long-horizon evaluation complete. Summary written to %s",
            self.results_dir / "long_horizon_summary.json",
        )

        return {
            "summary": summary,
            "n_users": len(users),
            "results_dir": str(self.results_dir),
        }

    def _write_json(self, data: Any, filename: str) -> None:
        path = self.results_dir / filename
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)
        logger.info("Wrote %s", path)


# ---------------------------------------------------------------------------
# Stress-test entry point (Deliverable 13)
# ---------------------------------------------------------------------------

@dataclass
class StressTestConfig:
    """Configuration for a single stress-test run."""

    n_users: int = 10
    max_turns: int = 500
    seed_base: int = 9999

    def label(self) -> str:
        return f"stress_{self.n_users}u_{self.max_turns}t"


def run_stress_tests(
    configs: list[StressTestConfig] | None = None,
    results_dir: Path = DEFAULT_RESULTS_DIR,
) -> dict[str, Any]:
    """Run multiple stress-test configurations and collect stability metrics."""
    if configs is None:
        configs = [
            StressTestConfig(n_users=100, max_turns=500),
            StressTestConfig(n_users=100, max_turns=750),
            StressTestConfig(n_users=100, max_turns=1000),
        ]

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    stress_results: list[dict[str, Any]] = []
    for cfg in configs:
        logger.info("Stress test: %s", cfg.label())
        runner = LongHorizonRunner(
            n_users=cfg.n_users,
            min_turns=cfg.max_turns,
            max_turns=cfg.max_turns,
            seed_base=cfg.seed_base,
            results_dir=results_dir / cfg.label(),
        )
        try:
            result = runner.run()
            stress_results.append({
                "label": cfg.label(),
                "n_users": cfg.n_users,
                "max_turns": cfg.max_turns,
                "status": "ok",
                "summary": result.get("summary", {}),
            })
        except Exception as exc:  # noqa: BLE001
            stress_results.append({
                "label": cfg.label(),
                "n_users": cfg.n_users,
                "max_turns": cfg.max_turns,
                "status": "error",
                "error": str(exc),
            })

    output = {"stress_tests": stress_results}
    path = results_dir / "stress_test_results.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2, default=str)

    return output


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    runner = LongHorizonRunner()
    runner.run()
