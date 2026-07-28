"""Long-horizon evaluation metrics — Deliverable 8."""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# Preference Retention
# ---------------------------------------------------------------------------

def preference_retention_at_turn(
    turn_records: list[dict[str, Any]],
    at_turn: int,
) -> float:
    """Fraction of turns ≤ at_turn where the retrieved preference matched ground truth."""
    relevant = [r for r in turn_records if r.get("turn_index", 0) <= at_turn]
    if not relevant:
        return 0.0
    hits = sum(
        1 for r in relevant
        if r.get("retrieved_preference") == r.get("ground_truth_preference")
        and r.get("ground_truth_preference", "") != ""
    )
    denom = sum(1 for r in relevant if r.get("ground_truth_preference", "") != "")
    return round(hits / denom, 6) if denom else 0.0


def preference_retention_curve(
    turn_records: list[dict[str, Any]],
    checkpoints: tuple[int, ...] = (10, 50, 100, 250, 500, 750, 1000),
) -> dict[int, float]:
    """Return preference retention at each checkpoint turn."""
    return {cp: preference_retention_at_turn(turn_records, cp) for cp in checkpoints}


# ---------------------------------------------------------------------------
# Identity Consistency
# ---------------------------------------------------------------------------

def identity_consistency_score(probe_results: list[dict[str, Any]]) -> float:
    """Fraction of identity probes answered consistently with ground truth.

    Each probe record must have fields:
        probe_turn, field, ground_truth, retrieved_value, consistent (bool)
    """
    if not probe_results:
        return 0.0
    consistent = sum(1 for r in probe_results if r.get("consistent", False))
    return round(consistent / len(probe_results), 6)


def identity_consistency_curve(
    probe_results: list[dict[str, Any]],
    checkpoints: tuple[int, ...] = (10, 50, 100, 250, 500),
) -> dict[int, float]:
    """Identity consistency per checkpoint."""
    result: dict[int, float] = {}
    for cp in checkpoints:
        at_cp = [r for r in probe_results if r.get("probe_turn", 0) <= cp]
        result[cp] = identity_consistency_score(at_cp)
    return result


# ---------------------------------------------------------------------------
# Fact Retention
# ---------------------------------------------------------------------------

def fact_retention_at_turn(
    fact_records: list[dict[str, Any]],
    at_turn: int,
) -> float:
    """Fraction of fact probes at exactly the given checkpoint that matched."""
    relevant = [r for r in fact_records if r.get("probe_turn", 0) == at_turn]
    if not relevant:
        return 0.0
    hits = sum(1 for r in relevant if r.get("correct", False))
    return round(hits / len(relevant), 6)


def fact_retention_curve(
    fact_records: list[dict[str, Any]],
    checkpoints: tuple[int, ...] = (10, 50, 100, 250, 500),
) -> dict[int, float]:
    """Fact retention rate at each checkpoint."""
    return {cp: fact_retention_at_turn(fact_records, cp) for cp in checkpoints}


# ---------------------------------------------------------------------------
# Emotion Continuity
# ---------------------------------------------------------------------------

def emotion_continuity_score(emotion_records: list[dict[str, Any]]) -> float:
    """Fraction of turns where retrieved emotion matches expected emotion trajectory."""
    if not emotion_records:
        return 0.0
    matches = sum(
        1 for r in emotion_records
        if r.get("retrieved_emotion") == r.get("expected_emotion")
        and r.get("expected_emotion", "neutral") != ""
    )
    denom = sum(1 for r in emotion_records if r.get("expected_emotion", "") != "")
    return round(matches / denom, 6) if denom else 0.0


def emotion_continuity_curve(
    emotion_records: list[dict[str, Any]],
    checkpoints: tuple[int, ...] = (10, 50, 100, 250, 500),
) -> dict[int, float]:
    """Emotion continuity at each checkpoint."""
    result: dict[int, float] = {}
    for cp in checkpoints:
        at_cp = [r for r in emotion_records if r.get("turn_index", 0) <= cp]
        result[cp] = emotion_continuity_score(at_cp)
    return result


# ---------------------------------------------------------------------------
# Memory Growth
# ---------------------------------------------------------------------------

def memory_growth_curve(
    turn_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return cumulative memory store size at each recorded turn."""
    cumulative = 0
    curve: list[dict[str, Any]] = []
    for record in sorted(turn_records, key=lambda r: r.get("turn_index", 0)):
        cumulative += int(record.get("memory_notes_created", 0))
        curve.append({
            "turn_index": record.get("turn_index", 0),
            "cumulative_memories": cumulative,
            "interactions": record.get("turn_index", 0) + 1,
            "memory_density": round(cumulative / max(1, record.get("turn_index", 0) + 1), 6),
        })
    return curve


def memory_density_at_turn(curve: list[dict[str, Any]], at_turn: int) -> float:
    """Return memory density (memories / interactions) at the given turn."""
    matches = [c for c in curve if c["turn_index"] <= at_turn]
    return float(matches[-1]["memory_density"]) if matches else 0.0


# ---------------------------------------------------------------------------
# Retrieval Stability
# ---------------------------------------------------------------------------

def retrieval_stability(turn_records: list[dict[str, Any]]) -> dict[str, float]:
    """Measure whether retrieval quality degrades as memory grows.

    Returns mean retrieval count in first/second/third thirds and the
    degradation ratio (final_third / first_third).
    """
    if not turn_records:
        return {"first_third": 0.0, "second_third": 0.0, "final_third": 0.0, "degradation_ratio": 1.0}

    sorted_records = sorted(turn_records, key=lambda r: r.get("turn_index", 0))
    n = len(sorted_records)
    third = max(1, n // 3)

    def mean_retrieval(subset: list[dict[str, Any]]) -> float:
        vals = [float(r.get("retrieval_count", 0)) for r in subset]
        return round(sum(vals) / len(vals), 6) if vals else 0.0

    first = mean_retrieval(sorted_records[:third])
    second = mean_retrieval(sorted_records[third : 2 * third])
    final = mean_retrieval(sorted_records[2 * third :])
    ratio = round(final / first, 6) if first > 0 else 1.0

    return {
        "first_third": first,
        "second_third": second,
        "final_third": final,
        "degradation_ratio": ratio,
    }


# ---------------------------------------------------------------------------
# Reflection Stability
# ---------------------------------------------------------------------------

def reflection_stability(turn_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Track reflection frequency, correction frequency, and utility across conversation."""
    if not turn_records:
        return {
            "reflection_frequency": 0.0,
            "correction_frequency": 0.0,
            "mean_utility": 0.0,
            "reflection_rate_curve": {},
        }

    sorted_records = sorted(turn_records, key=lambda r: r.get("turn_index", 0))
    total = len(sorted_records)

    reflections = sum(1 for r in sorted_records if r.get("reflection_triggered", False))
    corrections = sum(int(r.get("correction_count", 0)) for r in sorted_records)
    utilities = [float(r.get("reflection_utility_score", 0.0)) for r in sorted_records]

    # Rate by segment
    segment_size = max(1, total // 5)
    rate_curve: dict[int, float] = {}
    for seg in range(5):
        start = seg * segment_size
        end = min(start + segment_size, total)
        seg_records = sorted_records[start:end]
        rate = sum(1 for r in seg_records if r.get("reflection_triggered")) / len(seg_records)
        midpoint = sorted_records[min(end - 1, total - 1)].get("turn_index", 0)
        rate_curve[int(midpoint)] = round(rate, 6)

    return {
        "reflection_frequency": round(reflections / total, 6),
        "correction_frequency": round(corrections / total, 6),
        "mean_utility": round(sum(utilities) / len(utilities), 6),
        "reflection_rate_curve": rate_curve,
    }


# ---------------------------------------------------------------------------
# Forgetting Evaluation (Deliverable 9)
# ---------------------------------------------------------------------------

def forgetting_evaluation(
    forgotten_records: list[dict[str, Any]],
    probe_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure whether forgotten memories were appropriate.

    forgotten_records: list of {memory_id, content, forgotten_at_turn, reason}
    probe_records: list of {memory_id, probe_turn, retrieved (bool)}
    """
    if not forgotten_records:
        return {
            "total_forgotten": 0,
            "still_useful": 0,
            "should_have_retained": 0,
            "appropriate_forgetting_rate": 1.0,
        }

    forgotten_ids = {r["memory_id"] for r in forgotten_records}
    still_useful = sum(
        1 for p in probe_records
        if p["memory_id"] in forgotten_ids and not p.get("retrieved", True)
    )
    should_retain = sum(
        1 for p in probe_records
        if p["memory_id"] in forgotten_ids
        and not p.get("retrieved", True)
        and p.get("probe_turn", 0) < p.get("forgotten_at_turn", 0)
    )

    total = len(forgotten_records)
    appropriate = max(0, total - should_retain)
    return {
        "total_forgotten": total,
        "still_useful": still_useful,
        "should_have_retained": should_retain,
        "appropriate_forgetting_rate": round(appropriate / max(1, total), 6),
    }


# ---------------------------------------------------------------------------
# Memory Evolution (Deliverable 10)
# ---------------------------------------------------------------------------

def memory_evolution_analysis(
    turn_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Measure abstraction creation, semantic merging, lineage evolution, graph growth."""
    if not turn_records:
        return {
            "total_memories_created": 0,
            "abstraction_count": 0,
            "semantic_merge_count": 0,
            "lineage_depth_mean": 0.0,
            "graph_node_count": 0,
            "graph_edge_count": 0,
        }

    total_created = sum(int(r.get("memory_notes_created", 0)) for r in turn_records)
    abstraction_count = sum(int(r.get("abstraction_count", 0)) for r in turn_records)
    semantic_merge_count = sum(int(r.get("semantic_merge_count", 0)) for r in turn_records)
    lineage_depths = [float(r.get("lineage_depth", 1.0)) for r in turn_records if r.get("lineage_depth")]
    graph_nodes = max((int(r.get("graph_node_count", 0)) for r in turn_records), default=0)
    graph_edges = max((int(r.get("graph_edge_count", 0)) for r in turn_records), default=0)

    return {
        "total_memories_created": total_created,
        "abstraction_count": abstraction_count,
        "semantic_merge_count": semantic_merge_count,
        "lineage_depth_mean": round(sum(lineage_depths) / len(lineage_depths), 6) if lineage_depths else 0.0,
        "graph_node_count": graph_nodes,
        "graph_edge_count": graph_edges,
    }


# ---------------------------------------------------------------------------
# Latency Analysis
# ---------------------------------------------------------------------------

def latency_vs_memory(turn_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair each turn's latency with the cumulative memory count at that point."""
    cumulative = 0
    result: list[dict[str, Any]] = []
    for record in sorted(turn_records, key=lambda r: r.get("turn_index", 0)):
        cumulative += int(record.get("memory_notes_created", 0))
        result.append({
            "turn_index": record.get("turn_index", 0),
            "latency_ms": float(record.get("latency_ms", 0.0)),
            "cumulative_memories": cumulative,
        })
    return result


def average_latency(turn_records: list[dict[str, Any]]) -> float:
    """Return mean latency across all recorded turns."""
    lats = [float(r.get("latency_ms", 0.0)) for r in turn_records]
    return round(sum(lats) / len(lats), 6) if lats else 0.0


# ---------------------------------------------------------------------------
# Consolidated Summary (Deliverable 11)
# ---------------------------------------------------------------------------

def build_long_horizon_summary(
    user_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate per-user metrics into a cohort-level summary."""
    if not user_results:
        return {}

    def _mean(key: str) -> float:
        vals = [float(r.get(key, 0.0)) for r in user_results if r.get(key) is not None]
        return round(sum(vals) / len(vals), 6) if vals else 0.0

    def _mean_curve(key: str, checkpoints: tuple[int, ...]) -> dict[int, float]:
        aggregated: dict[int, list[float]] = {cp: [] for cp in checkpoints}
        for r in user_results:
            curve = r.get(key, {})
            if isinstance(curve, dict):
                for cp in checkpoints:
                    v = curve.get(str(cp)) or curve.get(cp)
                    if v is not None:
                        aggregated[cp].append(float(v))
        return {
            cp: round(sum(vs) / len(vs), 6) if vs else 0.0
            for cp, vs in aggregated.items()
        }

    checkpoints = (10, 50, 100, 250, 500)
    return {
        "total_users": len(user_results),
        "preference_retention": _mean("preference_retention"),
        "preference_retention_curve": _mean_curve("preference_retention_curve", checkpoints),
        "identity_consistency": _mean("identity_consistency"),
        "identity_consistency_curve": _mean_curve("identity_consistency_curve", checkpoints),
        "fact_retention_curve": _mean_curve("fact_retention_curve", checkpoints),
        "emotion_continuity": _mean("emotion_continuity"),
        "emotion_continuity_curve": _mean_curve("emotion_continuity_curve", checkpoints),
        "retrieval_recall": _mean("retrieval_hit_rate"),
        "reflection_utility": _mean("mean_reflection_utility"),
        "memory_growth_total": _mean("total_memories_created"),
        "average_latency_ms": _mean("average_latency_ms"),
        "retrieval_degradation_ratio": _mean("retrieval_degradation_ratio"),
    }
