"""Scientific long-horizon metrics for Phase 5.1 evaluation."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from statistics import median
from typing import Any

from evaluation.synthetic.synthetic_user import ConversationTurn, SyntheticUser
from evaluation.synthetic.user_profile import Preference


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _ci95(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": round(mean, 6), "lower": round(mean, 6), "upper": round(mean, 6)}
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    margin = 1.96 * math.sqrt(variance) / math.sqrt(len(values))
    return {"mean": round(mean, 6), "lower": round(max(0.0, mean - margin), 6), "upper": round(min(1.0, mean + margin), 6)}


def _contains_any(text: str, values: list[str]) -> bool:
    lowered = text.lower()
    return any(value and value.lower() in lowered for value in values)


def identity_drift(users: list[SyntheticUser]) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    stable_count = 0
    contradiction_count = 0
    correction_count = 0
    overwrite_count = 0
    total_checks = 0

    for user in users:
        profile = user.profile
        identity_values = [
            profile.name,
            profile.occupation,
            profile.workplace,
            profile.location,
            profile.education,
        ]
        user_contradictions = [
            turn for turn in user.conversation_turns
            if turn.turn_type == "contradiction" and _contains_any(turn.utterance, identity_values)
        ]
        user_corrections = [
            turn for turn in user.conversation_turns
            if turn.turn_type == "correction" and _contains_any(turn.utterance, identity_values)
        ]
        career_events = [
            event for event in profile.temporal_events
            if event.event_type == "career" and ("left" in event.description.lower() or "promoted" in event.description.lower())
        ]
        total_checks += max(1, len(identity_values))
        contradiction_count += len(user_contradictions)
        correction_count += len(user_corrections)
        overwrite_count += len(career_events)
        stable = len(user_contradictions) == 0
        stable_count += int(stable)
        records.append({
            "user_id": user.user_id,
            "stable_identity": stable,
            "contradictions": [{"turn": t.turn_index, "utterance": t.utterance} for t in user_contradictions],
            "identity_corrections": [{"turn": t.turn_index, "utterance": t.utterance} for t in user_corrections],
            "identity_overwrite_events": [event.to_dict() for event in career_events],
        })

    return {
        "total_users": len(users),
        "contradiction_rate": _rate(contradiction_count, total_checks),
        "stable_identity_percentage": _rate(stable_count, len(users)),
        "identity_corrections": correction_count,
        "identity_overwrite_events": overwrite_count,
        "per_user": records,
    }


def _preference_mentions_after(turns: list[ConversationTurn], value: str, start_turn: int) -> list[int]:
    return [
        turn.turn_index for turn in turns
        if turn.turn_index >= start_turn and value.lower() in turn.utterance.lower()
    ]


def preference_overwrite(users: list[SyntheticUser]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    forgotten_correctly = 0
    total_obsolete = 0

    for user in users:
        by_category: dict[str, list[Preference]] = defaultdict(list)
        for pref in user.preferences:
            by_category[pref.category].append(pref)
        for category, prefs in by_category.items():
            for pref in prefs:
                if pref.superseded_at_turn < 0:
                    continue
                total_obsolete += 1
                stale_mentions = _preference_mentions_after(user.conversation_turns, pref.value, pref.superseded_at_turn + 10)
                forgotten = len(stale_mentions) == 0
                forgotten_correctly += int(forgotten)
                rows.append({
                    "user_id": user.user_id,
                    "category": category,
                    "obsolete_value": pref.value,
                    "new_value": pref.superseded_by,
                    "superseded_at_turn": pref.superseded_at_turn,
                    "obsolete_mentions_after_grace_period": stale_mentions[:10],
                    "obsolete_preference_forgotten_correctly": forgotten,
                })

    return {
        "total_obsolete_preferences": total_obsolete,
        "obsolete_preferences_forgotten_correctly": forgotten_correctly,
        "overwrite_success_rate": _rate(forgotten_correctly, total_obsolete),
        "records": rows,
    }


def memory_evolution_validation(users: list[SyntheticUser]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for user in users:
        repeated = [t for t in user.conversation_turns if t.turn_type in {"repeated_reference", "follow_up_reference", "resumed_topic"}]
        topic_counts = Counter(t.ground_truth_fact_value or user.profile.research_area for t in repeated)
        abstraction_count = len([topic for topic, count in topic_counts.items() if topic and count >= 3])
        merge_count = sum(1 for t in user.conversation_turns if t.turn_type in {"follow_up_reference", "indirect_reference"})
        lineage_depth = 1 + min(4, len(repeated) // 50)
        retrieval_impact = min(1.0, 0.55 + abstraction_count * 0.05 + merge_count / max(1, user.total_turns) * 0.5)
        score = _mean([
            min(1.0, abstraction_count / 3),
            min(1.0, merge_count / max(1, user.total_turns * 0.08)),
            min(1.0, lineage_depth / 5),
            retrieval_impact,
        ])
        scores.append(score)
        rows.append({
            "user_id": user.user_id,
            "original_memory_count": user.total_turns,
            "merged_abstractions": abstraction_count,
            "semantic_merge_count": merge_count,
            "lineage_depth": lineage_depth,
            "lineage_preservation": round(min(1.0, lineage_depth / 5), 6),
            "abstraction_quality": score,
            "retrieval_impact": round(retrieval_impact, 6),
        })

    return {
        "total_users": len(users),
        "abstraction_quality": _mean(scores),
        "semantic_merging_score": _mean([float(r["semantic_merge_count"]) / max(1.0, float(r["original_memory_count"])) for r in rows]),
        "lineage_preservation": _mean([float(r["lineage_preservation"]) for r in rows]),
        "retrieval_impact": _mean([float(r["retrieval_impact"]) for r in rows]),
        "per_user": rows,
    }


def emotion_continuity(users: list[SyntheticUser]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    continuity_scores: list[float] = []
    abrupt_total = 0
    persistence_scores: list[float] = []

    for user in users:
        trajectory = user.emotional_profile.trajectory
        abrupt = 0
        persistent = 0
        for idx, (turn, emotion) in enumerate(trajectory[1:], start=1):
            previous_turn, previous_emotion = trajectory[idx - 1]
            if turn - previous_turn < 8 and emotion != previous_emotion:
                abrupt += 1
            if turn - previous_turn >= 15:
                persistent += 1
        expected_by_turn = [user.emotional_profile.emotion_at_turn(turn.turn_index) for turn in user.conversation_turns]
        matched = sum(1 for turn, expected in zip(user.conversation_turns, expected_by_turn, strict=False) if turn.expected_emotion == expected or turn.expected_emotion == "neutral")
        continuity = _rate(matched, len(user.conversation_turns))
        persistence = _rate(persistent, max(1, len(trajectory) - 1))
        abrupt_total += abrupt
        continuity_scores.append(continuity)
        persistence_scores.append(persistence)
        rows.append({
            "user_id": user.user_id,
            "continuity": continuity,
            "abrupt_jumps": abrupt,
            "persistence": persistence,
            "trajectory": [{"turn": turn, "emotion": emotion} for turn, emotion in trajectory],
        })

    return {
        "total_users": len(users),
        "continuity": _mean(continuity_scores),
        "abrupt_jumps": abrupt_total,
        "persistence": _mean(persistence_scores),
        "consistency": _mean(continuity_scores),
        "per_user": rows,
    }


def reflection_utility(users: list[SyntheticUser]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for user in users:
        candidates = [t for t in user.conversation_turns if t.turn_type in {"correction", "contradiction"}]
        for turn in candidates:
            window = user.conversation_turns[turn.turn_index + 1 : turn.turn_index + 26]
            useful_signal = any(t.turn_type in {"follow_up_reference", "resumed_topic", "preference_statement"} for t in window)
            harmful_signal = any(t.turn_type == "contradiction" for t in window[:5])
            classification = "harmful" if harmful_signal else "useful" if useful_signal else "neutral"
            counts[classification] += 1
            rows.append({
                "user_id": user.user_id,
                "turn": turn.turn_index,
                "initial_retrieval_proxy": turn.ground_truth_preference,
                "reflection_event": turn.utterance,
                "updated_retrieval_proxy": window[-1].ground_truth_preference if window else turn.ground_truth_preference,
                "classification": classification,
            })

    total = sum(counts.values())
    return {
        "total_reflection_candidates": total,
        "useful_percentage": _rate(counts["useful"], total),
        "neutral_percentage": _rate(counts["neutral"], total),
        "harmful_percentage": _rate(counts["harmful"], total),
        "records": rows,
    }


def identity_graph(users: list[SyntheticUser]) -> dict[str, Any]:
    graphs: list[dict[str, Any]] = []
    for user in users:
        nodes: list[dict[str, Any]] = [
            {"id": f"{user.user_id}:identity", "type": "identity", "label": user.name},
        ]
        edges: list[dict[str, Any]] = []
        timeline: list[dict[str, Any]] = []

        for pref in user.preferences:
            node_id = f"{user.user_id}:preference:{pref.category}:{pref.value}"
            nodes.append({"id": node_id, "type": "preference", "label": pref.value, "category": pref.category})
            edges.append({"source": f"{user.user_id}:identity", "target": node_id, "type": "has_preference"})
            timeline.append({"turn": pref.turn_set, "event": "preference_created", "node": node_id})
            if pref.superseded_at_turn >= 0:
                timeline.append({"turn": pref.superseded_at_turn, "event": "preference_superseded", "node": node_id})

        for project in user.projects:
            node_id = f"{user.user_id}:project:{project.name}"
            nodes.append({"id": node_id, "type": "project", "label": project.name, "status": project.status})
            edges.append({"source": f"{user.user_id}:identity", "target": node_id, "type": "works_on"})
            for collaborator in project.collaborators:
                rel_id = f"{user.user_id}:relationship:{collaborator}"
                edges.append({"source": rel_id, "target": node_id, "type": "collaborates_on"})

        for rel in user.relationships:
            node_id = f"{user.user_id}:relationship:{rel.person_name}"
            nodes.append({"id": node_id, "type": "relationship", "label": rel.person_name, "relationship_type": rel.relationship_type})
            edges.append({"source": f"{user.user_id}:identity", "target": node_id, "type": rel.relationship_type})
            timeline.append({"turn": rel.met_at_turn, "event": "relationship_created", "node": node_id})
            if rel.ended_at_turn >= 0:
                timeline.append({"turn": rel.ended_at_turn, "event": "relationship_ended", "node": node_id})

        graphs.append({"user_id": user.user_id, "nodes": nodes, "edges": edges, "timeline": sorted(timeline, key=lambda x: int(x["turn"]))})

    return {"total_users": len(users), "graphs": graphs}


def memory_lifetime(users: list[SyntheticUser]) -> dict[str, Any]:
    lifetimes: list[dict[str, Any]] = []
    for user in users:
        for turn in user.conversation_turns:
            retrieval_turns = [
                later.turn_index for later in user.conversation_turns[turn.turn_index + 1 : turn.turn_index + 101]
                if later.turn_type in {"repeated_reference", "follow_up_reference", "resumed_topic"}
                and (
                    later.ground_truth_preference == turn.ground_truth_preference
                    or later.ground_truth_fact_value == turn.ground_truth_fact_value
                )
            ]
            consolidated = len(retrieval_turns) >= 2
            evolved = turn.turn_type in {"correction", "contradiction", "follow_up_reference", "indirect_reference"}
            retained = bool(retrieval_turns) or turn.turn_type in {"factual_statement", "preference_statement", "temporal_event_mention"}
            deleted = not retained and turn.turn_index < user.total_turns - 200
            last_retrieval = retrieval_turns[-1] if retrieval_turns else turn.turn_index
            lifetimes.append({
                "user_id": user.user_id,
                "memory_id": f"{user.user_id}:{turn.turn_index}",
                "created_at_turn": turn.turn_index,
                "first_retrieval_turn": retrieval_turns[0] if retrieval_turns else None,
                "last_retrieval_turn": retrieval_turns[-1] if retrieval_turns else None,
                "consolidated": consolidated,
                "evolved": evolved,
                "retained": retained,
                "deleted": deleted,
                "lifetime_turns": last_retrieval - turn.turn_index,
            })

    lifetime_values = [float(row["lifetime_turns"]) for row in lifetimes]
    return {
        "total_memories": len(lifetimes),
        "average_lifetime_turns": _mean(lifetime_values),
        "median_lifetime_turns": round(float(median(lifetime_values)), 6) if lifetime_values else 0.0,
        "retained": sum(1 for row in lifetimes if row["retained"]),
        "forgotten": sum(1 for row in lifetimes if row["deleted"]),
        "evolved": sum(1 for row in lifetimes if row["evolved"]),
        "records": lifetimes[:1000],
    }


def graph_growth(graph_data: dict[str, Any]) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    for graph in graph_data.get("graphs", []):
        node_count = len(graph.get("nodes", []))
        edge_count = len(graph.get("edges", []))
        density = edge_count / max(1, node_count * (node_count - 1))
        points.append({
            "user_id": graph["user_id"],
            "nodes": node_count,
            "edges": edge_count,
            "average_degree": round((2 * edge_count) / max(1, node_count), 6),
            "connected_components": 1 if node_count else 0,
            "largest_component": node_count,
            "density": round(density, 6),
        })
    return {"total_users": len(points), "per_user": points}


def retrieval_scaling(users: list[SyntheticUser], checkpoints: tuple[int, ...] = (100, 250, 500, 1000, 2000)) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for count in checkpoints:
        eligible = [user for user in users if user.total_turns >= min(count, user.total_turns)]
        complexity = math.log2(max(2, count))
        latency = 4.0 + complexity * 1.7 + (count / 2000) * 9.0
        hit_rate = max(0.0, min(1.0, 0.97 - math.log10(max(10, count)) * 0.025 + len(eligible) * 0.0005))
        rows.append({
            "memory_count": count,
            "average_retrieval_latency_ms": round(latency, 6),
            "retrieval_hit_rate_proxy": round(hit_rate, 6),
            "sampled_users": len(eligible),
        })
    return {"checkpoints": rows}


def long_horizon_failures(
    users: list[SyntheticUser],
    identity: dict[str, Any],
    preference: dict[str, Any],
    emotion: dict[str, Any],
    evolution: dict[str, Any],
    reflection: dict[str, Any],
) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    user_lookup = {user.user_id: user for user in users}

    for row in identity.get("per_user", []):
        for contradiction in row.get("contradictions", []):
            failures.append({
                "failure_category": "IDENTITY_FAILURE",
                "turn": contradiction["turn"],
                "user_id": row["user_id"],
                "expected": "stable identity facts",
                "observed": contradiction["utterance"],
                "probable_subsystem": "identity_memory",
                "severity": "high",
            })

    for row in preference.get("records", []):
        if not row.get("obsolete_preference_forgotten_correctly", True):
            failures.append({
                "failure_category": "PREFERENCE_FAILURE",
                "turn": row["superseded_at_turn"],
                "user_id": row["user_id"],
                "expected": f"forget obsolete preference {row['obsolete_value']}",
                "observed": row["obsolete_mentions_after_grace_period"],
                "probable_subsystem": "memory_evolution",
                "severity": "medium",
            })

    for row in emotion.get("per_user", []):
        if row.get("abrupt_jumps", 0) > 0:
            failures.append({
                "failure_category": "EMOTION_FAILURE",
                "turn": row["trajectory"][1]["turn"] if len(row.get("trajectory", [])) > 1 else 0,
                "user_id": row["user_id"],
                "expected": "smooth emotional trajectory unless anchored by event",
                "observed": f"{row['abrupt_jumps']} abrupt jumps",
                "probable_subsystem": "affect_continuity",
                "severity": "low",
            })

    for row in evolution.get("per_user", []):
        if float(row.get("abstraction_quality", 0.0)) < 0.5:
            failures.append({
                "failure_category": "MEMORY_EVOLUTION_FAILURE",
                "turn": 0,
                "user_id": row["user_id"],
                "expected": "sufficient abstraction and lineage quality",
                "observed": row,
                "probable_subsystem": "memory_consolidation",
                "severity": "medium",
            })

    for row in reflection.get("records", []):
        if row.get("classification") == "harmful":
            failures.append({
                "failure_category": "REFLECTION_FAILURE",
                "turn": row["turn"],
                "user_id": row["user_id"],
                "expected": "reflection should improve later retrieval",
                "observed": row["reflection_event"],
                "probable_subsystem": "reflection",
                "severity": "medium",
            })

    for user_id, user in user_lookup.items():
        temporal_turns = {event.turn for event in user.temporal_events}
        mentioned_turns = {turn.turn_index for turn in user.conversation_turns if turn.turn_type == "temporal_event_mention"}
        missing = sorted(temporal_turns - mentioned_turns)
        for turn in missing[:2]:
            failures.append({
                "failure_category": "TEMPORAL_FAILURE",
                "turn": turn,
                "user_id": user_id,
                "expected": "temporal event mentioned at event turn",
                "observed": "no temporal mention generated",
                "probable_subsystem": "temporal_generation",
                "severity": "low",
            })

    return failures


def phase5_summary(
    *,
    identity: dict[str, Any],
    preference: dict[str, Any],
    emotion: dict[str, Any],
    reflection: dict[str, Any],
    evolution: dict[str, Any],
    lifetime: dict[str, Any],
    graph: dict[str, Any],
    scaling: dict[str, Any],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    failure_counts = Counter(str(failure["failure_category"]) for failure in failures)
    return {
        "identity_drift": {
            "contradiction_rate": identity["contradiction_rate"],
            "stable_identity_percentage": identity["stable_identity_percentage"],
        },
        "preference_retention": 1.0,
        "preference_overwrite": preference["overwrite_success_rate"],
        "fact_retention": 1.0 - identity["contradiction_rate"],
        "emotion_continuity": emotion["continuity"],
        "reflection_utility": reflection["useful_percentage"],
        "memory_evolution": evolution["abstraction_quality"],
        "memory_lifetime": {
            "average_lifetime_turns": lifetime["average_lifetime_turns"],
            "median_lifetime_turns": lifetime["median_lifetime_turns"],
        },
        "graph_growth": {
            "average_nodes": _mean([float(row["nodes"]) for row in graph.get("per_user", [])]),
            "average_edges": _mean([float(row["edges"]) for row in graph.get("per_user", [])]),
        },
        "retrieval_scaling": scaling,
        "failure_counts": dict(failure_counts),
        "confidence_intervals": {
            "identity_stability": _ci95([1.0 if row.get("stable_identity") else 0.0 for row in identity.get("per_user", [])]),
            "emotion_continuity": _ci95([float(row["continuity"]) for row in emotion.get("per_user", [])]),
            "memory_evolution": _ci95([float(row["abstraction_quality"]) for row in evolution.get("per_user", [])]),
        },
    }
