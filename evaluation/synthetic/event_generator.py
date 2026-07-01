"""Temporal event generation for synthetic users — Deliverable 3."""

from __future__ import annotations

import random
from typing import Any

from evaluation.synthetic.user_profile import TemporalEvent

# Emotion trajectories used across all users (indices map to phases)
EMOTION_TRAJECTORIES: list[list[str]] = [
    ["joy", "anticipation", "trust", "joy", "trust"],
    ["anticipation", "joy", "surprise", "trust", "joy"],
    ["joy", "anticipation", "fear", "sadness", "trust"],
    ["trust", "anticipation", "joy", "anticipation", "trust"],
    ["anticipation", "sadness", "fear", "trust", "joy"],
    ["joy", "surprise", "anger", "sadness", "trust"],
    ["trust", "joy", "anticipation", "sadness", "joy"],
    ["anticipation", "joy", "trust", "fear", "joy"],
]

# Event templates parameterised by occupation category
_TECH_EVENTS: list[dict[str, Any]] = [
    {"description": "Started a new internship at {workplace}.", "type": "career", "emotion": "anticipation"},
    {"description": "Internship mentor changed from {old_person} to {new_person}.", "type": "relationship", "emotion": "surprise"},
    {"description": "Completed first project milestone for {project}.", "type": "milestone", "emotion": "joy"},
    {"description": "Received a positive performance review.", "type": "career", "emotion": "joy"},
    {"description": "Changed primary research topic to {topic}.", "type": "academic", "emotion": "anticipation"},
    {"description": "Switched from {old_pref} to {new_pref} as primary tool.", "type": "preference", "emotion": "anticipation"},
    {"description": "Submitted conference paper on {topic}.", "type": "academic", "emotion": "anticipation"},
    {"description": "Paper accepted at {conference}.", "type": "academic", "emotion": "joy"},
    {"description": "Experienced significant crunch period on {project}.", "type": "stress", "emotion": "fear"},
    {"description": "Successfully deployed {project} to production.", "type": "milestone", "emotion": "joy"},
    {"description": "Started collaborating with {new_person} on {project}.", "type": "relationship", "emotion": "anticipation"},
    {"description": "Stopped working with {old_person} due to project restructuring.", "type": "relationship", "emotion": "sadness"},
    {"description": "Promoted to senior engineer.", "type": "career", "emotion": "joy"},
    {"description": "Left {workplace} and joined a new company.", "type": "career", "emotion": "surprise"},
    {"description": "Switched to remote work arrangement.", "type": "lifestyle", "emotion": "trust"},
    {"description": "Completed a major certification in {topic}.", "type": "academic", "emotion": "joy"},
    {"description": "Presented research results to the team.", "type": "academic", "emotion": "trust"},
    {"description": "Experienced burnout and took time off.", "type": "health", "emotion": "sadness"},
    {"description": "Returned from leave feeling refreshed.", "type": "health", "emotion": "joy"},
    {"description": "Launched a side project in {topic}.", "type": "project", "emotion": "anticipation"},
]

_ACADEMIC_EVENTS: list[dict[str, Any]] = [
    {"description": "Started new semester with focus on {topic}.", "type": "academic", "emotion": "anticipation"},
    {"description": "Passed qualifying examination.", "type": "academic", "emotion": "joy"},
    {"description": "Advisor changed to {new_person}.", "type": "relationship", "emotion": "surprise"},
    {"description": "Submitted dissertation chapter on {topic}.", "type": "academic", "emotion": "trust"},
    {"description": "Defended thesis proposal successfully.", "type": "academic", "emotion": "joy"},
    {"description": "Received fellowship grant.", "type": "career", "emotion": "joy"},
    {"description": "Attended {conference} and presented poster.", "type": "academic", "emotion": "anticipation"},
    {"description": "Published first co-authored paper.", "type": "academic", "emotion": "joy"},
    {"description": "Started teaching assistant role.", "type": "career", "emotion": "anticipation"},
    {"description": "Lab collaboration with {new_person} began.", "type": "relationship", "emotion": "anticipation"},
]

_GENERAL_EVENTS: list[dict[str, Any]] = [
    {"description": "Moved to a new apartment.", "type": "lifestyle", "emotion": "anticipation"},
    {"description": "Started a new hobby: {hobby}.", "type": "lifestyle", "emotion": "joy"},
    {"description": "Met {new_person} through a mutual friend.", "type": "relationship", "emotion": "surprise"},
    {"description": "Adopted a pet.", "type": "lifestyle", "emotion": "joy"},
    {"description": "Completed a long personal goal.", "type": "milestone", "emotion": "joy"},
    {"description": "Experienced a difficult week dealing with {topic}.", "type": "stress", "emotion": "sadness"},
    {"description": "Recovered from a difficult period.", "type": "health", "emotion": "trust"},
    {"description": "Started regular exercise routine.", "type": "lifestyle", "emotion": "anticipation"},
    {"description": "Reconnected with old friend {old_person}.", "type": "relationship", "emotion": "joy"},
]


def _fill_template(template: str, rng: random.Random, profile_hints: dict[str, str]) -> str:
    """Fill a template string with deterministic placeholder values."""
    hints = dict(profile_hints)
    hints.setdefault("old_person", rng.choice(["Alice", "Bob", "Carlos", "Diana", "Ethan"]))
    hints.setdefault("new_person", rng.choice(["Fiona", "George", "Hannah", "Ivan", "Julia"]))
    hints.setdefault("conference", rng.choice(["NeurIPS", "ICML", "CVPR", "ICLR", "CHI", "SOSP"]))
    hints.setdefault("old_pref", rng.choice(["VSCode", "Vim", "Emacs", "PyCharm"]))
    hints.setdefault("new_pref", rng.choice(["Neovim", "IntelliJ", "Helix", "Zed"]))
    try:
        return template.format(**hints)
    except KeyError:
        return template


def generate_temporal_events(
    rng: random.Random,
    total_turns: int,
    occupation: str,
    profile_hints: dict[str, str],
) -> list[TemporalEvent]:
    """Generate a list of temporal events spread across the conversation arc.

    Events are pinned to deterministic weeks (turns) so retrieval must reason
    over temporal ordering.
    """
    # Pick an event pool weighted by occupation
    occ_lower = occupation.lower()
    if any(kw in occ_lower for kw in ("engineer", "developer", "researcher", "scientist")):
        pool: list[dict[str, Any]] = _TECH_EVENTS + _GENERAL_EVENTS
    elif any(kw in occ_lower for kw in ("student", "phd", "professor", "academic")):
        pool = _ACADEMIC_EVENTS + _GENERAL_EVENTS
    else:
        pool = _GENERAL_EVENTS + _TECH_EVENTS[:8]

    # Target 1 event every ~35–50 turns for long conversations
    num_events = max(5, total_turns // 40)
    num_events = min(num_events, len(pool))
    selected_templates = rng.sample(pool, num_events)

    # Distribute events evenly with slight jitter
    events: list[TemporalEvent] = []
    spacing = max(1, total_turns // (num_events + 1))
    for idx, template in enumerate(selected_templates):
        base_turn = spacing * (idx + 1)
        jitter = rng.randint(-max(1, spacing // 4), max(1, spacing // 4))
        turn = max(1, min(total_turns - 1, base_turn + jitter))
        week = turn // 35 + 1  # roughly 35 turns per simulated week

        description = _fill_template(template["description"], rng, profile_hints)
        events.append(
            TemporalEvent(
                description=description,
                event_type=str(template["type"]),
                week=week,
                turn=turn,
                emotional_impact=str(template["emotion"]),
            )
        )

    events.sort(key=lambda e: e.turn)
    return events


def build_emotion_trajectory(
    rng: random.Random,
    total_turns: int,
    temporal_events: list[TemporalEvent],
    baseline_emotion: str,
) -> list[tuple[int, str]]:
    """Build an emotion trajectory anchored to temporal events.

    Each significant event shifts the dominant emotion. Between events the
    emotion gradually drifts back toward the baseline.
    """
    trajectory: list[tuple[int, str]] = [(0, baseline_emotion)]

    for event in temporal_events:
        if event.emotional_impact and event.emotional_impact != "neutral":
            trajectory.append((event.turn, event.emotional_impact))

        # Recovery step: 15–25 turns after the event, drift back toward baseline
        if event.emotional_impact in {"fear", "sadness", "anger"}:
            recovery_turn = event.turn + rng.randint(15, 25)
            if recovery_turn < total_turns:
                recovery = rng.choice(["trust", "anticipation", "joy"])
                trajectory.append((recovery_turn, recovery))

    trajectory.sort(key=lambda item: item[0])
    return trajectory
