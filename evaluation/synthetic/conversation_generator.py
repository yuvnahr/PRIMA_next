"""Conversation generator for synthetic users — Deliverables 2–6."""

from __future__ import annotations

import random
from typing import Any

from evaluation.synthetic.synthetic_user import ConversationTurn
from evaluation.synthetic.user_profile import (
    TemporalEvent,
    UserProfile,
)

# ---------------------------------------------------------------------------
# Turn-type weights: as turns increase, difficulty goes up (more contradictions,
# corrections, complex references).
# ---------------------------------------------------------------------------

_EARLY_WEIGHTS = {
    "factual_statement": 0.22,
    "preference_statement": 0.15,
    "emotional_event": 0.10,
    "project_discussion": 0.12,
    "relationship_mention": 0.10,
    "goal_statement": 0.10,
    "repeated_reference": 0.08,
    "temporal_event_mention": 0.08,
    "correction": 0.03,
    "contradiction": 0.02,
    "interruption": 0.02,
    "topic_shift": 0.02,
    "follow_up_reference": 0.01,
    "indirect_reference": 0.01,
    "resumed_topic": 0.00,
}

_LATE_WEIGHTS = {
    "factual_statement": 0.14,
    "preference_statement": 0.10,
    "emotional_event": 0.10,
    "project_discussion": 0.14,
    "relationship_mention": 0.10,
    "goal_statement": 0.08,
    "repeated_reference": 0.12,
    "temporal_event_mention": 0.10,
    "correction": 0.07,
    "contradiction": 0.05,
    "interruption": 0.05,
    "topic_shift": 0.06,
    "follow_up_reference": 0.08,
    "indirect_reference": 0.06,
    "resumed_topic": 0.05,
}


def _lerp_weights(turn: int, total: int) -> dict[str, float]:
    """Linearly interpolate turn-type weights between early and late phases."""
    t = min(1.0, turn / max(1, total))
    result: dict[str, float] = {}
    for key in _EARLY_WEIGHTS:
        result[key] = _EARLY_WEIGHTS[key] * (1 - t) + _LATE_WEIGHTS[key] * t
    return result


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    values = [weights[k] for k in keys]
    return rng.choices(keys, weights=values, k=1)[0]


# ---------------------------------------------------------------------------
# Utterance templates per turn type
# ---------------------------------------------------------------------------

_FACTUAL_TEMPLATES = [
    "My name is {name} and I work at {workplace}.",
    "I am a {occupation} based in {location}.",
    "I graduated from university with a degree in {topic}.",
    "I have been working at {workplace} for {years} years.",
    "My current role is {occupation} at {workplace}.",
    "I live in {location} and commute to {workplace}.",
    "I specialize in {topic} as part of my work.",
    "My team at {workplace} focuses on {topic}.",
    "I started my career in {topic} about {years} years ago.",
    "My main expertise is in {topic}.",
]

_PREFERENCE_TEMPLATES = [
    "I prefer using {tool} for all my development work.",
    "My favorite editor these days is {tool}.",
    "I switched to {tool} because it is faster for my workflow.",
    "I like {hobby} more than anything else in my free time.",
    "My go-to programming language is {lang}.",
    "I always use {tool} when working on {topic} projects.",
    "I prefer {style} architecture over {old_style}.",
    "Lately I have been enjoying {hobby} a lot.",
    "I like working with {tool} because of its extensibility.",
    "My preferred way to manage notes is with {tool}.",
]

_EMOTIONAL_TEMPLATES = [
    "I am feeling really excited about {topic} this week.",
    "Honestly I am quite stressed about the upcoming deadline for {project}.",
    "I feel confident about my progress on {project}.",
    "I have been feeling burned out lately and need a break.",
    "I am finally starting to recover after a rough period.",
    "I feel anxious about the upcoming presentation on {topic}.",
    "I am thrilled about the positive feedback I received on {project}.",
    "I am disappointed that {topic} did not work out as expected.",
    "I feel very optimistic about where {project} is heading.",
    "I am overwhelmed by the amount of work on my plate.",
]

_PROJECT_TEMPLATES = [
    "I am currently working on {project} which involves {topic}.",
    "The {project} project is progressing well — we hit a key milestone.",
    "I need to finish the {topic} component of {project} by next week.",
    "{project} requires deep knowledge of {topic}, which I am developing.",
    "I collaborated with {person} on the {project} implementation.",
    "The {project} project went live and the results are impressive.",
    "We restructured the {project} codebase to use a cleaner {topic} approach.",
    "I am writing documentation for {project} related to {topic}.",
    "The hardest part of {project} has been the {topic} integration.",
    "I presented the {project} results at the team meeting.",
]

_RELATIONSHIP_TEMPLATES = [
    "I met {person} today and we had a great conversation about {topic}.",
    "I have been collaborating with {person} on {project}.",
    "I stopped working with {person} after the project ended.",
    "{person} gave me useful feedback on my {topic} approach.",
    "I started working more closely with {person} this week.",
    "My mentor {person} helped me think through the {topic} challenge.",
    "{person} and I are co-authoring a paper on {topic}.",
    "I ran into {person} and we caught up about our respective projects.",
    "{person} is no longer on the team.",
    "I am now reporting to {person} on the {project} effort.",
]

_GOAL_TEMPLATES = [
    "My long-term goal is to {goal}.",
    "I want to {goal} within the next few years.",
    "One of my main ambitions is to {goal}.",
    "I am actively working toward {goal}.",
    "I hope to {goal} before I turn 35.",
    "My career goal remains to {goal}.",
    "I keep reminding myself that {goal} is what I am working toward.",
    "Eventually I want to {goal}.",
    "The reason I am focused on {topic} is because it helps me {goal}.",
    "Progress toward {goal} has been steady but slow.",
]

_REPEATED_TEMPLATES = [
    "As I mentioned earlier, I work at {workplace}.",
    "I still prefer {tool} for my day-to-day work.",
    "My focus on {topic} continues to be my main priority.",
    "{project} is still my biggest ongoing project.",
    "I remain committed to {goal}.",
    "My relationship with {person} is still going strong.",
    "I keep coming back to {topic} in my research.",
    "The core challenge in {project} is still {topic}.",
    "I continue to work in {location} and love the environment.",
    "I still use {tool} every single day.",
]

_TEMPORAL_TEMPLATES = [
    "This week I {event_description}",
    "Recently I {event_description}",
    "A few weeks ago I {event_description}",
    "Looking back, {event_description}",
    "A significant event just happened: {event_description}",
]

_CORRECTION_TEMPLATES = [
    "Actually I need to correct something — I use {new_val}, not {old_val}.",
    "I should clarify: my current role is {new_val}, not {old_val}.",
    "Let me update what I said earlier: I am now using {new_val} instead of {old_val}.",
    "I realized I misspoke — {new_val} is what I meant, not {old_val}.",
    "To be precise: {new_val} is my correct {category}.",
]

_CONTRADICTION_TEMPLATES = [
    "Actually I think I prefer {contradicting_val} now.",
    "Wait — I changed my mind about {category}; it is {contradicting_val}.",
    "I was wrong earlier — {contradicting_val} is better for {topic}.",
    "My opinion on {category} has shifted — I now favour {contradicting_val}.",
]


_NATURAL_THREAD_TEMPLATES = {
    "interruption": [
        "Before I forget, the thing with {person} changed how I am thinking about {project}.",
        "Quick detour: that deadline made me rethink whether {tool} is still the right setup.",
        "Hold on, there is another detail from this week that matters for {topic}.",
    ],
    "topic_shift": [
        "Different topic for a second: {family_member} has been asking how {project} is going.",
        "Switching gears, I booked travel to {travel_place} after the {topic} review wraps up.",
        "On a personal note, {habit} has been helping more than I expected.",
    ],
    "follow_up_reference": [
        "That earlier issue with {project} is still unresolved, but {person}'s suggestion helped.",
        "Following up on the tool switch, I am faster in {tool} when the work involves {topic}.",
        "The concern I mentioned before is less sharp now, mostly because {habit}.",
    ],
    "indirect_reference": [
        "It is the same recurring problem from the paper work, just showing up in a new place.",
        "That collaborator I mentioned earlier is now central to the {topic} plan.",
        "The old workflow is becoming a bottleneck again, especially around {project}.",
    ],
    "resumed_topic": [
        "Coming back to {project}, the next milestone depends on {topic}.",
        "I want to return to what I said about {person}; the relationship matters more now.",
        "Picking up the thread from last time, {tool} is still part of the decision.",
    ],
}


# ---------------------------------------------------------------------------
# Fill helpers
# ---------------------------------------------------------------------------

def _pick(rng: random.Random, lst: list[Any]) -> Any:
    return rng.choice(lst) if lst else ""


def _fill(template: str, rng: random.Random, ctx: dict[str, Any]) -> str:
    try:
        return template.format(**ctx)
    except (KeyError, IndexError):
        return template


def _build_context(profile: UserProfile, rng: random.Random, turn: int) -> dict[str, Any]:
    """Build a template-filling context from the user profile."""
    active_prefs = profile.active_preferences(turn)
    active_rels = profile.active_relationships(turn)
    projects = [p for p in profile.projects if p.status == "active"]
    ctx: dict[str, Any] = {
        "name": profile.name,
        "workplace": profile.workplace,
        "occupation": profile.occupation,
        "location": profile.location,
        "topic": _pick(rng, profile.recurring_topics) or "machine learning",
        "project": _pick(rng, [p.name for p in projects]) or "the main project",
        "person": _pick(rng, [r.person_name for r in active_rels]) or "a colleague",
        "tool": _pick(rng, [p.value for p in active_prefs if p.category == "editor"]) or "VSCode",
        "lang": _pick(rng, [p.value for p in active_prefs if p.category == "language"]) or "Python",
        "style": _pick(rng, [p.value for p in active_prefs if p.category == "architecture"]) or "microservices",
        "old_style": "monolith",
        "hobby": _pick(rng, profile.hobbies) or "reading",
        "goal": _pick(rng, profile.long_term_goals) or "lead a research team",
        "years": rng.randint(1, 10),
        "event_description": "",
        "old_val": "",
        "new_val": "",
        "category": "",
        "contradicting_val": "",
        "old_pref": "",
        "new_pref": "",
        "family_member": _pick(rng, profile.family) or "my family",
        "travel_place": _pick(rng, profile.travel_history) or profile.location,
        "habit": _pick(rng, profile.habits) or "keeping a weekly note",
    }
    return ctx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_conversation(  # noqa: PLR0912, PLR0915
    profile: UserProfile,
    rng: random.Random,
    total_turns: int,
    temporal_events: list[TemporalEvent],
) -> tuple[list[ConversationTurn], dict[int, dict[str, str]], dict[int, dict[str, str]], dict[int, dict[str, str]], dict[int, str]]:
    """Generate a full conversation for one synthetic user.

    Returns
    -------
    turns:
        All conversation turns.
    identity_probes:
        Mapping turn → {question: expected_answer} for identity consistency.
    preference_probes:
        Mapping turn → {category: expected_value}.
    fact_probes:
        Mapping turn → {fact_key: fact_value}.
    emotion_probes:
        Mapping turn → expected_dominant_emotion.
    """
    turns: list[ConversationTurn] = []
    identity_probes: dict[int, dict[str, str]] = {}
    preference_probes: dict[int, dict[str, str]] = {}
    fact_probes: dict[int, dict[str, str]] = {}
    emotion_probes: dict[int, str] = {}

    # Build an event lookup by turn
    event_by_turn: dict[int, TemporalEvent] = {e.turn: e for e in temporal_events}

    # Track preference evolution for probing correctness
    pref_state: dict[str, str] = {
        p.category: p.value for p in profile.preferences if p.turn_set == 0
    }

    for t in range(total_turns):
        weights = _lerp_weights(t, total_turns)
        ctx = _build_context(profile, rng, t)

        # If a temporal event fires at this turn, inject it
        if t in event_by_turn:
            event = event_by_turn[t]
            ctx["event_description"] = event.description.lower()
            turn_type = "temporal_event_mention"
            template = _pick(rng, _TEMPORAL_TEMPLATES)
            utterance = _fill(template, rng, ctx)
            expected_emotion = event.emotional_impact
        else:
            turn_type = _weighted_choice(rng, weights)

            if turn_type == "factual_statement":
                template = _pick(rng, _FACTUAL_TEMPLATES)
                utterance = _fill(template, rng, ctx)
                expected_emotion = "neutral"

            elif turn_type == "preference_statement":
                # Possibly evolve a preference
                evolve_prefs = [p for p in profile.preferences if p.turn_set <= t and p.turn_set > 0]
                if evolve_prefs and rng.random() < 0.4:
                    pref = _pick(rng, evolve_prefs)
                    pref_state[pref.category] = pref.value
                    ctx["tool"] = pref.value
                template = _pick(rng, _PREFERENCE_TEMPLATES)
                utterance = _fill(template, rng, ctx)
                expected_emotion = "anticipation"

            elif turn_type == "emotional_event":
                template = _pick(rng, _EMOTIONAL_TEMPLATES)
                utterance = _fill(template, rng, ctx)
                expected_emotion = profile.emotional_profile.emotion_at_turn(t)

            elif turn_type == "project_discussion":
                template = _pick(rng, _PROJECT_TEMPLATES)
                utterance = _fill(template, rng, ctx)
                expected_emotion = "anticipation"

            elif turn_type == "relationship_mention":
                # Possibly evolve relationships
                new_rels = [r for r in profile.relationships if r.met_at_turn <= t and r.met_at_turn > 0]
                if new_rels:
                    ctx["person"] = _pick(rng, [r.person_name for r in new_rels])
                template = _pick(rng, _RELATIONSHIP_TEMPLATES)
                utterance = _fill(template, rng, ctx)
                expected_emotion = "trust"

            elif turn_type == "goal_statement":
                template = _pick(rng, _GOAL_TEMPLATES)
                utterance = _fill(template, rng, ctx)
                expected_emotion = "anticipation"

            elif turn_type == "repeated_reference":
                template = _pick(rng, _REPEATED_TEMPLATES)
                utterance = _fill(template, rng, ctx)
                expected_emotion = "trust"

            elif turn_type == "correction":
                cats = list(pref_state.keys())
                if cats:
                    cat = _pick(rng, cats)
                    old_val = pref_state[cat]
                    new_options = [p.value for p in profile.preferences if p.category == cat and p.value != old_val]
                    new_val = _pick(rng, new_options) if new_options else f"{old_val}_v2"
                    pref_state[cat] = new_val
                    ctx["old_val"] = old_val
                    ctx["new_val"] = new_val
                    ctx["category"] = cat
                    template = _pick(rng, _CORRECTION_TEMPLATES)
                    utterance = _fill(template, rng, ctx)
                else:
                    utterance = "Actually, let me clarify what I said before."
                expected_emotion = "neutral"

            elif turn_type == "contradiction":
                cats = list(pref_state.keys())
                if cats:
                    cat = _pick(rng, cats)
                    old_val = pref_state[cat]
                    contradicting_val = f"alternative_{old_val[:4]}"
                    ctx["category"] = cat
                    ctx["contradicting_val"] = contradicting_val
                    template = _pick(rng, _CONTRADICTION_TEMPLATES)
                    utterance = _fill(template, rng, ctx)
                else:
                    utterance = "I'm not sure what I prefer anymore."
                expected_emotion = "surprise"

            elif turn_type in _NATURAL_THREAD_TEMPLATES:
                template = _pick(rng, _NATURAL_THREAD_TEMPLATES[turn_type])
                utterance = _fill(template, rng, ctx)
                expected_emotion = profile.emotional_profile.emotion_at_turn(t)

            else:
                utterance = f"I wanted to share an update about {ctx['topic']}."
                expected_emotion = "neutral"

        # Determine ground-truth values for evaluation probes
        active_prefs = profile.active_preferences(t)
        editor_prefs = [p for p in active_prefs if p.category == "editor"]
        gt_pref = editor_prefs[-1].value if editor_prefs else ""

        active_rels = profile.active_relationships(t)
        gt_rel = ", ".join(r.person_name for r in active_rels[:2]) if active_rels else ""

        gt_fact_key = "occupation"
        gt_fact_val = profile.occupation

        turns.append(
            ConversationTurn(
                turn_index=t,
                utterance=utterance,
                turn_type=turn_type,
                expected_memory_content=utterance[:80],
                expected_emotion=expected_emotion,
                ground_truth_preference=gt_pref,
                ground_truth_relationship=gt_rel,
                ground_truth_fact_key=gt_fact_key,
                ground_truth_fact_value=gt_fact_val,
                temporal_week=t // 35 + 1,
            )
        )

        # Register evaluation probes at key milestones
        if t in {10, 50, 100, 250, 500, 750, 1000} and t < total_turns:
            identity_probes[t] = {
                "name": profile.name,
                "occupation": profile.occupation,
                "workplace": profile.workplace,
                "location": profile.location,
                "goal": profile.long_term_goals[0] if profile.long_term_goals else "",
            }
            preference_probes[t] = dict(pref_state)
            fact_probes[t] = {
                "occupation": profile.occupation,
                "workplace": profile.workplace,
                "name": profile.name,
            }
            emotion_probes[t] = profile.emotional_profile.emotion_at_turn(t)

    return turns, identity_probes, preference_probes, fact_probes, emotion_probes
