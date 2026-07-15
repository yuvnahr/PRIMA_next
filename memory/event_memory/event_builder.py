"""Build event memories from event segments."""

from __future__ import annotations

from typing import Any

from memory.event_memory.event import EventMemory
from memory.event_memory.event_metadata import EventMetadata, unique_sorted
from memory.event_memory.event_segmenter import EventSegment
from memory.memory_note import extract_dominant_context_chain, tokenize

RELATION_TERMS = {
    "aunt", "brother", "coworker", "daughter", "father", "friend", "husband", "manager",
    "mentor", "mother", "neighbor", "partner", "sister", "son", "teammate", "uncle", "wife",
}
PREFERENCE_TERMS = {"favorite", "likes", "liked", "love", "loves", "prefers", "enjoys", "hates"}
EMOTION_TERMS = {"happy", "sad", "angry", "afraid", "excited", "worried", "proud", "upset"}
TEMPORAL_TERMS = {
    "after", "before", "during", "earlier", "first", "last", "later", "month", "today",
    "tomorrow", "week", "when", "while", "year", "yesterday",
}
LOCATION_PREPOSITIONS = {"at", "in", "near", "to"}


class EventMemoryBuilder:
    def build(self, segment: EventSegment) -> EventMemory:
        turn_texts = [str(getattr(turn, "text", "")) for turn in segment.turns]
        speakers = [str(getattr(turn, "speaker", "")) for turn in segment.turns if str(getattr(turn, "speaker", "")).strip()]
        joined = " ".join(turn_texts)
        participants = unique_sorted([*speakers, *_capitalized_terms(joined)])
        temporal = unique_sorted(_temporal_terms(joined))
        relationships = unique_sorted(_matching_terms(joined, RELATION_TERMS))
        preferences = unique_sorted(_matching_terms(joined, PREFERENCE_TERMS))
        objects = unique_sorted(_objects(joined, participants, temporal, relationships, preferences))
        locations = unique_sorted(_locations(joined))
        emotion = next(iter(_matching_terms(joined, EMOTION_TERMS)), None)
        summary = _summary(segment, turn_texts, participants, objects)
        turn_positions = [_turn_position(turn, index) for index, turn in enumerate(segment.turns, start=1)]
        metadata = EventMetadata(
            event_type=_event_type(relationships, preferences, temporal),
            participants=participants,
            objects=objects,
            locations=locations,
            temporal_expressions=temporal,
            relationships=relationships,
            preferences=preferences,
            emotion=emotion,
            importance=_importance(len(segment.turns), participants, temporal, relationships, preferences),
            source_turn_range=(min(turn_positions), max(turn_positions)),
        )
        return EventMemory(
            event_id=f"{segment.conversation_id}:event:{segment.index}",
            conversation_id=segment.conversation_id,
            summary=summary,
            turns_included=segment.turn_ids,
            metadata=metadata,
            embedding_text=_embedding_text(summary, metadata, turn_texts),
        )


def _summary(segment: EventSegment, turn_texts: list[str], participants: tuple[str, ...], objects: tuple[str, ...]) -> str:
    subject = ", ".join(participants[:3]) if participants else "Participants"
    focus = ", ".join(objects[:5]) if objects else "the event"
    snippets = " ".join(turn_texts)
    return f"{subject} discussed {focus}. {snippets}".strip()


def _embedding_text(summary: str, metadata: EventMetadata, turn_texts: list[str]) -> str:
    fields = [
        summary,
        f"participants: {' '.join(metadata.participants)}",
        f"objects: {' '.join(metadata.objects)}",
        f"relationships: {' '.join(metadata.relationships)}",
        f"preferences: {' '.join(metadata.preferences)}",
        f"temporal: {' '.join(metadata.temporal_expressions)}",
        f"locations: {' '.join(metadata.locations)}",
        "turns: " + " ".join(turn_texts),
    ]
    return "\n".join(field for field in fields if field.strip())


def _event_type(relationships: tuple[str, ...], preferences: tuple[str, ...], temporal: tuple[str, ...]) -> str:
    if relationships:
        return "relationship_event"
    if preferences:
        return "preference_event"
    if temporal:
        return "temporal_event"
    return "conversation_event"


def _importance(turn_count: int, participants: tuple[str, ...], temporal: tuple[str, ...], relationships: tuple[str, ...], preferences: tuple[str, ...]) -> float:
    score = 0.10 + min(0.35, turn_count * 0.06)
    score += min(0.20, len(participants) * 0.03)
    score += 0.12 if temporal else 0.0
    score += 0.12 if relationships else 0.0
    score += 0.10 if preferences else 0.0
    return round(min(1.0, score), 6)


def _turn_position(turn: Any, fallback: int) -> int:
    raw_id = str(getattr(turn, "turn_id", "") or fallback)
    digits = "".join(character for character in raw_id if character.isdigit())
    return int(digits) if digits else fallback


def _capitalized_terms(text: str) -> list[str]:
    return [token.strip(".,!?;:") for token in text.split() if token[:1].isupper()]


def _matching_terms(text: str, terms: set[str]) -> list[str]:
    tokens = set(tokenize(text))
    return sorted(tokens & terms)


def _temporal_terms(text: str) -> list[str]:
    tokens = set(tokenize(text))
    years = [token for token in tokens if token.isdigit() and len(token) == 4]
    return sorted((tokens & TEMPORAL_TERMS) | set(years))


def _objects(text: str, participants: tuple[str, ...], *excluded_groups: tuple[str, ...]) -> list[str]:
    excluded = {item.lower() for item in participants}
    for group in excluded_groups:
        excluded.update(item.lower() for item in group)
    keywords = extract_dominant_context_chain(text, top_k=12)
    return [keyword for keyword in keywords if keyword.lower() not in excluded]


def _locations(text: str) -> list[str]:
    tokens = text.split()
    locations: list[str] = []
    for index, token in enumerate(tokens[:-1]):
        if token.lower().strip(".,!?;:") in LOCATION_PREPOSITIONS and tokens[index + 1][:1].isupper():
            locations.append(tokens[index + 1].strip(".,!?;:"))
    return locations
