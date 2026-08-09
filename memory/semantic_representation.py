"""Deterministic semantic memory representation for embedding ablations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from memory.identity_normalization import IdentityNormalizer
from memory.memory_note import extract_dominant_context_chain, tokenize

RELATION_TERMS = {
    "aunt", "brother", "coworker", "daughter", "father", "friend", "husband", "manager",
    "mentor", "mother", "neighbor", "partner", "sister", "son", "teammate", "uncle", "wife",
}
PREFERENCE_TERMS = {"favorite", "likes", "liked", "love", "loves", "prefers", "enjoys", "hates"}
EMOTION_TERMS = {"happy", "sad", "angry", "afraid", "excited", "worried", "proud", "upset"}
ACTION_TERMS = {
    "bought", "called", "cooked", "discussed", "gave", "gifted", "helped", "liked",
    "met", "moved", "planned", "played", "recommended", "shared", "studied", "visited",
}
TEMPORAL_TERMS = {
    "after", "before", "during", "earlier", "first", "last", "later", "month", "today",
    "tomorrow", "tonight", "week", "weekend", "when", "while", "year", "yesterday",
}
LOCATION_PREPOSITIONS = {"at", "in", "near", "to"}


@dataclass(frozen=True, slots=True)
class SemanticMemoryRepresentation:
    summary: str
    participants: tuple[str, ...]
    objects: tuple[str, ...]
    locations: tuple[str, ...]
    temporal_expressions: tuple[str, ...]
    actions: tuple[str, ...]
    preferences: tuple[str, ...]
    identity_references: tuple[str, ...]
    relationships: tuple[str, ...]
    emotion: str | None
    original_text: str

    def serialize(self) -> str:
        fields = [
            ("Summary", (self.summary,)),
            ("Participants", self.participants),
            ("Objects", self.objects),
            ("Locations", self.locations),
            ("Temporal Expressions", self.temporal_expressions),
            ("Actions", self.actions),
            ("Preferences", self.preferences),
            ("Identity References", self.identity_references),
            ("Relationships", self.relationships),
            ("Emotion", (self.emotion,) if self.emotion else ()),
            ("Original", (self.original_text,)),
        ]
        lines: list[str] = []
        for label, values in fields:
            clean_values = [str(value).strip() for value in values if str(value).strip()]
            if clean_values:
                lines.append(f"{label}:")
                lines.extend(clean_values)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary,
            "participants": list(self.participants),
            "objects": list(self.objects),
            "locations": list(self.locations),
            "temporal_expressions": list(self.temporal_expressions),
            "actions": list(self.actions),
            "preferences": list(self.preferences),
            "identity_references": list(self.identity_references),
            "relationships": list(self.relationships),
            "emotion": self.emotion,
            "original_text": self.original_text,
            "serialized": self.serialize(),
        }


def build_semantic_representation(
    text: str,
    speaker: str | None = None,
    affective_state: dict[str, Any] | None = None,
    identity_normalizer: IdentityNormalizer | None = None,
    normalize_identities: bool = False,
) -> SemanticMemoryRepresentation:
    original_text = str(text)
    working_text = original_text
    identity_references: list[str] = []
    if normalize_identities:
        normalizer = identity_normalizer or IdentityNormalizer()
        identity_result = normalizer.observe(original_text, speaker=speaker)
        working_text = identity_result.text
        identity_references = [f"{alias} -> {canonical}" for alias, canonical in identity_result.replacements.items() if alias != canonical]
    participants = _unique([item for item in [speaker, *_capitalized_terms(working_text)] if item])
    temporal = _unique(_temporal_terms(working_text))
    actions = _unique(_matching_terms(working_text, ACTION_TERMS))
    preferences = _unique(_matching_terms(working_text, PREFERENCE_TERMS))
    relationships = _unique(_relationships(working_text, participants))
    locations = _unique(_locations(working_text))
    objects = _unique(_objects(working_text, participants, temporal, actions, preferences, relationships, locations))
    emotion = _emotion(working_text, affective_state)
    return SemanticMemoryRepresentation(
        summary=_summary(working_text, participants, actions, objects),
        participants=tuple(participants),
        objects=tuple(objects),
        locations=tuple(locations),
        temporal_expressions=tuple(temporal),
        actions=tuple(actions),
        preferences=tuple(preferences),
        identity_references=tuple(identity_references),
        relationships=tuple(relationships),
        emotion=emotion,
        original_text=original_text,
    )


def semantic_representation_analysis_row(memory_id: str, original_text: str, representation: SemanticMemoryRepresentation) -> dict[str, Any]:
    original_tokens = len(tokenize(original_text))
    structured_tokens = len(tokenize(representation.serialize()))
    return {
        "memory_id": memory_id,
        "original_token_count": original_tokens,
        "structured_token_count": structured_tokens,
        "entities_extracted": len(set(representation.participants) | set(representation.identity_references)),
        "temporal_expressions_extracted": len(representation.temporal_expressions),
        "relationships_extracted": len(representation.relationships),
        "objects_extracted": len(representation.objects),
        "participants_extracted": len(representation.participants),
        "compression_ratio": round(structured_tokens / max(1, original_tokens), 6),
        "serialization_consistency": representation.serialize() == representation.serialize(),
    }


def _summary(text: str, participants: list[str], actions: list[str], objects: list[str]) -> str:
    subject = ", ".join(participants[:3]) if participants else "Participants"
    action = ", ".join(actions[:3]) if actions else "discussed"
    focus = ", ".join(objects[:5]) if objects else "the conversation"
    return f"{subject} {action} {focus}."


def _capitalized_terms(text: str) -> list[str]:
    return [token.strip(".,!?;:") for token in text.split() if token[:1].isupper()]


def _matching_terms(text: str, terms: set[str]) -> list[str]:
    return sorted(set(tokenize(text)) & terms)


def _temporal_terms(text: str) -> list[str]:
    tokens = set(tokenize(text))
    years = set(re.findall(r"\b\d{4}\b", text))
    return sorted((tokens & TEMPORAL_TERMS) | years)


def _relationships(text: str, participants: list[str]) -> list[str]:
    relationships = _matching_terms(text, RELATION_TERMS)
    if len(participants) >= 2:
        relationships.extend(f"{participants[0]} -> {participants[1]}" for _ in relationships[:1])
    return relationships


def _objects(text: str, *excluded_groups: list[str]) -> list[str]:
    excluded = {item.lower() for group in excluded_groups for item in group}
    return [keyword for keyword in extract_dominant_context_chain(text, top_k=12) if keyword.lower() not in excluded]


def _locations(text: str) -> list[str]:
    tokens = text.split()
    locations: list[str] = []
    for index, token in enumerate(tokens[:-1]):
        if token.lower().strip(".,!?;:") in LOCATION_PREPOSITIONS and tokens[index + 1][:1].isupper():
            locations.append(tokens[index + 1].strip(".,!?;:"))
    return locations


def _emotion(text: str, affective_state: dict[str, Any] | None) -> str | None:
    if affective_state:
        top = affective_state.get("top_emotions")
        if isinstance(top, list) and top:
            label = top[0].get("label") if isinstance(top[0], dict) else None
            if label:
                return str(label)
    matches = _matching_terms(text, EMOTION_TERMS)
    return matches[0] if matches else None


def _unique(values: list[str]) -> list[str]:
    return list(dict.fromkeys(str(value).strip() for value in values if str(value).strip()))
