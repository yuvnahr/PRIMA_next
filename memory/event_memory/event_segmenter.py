"""Conversation-to-event segmentation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from memory.memory_note import tokenize

STOPWORDS = {
    "about", "after", "also", "and", "are", "but", "did", "for", "from", "had", "has",
    "have", "her", "him", "his", "how", "into", "our", "she", "that", "the", "their",
    "then", "they", "this", "was", "were", "what", "when", "where", "who", "with", "you",
}

BRIDGE_TERMS = {
    "also", "then", "after", "before", "because", "there", "that", "it", "they", "he",
    "she", "we", "same", "again", "later", "while",
}


@dataclass(frozen=True, slots=True)
class EventSegment:
    conversation_id: str
    index: int
    turns: tuple[Any, ...]

    @property
    def turn_ids(self) -> tuple[str, ...]:
        return tuple(str(getattr(turn, "turn_id", "") or position + 1) for position, turn in enumerate(self.turns))


class EventSegmenter:
    """Group adjacent turns that appear to describe one semantic event."""

    def __init__(self, min_overlap: float = 0.18, max_turns: int = 6) -> None:
        self.min_overlap = max(0.0, min_overlap)
        self.max_turns = max(1, max_turns)

    def segment(self, conversation: Any) -> list[EventSegment]:
        segments: list[EventSegment] = []
        current: list[Any] = []
        current_terms: set[str] = set()
        conversation_id = str(getattr(conversation, "id", "conversation"))

        for turn in getattr(conversation, "turns", ()):
            terms = _content_terms(str(getattr(turn, "text", "")))
            if current and not self._belongs(current_terms, terms, turn, current[-1]):
                segments.append(EventSegment(conversation_id, len(segments) + 1, tuple(current)))
                current = []
                current_terms = set()
            current.append(turn)
            current_terms.update(terms)
            if len(current) >= self.max_turns:
                segments.append(EventSegment(conversation_id, len(segments) + 1, tuple(current)))
                current = []
                current_terms = set()
        if current:
            segments.append(EventSegment(conversation_id, len(segments) + 1, tuple(current)))
        return segments

    def _belongs(self, current_terms: set[str], next_terms: set[str], turn: Any, previous_turn: Any) -> bool:
        if not current_terms or not next_terms:
            return False
        overlap = len(current_terms & next_terms) / max(1, min(len(current_terms), len(next_terms)))
        same_session = getattr(turn, "session_id", None) == getattr(previous_turn, "session_id", None)
        has_bridge = bool(next_terms & BRIDGE_TERMS)
        shared_entity = bool(_capitalized_terms(str(getattr(turn, "text", ""))) & _capitalized_terms(str(getattr(previous_turn, "text", ""))))
        return same_session and (overlap >= self.min_overlap or has_bridge or shared_entity)


def _content_terms(text: str) -> set[str]:
    return {token for token in tokenize(text) if len(token) > 2 and token not in STOPWORDS}


def _capitalized_terms(text: str) -> set[str]:
    return {token.strip(".,!?;:") for token in text.split() if token[:1].isupper()}
