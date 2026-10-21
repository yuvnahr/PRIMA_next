"""Deterministic semantic abstraction engine."""

from __future__ import annotations

import re
from collections import Counter

from memory.memory_note import MemoryNote, extract_dominant_context_chain


def factual_evidence_eligible(note: MemoryNote) -> bool:
    if note.context.get("evidence_eligible") is False:
        return False
    return not (note.context.get("source") == "memory_evolution"
                and note.context.get("type") == "semantic_abstraction"
                and re.fullmatch(r"I have \d+ related memories about [a-z0-9', -]+\.", note.content))


class SemanticAbstractionEngine:
    def summarize_cluster(self, notes: list[MemoryNote], max_words: int = 40) -> str:
        """Create a provenance-preserving summary without LLM dependency."""
        keywords: Counter[str] = Counter()
        for note in notes:
            keywords.update(note.keywords)
        top_terms = [term for term, _ in keywords.most_common(5)]
        source_count = len(notes)
        anchor = ", ".join(top_terms) if top_terms else "related experiences"
        summary = f"I have {source_count} related memories about {anchor}."
        words = summary.split()
        return " ".join(words[:max_words])

    def extract_concepts(self, note: MemoryNote) -> list[str]:
        return extract_dominant_context_chain(note.content)
