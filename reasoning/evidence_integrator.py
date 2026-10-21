"""Bounded evidence integration with source provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass

from memory.evolution.semantic_abstraction_engine import factual_evidence_eligible
from memory.retrieval.retrieval_controller import RetrievalResponse
from reasoning.models import EvidenceItem, EvidenceState


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass(frozen=True, slots=True)
class IntegrationResult:
    added_count: int
    duplicate_count: int

    excluded_count: int = 0

class EvidenceIntegrator:
    def integrate(self, state: EvidenceState, response: RetrievalResponse, *, hop: int, query: str) -> IntegrationResult:
        seen_sources = {item.source_id for item in state.evidence_items}
        seen_text = {_normalise(item.text) for item in state.evidence_items}
        added = duplicates = excluded = 0
        for result in response.results:
            source_id = result.note.id
            if not result.note.content.strip() or not factual_evidence_eligible(result.note):
                excluded += 1
                continue
            normalised = _normalise(result.note.content)
            if source_id in seen_sources or normalised in seen_text or len(state.evidence_items) >= state.request.budget.max_documents:
                duplicates += 1
                continue
            source_time = result.note.context.get("source_timestamp")
            text = result.note.content
            if source_time and str(source_time) not in text:
                text = f"Source timestamp: {source_time}\n{text}"
            state.evidence_items.append(EvidenceItem(
                evidence_id=f"{source_id}:{hop}", text=text, source_id=source_id,
                source_type=result.note.memory_type.value, retrieval_score=float(result.score), hop=hop, query=query,
                provenance={"strategy_scores": dict(result.strategy_scores), **dict(result.note.context)}, result=result,
            ))
            seen_sources.add(source_id)
            seen_text.add(normalised)
            added += 1
        state.confidence = max(state.confidence, float(response.confidence.confidence))
        state.no_progress_hops = 0 if added else state.no_progress_hops + 1
        return IntegrationResult(added, duplicates, excluded)
