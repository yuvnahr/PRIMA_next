"""Bounded evidence integration with source provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass

from memory.retrieval.retrieval_controller import RetrievalResponse
from reasoning.models import EvidenceItem, EvidenceState


def _normalise(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


@dataclass(frozen=True, slots=True)
class IntegrationResult:
    added_count: int
    duplicate_count: int


class EvidenceIntegrator:
    def integrate(self, state: EvidenceState, response: RetrievalResponse, *, hop: int, query: str) -> IntegrationResult:
        seen_sources = {item.source_id for item in state.evidence_items}
        seen_text = {_normalise(item.text) for item in state.evidence_items}
        added = duplicates = 0
        for result in response.results:
            source_id = result.note.id
            normalised = _normalise(result.note.content)
            if source_id in seen_sources or normalised in seen_text or len(state.evidence_items) >= state.request.budget.max_documents:
                duplicates += 1
                continue
            state.evidence_items.append(EvidenceItem(
                evidence_id=f"{source_id}:{hop}", text=result.note.content, source_id=source_id,
                source_type=result.note.memory_type.value, retrieval_score=float(result.score), hop=hop, query=query,
                provenance={"strategy_scores": dict(result.strategy_scores)}, result=result,
            ))
            seen_sources.add(source_id)
            seen_text.add(normalised)
            added += 1
        state.confidence = max(state.confidence, float(response.confidence.confidence))
        state.no_progress_hops = 0 if added else state.no_progress_hops + 1
        return IntegrationResult(added, duplicates)
