"""Sparse lexical retrieval strategy."""

from __future__ import annotations

from memory.memory_note import calculate_smart_overlap, extract_dominant_context_chain, tokenize
from memory.memory_repository import MemoryRepository
from memory.retrieval.retrieval_request import RetrievalRequest
from memory.retrieval.retrieval_result import RetrievalResult
from memory.retrieval.retrieval_strategy import RetrievalStrategy


class SparseRetrievalStrategy(RetrievalStrategy):
    name = "sparse"

    def retrieve(self, request: RetrievalRequest, repository: MemoryRepository) -> list[RetrievalResult]:
        query_terms = self._query_terms(request)
        required_terms = self._required_terms(request)
        entities = self._entities(request)
        results: list[RetrievalResult] = []
        for memory_type in request.memory_types:
            for note in repository.list(memory_type):
                memory_terms = set(note.keywords) | set(tokenize(note.content))
                memory_text = note.content.lower()
                overlap = calculate_smart_overlap(query_terms, tuple(memory_terms))
                exact_overlap = len(set(query_terms) & memory_terms)
                required_overlap = _term_match(required_terms, memory_terms, memory_text)
                entity_overlap = _term_match(entities, memory_terms, memory_text)
                if overlap <= 0 and exact_overlap <= 0 and required_overlap <= 0:
                    continue
                lexical_score = min(1.0, (overlap + exact_overlap * 0.5) / max(1, len(set(query_terms))))
                score = min(1.0, lexical_score * 0.40 + required_overlap * 0.50 + entity_overlap * 0.10)
                results.append(
                    RetrievalResult(
                        note=note,
                        score=score,
                        strategy_scores={self.name: score},
                        explanation={
                            "query_terms": query_terms,
                            "required_terms": sorted(required_terms),
                            "overlap": overlap,
                            "exact_overlap": exact_overlap,
                            "required_overlap": round(required_overlap, 6),
                            "entity_overlap": round(entity_overlap, 6),
                        },
                    )
                )
        return sorted(results, key=lambda item: item.score, reverse=True)[: request.top_k]

    def _query_terms(self, request: RetrievalRequest) -> list[str]:
        analysis = request.analyzed_query
        expanded = request.expanded_query
        terms: list[str] = []
        terms.extend(extract_dominant_context_chain(request.query, top_k=12))
        if analysis is not None:
            terms.extend(analysis.keywords)
            terms.extend(analysis.entities)
            terms.extend(analysis.relations)
            terms.extend(analysis.preference_terms)
            terms.extend(analysis.identity_attributes)
            terms.extend(analysis.events)
            terms.extend(analysis.intents)
            terms.extend(constraint.operator for constraint in analysis.temporal_constraints)
            terms.extend(constraint.anchor for constraint in analysis.temporal_constraints if constraint.anchor)
        if expanded is not None:
            terms.extend(expanded.terms)
        terms.extend(tokenize(request.lexical_query()))
        return [term.lower() for term in dict.fromkeys(str(term).strip() for term in terms if str(term).strip())]

    def _required_terms(self, request: RetrievalRequest) -> set[str]:
        analysis = request.analyzed_query
        if analysis is None:
            return set()
        terms = set(analysis.relations) | set(analysis.preference_terms) | set(analysis.events)
        if "relationship" in analysis.intents:
            terms.update({"friend", "coworker", "colleague", "teammate", "neighbor", "family", "spouse", "mentor", "student", "manager"})
        elif "identity" in analysis.intents:
            terms.update(analysis.identity_attributes)
        if "travel" in analysis.intents:
            terms.update({"travel", "traveled", "visited", "trip", "vacation", "destination"})
        if "gift" in analysis.intents:
            terms.update({"gift", "gave", "received", "present", "borrowed", "owned"})
        return {term.lower() for term in terms if term}

    def _entities(self, request: RetrievalRequest) -> set[str]:
        analysis = request.analyzed_query
        if analysis is None:
            return set()
        entities = set(analysis.entities)
        for entity, aliases in analysis.aliases.items():
            if entity in entities:
                entities.update(aliases)
        return {entity.lower() for entity in entities if entity}


def _term_match(terms: set[str], memory_terms: set[str], memory_text: str) -> float:
    if not terms:
        return 0.5
    matches = sum(1 for term in terms if term in memory_terms or term in memory_text)
    return matches / max(1, len(terms))

