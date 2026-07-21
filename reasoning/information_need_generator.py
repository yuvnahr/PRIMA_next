"""Generate one explicit generic follow-up need from retrieved evidence."""

from __future__ import annotations

import re
from uuid import uuid4

from reasoning.models import EvidenceState, InformationNeed, SufficiencyDecision


class InformationNeedGenerator:
    def generate(self, state: EvidenceState, decision: SufficiencyDecision) -> InformationNeed | None:
        attempted = {" ".join(query.lower().split()) for query in state.attempted_queries}
        question = state.request.question.lower()
        for item in reversed(state.evidence_items):
            candidates = re.findall(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*)*\b", item.text)
            for candidate in candidates:
                query = candidate.strip()
                if query.lower() not in question and " ".join(query.lower().split()) not in attempted:
                    return InformationNeed(
                        need_id=f"need_{uuid4()}",
                        description=f"Retrieve evidence about {query} required by the current evidence.",
                        query=query,
                        relation_target=query,
                    )
        return None
