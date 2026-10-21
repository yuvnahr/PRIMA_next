"""Bounded original-question expansions using source-supported bridges."""

from __future__ import annotations

import re
from uuid import uuid4

from memory.retrieval.query_analysis import PRONOUNS, STOPWORDS, QueryAnalyzer
from reasoning.models import EvidenceState, InformationNeed, SufficiencyDecision

_GENERIC = STOPWORDS | PRONOUNS | {"wow", "your", "yours", "you", "it", "its", "i", "my", "me", "we", "our",
                                  "thanks", "thank", "sounds", "sound", "yes", "no", "okay", "ok", "great", "nice"}
_BRIDGE = re.compile(r"\b(?:identifies|named|called|knows|met|linked to|works with|friend is|partner is)\s+"
                     r"([A-Za-z][\w'-]*(?:[ \t]+[A-Za-z][\w'-]*){0,5})", re.IGNORECASE)


def normalize_query(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold()))


class InformationNeedGenerator:
    def generate(self, state: EvidenceState, decision: SufficiencyDecision) -> InformationNeed | None:
        if not decision.missing_information and not decision.unresolved_entities:
            return None
        attempted = {normalize_query(query) for query in state.attempted_queries}
        original = normalize_query(state.request.question)
        for candidate in reversed(self.supported_bridges(state, decision)):
            if normalize_query(candidate) in original:
                continue
            query = f"{state.request.question.strip()} Source bridge: {candidate}"
            if normalize_query(query) not in attempted:
                return InformationNeed(f"need_{uuid4()}", "; ".join(decision.missing_information), query, candidate)
        return None

    def supported_bridges(self, state: EvidenceState, decision: SufficiencyDecision) -> list[str]:
        analysis = QueryAnalyzer().analyze(state.request.question)
        anchors = [normalize_query(entity) for entity in analysis.entities if normalize_query(entity) not in _GENERIC]
        if not anchors:
            anchors = [term for term in analysis.keywords if term not in _GENERIC]
        bridges: list[str] = []
        for item in state.evidence_items:
            if not item.provenance.get("evidence_eligible", True):
                continue
            for span in re.split(r"[.!?\n]", item.text):
                normalized = normalize_query(span)
                if not any(re.search(rf"\b{re.escape(anchor)}\b", normalized) for anchor in anchors):
                    continue
                structured = item.provenance.get("entities", ())
                if not isinstance(structured, (tuple, list)):
                    structured = ()
                candidates = [*decision.unresolved_entities, *(value for value in structured if isinstance(value, str)),
                              *(match.group(1) for match in _BRIDGE.finditer(span))]
                for value in candidates:
                    candidate = " ".join(value.strip(" ,;:\"'").split())
                    words = normalize_query(candidate).split()
                    if (not words or len(words) > 6 or len(candidate) > 80 or words[0] in _GENERIC
                            or re.sub(r"[\W_]+", "", candidate.casefold()) in _GENERIC):
                        continue
                    if not re.search(rf"\b{re.escape(normalize_query(candidate))}\b", normalized):
                        continue
                    if candidate not in bridges:
                        bridges.append(candidate)
                        anchors.append(normalize_query(candidate))
        return bridges

    def supported_query(self, state: EvidenceState, decision: SufficiencyDecision, query: str) -> str | None:
        """Reflection can select a supported bridge without discarding the question."""
        normalized = normalize_query(query)
        for bridge in reversed(self.supported_bridges(state, decision)):
            if re.search(rf"\b{re.escape(normalize_query(bridge))}\b", normalized):
                return f"{state.request.question.strip()} Source bridge: {bridge}"
        return None
