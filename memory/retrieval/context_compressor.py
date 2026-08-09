"""Traceable evidence compression with stable citation identities."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from reasoning.models import EvidenceItem


@dataclass(frozen=True, slots=True)
class CompressedContext:
    """Evidence retained under a token budget with citation lineage."""

    evidence: tuple[EvidenceItem, ...]
    citation_map: dict[str, str]
    dropped_evidence_ids: tuple[str, ...]
    original_tokens: int
    compressed_tokens: int
    token_budget: int
    enabled: bool

    @property
    def compression_loss(self) -> float:
        """Return the fraction of approximate tokens removed."""
        if not self.original_tokens:
            return 0.0
        return round(1.0 - self.compressed_tokens / self.original_tokens, 6)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the schema-versioned compression audit record."""
        return {
            "schema_version": "1.0",
            "enabled": self.enabled,
            "token_budget": self.token_budget,
            "original_tokens": self.original_tokens,
            "compressed_tokens": self.compressed_tokens,
            "compression_loss": self.compression_loss,
            "retained_evidence_ids": [item.evidence_id for item in self.evidence],
            "dropped_evidence_ids": list(self.dropped_evidence_ids),
            "citation_map": dict(self.citation_map),
        }


@dataclass(frozen=True, slots=True)
class ContextCompressor:
    """Apply a deterministic word-token budget without changing evidence IDs."""

    enabled: bool = True

    def compress(self, evidence: tuple[EvidenceItem, ...], token_budget: int) -> CompressedContext:
        """Retain evidence in rank order and truncate only at the final boundary."""
        if token_budget < 1:
            raise ValueError("Context token budget must be at least one.")
        original_tokens = sum(_token_count(item.text) for item in evidence)
        citation_map = {item.evidence_id: item.source_id for item in evidence}
        if not self.enabled:
            return CompressedContext(
                evidence, citation_map, (), original_tokens, original_tokens, token_budget, False
            )
        retained: list[EvidenceItem] = []
        dropped: list[str] = []
        remaining = token_budget
        for item in evidence:
            words = item.text.split()
            if remaining <= 0:
                dropped.append(item.evidence_id)
                continue
            selected = words[:remaining]
            if not selected:
                dropped.append(item.evidence_id)
                continue
            provenance = dict(item.provenance)
            if len(selected) < len(words):
                provenance["context_compressed"] = True
            retained.append(replace(item, text=" ".join(selected), provenance=provenance))
            remaining -= len(selected)
        compressed_tokens = sum(_token_count(item.text) for item in retained)
        retained_ids = {item.evidence_id for item in retained}
        dropped.extend(item.evidence_id for item in evidence if item.evidence_id not in retained_ids and item.evidence_id not in dropped)
        return CompressedContext(
            tuple(retained), citation_map, tuple(dropped), original_tokens,
            compressed_tokens, token_budget, True,
        )


def _token_count(text: str) -> int:
    return len(text.split())
