"""Compact evidence identities; memory IDs and conversation-qualified turns stay separate."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from memory.retrieval.retrieval_controller import RetrievalResponse
from reasoning.models import EvidenceItem

EVIDENCE_AUDIT_VERSION = "1.0"
STAGES = {
    "dense": "dense_candidates", "sparse": "sparse_candidates", "fused": "hybrid_candidates",
    "reranked": "reranked_candidates", "returned": "final_candidates",
}
LEGACY_STAGES = dict(zip(STAGES, ("dense_top30", "sparse_top30", "fused_top30", "reranked_top30", "final_candidates"), strict=True))


def source_turn_id(provenance: dict[str, Any], conversation_id: str) -> str | None:
    turn = provenance.get("source_turn_id")
    return f"{conversation_id}:{turn}" if turn and conversation_id else str(turn) if turn else None


def evidence_ids(items: Iterable[EvidenceItem], conversation_id: str) -> dict[str, Any]:
    records = list(items)
    return {
        "memory_ids": sorted({item.source_id for item in records}),
        "source_turn_ids": sorted({turn for item in records if (turn := source_turn_id(item.provenance, conversation_id))}),
        "untraceable_memory_count": len({item.source_id for item in records if not source_turn_id(item.provenance, conversation_id)}),
    }


def retrieval_hop(response: RetrievalResponse, *, conversation_id: str, query: str, hop: int, elapsed_ms: float) -> dict[str, Any]:
    diagnostics = response.diagnostics
    sources: dict[str, list[str] | None] = {}
    memories: dict[str, list[str] | None] = {}
    for stage, key in STAGES.items():
        rows = diagnostics.get(key, diagnostics.get(LEGACY_STAGES[stage]))
        if stage == "returned":
            rows = [{"id": result.note.id, **result.note.context} for result in response.results]
        sources[stage] = None if rows is None else sorted({turn for row in rows if (turn := source_turn_id(row, conversation_id))})
        memories[stage] = None if rows is None else sorted({str(row["id"]) for row in rows if row.get("id")})
    return {
        "hop_index": hop, "attempt_kind": "initial" if hop == 0 else "follow_up", "query": query,
        "stage_source_ids": sources, "stage_memory_ids": memories,
        "stage_limits": dict(diagnostics.get("stage_limits", {})),
        "added_source_turn_ids": [], "added_memory_ids": [], "excluded_memory_ids": [],
        "untraceable_added_memory_count": 0, "untraceable_added_memory_ids": [],
        "elapsed_ms": elapsed_ms, "stop_reason": None,
    }


def summarize_hops(hops: list[dict[str, Any]]) -> dict[str, Any]:
    stage_unions = {
        stage: sorted({turn for hop in hops for turn in hop["stage_source_ids"][stage] or ()})
        if all(hop["stage_source_ids"][stage] is not None for hop in hops) else None
        for stage in STAGES
    }
    return {
        "schema_version": EVIDENCE_AUDIT_VERSION, "scope": "request_all_hops", "hops": hops,
        "stage_source_turn_ids": stage_unions,
        "candidate_source_turn_ids": sorted({turn for ids in stage_unions.values() for turn in ids or ()})
        if all(ids is not None for ids in stage_unions.values()) else None,
        "accumulated_source_turn_ids": sorted({turn for hop in hops for turn in hop["added_source_turn_ids"]}),
        "accumulated_memory_ids": sorted({memory for hop in hops for memory in hop["added_memory_ids"]}),
    }
