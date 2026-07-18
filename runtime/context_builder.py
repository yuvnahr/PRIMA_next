"""Build bounded answering context from retrieved PRIMA memories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from memory.retrieval.retrieval_result import RetrievalResult


def estimate_tokens(text: str) -> int:
    """Approximate token count without adding a tokenizer dependency."""

    return max(1, len(text.split()))


@dataclass(frozen=True, slots=True)
class ContextMemory:
    """Serializable memory context included in an answering prompt."""

    memory_id: str
    text: str
    timestamp: str
    importance_score: float
    retrieval_score: float
    emotional_metadata: dict[str, Any] = field(default_factory=dict)
    linked_memories: dict[str, Any] = field(default_factory=dict)
    reflection_summary: str | None = None


@dataclass(frozen=True, slots=True)
class AnswerContext:
    """Final context window used to answer a question."""

    question: str
    memories: tuple[ContextMemory, ...]
    context_text: str
    token_count: int
    graph_links_traversed: tuple[dict[str, Any], ...] = ()
    reflection_used: bool = False


class RuntimeContextBuilder:
    """Construct a token-budgeted context window from retrieved memories."""

    def __init__(self, token_budget: int = 1600) -> None:
        self.token_budget = max(128, int(token_budget))

    def build(self, question: str, retrieved: tuple[RetrievalResult, ...]) -> AnswerContext:
        """Build a temporally ordered context window constrained by token budget."""

        ordered = sorted(retrieved, key=lambda item: item.score, reverse=True)
        memories: list[ContextMemory] = []
        sections: list[str] = []
        graph_links: list[dict[str, Any]] = []
        used_tokens = estimate_tokens(question)
        reflection_used = False

        for index, result in enumerate(ordered, start=1):
            note = result.note
            reflection_summary = self._reflection_summary(note)
            reflection_used = reflection_used or reflection_summary is not None
            context_memory = ContextMemory(
                memory_id=note.id,
                text=note.content,
                timestamp=note.timestamp.isoformat(),
                importance_score=round(float(note.salience_score), 6),
                retrieval_score=round(float(result.score), 6),
                emotional_metadata=dict(note.affective_state),
                linked_memories=dict(note.graph_links),
                reflection_summary=reflection_summary,
            )
            section = self._format_memory(index, context_memory)
            next_tokens = estimate_tokens(section)
            if used_tokens + next_tokens > self.token_budget and memories:
                break
            memories.append(context_memory)
            sections.append(section)
            used_tokens += next_tokens
            graph_links.extend(self._graph_links(note.id, note.graph_links))

        context_text = "\n\n".join(sections)
        return AnswerContext(
            question=question,
            memories=tuple(memories),
            context_text=context_text,
            token_count=estimate_tokens(context_text),
            graph_links_traversed=tuple(graph_links),
            reflection_used=reflection_used,
        )

    def _format_memory(self, index: int, memory: ContextMemory) -> str:
        lines = [
            f"[M{index}] {memory.text}",
            f"importance={memory.importance_score} timestamp={memory.timestamp}",
        ]
        if memory.reflection_summary:
            lines.append(f"reflection={memory.reflection_summary}")
        return "\n".join(lines)

    def _reflection_summary(self, note: Any) -> str | None:
        context = getattr(note, "context", {})
        if context.get("type") in {"reflection_memory", "expel_rule"}:
            return str(note.content)
        metadata = getattr(note, "evolution_metadata", {})
        if "reflection_lineage" in metadata:
            return str(metadata["reflection_lineage"])
        return None

    def _graph_links(self, memory_id: str, graph_links: dict[str, Any]) -> list[dict[str, Any]]:
        links: list[dict[str, Any]] = []
        for key, value in graph_links.items():
            if isinstance(value, list):
                linked_ids = [str(item) for item in value]
            else:
                linked_ids = [str(value)]
            for linked_id in linked_ids:
                links.append({"source_memory_id": memory_id, "relation": str(key), "target_memory_id": linked_id})
        return links



