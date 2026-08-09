"""Cold-path adapter over PRIMA's existing memory maintenance engines."""

from __future__ import annotations

import asyncio
from collections import deque
from typing import Any

from events.event import Event
from events.event_bus import EventBus
from events.event_types import EventType
from events.maintenance_events import maintenance_event
from memory.embedding_pipeline import get_embedding_pipeline
from memory.evolution.memory_evolution_engine import MemoryEvolutionEngine
from memory.maintenance.consolidation_engine import ConsolidationEngine
from memory.maintenance.salience_manager import SalienceManager
from memory.memory_index import MemoryIndex


class MaintenancePipeline:
    """Run consolidation, abstraction, decay, and derived-index updates."""

    def __init__(self, memory_index: MemoryIndex, event_bus: EventBus) -> None:
        if memory_index.graph_repository is None:
            raise ValueError("MaintenancePipeline requires a graph-capable MemoryIndex.")
        self.memory_index = memory_index
        self.graph_repository = memory_index.graph_repository
        self.event_bus = event_bus
        self.salience = SalienceManager()
        self.consolidation = ConsolidationEngine(memory_index)
        self.evolution = MemoryEvolutionEngine(memory_index, self.graph_repository)
        self.short_term_buffer: deque[str] = deque(maxlen=256)

    async def handle(self, event: Event) -> None:
        """Process one admitted memory outside the hot request task."""

        if event.event_type is not EventType.MEMORY_ADMITTED:
            return
        memory_id = str(event.payload["memory_id"])
        note = self.memory_index.get(memory_id)
        if note is None:
            raise KeyError(f"Maintenance memory not found: {memory_id}")
        embedded = await asyncio.to_thread(get_embedding_pipeline().embed_memory, note.content)
        note = note.with_updates(
            embedding=embedded.vector,
            retrieval_metadata={**note.retrieval_metadata, **embedded.metadata},
        )
        await asyncio.to_thread(self.memory_index.update, note)
        await self._emit(EventType.MEMORY_ENCODED, event, {"embedding_dimensions": len(note.embedding)})

        salience = await asyncio.to_thread(
            self.salience.score,
            note.salience_score,
            0.0,
            float(event.payload.get("novelty_score", 0.0)),
            float(event.payload.get("user_relevance_score", 0.0)),
        )
        await asyncio.to_thread(self.memory_index.update, note.with_updates(salience_score=salience))
        await self._emit(EventType.SALIENCE_SCORED, event, {"salience_score": salience})
        self.short_term_buffer.append(memory_id)
        await self._emit(
            EventType.SHORT_TERM_BUFFERED,
            event,
            {"buffer": "maintenance_short_term", "buffer_size": len(self.short_term_buffer)},
        )
        await self._emit(EventType.CONSOLIDATION_REQUESTED, event)
        maintained = await asyncio.to_thread(self.consolidation.run)
        await self._emit(EventType.CONSOLIDATION_COMPLETED, event, {"updated_count": len(maintained)})
        await self._emit(EventType.DECAY_APPLIED, event, {"updated_count": len(maintained)})
        await self._emit(EventType.ABSTRACTION_REQUESTED, event)
        evolved = await asyncio.to_thread(self.evolution.evolve)
        await self._emit(
            EventType.ABSTRACTION_COMPLETED,
            event,
            {"created_memory_ids": [note.id for note in evolved.evolved_memories]},
        )
        await self._emit(
            EventType.GRAPH_INDEX_UPDATED,
            event,
            {
                "node_count": len(self.graph_repository.nodes),
                "edge_count": len(self.graph_repository.edges),
            },
        )

    async def _emit(
        self,
        event_type: EventType,
        cause: Event,
        payload: dict[str, Any] | None = None,
    ) -> None:
        await self.event_bus.publish(
            maintenance_event(
                event_type,
                memory_id=str(cause.payload["memory_id"]),
                source="memory_maintenance_pipeline",
                execution_id=cause.execution_id,
                causation_id=cause.event_id,
                payload=payload,
            )
        )
