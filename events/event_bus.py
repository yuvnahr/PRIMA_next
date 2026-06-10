"""Async in-process event bus for PRIMA-NEXT."""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any

from events.event import Event
from events.event_store import EventStore
from events.event_types import EventType
from events.subscribers import EventHandler, Subscriber, SubscriberRegistry


@dataclass(frozen=True, slots=True)
class PublishResult:
    """Result of publishing an event."""

    event: Event
    delivered_count: int
    errors: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        """Return whether all matching subscribers handled the event."""
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize the publish result into plain Python values."""
        return {
            "event": self.event.to_dict(),
            "delivered_count": self.delivered_count,
            "errors": list(self.errors),
            "succeeded": self.succeeded,
        }


@dataclass(slots=True)
class EventBus:
    """Async-safe in-process event bus.

    This bus deliberately avoids external brokers. It is intended as the local
    communication boundary that replaces direct subsystem calls.
    """

    store: EventStore = field(default_factory=EventStore)
    subscribers: SubscriberRegistry = field(default_factory=SubscriberRegistry)
    raise_subscriber_errors: bool = False

    @property
    def events(self) -> list[Event]:
        """Return the underlying event list for backward-compatible inspection."""
        return list(self.store.all())

    def subscribe(
        self,
        handler: EventHandler,
        *,
        event_types: tuple[EventType, ...] = (),
        topics: tuple[str, ...] = (),
        sources: tuple[str, ...] = (),
        name: str = "",
    ) -> Subscriber:
        """Register a handler and return its subscriber record."""
        subscriber = Subscriber(
            handler=handler,
            event_types=event_types,
            topics=topics,
            sources=sources,
            name=name,
        )
        self.subscribers.subscribe(subscriber)
        return subscriber

    def unsubscribe(self, subscriber: Subscriber) -> None:
        """Remove a previously registered subscriber."""
        self.subscribers.unsubscribe(subscriber)

    async def publish(self, event: Event) -> PublishResult:
        """Publish an event, persist it, and await matching subscribers."""
        self.store.append(event)
        delivered_count = 0
        errors: list[str] = []
        for subscriber in self.subscribers.matching(event):
            try:
                result = subscriber.handler(event)
                if inspect.isawaitable(result):
                    await result
                delivered_count += 1
            except Exception as exc:
                errors.append(str(exc))
                if self.raise_subscriber_errors:
                    raise
        return PublishResult(event=event, delivered_count=delivered_count, errors=tuple(errors))

    async def publish_type(
        self,
        event_type: EventType,
        *,
        source: str,
        payload: dict[str, Any] | None = None,
        topic: str = "default",
        execution_id: str | None = None,
        correlation_id: str | None = None,
        causation_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> PublishResult:
        """Create and publish an event in one call."""
        event = Event.create(
            event_type=event_type,
            source=source,
            payload=payload or {},
            topic=topic,
            execution_id=execution_id,
            correlation_id=correlation_id,
            causation_id=causation_id,
            metadata=metadata or {},
        )
        return await self.publish(event)
