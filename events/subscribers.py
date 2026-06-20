"""Subscriber definitions for the PRIMA-NEXT event bus."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from events.event import Event
from events.event_types import EventType

EventHandler = Callable[[Event], None | Awaitable[None]]


@dataclass(frozen=True, slots=True)
class Subscriber:
    """A handler plus optional routing filters."""

    handler: EventHandler
    event_types: tuple[EventType, ...] = ()
    topics: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()
    name: str = ""
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "event_types", tuple(EventType(item) for item in self.event_types))
        object.__setattr__(self, "topics", tuple(str(item) for item in self.topics))
        object.__setattr__(self, "sources", tuple(str(item) for item in self.sources))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def matches(self, event: Event) -> bool:
        """Return whether this subscriber should receive an event."""
        if self.event_types and event.event_type not in self.event_types:
            return False
        if self.topics and event.topic not in self.topics:
            return False
        if self.sources and event.source not in self.sources:
            return False
        return True


@dataclass(slots=True)
class SubscriberRegistry:
    """Mutable registry of event subscribers."""

    _subscribers: list[Subscriber] = field(default_factory=list)

    def subscribe(self, subscriber: Subscriber) -> None:
        """Register a subscriber."""
        self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: Subscriber) -> None:
        """Remove a subscriber when present."""
        self._subscribers = [item for item in self._subscribers if item is not subscriber]

    def matching(self, event: Event) -> tuple[Subscriber, ...]:
        """Return subscribers matching an event."""
        return tuple(subscriber for subscriber in self._subscribers if subscriber.matches(event))

    def all(self) -> tuple[Subscriber, ...]:
        """Return all registered subscribers."""
        return tuple(self._subscribers)
