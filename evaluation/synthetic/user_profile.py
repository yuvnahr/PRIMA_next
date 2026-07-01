"""Structured long-term attribute types for synthetic users."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Preference:
    """A single user preference with temporal evolution support."""

    category: str
    value: str
    turn_set: int = 0
    superseded_by: str = ""
    superseded_at_turn: int = -1

    def is_active(self, at_turn: int) -> bool:
        """Return True if this preference is still active at the given turn."""
        if self.superseded_at_turn < 0:
            return at_turn >= self.turn_set
        return self.turn_set <= at_turn < self.superseded_at_turn

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category,
            "value": self.value,
            "turn_set": self.turn_set,
            "superseded_by": self.superseded_by,
            "superseded_at_turn": self.superseded_at_turn,
        }


@dataclass(slots=True)
class Project:
    """A project the user is working on."""

    name: str
    description: str
    status: str = "active"
    started_at_turn: int = 0
    completed_at_turn: int = -1
    collaborators: list[str] = field(default_factory=list)
    milestones: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "status": self.status,
            "started_at_turn": self.started_at_turn,
            "completed_at_turn": self.completed_at_turn,
            "collaborators": list(self.collaborators),
            "milestones": list(self.milestones),
        }


@dataclass(slots=True)
class Relationship:
    """A relationship between the user and another person."""

    person_name: str
    relationship_type: str
    status: str = "active"
    met_at_turn: int = 0
    ended_at_turn: int = -1
    context: str = ""
    evolution_history: list[dict[str, object]] = field(default_factory=list)

    def is_active(self, at_turn: int) -> bool:
        """Return True if the relationship is active at the given turn."""
        if self.ended_at_turn < 0:
            return at_turn >= self.met_at_turn
        return self.met_at_turn <= at_turn < self.ended_at_turn

    def to_dict(self) -> dict[str, object]:
        return {
            "person_name": self.person_name,
            "relationship_type": self.relationship_type,
            "status": self.status,
            "met_at_turn": self.met_at_turn,
            "ended_at_turn": self.ended_at_turn,
            "context": self.context,
            "evolution_history": list(self.evolution_history),
        }


@dataclass(slots=True)
class TemporalEvent:
    """A significant life event tied to a simulated week/turn."""

    description: str
    event_type: str
    week: int
    turn: int
    emotional_impact: str = "neutral"
    affects_preferences: list[str] = field(default_factory=list)
    affects_relationships: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "description": self.description,
            "event_type": self.event_type,
            "week": self.week,
            "turn": self.turn,
            "emotional_impact": self.emotional_impact,
            "affects_preferences": list(self.affects_preferences),
            "affects_relationships": list(self.affects_relationships),
        }


@dataclass(slots=True)
class EmotionalProfile:
    """Emotional baseline and trajectory for a synthetic user."""

    baseline_emotion: str
    current_emotion: str
    trajectory: list[tuple[int, str]] = field(default_factory=list)
    volatility: float = 0.3
    recovery_rate: float = 0.5

    def emotion_at_turn(self, turn: int) -> str:
        """Return the most recent emotion recorded at or before the given turn."""
        current = self.baseline_emotion
        for t, emotion in self.trajectory:
            if t <= turn:
                current = emotion
            else:
                break
        return current

    def to_dict(self) -> dict[str, object]:
        return {
            "baseline_emotion": self.baseline_emotion,
            "current_emotion": self.current_emotion,
            "trajectory": [(t, e) for t, e in self.trajectory],
            "volatility": self.volatility,
            "recovery_rate": self.recovery_rate,
        }


@dataclass(slots=True)
class PersonalityTraits:
    """Big-Five-inspired personality axes (0.0–1.0)."""

    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5

    def to_dict(self) -> dict[str, float]:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
        }


@dataclass(slots=True)
class UserProfile:
    """Complete structured long-term profile for a synthetic user."""

    user_id: str
    name: str
    occupation: str
    workplace: str
    location: str
    hobbies: list[str]
    long_term_goals: list[str]
    recurring_topics: list[str]
    preferences: list[Preference]
    projects: list[Project]
    relationships: list[Relationship]
    temporal_events: list[TemporalEvent]
    emotional_profile: EmotionalProfile
    personality_traits: PersonalityTraits
    known_facts: dict[str, str]
    research_area: str = ""
    education: str = ""
    interests: list[str] = field(default_factory=list)
    family: list[str] = field(default_factory=list)
    friends: list[str] = field(default_factory=list)
    colleagues: list[str] = field(default_factory=list)
    communication_style: str = ""
    tech_stack: list[str] = field(default_factory=list)
    travel_history: list[str] = field(default_factory=list)
    health_context: list[str] = field(default_factory=list)
    financial_goals: list[str] = field(default_factory=list)
    relationship_timeline: list[dict[str, object]] = field(default_factory=list)
    career_changes: list[dict[str, object]] = field(default_factory=list)
    milestones: list[dict[str, object]] = field(default_factory=list)
    habits: list[str] = field(default_factory=list)

    def active_preferences(self, at_turn: int) -> list[Preference]:
        """Return all preferences that are active at the given turn."""
        return [p for p in self.preferences if p.is_active(at_turn)]

    def active_relationships(self, at_turn: int) -> list[Relationship]:
        """Return relationships active at the given turn."""
        return [r for r in self.relationships if r.is_active(at_turn)]

    def events_up_to_turn(self, turn: int) -> list[TemporalEvent]:
        """Return all events that have occurred by the given turn."""
        return [e for e in self.temporal_events if e.turn <= turn]

    def to_dict(self) -> dict[str, object]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "occupation": self.occupation,
            "workplace": self.workplace,
            "location": self.location,
            "hobbies": list(self.hobbies),
            "long_term_goals": list(self.long_term_goals),
            "recurring_topics": list(self.recurring_topics),
            "preferences": [p.to_dict() for p in self.preferences],
            "projects": [p.to_dict() for p in self.projects],
            "relationships": [r.to_dict() for r in self.relationships],
            "temporal_events": [e.to_dict() for e in self.temporal_events],
            "emotional_profile": self.emotional_profile.to_dict(),
            "personality_traits": self.personality_traits.to_dict(),
            "known_facts": dict(self.known_facts),
            "research_area": self.research_area,
            "education": self.education,
            "interests": list(self.interests),
            "family": list(self.family),
            "friends": list(self.friends),
            "colleagues": list(self.colleagues),
            "communication_style": self.communication_style,
            "tech_stack": list(self.tech_stack),
            "travel_history": list(self.travel_history),
            "health_context": list(self.health_context),
            "financial_goals": list(self.financial_goals),
            "relationship_timeline": list(self.relationship_timeline),
            "career_changes": list(self.career_changes),
            "milestones": list(self.milestones),
            "habits": list(self.habits),
        }
