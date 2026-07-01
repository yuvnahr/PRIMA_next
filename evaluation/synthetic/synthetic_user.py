"""SyntheticUser: container for a complete user simulation state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from evaluation.synthetic.user_profile import (
    EmotionalProfile,
    PersonalityTraits,
    Preference,
    Project,
    Relationship,
    TemporalEvent,
    UserProfile,
)


@dataclass(slots=True)
class ConversationTurn:
    """A single simulated conversation turn."""

    turn_index: int
    utterance: str
    turn_type: str
    expected_memory_content: str = ""
    expected_emotion: str = "neutral"
    ground_truth_preference: str = ""
    ground_truth_relationship: str = ""
    ground_truth_fact_key: str = ""
    ground_truth_fact_value: str = ""
    temporal_week: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_index": self.turn_index,
            "utterance": self.utterance,
            "turn_type": self.turn_type,
            "expected_memory_content": self.expected_memory_content,
            "expected_emotion": self.expected_emotion,
            "ground_truth_preference": self.ground_truth_preference,
            "ground_truth_relationship": self.ground_truth_relationship,
            "ground_truth_fact_key": self.ground_truth_fact_key,
            "ground_truth_fact_value": self.ground_truth_fact_value,
            "temporal_week": self.temporal_week,
        }


@dataclass(slots=True)
class SyntheticUser:
    """A fully specified synthetic user with reproducible conversation history."""

    user_id: str
    name: str
    seed: int
    profile: UserProfile
    conversation_turns: list[ConversationTurn] = field(default_factory=list)

    # Ground-truth snapshots for evaluation probes (turn_index → answer)
    identity_probes: dict[int, dict[str, str]] = field(default_factory=dict)
    preference_probes: dict[int, dict[str, str]] = field(default_factory=dict)
    fact_probes: dict[int, dict[str, str]] = field(default_factory=dict)
    emotion_probes: dict[int, str] = field(default_factory=dict)

    @property
    def total_turns(self) -> int:
        """Number of conversation turns."""
        return len(self.conversation_turns)

    @property
    def preferences(self) -> list[Preference]:
        return self.profile.preferences

    @property
    def projects(self) -> list[Project]:
        return self.profile.projects

    @property
    def relationships(self) -> list[Relationship]:
        return self.profile.relationships

    @property
    def temporal_events(self) -> list[TemporalEvent]:
        return self.profile.temporal_events

    @property
    def emotional_profile(self) -> EmotionalProfile:
        return self.profile.emotional_profile

    @property
    def personality_traits(self) -> PersonalityTraits:
        return self.profile.personality_traits

    @property
    def occupation(self) -> str:
        return self.profile.occupation

    @property
    def hobbies(self) -> list[str]:
        return self.profile.hobbies

    @property
    def long_term_goals(self) -> list[str]:
        return self.profile.long_term_goals

    @property
    def recurring_topics(self) -> list[str]:
        return self.profile.recurring_topics

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "name": self.name,
            "seed": self.seed,
            "profile": self.profile.to_dict(),
            "total_turns": self.total_turns,
            "conversation_turns": [t.to_dict() for t in self.conversation_turns],
            "identity_probes": {str(k): v for k, v in self.identity_probes.items()},
            "preference_probes": {str(k): v for k, v in self.preference_probes.items()},
            "fact_probes": {str(k): v for k, v in self.fact_probes.items()},
            "emotion_probes": {str(k): v for k, v in self.emotion_probes.items()},
        }
