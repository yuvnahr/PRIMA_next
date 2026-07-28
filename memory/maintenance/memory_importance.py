"""Adaptive memory importance scoring and admission policy."""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from memory.embedding_pipeline import get_embedding_pipeline
from memory.maintenance.importance_score import ImportanceScore, clamp_score
from memory.maintenance.importance_types import ImportanceWeights, MemoryAdmissionDecision, MemoryImportanceConfig
from memory.memory_note import extract_dominant_context_chain
from memory.memory_repository import MemoryRepository
from memory.memory_types import MemoryType

DEFAULT_CONFIG_PATH = Path("config/memory_importance.yaml")

HIGH_SALIENCE_EMOTIONS = {"joy", "fear", "anger", "sadness", "surprise"}
LOW_SALIENCE_EMOTIONS = {"neutral", "trust", "acceptance", "senerity", "serenity"}
USER_RELEVANCE_TERMS = {
    "allergy",
    "allergies",
    "career",
    "goal",
    "goals",
    "important",
    "interest",
    "interests",
    "job",
    "love",
    "partner",
    "plan",
    "plans",
    "prefer",
    "preference",
    "preferences",
    "project",
    "relationship",
    "research",
    "remember",
    "study",
    "work",
}
LOW_RELEVANCE_TERMS = {
    "weather",
    "traffic",
    "line",
    "queue",
    "temporary",
    "currently",
    "today",
    "tonight",
    "casual",
}
RECURRENCE_STOPWORDS = {
    "and",
    "for",
    "from",
    "the",
    "with",
    "about",
    "after",
    "before",
    "into",
    "this",
    "that",
}


class MemoryImportanceEngine:
    """Score turn importance and decide whether to admit memory."""

    def __init__(
        self,
        repository: MemoryRepository,
        config: MemoryImportanceConfig | None = None,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
    ) -> None:
        self.repository = repository
        self.config = config or load_memory_importance_config(config_path)

    def score(
        self,
        query: str,
        *,
        affect_update: Any | None = None,
        reflection_result: Any | None = None,
    ) -> ImportanceScore:
        novelty = self.novelty_score(query)
        salience = self.emotional_salience_score(query, affect_update)
        relevance = self.user_relevance_score(query)
        recurrence = self.recurrence_score(query)
        reflection = self.reflection_score(reflection_result)
        weights = self.config.weights.normalized()
        total = (
            weights.novelty * novelty
            + weights.emotional_salience * salience
            + weights.user_relevance * relevance
            + weights.recurrence * recurrence
            + weights.reflection * reflection
        )
        return ImportanceScore(
            novelty_score=novelty,
            emotional_salience_score=salience,
            user_relevance_score=relevance,
            recurrence_score=recurrence,
            reflection_score=reflection,
            total_score=total,
        )

    def decide(
        self,
        query: str,
        *,
        affect_update: Any | None = None,
        reflection_result: Any | None = None,
    ) -> MemoryAdmissionDecision:
        score = self.score(query, affect_update=affect_update, reflection_result=reflection_result)
        stored = score.total_score >= self.config.threshold
        reason = "importance_threshold_met" if stored else "importance_below_threshold"
        return MemoryAdmissionDecision(
            query=query,
            score=score,
            stored=stored,
            threshold=self.config.threshold,
            reason=reason,
        )

    def novelty_score(self, query: str) -> float:
        results = self.repository.query(get_embedding_pipeline().embed_query(query).vector, memory_type=MemoryType.EPISODIC, limit=self.config.top_k)
        if not results:
            return 1.0
        query_tokens = set(_tokens(query))
        if not query_tokens:
            return 0.0
        overlaps = []
        for note, _ in results:
            note_tokens = set(_tokens(note.content))
            overlaps.append(len(query_tokens & note_tokens) / max(1, len(query_tokens | note_tokens)))
        return clamp_score(1.0 - max(overlaps))

    def emotional_salience_score(self, query: str, affect_update: Any | None = None) -> float:
        profile = getattr(affect_update, "profile", None)
        if profile is None and isinstance(affect_update, Mapping):
            profile = affect_update.get("profile")
        emotion = _profile_value(profile, "dominant_emotion", "neutral")
        confidence = float(_profile_value(profile, "confidence", 0.0) or 0.0)
        pad_state = getattr(affect_update, "pad_state", None)
        arousal = abs(float(getattr(pad_state, "arousal", 0.0) or 0.0))
        salience = float(getattr(affect_update, "salience_score", 0.0) or 0.0)
        if not salience:
            salience = _lexical_emotional_salience(query)
        emotion_weight = 0.85 if emotion in HIGH_SALIENCE_EMOTIONS else 0.35 if emotion in LOW_SALIENCE_EMOTIONS else 0.55
        return clamp_score(0.10 + 0.45 * salience + 0.25 * confidence + 0.20 * arousal + 0.10 * emotion_weight)

    def user_relevance_score(self, query: str) -> float:
        tokens = set(_tokens(query))
        score = 0.0
        if tokens & USER_RELEVANCE_TERMS:
            score += 0.55
        if re.search(r"\b(i|my|we|our)\b", query.lower()):
            score += 0.30
        if re.search(r"\b(i need|i want|i prefer|remember that|please remember|i am working on)\b", query.lower()):
            score += 0.25
        if tokens & LOW_RELEVANCE_TERMS:
            score -= 0.20
        return clamp_score(score)

    def recurrence_score(self, query: str) -> float:
        keywords = [keyword for keyword in extract_dominant_context_chain(query) if keyword not in RECURRENCE_STOPWORDS]
        if not keywords:
            return 0.0
        notes = self.repository.list(MemoryType.EPISODIC)
        if not notes:
            return 0.0
        mentions = 0
        for note in notes:
            note_keywords = set(note.keywords)
            if any(keyword in note_keywords or keyword in note.content.lower() for keyword in keywords):
                mentions += 1
        return clamp_score(mentions / 5.0)

    def reflection_score(self, reflection_result: Any | None = None) -> float:
        if reflection_result is None:
            return 0.0
        score = 0.0
        if bool(getattr(reflection_result, "should_reflect", False)):
            score += 0.45
        if bool(getattr(reflection_result, "correction_applied", False)):
            score += 0.25
        reasons = getattr(reflection_result, "trigger_reasons", ()) or ()
        for reason in reasons:
            reason_name = str(reason.get("reason", "") if isinstance(reason, dict) else "")
            if reason_name in {"low_confidence", "retrieval_ambiguity", "contradiction"} and bool(reason.get("triggered", False)):
                score += 0.15
        return clamp_score(score)


def load_memory_importance_config(path: str | Path = DEFAULT_CONFIG_PATH) -> MemoryImportanceConfig:
    payload = _read_simple_yaml(Path(path))
    weights_payload = payload.get("weights", {})
    weights = ImportanceWeights(
        novelty=float(weights_payload.get("novelty", 0.35)),
        emotional_salience=float(weights_payload.get("emotional_salience", 0.25)),
        user_relevance=float(weights_payload.get("user_relevance", 0.15)),
        recurrence=float(weights_payload.get("recurrence", 0.15)),
        reflection=float(weights_payload.get("reflection", 0.10)),
    )
    return MemoryImportanceConfig(
        threshold=float(payload.get("threshold", 0.55)),
        top_k=int(payload.get("top_k", 5)),
        weights=weights,
    )


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    root: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if not line.startswith(" ") and line.endswith(":"):
            section_name = line[:-1].strip()
            current_section = {}
            root[section_name] = current_section
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        target = current_section if raw_line.startswith(" ") and current_section is not None else root
        target[key.strip()] = _parse_scalar(value.strip())
    return root


def _parse_scalar(value: str) -> Any:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value.strip('"').strip("'")


def _profile_value(profile: Any, key: str, default: Any) -> Any:
    if isinstance(profile, Mapping):
        return profile.get(key, default)
    return getattr(profile, key, default)


def _lexical_emotional_salience(query: str) -> float:
    lowered = query.lower()
    high_terms = (
        "afraid",
        "angry",
        "anxious",
        "devastated",
        "excited",
        "furious",
        "grief",
        "happy",
        "panic",
        "sad",
        "surprise",
        "terrified",
        "victory",
    )
    return clamp_score(sum(1 for term in high_terms if term in lowered) / 3.0)


def _tokens(query: str) -> list[str]:
    return [token.lower() for token in re.findall(r"[a-zA-Z][a-zA-Z']+", query)]
