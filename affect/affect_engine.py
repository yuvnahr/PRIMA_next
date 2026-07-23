"""Dynamic PRIMA-NEXT affect engine."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from affect.affect_evolution import (
    compute_stability,
    compute_volatility,
    detect_dissonance,
    evolution_metrics,
    update_momentum,
)
from affect.affect_types import AffectUpdate
from affect.classifier_factory import create_affect_classifier
from affect.interfaces import EmotionClassifier
from affect.emotion_history import EmotionHistory
from affect.emotion_prediction import EmotionPrediction
from affect.emotion_profile import EmotionProfile
from affect.emotional_memory_adapter import EmotionalMemoryAdapter
from affect.pad_model import PADState
from affect.reflection_triggers import generate_reflection_signals
from affect.retrieval_priors import generate_retrieval_priors
from affect.salience_modulation import compute_salience
from state.cognitive_state import CognitiveState
from state.emotional_state import EmotionalState


class DynamicAffectEngine:
    """Stateful affect component that emits structured cognitive metadata."""

    def __init__(
        self,
        classifier: EmotionClassifier | None = None,
        emotional_state: EmotionalState | None = None,
        history: EmotionHistory | None = None,
        memory_adapter: EmotionalMemoryAdapter | None = None,
    ) -> None:
        self.classifier = classifier or create_affect_classifier()
        self.emotional_state = emotional_state or EmotionalState()
        self.history = history or EmotionHistory(maxlen=50)
        self.memory_adapter = memory_adapter or EmotionalMemoryAdapter()

    def process(
        self,
        text: str,
        cognitive_state: CognitiveState | None = None,
        memory_context: dict[str, Any] | None = None,
    ) -> AffectUpdate:
        """Classify text, update affect state, and emit PRIMA-facing metadata."""
        profile = self.classifier.classify(text)
        profile_pad = PADState(profile.valence, profile.arousal, profile.dominance)
        state_source = cognitive_state.emotional_state if cognitive_state else self.emotional_state
        decayed_current = state_source.current_pad.decay(0.92, state_source.baseline_pad)
        next_pad = decayed_current.merge(profile_pad, weight=0.45 + profile.confidence * 0.25)

        dissonance_score = detect_dissonance(self.history, profile_pad, profile.confidence)
        self.history.add(profile, next_pad)

        volatility = compute_volatility(self.history)
        stability = compute_stability(volatility, profile.confidence)
        momentum = update_momentum(state_source.emotional_momentum, profile)
        evolution = evolution_metrics(self.history, state_source.baseline_pad)
        novelty = next_pad.distance(self.history.average_pad(self.history.get_recent(5)))
        intensity = max(profile.emotions.values()) if profile.emotions else 0.0
        salience_score = compute_salience(intensity, novelty, volatility)

        updated_state = EmotionalState(
            current_pad=next_pad,
            baseline_pad=state_source.baseline_pad,
            dominant_emotion=profile.dominant_emotion,
            emotional_momentum=momentum,
            emotional_stability=stability,
            emotional_volatility=volatility,
            confidence=profile.confidence,
            last_update_time=datetime.now(timezone.utc),
        )

        self.emotional_state = updated_state
        if cognitive_state is not None:
            cognitive_state.emotional_state = updated_state

        retrieval_priors = generate_retrieval_priors(profile, salience_score, volatility)
        reflection_signals = generate_reflection_signals(profile, dissonance_score, evolution, volatility)
        memory_metadata = self.memory_adapter.to_metadata(profile, salience_score, dissonance_score)
        prediction = getattr(self.classifier, "last_prediction", None)
        if isinstance(prediction, EmotionPrediction):
            memory_metadata.update({
                "affect_backend": "goemotions",
                "affect_model_id": prediction.model_id,
                "affect_model_revision": prediction.model_revision,
                "affect_taxonomy": prediction.taxonomy,
                "fine_grained_emotions": dict(prediction.probabilities),
                "selected_fine_grained_labels": list(prediction.selected_labels),
                "prima_core_emotions": dict(profile.emotions),
                "entropy": prediction.entropy,
                "uncertainty": prediction.uncertainty,
                "margin": prediction.margin,
            })
        if memory_context:
            memory_metadata["memory_context_keys"] = sorted(memory_context.keys())

        return AffectUpdate(
            profile=profile,
            pad_state=next_pad,
            emotional_state=updated_state,
            retrieval_priors=retrieval_priors,
            reflection_signals=reflection_signals,
            memory_metadata=memory_metadata,
            salience_score=salience_score,
            dissonance_score=dissonance_score,
            evolution=evolution,
        )

    def get_emotion_profile(self, text: str) -> EmotionProfile:
        return self.classifier.classify(text)


_DEFAULT_ENGINE = DynamicAffectEngine()


def get_emotion_profile(text: str) -> EmotionProfile:
    """Backward-compatible module-level helper."""
    return _DEFAULT_ENGINE.get_emotion_profile(text)
