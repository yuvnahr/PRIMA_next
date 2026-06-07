import unittest

from affect.emotion_classifier import get_emotion_profile, map_opposite_emotions, resolve_modifiers_and_negations
from affect.emotion_profile import EmotionProfile
from affect.pad_model import PADState
from state.emotional_state import EmotionalState


class DomainModelsTest(unittest.TestCase):
    def test_pad_clamps_updates_decays_and_distances(self) -> None:
        pad = PADState(2.0, -2.0, 0.5)
        self.assertEqual(pad.pleasure, 1.0)
        self.assertEqual(pad.arousal, -1.0)

        updated = pad.update(PADState(-0.5, 0.5, 0.5), rate=0.5)
        self.assertEqual(updated.pleasure, 0.75)
        self.assertEqual(updated.dominance, 0.75)
        self.assertEqual(updated.decay(0.0), PADState())
        self.assertEqual(updated.distance(updated), 0.0)

    def test_emotion_profile_is_immutable_and_serializable(self) -> None:
        profile = EmotionProfile.from_scores({"joy": 0.7, "fear": 0.3}, ("happy",))

        self.assertEqual(profile.dominant_emotion, "joy")
        self.assertEqual(profile.to_dict()["emotional_keywords"], ["happy"])
        with self.assertRaises(TypeError):
            profile.emotions["fear"] = 1.0  # type: ignore[index]

    def test_emotional_state_serialization_roundtrip(self) -> None:
        state = EmotionalState(current_pad=PADState(0.1, 0.2, 0.3), dominant_emotion="joy")
        restored = EmotionalState.from_dict(state.to_dict())

        self.assertEqual(restored.current_pad, state.current_pad)
        self.assertEqual(restored.dominant_emotion, "joy")

    def test_modifier_and_negation_compatibility(self) -> None:
        intensified = resolve_modifiers_and_negations(
            ["this", "is", "very", "bad"],
            {"joy": 0.4, "fear": 0.6},
        )
        self.assertGreater(intensified["fear"], 0.6)

        negated = map_opposite_emotions({"joy": 1.0})
        self.assertEqual(negated, {"sad": 1.0})

    def test_backward_compatible_profile_helper(self) -> None:
        profile = get_emotion_profile("I feel happy")

        self.assertEqual(profile.dominant_emotion, "joy")
        self.assertIn("happy", profile.emotional_keywords)


if __name__ == "__main__":
    unittest.main()
