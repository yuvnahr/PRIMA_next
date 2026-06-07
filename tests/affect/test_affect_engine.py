import unittest

from affect.affect_engine import DynamicAffectEngine


class AffectEngineTest(unittest.TestCase):
    def test_engine_returns_structured_update_without_side_effecting_retrieval(self) -> None:
        engine = DynamicAffectEngine()

        update = engine.process("I am really scared and nervous about my exam")

        self.assertEqual(update.profile.dominant_emotion, "fear")
        self.assertIn("boost_emotional_memories", update.retrieval_priors)
        self.assertEqual(update.memory_metadata["emotion"], "fear")
        self.assertGreater(update.pad_state.arousal, 0)

    def test_momentum_accumulates_for_repeated_emotion(self) -> None:
        engine = DynamicAffectEngine()

        first = engine.process("I feel sad")
        third = engine.process("I feel sad again")
        fourth = engine.process("I am still sad")

        self.assertGreater(
            third.emotional_state.emotional_momentum["sadness"],
            first.emotional_state.emotional_momentum["sadness"],
        )
        self.assertEqual(fourth.emotional_state.dominant_emotion, "sadness")

    def test_volatility_increases_with_oscillation(self) -> None:
        engine = DynamicAffectEngine()

        calm = engine.process("I feel happy")
        engine.process("I am angry")
        engine.process("I feel happy again")
        volatile = engine.process("I am anxious")

        self.assertGreater(volatile.emotional_state.emotional_volatility, calm.emotional_state.emotional_volatility)

    def test_dissonance_signal_after_conflicting_history(self) -> None:
        engine = DynamicAffectEngine()

        engine.process("I feel sad")
        engine.process("I am still sad")
        engine.process("I feel sad again")
        update = engine.process("I am completely fine")

        self.assertGreaterEqual(update.dissonance_score, 0.55)
        self.assertTrue(any(signal.signal_type == "emotional_dissonance" for signal in update.reflection_signals))


if __name__ == "__main__":
    unittest.main()
