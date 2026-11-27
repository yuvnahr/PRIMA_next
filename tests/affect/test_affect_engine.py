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


def test_session_history_interleaving_reopen_and_legacy(tmp_path):
    from affect.emotion_history import EmotionHistory
    from state import CognitiveState, JsonStateManager

    injected = EmotionHistory(2)
    shared = DynamicAffectEngine(history=injected)
    manager = JsonStateManager(tmp_path / "state.json")
    standalone = DynamicAffectEngine()
    for i, utterance in enumerate(["I feel happy"] * 4 + ["I feel angry"]):
        state = manager.load("a")
        actual = shared.process(utterance, state)
        expected = standalone.process(utterance)
        assert actual.evolution == expected.evolution
        assert actual.dissonance_score == expected.dissonance_score
        assert actual.salience_score == expected.salience_score
        assert actual.reflection_signals == expected.reflection_signals
        manager.save("a", state, i)
        shared.process("I feel afraid", CognitiveState())
        manager = JsonStateManager(tmp_path / "state.json")
    assert len(injected) == 0
    shared.process("I feel sad")
    assert len(injected) == 1
    manager.reset("a")
    assert (
        shared.process("I feel angry", manager.load("a")).evolution
        == DynamicAffectEngine().process("I feel angry").evolution
    )


def test_rolling_history_is_owned_by_successful_commit(tmp_path):
    import pytest

    from state import JsonStateManager, StateVersionConflict

    path = tmp_path / "state.json"
    manager = JsonStateManager(path)
    engine = DynamicAffectEngine()
    state = manager.load("s")
    loser = manager.load("s")
    for _ in range(51):
        engine.process("I feel calm", state)
    engine.process("I am angry", loser)
    saved = manager.save("s", state, 0)
    assert len(saved.emotional.history) == 50
    with pytest.raises(StateVersionConflict):
        manager.save("s", loser, 0)
    assert JsonStateManager(path).load("s").emotional.to_dict() == saved.emotional.to_dict()
