import unittest

from affect.emotion_history import EmotionHistory
from affect.emotion_profile import EmotionProfile
from affect.pad_model import PADState
from affect.retrieval_priors import generate_retrieval_priors
from affect.salience_modulation import compute_salience


class HistoryAndPriorsTest(unittest.TestCase):
    def test_history_velocity_acceleration_and_drift_are_deterministic(self) -> None:
        history = EmotionHistory(maxlen=50)
        profile = EmotionProfile.from_scores({"fear": 1.0})

        history.add(profile, PADState(0.0, 0.0, 0.0))
        history.add(profile, PADState(0.0, 0.5, 0.0))
        history.add(profile, PADState(0.0, 1.0, 0.0))

        self.assertEqual(history.compute_velocity(), 0.5)
        self.assertEqual(history.compute_acceleration(), 0.0)
        self.assertEqual(history.compute_drift(PADState()), 1.0)

    def test_salience_and_retrieval_priors_are_bounded(self) -> None:
        profile = EmotionProfile.from_scores({"sadness": 0.75, "trust": 0.25})
        salience = compute_salience(intensity=0.75, novelty=0.8, volatility=0.4)
        priors = generate_retrieval_priors(profile, salience, volatility=0.4)

        self.assertGreaterEqual(salience, 0.0)
        self.assertLessEqual(salience, 1.0)
        self.assertEqual(priors["boost_negative_experiences"], 0.75)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in priors.values()))


if __name__ == "__main__":
    unittest.main()
