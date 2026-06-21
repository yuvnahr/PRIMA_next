import unittest

from affect.affect_types import ReflectionSignal as AffectReflectionSignal
from memory.memory_repository import InMemoryMemoryRepository
from memory.retrieval.retrieval_confidence import RetrievalConfidence
from reflection.adaptive_reflection_pipeline import AdaptiveReflectionPipeline
from reflection.failure_classifier import FailureClassifier
from reflection.reflection_context import ReflectionContext
from reflection.reflection_engine import ReflectionEngine
from reflection.reflection_engine import ReflectionTriggerWeights
from reflection.reflection_repository import ReflectionRepository
from reflection.reflection_signal import ReflectionSignal
from reflection.reflection_types import FailureType, ReflectionSignalType
from reflection.rule_extractor import RuleExtractor
from reflection.verifier_adapter import VerifierAdapter, fuzzy_match, is_answer_in_content, parse_action
from state.cognitive_state import CognitiveState


class ReflectionPillarTest(unittest.TestCase):
    def test_failure_classification_and_signals(self) -> None:
        classifier = FailureClassifier()
        self.assertEqual(classifier.classify("low confidence retrieval with ambiguous memories"), FailureType.RETRIEVAL_FAILURE)

        signal = ReflectionSignal.create(
            ReflectionSignalType.RETRIEVAL_AMBIGUITY,
            severity=0.8,
            confidence=0.9,
            source="test",
        )
        self.assertEqual(signal.signal_type, ReflectionSignalType.RETRIEVAL_AMBIGUITY)
        self.assertLessEqual(signal.severity, 1.0)

    def test_verifier_preserves_fuzzy_and_grounded_checks(self) -> None:
        self.assertEqual(parse_action("Thought: done\nAction: Finish[Richard Nixon]"), ("Finish", "Richard Nixon"))
        self.assertTrue(fuzzy_match("Richard M Nixon", "Richard Nixon"))
        self.assertTrue(is_answer_in_content("Richard Nixon", "The answer is Richard Milhous Nixon."))

        verifier = VerifierAdapter()
        result = verifier.verify_answer("Richard M Nixon", "Richard Nixon", "Richard Nixon was president.")
        self.assertTrue(result.is_verified)
        self.assertEqual(result.match_type, "fuzzy")

    def test_reflection_trigger_consumes_state_retrieval_and_affect(self) -> None:
        memory_repository = InMemoryMemoryRepository()
        reflection_repository = ReflectionRepository(memory_repository)
        engine = ReflectionEngine(reflection_repository=reflection_repository)
        cognitive_state = CognitiveState(
            goal_state={"active_goal": "answer_question"},
            task_state={"task": "qa"},
            confidence_state={"confidence": 0.2},
        )
        affect_signal = AffectReflectionSignal(
            signal_type="emotional_dissonance",
            strength=0.8,
            reason="Recent affect conflicts with current statement.",
        )
        context = ReflectionContext(
            query="Who named Milhouse?",
            retrieval_confidence=RetrievalConfidence(0.2, 0.8, 0.2, 0.2),
            cognitive_state=cognitive_state,
            failure_metadata={"reason": "unsupported answer, not grounded", "severity": 0.9},
            affect_signals=(affect_signal,),
        )

        result = engine.evaluate(context)

        self.assertTrue(result.should_reflect)
        self.assertIsNotNone(result.reflection_memory)
        self.assertTrue(any(signal.signal_type == ReflectionSignalType.EMOTIONAL_DISSONANCE for signal in result.signals))
        self.assertEqual(len(memory_repository.list()), 1)
        self.assertEqual(result.state_updates["last_failure_type"], FailureType.HALLUCINATION_RISK.value)

    def test_affect_uncertainty_can_trigger_reflection_without_retrieval_pressure(self) -> None:
        engine = ReflectionEngine()
        context = ReflectionContext(
            query="Why did the model hesitate?",
            affect_confidence=0.08,
            retrieval_confidence=RetrievalConfidence(0.92, 0.92, 0.92, 0.92),
            failure_metadata={"reason": "routine workflow reflection checkpoint", "severity": 0.1},
        )

        result = engine.evaluate(context)

        self.assertTrue(result.should_reflect)
        self.assertGreaterEqual(result.trigger_score, engine.trigger_threshold)
        self.assertTrue(any(reason.get("reason") == "affect_uncertainty" and reason.get("triggered") for reason in result.trigger_reasons))

    def test_custom_trigger_weights_are_configurable(self) -> None:
        engine = ReflectionEngine(trigger_weights=ReflectionTriggerWeights(affect_weight=0.5, retrieval_weight=0.4, contradiction_weight=0.1))
        context = ReflectionContext(
            query="Why did the model hesitate?",
            affect_confidence=0.2,
            retrieval_confidence=RetrievalConfidence(0.9, 0.9, 0.9, 0.9),
            failure_metadata={"reason": "routine workflow reflection checkpoint", "severity": 0.1},
        )

        score = engine.compute_trigger_score(context, ())

        self.assertAlmostEqual(score, 0.44, places=6)

    def test_rule_extraction_and_repository(self) -> None:
        repository = ReflectionRepository()
        rule = RuleExtractor().extract_rule(
            "Which film came first?",
            "Action: Search[Film A]\nObservation: date\nAction: Lookup[started]\nObservation: answer",
            source_failures=("failure_a",),
        )
        repository.save_rule(rule)

        self.assertIn("lookup", rule.rule_text.lower())
        self.assertEqual(repository.get_rule(rule.rule_id), rule)

    def test_adaptive_pipeline_retries_and_expel_on_success(self) -> None:
        memory_repository = InMemoryMemoryRepository()
        reflection_repository = ReflectionRepository(memory_repository)
        engine = ReflectionEngine(reflection_repository=reflection_repository)
        pipeline = AdaptiveReflectionPipeline(reflection_engine=engine, max_retries=2)
        base_context = ReflectionContext(
            query="Who was Milhouse named after?",
            retrieval_confidence=RetrievalConfidence(0.3, 0.7, 0.5, 0.3),
        )

        result = pipeline.run_answer_trial(
            query="Who was Milhouse named after?",
            proposals=["Abraham Lincoln", "Richard Nixon"],
            ground_truth="Richard Nixon",
            retrieved_content="Milhouse was named after Richard Nixon.",
            base_context=base_context,
        )

        self.assertTrue(result.is_correct)
        self.assertEqual(result.attempts, 2)
        self.assertTrue(result.reflections)
        self.assertTrue(result.extracted_rules)
        self.assertTrue(reflection_repository.rules())


if __name__ == "__main__":
    unittest.main()
