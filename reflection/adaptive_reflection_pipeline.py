"""Retry-gated adaptive reflection pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from reflection.reflection_context import ReflectionContext
from reflection.reflection_engine import ReflectionEngine
from reflection.reflection_result import ReflectionResult
from reflection.verifier_adapter import VerifierAdapter, VerificationResult


@dataclass(frozen=True, slots=True)
class PipelineRunResult:
    final_answer: str
    is_correct: bool
    attempts: int
    reflections: tuple[ReflectionResult, ...]
    extracted_rules: tuple[str, ...]


class AdaptiveReflectionPipeline:
    """Preserves verifier -> reflector -> retry -> ExpeL flow without globals."""

    def __init__(
        self,
        reflection_engine: ReflectionEngine | None = None,
        verifier: VerifierAdapter | None = None,
        max_retries: int = 2,
    ) -> None:
        self.reflection_engine = reflection_engine or ReflectionEngine()
        self.verifier = verifier or VerifierAdapter()
        self.max_retries = max_retries

    def run_answer_trial(
        self,
        query: str,
        proposals: list[str],
        ground_truth: str,
        retrieved_content: str = "",
        base_context: ReflectionContext | None = None,
    ) -> PipelineRunResult:
        reflections: list[ReflectionResult] = []
        extracted_rules: list[str] = []
        final_answer = ""

        for attempt, proposal in enumerate(proposals[: self.max_retries + 1], start=1):
            final_answer = proposal
            verification = self.verifier.verify_answer(proposal, ground_truth, retrieved_content)
            if verification.is_verified:
                success_context = self._context_from_verification(
                    query,
                    verification,
                    proposal,
                    base_context,
                    successful_trace=f"Question: {query}\nFinish[{proposal}]",
                )
                result = self.reflection_engine.evaluate(success_context)
                rule = self.reflection_engine.rule_extractor.extract_rule(
                    query,
                    str(success_context.failure_metadata["successful_trace"]),
                    source_failures=(),
                )
                self.reflection_engine.reflection_repository.save_rule(rule)
                extracted_rules.append(rule.rule_text)
                extracted_rules.extend(rule.rule_text for rule in result.rules)
                return PipelineRunResult(proposal, True, attempt, tuple(reflections), tuple(extracted_rules))

            context = self._context_from_verification(query, verification, proposal, base_context)
            result = self.reflection_engine.evaluate(context)
            reflections.append(result)
            if result.rules:
                extracted_rules.extend(rule.rule_text for rule in result.rules)

        return PipelineRunResult(final_answer, False, min(len(proposals), self.max_retries + 1), tuple(reflections), tuple(extracted_rules))

    def _context_from_verification(
        self,
        query: str,
        verification: VerificationResult,
        rejected_action: str,
        base_context: ReflectionContext | None,
        successful_trace: str | None = None,
    ) -> ReflectionContext:
        metadata = {
            "reason": verification.reason,
            "match_type": verification.match_type,
            "rejected_action": rejected_action,
            "severity": 0.85 if not verification.is_verified else 0.25,
        }
        if successful_trace:
            metadata["successful_trace"] = successful_trace
        if base_context is None:
            return ReflectionContext(query=query, failure_metadata=metadata)
        return ReflectionContext(
            query=query,
            retrieved_memories=base_context.retrieved_memories,
            retrieval_confidence=base_context.retrieval_confidence,
            cognitive_state=base_context.cognitive_state,
            emotional_state=base_context.emotional_state,
            reflection_history=base_context.reflection_history,
            failure_type=base_context.failure_type,
            failure_metadata={**base_context.failure_metadata, **metadata},
            affect_signals=base_context.affect_signals,
        )
