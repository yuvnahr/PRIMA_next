"""Verifier behavior preserved from Reflexion benchmarks."""

from __future__ import annotations

import re
import string
from dataclasses import dataclass
from difflib import SequenceMatcher


def parse_action(action_string: str) -> tuple[str | None, str | None]:
    matches = re.findall(r"(\w+)\[(.+?)\]", action_string)
    if matches:
        return matches[-1][0], matches[-1][1]
    return None, None


def normalize_answer(value: str) -> str:
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def remove_punc(text: str) -> str:
        return "".join(char for char in text if char not in set(string.punctuation))

    return " ".join(remove_articles(remove_punc(value.lower())).split())


def fuzzy_match(prediction: str, truth: str, threshold: float = 0.85) -> bool:
    normalized_prediction = normalize_answer(prediction)
    normalized_truth = normalize_answer(truth)
    if not normalized_prediction or not normalized_truth:
        return False
    if normalized_prediction == normalized_truth:
        return True
    if normalized_prediction in normalized_truth or normalized_truth in normalized_prediction:
        return True
    return SequenceMatcher(None, normalized_prediction, normalized_truth).ratio() >= threshold


def is_answer_in_content(answer: str, content: str, threshold: float = 0.7) -> bool:
    answer_tokens = [token for token in normalize_answer(answer).split() if len(token) > 2]
    if not answer_tokens:
        return False
    content_text = normalize_answer(content)
    covered = sum(1 for token in answer_tokens if token in content_text)
    return covered / len(answer_tokens) >= threshold


@dataclass(frozen=True, slots=True)
class VerificationResult:
    is_verified: bool
    reason: str
    match_type: str = "none"


class VerifierAdapter:
    """Verifier adapter preserving PASS parsing, fuzzy matching, and grounding."""

    def parse_verifier_response(self, response: str) -> VerificationResult:
        upper_response = response.upper()
        first_line = upper_response.split("\n")[0]
        is_pass = "JUDGMENT: PASS" in upper_response or "PASS" in first_line
        reason = response
        if "Reason:" in response:
            reason = response.split("Reason:", 1)[1].strip()
        return VerificationResult(is_pass, reason, "verifier_text")

    def verify_answer(self, prediction: str, truth: str, retrieved_content: str = "") -> VerificationResult:
        if fuzzy_match(prediction, truth):
            grounded = is_answer_in_content(prediction, retrieved_content) if retrieved_content else True
            if grounded:
                return VerificationResult(True, "Answer matched expected value and is grounded.", "fuzzy")
            return VerificationResult(False, "Answer matched but was not grounded in retrieved content.", "ungrounded")
        return VerificationResult(False, "Answer did not match expected value.", "mismatch")

    def verify_step(self, proposal: str, observation: str, scratchpad: str = "") -> VerificationResult:
        action_type, _ = parse_action(proposal)
        if not action_type:
            return VerificationResult(False, "No parseable action found.", "parse")
        if action_type == "Finish":
            return VerificationResult(True, "Final answer proposed. Awaiting answer verification.", "finish")
        if "nothing happens" in observation.lower() or "you can't do that" in observation.lower():
            return VerificationResult(False, f"Simulator/tool rejected action: {observation}", "tool")
        if "could not find" in observation.lower() or "no page found" in observation.lower():
            return VerificationResult(True, "Search miss should pass so proposer can use suggestions.", "search_miss")
        if observation.strip():
            return VerificationResult(True, "Action produced an observation.", "observation")
        return VerificationResult(False, "Action produced no useful observation.", "empty")
