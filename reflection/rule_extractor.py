"""ExpeL rule extraction preservation."""

from __future__ import annotations

import re

from reflection.reflection_lineage import ReflectionLineage
from reflection.reflection_memory import Rule


META_INSTRUCTION_RE = re.compile(r"<([A-Z_]+)>")


class RuleExtractor:
    def extract_rule(
        self,
        question: str,
        successful_trace: str,
        source_failures: tuple[str, ...] = (),
        lineage: ReflectionLineage | None = None,
    ) -> Rule:
        text = successful_trace.strip()
        if not text:
            rule_text = "Verify each intermediate step before finalizing an answer."
        elif "lookup" in text.lower():
            rule_text = "When a search finds the right entity but not the answer, use lookup with a simpler keyword before finishing."
        elif "search" in text.lower():
            rule_text = "Search the key entities separately, compare the verified observations, and only then finish with the exact answer."
        else:
            rule_text = "Use verified observations from the trace to guide the next attempt and avoid repeating rejected actions."

        applicability = tuple(match.group(1).lower() for match in META_INSTRUCTION_RE.finditer(text))
        if not applicability:
            applicability = ("general_reflection",)
        return Rule.create(
            rule_text=rule_text,
            source_failures=source_failures,
            confidence=0.85,
            applicability=applicability,
            lineage=lineage,
            creation_context={"question": question},
        )

    def extract_meta_instruction(self, failure_reason: str) -> str:
        lowered = failure_reason.lower()
        if "not found" in lowered or "could not find" in lowered:
            return "<BROADEN_SEARCH>: Use a broader or simpler search term instead of repeating the exact failed title."
        if "wrong type" in lowered or "semantic drift" in lowered:
            return "<PIVOT_ENTITY>: The current entity path drifted; search the alternate entity before concluding."
        if "already" in lowered or "answer is in" in lowered:
            return "<STRICT_FINISH>: The answer appears supported; finish with the exact brief answer."
        if "lookup" in lowered:
            return "<CHANGE_KEYWORD>: Use a simpler lookup keyword grounded in the retrieved page."
        if "nothing happens" in lowered or "can't do that" in lowered:
            return "<FIX_SYNTAX>: Use a valid action format and avoid repeating rejected simulator actions."
        return "<VERIFY_STEP>: Re-check the evidence, avoid repeating rejected actions, and choose the next grounded step."
