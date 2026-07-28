"""Deterministic GoEmotions classification prompt."""

from __future__ import annotations

from affect.taxonomies.goemotions import DEFINITIONS


def build_prompt(text: str, labels: list[str], *, include_definitions: bool = False, structured: bool = False) -> str:
    definitions = "\n".join(f"- {label}: {DEFINITIONS[label]}" for label in labels) if include_definitions else ""
    definition_block = f"Definitions:\n{definitions}\n" if definitions else ""
    prompt = (
        "Classify the Reddit comment using the GoEmotions taxonomy.\n"
        "Return every applicable label and no labels that are not applicable.\n"
        "Return exactly one JSON object and nothing else: {\"labels\": [\"label\"]}.\n"
        "Labels are independent; neutral may coexist with emotion labels.\n"
        f"Allowed labels: {', '.join(labels)}\n"
        f"{definition_block}\n"
        f"Comment: {text}"
    )
    if not structured:
        return prompt
    return prompt.replace("Return exactly one JSON object and nothing else", "Return one schema-valid JSON object and nothing else")
