"""Deterministic GoEmotions classification prompt."""

from __future__ import annotations

from affect.taxonomies.goemotions import DEFINITIONS

PROMPT_VARIANTS = ("definitions_schema", "expressed_evidence", "taxonomy_distinctions")


def build_prompt(text: str, labels: list[str], *, include_definitions: bool = False, structured: bool = False,
                 variant: str = "definitions_schema") -> str:
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"Unknown GoEmotions prompt variant: {variant}")
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
    if variant != "definitions_schema":
        prompt = prompt.replace(
            "Return every applicable label and no labels that are not applicable.",
            "Select every emotion expressed by the comment's author, using evidence in the comment. "
            "A topic or quoted emotion alone is not evidence of the author's emotion. "
            "Avoid speculative extra labels; preserve all independently supported labels.",
            1,
        )
    if variant == "taxonomy_distinctions":
        prompt = prompt.replace("Allowed labels:", (
            "Disapproval is an unfavorable opinion; annoyance is irritation. "
            "Admiration finds something impressive; approval expresses a favorable opinion. "
            "Neutral means no particular emotion is expressed; it is not a default extra label. "
            "Retain neutral together with emotion labels when independently supported.\nAllowed labels:"
        ), 1)
    if not structured:
        return prompt
    return prompt.replace("Return exactly one JSON object and nothing else", "Return one schema-valid JSON object and nothing else")
