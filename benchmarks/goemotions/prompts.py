"""Deterministic GoEmotions classification prompt."""

from __future__ import annotations


def build_prompt(text: str, labels: list[str], *, structured: bool = False) -> str:
    prompt = (
        "Classify the Reddit comment using the GoEmotions taxonomy.\n"
        "Return every applicable label and no labels that are not applicable.\n"
        "Return exactly one JSON object and nothing else: {\"labels\": [\"label\"]}.\n"
        "Use [\"neutral\"] only when no emotion label applies.\n"
        f"Allowed labels: {', '.join(labels)}\n\n"
        f"Comment: {text}"
    )
    if not structured:
        return prompt
    return (
        "Classify the Reddit comment using the GoEmotions taxonomy. Return one JSON object only.\n"
        "Choose labels only when supported by the comment; avoid speculative secondary labels. "
        "neutral means no emotion label applies and cannot appear with another label.\n"
        "Use these distinctions: annoyance is mild irritation; anger is hostility; "
        "nervousness is anxious unease; fear is a threat response; realization is coming to understand.\n"
        f"Allowed labels: {', '.join(labels)}\n\nComment: {text}"
    )
