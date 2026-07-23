"""Deterministic GoEmotions classification prompt."""

from __future__ import annotations


def build_prompt(text: str, labels: list[str]) -> str:
    return (
        "Classify the Reddit comment using the GoEmotions taxonomy.\n"
        "Return every applicable label and no labels that are not applicable.\n"
        "Return exactly one JSON object and nothing else: {\"labels\": [\"label\"]}.\n"
        "Use [\"neutral\"] only when no emotion label applies.\n"
        f"Allowed labels: {', '.join(labels)}\n\n"
        f"Comment: {text}"
    )
