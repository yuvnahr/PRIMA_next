"""Ollama-compatible constrained output schema for GoEmotions labels."""

from __future__ import annotations

import json

from affect.taxonomies.goemotions import LABELS

GOEMOTIONS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "labels": {
            "type": "array",
            "items": {"type": "string", "enum": list(LABELS)},
            "minItems": 1,
            "uniqueItems": True,
        }
    },
    "required": ["labels"],
    "additionalProperties": False,
}


def parse_label_response(
    response: str, labels: list[str] | tuple[str, ...] = LABELS
) -> tuple[frozenset[str], str | None]:
    try:
        content = response.strip()
        if content.startswith("```json"):
            content = content[len("```json") :]
        elif content.startswith("```"):
            content = content[3:]
        if content.rstrip().endswith("```"):
            content = content.rstrip()[:-3]
        payload = json.loads(content.strip())
        values = payload.get("labels") if isinstance(payload, dict) else None
        if not isinstance(payload, dict) or set(payload) != {"labels"}:
            raise ValueError("Response must contain only the labels field.")
        if not isinstance(values, list) or not values or not all(isinstance(value, str) for value in values):
            raise ValueError("Response must contain a non-empty string labels list.")
        unknown = sorted(set(values) - set(labels))
        if unknown:
            raise ValueError(f"Unknown labels: {', '.join(unknown)}")
        if len(values) != len(set(values)):
            raise ValueError("Duplicate labels are not allowed.")
        return frozenset(values), None
    except (json.JSONDecodeError, ValueError) as exc:
        return frozenset(), str(exc)
