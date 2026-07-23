"""Ollama-compatible constrained output schema for GoEmotions labels."""

from __future__ import annotations

from affect.taxonomies.goemotions import LABELS

GOEMOTIONS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {"labels": {"type": "array", "items": {"type": "string", "enum": list(LABELS)}, "minItems": 1, "uniqueItems": True}},
    "required": ["labels"],
    "additionalProperties": False,
    "allOf": [
        {"not": {"properties": {"labels": {"contains": {"const": "neutral"}, "minContains": 1, "minItems": 2}}}}
    ],
}
