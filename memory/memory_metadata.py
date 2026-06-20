"""Metadata helpers for Chroma-compatible storage."""

from __future__ import annotations

import json
from typing import Any

JSON_FIELDS = {
    "affective_state",
    "context",
    "lineage",
    "graph_links",
    "retrieval_metadata",
    "evolution_metadata",
    "state_snapshot",
    "keywords",
    "tags",
    "labels",
    "metadata",
}


def encode_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    encoded: dict[str, Any] = {}
    for key, value in metadata.items():
        if key in JSON_FIELDS:
            encoded[key] = json.dumps(value, sort_keys=True)
        else:
            encoded[key] = value
    return encoded


def decode_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(metadata)
    for key in JSON_FIELDS:
        value = decoded.get(key)
        if isinstance(value, str):
            try:
                decoded[key] = json.loads(value)
            except json.JSONDecodeError:
                decoded[key] = value
    return decoded
