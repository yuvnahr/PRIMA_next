"""Frozen selection and independent scoring for paired research campaigns."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from config.frozen_json import freeze_json

PARTITIONS = ("development", "validation", "final")


def freeze_selection(
    path: Path, cases: dict[str, str], partitions: dict[str, list[str]],
    identities: dict[str, str], caps: dict[str, int | float],
) -> dict[str, Any]:
    """Cases map to independent units. Partitions contain ordered case IDs."""
    if set(partitions) != set(PARTITIONS):
        raise ValueError("Declare development, validation and final partitions")
    selected = [case for partition in PARTITIONS for case in partitions[partition]]
    if len(selected) != len(set(selected)) or set(selected) != set(cases):
        raise ValueError("Selected cases must be complete, unique and disjoint")
    units = set()
    for partition in PARTITIONS:
        local_units = {cases[case] for case in partitions[partition]}
        if "" in local_units or units & local_units:
            raise ValueError("Independent units must stay within a single partition")
        units.update(local_units)
    if not all(identities.get(name) for name in ("source", "data", "model", "config", "scorer")):
        raise ValueError("Freeze source, data, model, config and scorer identities")
    return freeze_json(path, {"schema_version": 1, "case_units": cases, "partitions": partitions,
                              "identities": identities, "caps": caps})


def blind_order(case_id: str, arm_ids: list[str], *, seed: int = 20261127) -> list[str]:
    ordered = sorted(arm_ids)
    random.Random(f"{seed}:{case_id}").shuffle(ordered)  # noqa: S311  # nosec B311
    return ordered


def evidence_checks(
    retrieved: list[str], cited: list[str], gold_support: set[str], available: set[str],
    judged_supported: set[str],
) -> dict[str, Any]:
    """Gold relevance and independently adjudicated claim support are separate."""
    citations = set(cited)
    return {"retrieval_recall": len(set(retrieved) & gold_support) / len(gold_support) if gold_support else None,
            "invalid_citations": sorted(citations - available),
            "unsupported_real_citations": sorted((citations & available) - judged_supported),
            "supported_citations": sorted(citations & available & judged_supported)}


def action_outcomes(attempts: list[dict[str, Any] | None], expected: dict[str, Any]) -> dict[str, Any]:
    if not attempts:
        raise ValueError("Record at least the scheduled first attempt")
    solved = [state is not None and state == expected for state in attempts]
    return {"attempts": len(attempts), "failed": attempts.count(None), "first_solved": solved[0],
            "final_solved": solved[-1], "ever_solved": any(solved)}


def lifecycle_cost(
    construction_ms: float | None, maintenance_ms: float | None, retrieval_ms: float | None,
    generation_ms: float | None, *, reuse_count: int, cache_state: str,
    calls: int | None, input_tokens: int | None, output_tokens: int | None, retries: int | None,
) -> dict[str, Any]:
    if reuse_count < 1 or cache_state not in {"cold", "warm"}:
        raise ValueError("Record actual reuse and cold/warm state")
    return {"construction_ms": construction_ms, "maintenance_ms": maintenance_ms,
            "retrieval_ms": retrieval_ms, "generation_ms": generation_ms,
            "amortized_construction_ms": construction_ms / reuse_count if construction_ms is not None else None,
            "reuse_count": reuse_count, "cache_state": cache_state, "calls": calls, "input_tokens": input_tokens,
            "output_tokens": output_tokens, "retries": retries}


def judge_agreement(
    adjudicated: dict[str, str], judged: dict[str, str], identity: dict[str, str],
) -> dict[str, Any]:
    if not adjudicated or set(adjudicated) != set(judged):
        raise ValueError("Judge results must exactly match the blinded calibration IDs")
    if not all(identity.get(field) for field in ("model", "revision", "prompt")):
        raise ValueError("Judge model, revision and prompt identity are required")
    confusion: dict[str, dict[str, int]] = {}
    correct = 0
    for case_id, expected in adjudicated.items():
        guess = judged[case_id]
        row = confusion.setdefault(expected, {})
        row[guess] = row.get(guess, 0) + 1
        correct += int(guess == expected)
    return {"judge": identity, "cases": len(adjudicated), "agreement": correct / len(adjudicated),
            "confusion": confusion}
