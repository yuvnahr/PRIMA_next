"""Core scoring adapted from snap-research/LoCoMo (CC BY-NC 4.0).

Source: https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py
Changes: isolate core F1/category dispatch, omit optional metrics and file I/O,
return Python floats. Upstream license: official_scoring_LICENSE.txt.
"""

from __future__ import annotations

import hashlib
import string
from collections import Counter
from pathlib import Path
from typing import Any

import regex  # type: ignore[import-untyped]
from nltk.stem import PorterStemmer

UPSTREAM_COMMIT = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
UPSTREAM_BLOB = "8f597dd687e66832da1f6f04a169e05622049576"
UPSTREAM_SOURCE_SHA256 = "8e3be5d57ff2ff9ec5cd05939592f468c5f3f1fd95d13e431932bdf6bf0fd6fd"
_STEMMER = PorterStemmer()


def scorer_identity() -> dict[str, str]:
    return {
        "version": "official-locomo-core-1.0", "upstream_commit": UPSTREAM_COMMIT,
        "upstream_blob": UPSTREAM_BLOB, "upstream_scorer_sha256_lf": UPSTREAM_SOURCE_SHA256,
        "adapted_scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "license": "CC-BY-NC-4.0", "source": "snap-research/LoCoMo/task_eval/evaluation.py",
    }


def normalize_answer(text: str) -> str:
    text = text.replace(",", "").lower()
    text = "".join(character for character in text if character not in string.punctuation)
    return " ".join(regex.sub(r"\b(a|an|the|and)\b", " ", text).split())


def f1_score(prediction: str, ground_truth: str) -> float:
    predicted = [_STEMMER.stem(word) for word in normalize_answer(prediction).split()]
    expected = [_STEMMER.stem(word) for word in normalize_answer(ground_truth).split()]
    common = sum((Counter(predicted) & Counter(expected)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(predicted), common / len(expected)
    return 2 * precision * recall / (precision + recall)


def official_locomo_score(prediction: str, answer: Any, category: str | int) -> float:
    category = str(category)
    reference = str(answer)
    if category == "3":
        reference = reference.split(";")[0].strip()
    if category in {"2", "3", "4"}:
        return f1_score(prediction, reference)
    if category == "1":
        predictions = [part.strip() for part in prediction.split(",")]
        references = [part.strip() for part in reference.split(",")]
        return sum(max(f1_score(part, gold) for part in predictions) for gold in references) / len(references)
    if category == "5":
        return float("no information available" in prediction.lower() or "not mentioned" in prediction.lower())
    raise ValueError(f"Unsupported LoCoMo category: {category!r}")
