"""Dependency-free trainable 28-label GoEmotions probability baseline."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from affect.emotion_prediction import EmotionPrediction
from affect.taxonomies.goemotions import LABELS
from benchmarks.goemotions.dataset import GoEmotionsExample

_TOKEN = re.compile(r"[a-z]+(?:'[a-z]+)?")


@dataclass(slots=True)
class GoEmotionsNaiveBayes:
    """One independent multinomial NB classifier per GoEmotions label."""

    document_count: int = 0
    label_documents: Counter[str] = field(default_factory=Counter)
    total_tokens: Counter[str] = field(default_factory=Counter)
    label_tokens: dict[str, Counter[str]] = field(default_factory=lambda: {label: Counter() for label in LABELS})
    thresholds: dict[str, float] = field(default_factory=lambda: {label: 0.5 for label in LABELS})

    def fit(self, rows: list[GoEmotionsExample]) -> GoEmotionsNaiveBayes:
        self.document_count = len(rows)
        for row in rows:
            tokens = Counter(_TOKEN.findall(row.text.lower()))
            self.total_tokens.update(tokens)
            for label in row.labels:
                self.label_documents[label] += 1
                self.label_tokens[label].update(tokens)
        return self

    def probabilities(self, text: str) -> dict[str, float]:
        if not self.document_count:
            raise RuntimeError("Fit GoEmotionsNaiveBayes before prediction.")
        tokens, vocabulary = Counter(_TOKEN.findall(text.lower())), max(1, len(self.total_tokens))
        result: dict[str, float] = {}
        all_total = sum(self.total_tokens.values())
        for label in LABELS:
            positive_docs = self.label_documents[label]
            positive_total = sum(self.label_tokens[label].values())
            negative_docs, negative_total = self.document_count - positive_docs, all_total - positive_total
            log_odds = math.log((positive_docs + 1) / (negative_docs + 1))
            for token, count in tokens.items():
                positive = self.label_tokens[label][token]
                negative = self.total_tokens[token] - positive
                log_odds += count * math.log((positive + 1) / (positive_total + vocabulary) * (negative_total + vocabulary) / (negative + 1))
            result[label] = 1 / (1 + math.exp(-max(-60.0, min(60.0, log_odds))))
        return result

    def predict(self, text: str) -> EmotionPrediction:
        return EmotionPrediction.from_probabilities(self.probabilities(text), self.thresholds, "prima-goemotions-naive-bayes", metadata={"learned_from": "train.tsv", "calibrated": False})

    def set_thresholds(self, thresholds: dict[str, float]) -> None:
        self.thresholds = {label: float(thresholds[label]) for label in LABELS}

    def save(self, path: Path) -> None:
        import json

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"document_count": self.document_count, "label_documents": self.label_documents, "total_tokens": self.total_tokens, "label_tokens": self.label_tokens, "thresholds": self.thresholds}, sort_keys=True), encoding="utf-8")
