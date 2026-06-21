"""Emotion evaluation runner tests."""

from __future__ import annotations

import json
from pathlib import Path

from evaluation.runners.emotion_eval_runner import EmotionEvalRunner, KeywordBaseline


class StaticModel:
    name = "static_model"

    def predict(self, text: str) -> str:
        if "promoted" in text:
            return "joy"
        return "sadness"


def test_emotion_eval_runner_writes_supervised_metrics(tmp_path: Path) -> None:
    dataset_path = tmp_path / "emotion_gold.json"
    output_path = tmp_path / "emotion_eval_results.json"
    dataset_path.write_text(
        json.dumps(
            [
                {"text": "I got promoted today", "emotion": "joy"},
                {"text": "My friend died yesterday", "emotion": "sadness"},
            ]
        ),
        encoding="utf-8",
    )

    result = EmotionEvalRunner(
        dataset_path=dataset_path,
        output_path=output_path,
        models=(StaticModel(),),
    ).run()

    assert result["sample_count"] == 2
    assert result["models"]["static_model"]["accuracy"] == 1.0
    assert result["models"]["static_model"]["macro_f1"] == 1.0
    assert output_path.exists()


def test_keyword_baseline_uses_canonical_labels() -> None:
    model = KeywordBaseline("baseline", {"joy_ecstasy": ("thrilled",)})

    assert model.predict("I am thrilled") == "joy"
