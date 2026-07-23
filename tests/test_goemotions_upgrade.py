from affect.classifiers.goemotions_adapter import GoEmotionsProfileAdapter
from affect.affect_engine import DynamicAffectEngine
from affect.emotion_prediction import EmotionPrediction
from affect.affect_perception import GoEmotionsDecisionController
from affect.taxonomies.goemotions import LABELS, PRIMA_CORE_MAP, validate_taxonomy
from benchmarks.goemotions.schemas import GOEMOTIONS_RESPONSE_SCHEMA
from benchmarks.goemotions.training.thresholds import select_thresholds
from llm.llm_types import LLMRequest
from llm.provider import OllamaProvider


def _prediction() -> EmotionPrediction:
    return EmotionPrediction.from_probabilities({label: 0.8 if label == "joy" else 0.0 for label in LABELS}, {}, "fixture")


def test_taxonomy_is_complete_and_prediction_is_neutral_exclusive() -> None:
    assert validate_taxonomy()["label_count"] == 28
    assert set(PRIMA_CORE_MAP) == set(LABELS)
    assert _prediction().selected_labels == ("joy",)


def test_unknown_or_mixed_neutral_predictions_are_rejected() -> None:
    values = {label: 0.0 for label in LABELS}
    try:
        EmotionPrediction(values, ("neutral", "joy"), "joy", 0.0, 0.0, 0.0, 0.0, {}, "fixture")
    except ValueError:
        pass
    else:
        raise AssertionError("neutral must not coexist with non-neutral labels")


def test_schema_reaches_ollama_payload(monkeypatch) -> None:
    captured = {}

    def fake_post(url, payload, headers=None, timeout=60):
        captured.update(payload)
        return {"response": '{"labels":["joy"]}'}

    monkeypatch.setattr("llm.provider.post_json", fake_post)
    OllamaProvider().send(LLMRequest("qwen3.5:4b", "classify", response_schema=GOEMOTIONS_RESPONSE_SCHEMA))
    assert captured["format"] == GOEMOTIONS_RESPONSE_SCHEMA


def test_threshold_selection_uses_only_supplied_development_rows() -> None:
    labels = ("joy", "neutral")
    values = [{"joy": 0.9, "neutral": 0.1}, {"joy": 0.1, "neutral": 0.9}]
    thresholds = select_thresholds(values, [frozenset({"joy"}), frozenset({"neutral"})], labels, min_support=1)
    assert thresholds["thresholds"]["joy"] == 0.15


def test_learned_prediction_drives_core_prima_profile() -> None:
    class FixturePredictor:
        def predict(self, text):
            return _prediction()

    profile = GoEmotionsProfileAdapter(FixturePredictor()).classify("ignored")
    assert profile.dominant_emotion == "joy"


def test_learned_prediction_preserves_fine_grained_metadata() -> None:
    class FixturePredictor:
        def predict(self, text):
            return _prediction()

    update = DynamicAffectEngine(classifier=GoEmotionsProfileAdapter(FixturePredictor())).process("ignored")
    assert update.memory_metadata["selected_fine_grained_labels"] == ["joy"]
    assert update.memory_metadata["affect_taxonomy"] == "goemotions"


def test_prima_controller_generates_labels_from_probabilities() -> None:
    probabilities = {label: 0.0 for label in LABELS}
    probabilities["joy"] = 0.8
    assert GoEmotionsDecisionController({"joy": 0.5}).decide(probabilities, model_id="fixture").selected_labels == ("joy",)
    assert GoEmotionsDecisionController({"joy": 0.9}).decide(probabilities, model_id="fixture").selected_labels == ("neutral",)


def test_prima_controller_uses_calibrated_neutral_override() -> None:
    probabilities = {label: 0.0 for label in LABELS}
    probabilities.update(joy=0.55, neutral=0.7)
    prediction = GoEmotionsDecisionController({"joy": 0.5, "neutral": 0.4}).decide(probabilities, model_id="fixture")
    assert prediction.selected_labels == ("neutral",)
    assert prediction.metadata["change_reason"] == "neutral_override"
