import json

from affect.classifiers.goemotions_adapter import GoEmotionsProfileAdapter
from affect.affect_engine import DynamicAffectEngine
from affect.emotion_prediction import EmotionPrediction
from affect.affect_perception import GoEmotionsDecisionController
from affect.taxonomies.goemotions import LABELS, PRIMA_CORE_MAP, validate_taxonomy
from benchmarks.goemotions.schemas import GOEMOTIONS_RESPONSE_SCHEMA
from benchmarks.goemotions.systems import PrimaQwenSystem, QwenWithPrimaTelemetrySystem
from benchmarks.goemotions.training.config import TrainingConfig
from benchmarks.goemotions.training.thresholds import select_thresholds
from llm.llm_types import LLMRequest, LLMResponse
from llm.provider import OllamaProvider


def _prediction() -> EmotionPrediction:
    return EmotionPrediction.from_probabilities({label: 0.8 if label == "joy" else 0.0 for label in LABELS}, {}, "fixture")


def test_taxonomy_is_complete() -> None:
    assert validate_taxonomy()["label_count"] == 28
    assert set(PRIMA_CORE_MAP) == set(LABELS)
    assert _prediction().selected_labels == ("joy",)


def test_official_mixed_neutral_predictions_are_preserved() -> None:
    values = {label: 0.0 for label in LABELS}
    prediction = EmotionPrediction(values, ("neutral", "joy"), "joy", 0.0, 0.0, 0.0, 0.0, {}, "fixture")
    assert prediction.selected_labels == ("joy", "neutral")


def test_schema_reaches_ollama_payload(monkeypatch) -> None:
    captured = {}

    def fake_post(url, payload, headers=None, timeout=60):
        captured.update(payload)
        return {"response": '{"labels":["joy"]}'}

    monkeypatch.setattr("llm.provider.post_json", fake_post)
    OllamaProvider().send(LLMRequest("qwen3.5:4b", "classify", response_schema=GOEMOTIONS_RESPONSE_SCHEMA))
    assert captured["format"] == GOEMOTIONS_RESPONSE_SCHEMA
    assert "allOf" not in GOEMOTIONS_RESPONSE_SCHEMA


def test_training_revision_must_be_an_immutable_commit() -> None:
    try:
        TrainingConfig(revision="main")
    except ValueError:
        pass
    else:
        raise AssertionError("Mutable Hugging Face revisions must be rejected.")


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


def test_prima_controller_preserves_independent_neutral_label() -> None:
    probabilities = {label: 0.0 for label in LABELS}
    probabilities.update(joy=0.8, neutral=0.7)
    prediction = GoEmotionsDecisionController({"joy": 0.5, "neutral": 0.4}).decide(probabilities, model_id="fixture")
    assert prediction.selected_labels == ("joy", "neutral")
    assert prediction.metadata["change_reason"] == "official_multilabel_threshold"


def test_prima_qwen_is_order_independent_and_auditable(monkeypatch) -> None:
    def fake_chat(self, prompt, **kwargs):
        label = "neutral" if "I am angry" in prompt else "joy"
        return LLMResponse(text=f'{{"labels":["{label}"]}}', raw={"fixture": True})

    monkeypatch.setattr("llm.llm_client.LLMClient.chat", fake_chat)
    texts = ("I am angry.", "I am happy.")
    forward = {text: PrimaQwenSystem().predict(text, list(LABELS)) for text in texts}
    reverse = {text: PrimaQwenSystem().predict(text, list(LABELS)) for text in reversed(texts)}
    assert {text: result[0] for text, result in forward.items()} == {text: result[0] for text, result in reverse.items()}
    angry = forward["I am angry."]
    assert set(json.loads(angry[0])["labels"]) == {"anger", "neutral"}
    assert angry[1]["prima_decision"]["reason"] == "add_prima_lexical_emotion"
    assert forward["I am happy."][1]["prima_decision"]["reason"] == "preserve_qwen_labels"


def test_prima_telemetry_preserves_qwen_labels(monkeypatch) -> None:
    monkeypatch.setattr("llm.llm_client.LLMClient.chat", lambda self, prompt, **kwargs: LLMResponse(text='{"labels":["joy","neutral"]}', raw={}))
    raw, metadata = QwenWithPrimaTelemetrySystem().predict("mixed", list(LABELS))
    assert set(json.loads(raw)["labels"]) == {"joy", "neutral"}
    assert metadata["prima_decision"]["reason"] == "telemetry_only_preserve_qwen"
