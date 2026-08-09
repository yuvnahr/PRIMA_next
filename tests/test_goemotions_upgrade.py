import json
from pathlib import Path

from affect.affect_engine import DynamicAffectEngine
from affect.affect_perception import GoEmotionsDecisionController
from affect.classifiers.goemotions_adapter import GoEmotionsProfileAdapter
from affect.emotion_prediction import EmotionPrediction
from affect.taxonomies.goemotions import LABELS, PRIMA_CORE_MAP, validate_taxonomy
from benchmarks.goemotions.schemas import GOEMOTIONS_RESPONSE_SCHEMA
from benchmarks.goemotions.systems import PrimaQwenSystem, QwenWithPrimaTelemetrySystem
from benchmarks.goemotions.training.classical import _runtime_predictions
from benchmarks.goemotions.training.config import TrainingConfig
from benchmarks.goemotions.training.thresholds import select_thresholds
from llm.generation_config import GenerationConfig, StructuredOutputMode
from llm.llm_types import LLMRequest, LLMResponse
from llm.provider import OllamaProvider


def _prediction() -> EmotionPrediction:
    return EmotionPrediction.from_probabilities(
        {label: 0.8 if label == "joy" else 0.0 for label in LABELS}, {}, "fixture"
    )


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
    OllamaProvider().send(
        LLMRequest(
            prompt="classify",
            generation=GenerationConfig(
                model="qwen3.5:4b",
                provider="ollama",
                structured_output=StructuredOutputMode.JSON_SCHEMA,
            ),
            response_schema=GOEMOTIONS_RESPONSE_SCHEMA,
        )
    )
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
    assert GoEmotionsDecisionController({"joy": 0.5}).decide(probabilities, model_id="fixture").selected_labels == (
        "joy",
    )
    assert GoEmotionsDecisionController({"joy": 0.9}).decide(probabilities, model_id="fixture").selected_labels == (
        "neutral",
    )


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
    assert {text: result[0] for text, result in forward.items()} == {
        text: result[0] for text, result in reverse.items()
    }
    angry = forward["I am angry."]
    assert set(json.loads(angry[0])["labels"]) == {"anger", "neutral"}
    assert angry[1]["prima_decision"]["reason"] == "add_prima_lexical_emotion"
    assert forward["I am happy."][1]["prima_decision"]["reason"] == "preserve_qwen_labels"


def test_prima_telemetry_preserves_qwen_labels(monkeypatch) -> None:
    monkeypatch.setattr(
        "llm.llm_client.LLMClient.chat",
        lambda self, prompt, **kwargs: LLMResponse(text='{"labels":["joy","neutral"]}', raw={}),
    )
    raw, metadata = QwenWithPrimaTelemetrySystem().predict("mixed", list(LABELS))
    assert set(json.loads(raw)["labels"]) == {"joy", "neutral"}
    assert metadata["prima_decision"]["reason"] == "telemetry_only_preserve_qwen"


def test_runtime_metadata_is_truthfully_bounded(monkeypatch) -> None:
    monkeypatch.setattr(
        "llm.llm_client.LLMClient.chat", lambda self, prompt, **kwargs: LLMResponse(text='{"labels":["joy"]}', raw={})
    )
    _, metadata = QwenWithPrimaTelemetrySystem().predict("happy", list(LABELS))
    runtime = metadata["runtime"]
    assert runtime["task_kind"] == "emotion_classification"
    assert runtime["profile"] == "affect_only"
    assert not (
        {"dense_retrieval", "world_model", "tool_executor", "model_executor"} & set(runtime["executed_components"])
    )
    assert "not retrieval, QA, tools, or full architecture" in metadata["scope"]


def test_bounded_layer_reports_parse_recovery_separately(monkeypatch) -> None:
    monkeypatch.setattr("llm.llm_client.LLMClient.chat", lambda self, prompt, **kwargs: LLMResponse(text="", raw={}))
    raw, metadata = PrimaQwenSystem().predict("I am angry.", list(LABELS))
    assert metadata["baseline_parse_error"]
    assert set(json.loads(raw)["labels"]) == {"anger"}
    assert metadata["affect_decision"]["reason"] == "add_bounded_lexical_emotion"


def test_trained_baseline_uses_canonical_classification_route() -> None:
    probabilities = [{label: 0.9 if label == "joy" else 0.0 for label in LABELS}]
    predictions, responses = _runtime_predictions(probabilities, {"joy": 0.5}, return_updates=True)
    assert predictions == [frozenset({"joy"})]
    assert responses[0].diagnostics.route_name == "emotion_classification:affect_only"


def test_per_example_resume(monkeypatch, tmp_path) -> None:
    from benchmarks.goemotions import experiment

    class InterruptingSystem:
        name = "model_only_zero_shot"
        system_family = "model_only"

        def __init__(self, interrupt: bool) -> None:
            self.calls = 0
            self.interrupt = interrupt

        def predict(self, text, labels):
            self.calls += 1
            if self.interrupt and self.calls == 2:
                raise KeyboardInterrupt
            return '{"labels":["joy"]}', {
                "baseline_labels": ["joy"],
                "baseline_parse_error": None,
                "provider_usage": {},
                "final_scored_labels": ["joy"],
            }

    source = Path(__file__).parent / "fixtures" / "goemotions"
    data = tmp_path / "data"
    data.mkdir()
    (data / "emotions.txt").write_text((source / "emotions.txt").read_text(encoding="utf-8"), encoding="utf-8")
    (data / "train.tsv").write_text("train\t17\ttrain-id\n", encoding="utf-8")
    (data / "dev.tsv").write_text("dev\t17\tdev-id\n", encoding="utf-8")
    (data / "test.tsv").write_text("one\t17\ttest-1\ntwo\t17\ttest-2\n", encoding="utf-8")
    fixture = data / "test.tsv"
    monkeypatch.setattr(experiment, "_preflight_provider", lambda *args: None)
    first = InterruptingSystem(True)
    monkeypatch.setattr(experiment, "_system", lambda *args: first)
    try:
        experiment.run_goemotions_experiment(
            dataset_path=fixture,
            output_path=tmp_path / "out",
            system="model_only_zero_shot",
            split="test",
            progress=False,
            bootstrap_samples=10,
        )
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("Fixture must interrupt after the first checkpoint.")
    manifest_path = tmp_path / "out" / "model_only_zero_shot" / "manifest.json"
    first_run_id = json.loads(manifest_path.read_text(encoding="utf-8"))["campaign_id"]
    second = InterruptingSystem(False)
    monkeypatch.setattr(experiment, "_system", lambda *args: second)
    result = experiment.run_goemotions_experiment(
        dataset_path=fixture,
        output_path=tmp_path / "out",
        system="model_only_zero_shot",
        split="test",
        progress=False,
        resume=True,
        bootstrap_samples=10,
    )
    assert result["samples"] == 2
    assert second.calls == 1
    assert result["run_id"].startswith("goemotions-")
    assert result["run_id"] != first_run_id
    resumed_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert resumed_manifest["benchmark_config"]["dataset_scope"] == "full_split"
    assert set(resumed_manifest["benchmark_config"]["split_validation"]["hashes"]) == {"train", "dev", "test"}
    assert resumed_manifest["resume_lineage"][0]["campaign_id"] == first_run_id


def test_metrics_round_only_for_serialization() -> None:
    from benchmarks.goemotions.experiment import _round_floats

    value = 1 / 3
    assert value != round(value, 6)
    assert _round_floats({"metric": value}) == {"metric": 0.333333}
