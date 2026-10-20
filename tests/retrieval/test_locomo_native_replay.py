"""Goal 2 contract checks; synthetic models are not semantic validation evidence."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from benchmarks.locomo.adapter import LoCoMoAdapter
from benchmarks.locomo.experiment import IngestionPolicy, _turn_request, _turn_text, representation_identity
from benchmarks.locomo.retrieval_development import development_manifest, run_development
from config.runtime_config import RuntimeConfig
from llm.generation_config import GenerationConfig
from memory import embedding_backend as embeddings
from memory.embedding_pipeline import CanonicalEmbeddingPipeline, current_embedding_metadata, use_embedding_pipeline
from memory.embedding_rebuild import rebuild_embeddings
from memory.maintenance.importance_score import ImportanceScore
from memory.maintenance.importance_types import MemoryAdmissionDecision
from memory.memory_metadata import decode_metadata, encode_metadata
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from memory.retrieval import reranker as reranker_module
from memory.retrieval.hybrid_fusion import HybridFusion, HybridFusionConfig
from memory.retrieval.query_analysis import QueryAnalyzer, temporal_agreement
from memory.retrieval.reranker import Reranker
from memory.retrieval.retrieval_controller import RetrievalController
from memory.retrieval.retrieval_request import RetrievalRequest
from memory.retrieval.retrieval_result import RetrievalResult
from memory.retrieval.temporal_strategy import TemporalRetrievalStrategy
from runtime.contracts import ExecutionProfile, ExecutionStatus, PrimaRequest, TaskKind
from runtime.prima_runtime import PrimaRuntime

REVISION = "a" * 40


def conversation():
    return LoCoMoAdapter().adapt([{
        "sample_id": "conv-a", "conversation": {"speaker_a": "A", "speaker_b": "B", "session_1_date_time": "8 May 2023 01:56 PM",
        "session_1": [{"speaker": "A", "dia_id": "D1:1", "text": "I remember loving green tea.", "blip_caption": "a cup on a table"}]},
        "qa": [{"question": "What drink?", "answer": "green tea", "category": 4, "evidence": ["D1:1"]}],
    }])[0]


class NoModel:
    provider_name = "test"

    def chat(self, **kwargs):
        raise AssertionError("Replay/retrieval checks must never generate an answer")


@pytest.mark.parametrize("policy", list(IngestionPolicy))
def test_historical_provenance_survives_both_ingestion_paths_and_record_reload(tmp_path, monkeypatch, policy):
    repository = InMemoryMemoryRepository()
    runtime = PrimaRuntime(llm_client=NoModel(), memory_repository=repository, maintenance_enabled=False, log_path=tmp_path / "runtime.log")
    monkeypatch.setattr(type(runtime.memory_importance_engine), "decide", lambda self, query, **kwargs: MemoryAdmissionDecision(
        query, ImportanceScore(1, 1, 1, 1, 1, 1), True, 0.55, "synthetic admission"))
    turn = conversation().turns[0]
    response = asyncio.run(runtime.execute(_turn_request("conv-a", turn, policy, GenerationConfig(model="fake", provider="test"))))
    assert response.status is ExecutionStatus.COMPLETED
    note = repository.list()[0]
    assert note.timestamp == datetime(2023, 5, 8, 13, 56, tzinfo=UTC)
    assert note.context["source_conversation_id"] == "conv-a"
    assert note.context["source_session_id"] == "session_1"
    assert note.context["source_turn_id"] == "D1:1" and note.context["speaker"] == "A"
    assert note.context["source_timestamp_original"] == "8 May 2023 01:56 PM"
    restored = MemoryNote.from_record(note.to_record())
    assert restored.timestamp == note.timestamp and restored.context == note.context
    record = note.to_record()
    record["metadata"] = decode_metadata(encode_metadata(record["metadata"]))
    persisted = MemoryNote.from_record(record)
    assert persisted.timestamp == note.timestamp and persisted.context == note.context
    assert response.diagnostics.model_call_count == 0


@pytest.mark.parametrize("stamp", ["yesterday", "", None, 123])
@pytest.mark.parametrize("controlled", [False, True])
def test_malformed_replay_time_fails_before_admission_or_state_commit(tmp_path, stamp, controlled):
    repository = InMemoryMemoryRepository()
    runtime = PrimaRuntime(llm_client=NoModel(), memory_repository=repository, maintenance_enabled=False, log_path=tmp_path / "runtime.log")
    metadata = {"source_timestamp": stamp}
    result = asyncio.run(runtime.execute(PrimaRequest(
        task_kind=TaskKind.DOCUMENT_INGESTION if controlled else TaskKind.HISTORICAL_REPLAY,
        profile=ExecutionProfile.INGESTION_ONLY if controlled else ExecutionProfile.PRIMA_FULL,
        input_text="Undated source text", session_id="conv", metadata={"document_metadata": metadata} if controlled else metadata)))
    assert result.status is ExecutionStatus.FAILED and repository.list() == []
    assert any("timestamp" in error for error in result.errors)


def test_live_document_keeps_current_time_default_and_scoped_sources_do_not_leak(tmp_path):
    repository = InMemoryMemoryRepository()
    runtime = PrimaRuntime(llm_client=NoModel(), memory_repository=repository, maintenance_enabled=False, log_path=tmp_path / "runtime.log")
    start = datetime.now(UTC)
    for session in ("conv-a", "conv-b"):
        response = asyncio.run(runtime.execute(PrimaRequest(task_kind=TaskKind.DOCUMENT_INGESTION, profile=ExecutionProfile.INGESTION_ONLY,
                                                           input_text=f"Green tea belongs to {session}", session_id=session)))
        assert response.status is ExecutionStatus.COMPLETED
    assert all(note.timestamp >= start for note in repository.list())
    for session in ("conv-a", "conv-b"):
        result = runtime.retrieval_controller.retrieve(RetrievalRequest("green tea", profile="simple_rag", session_id=session))
        assert result.results and {item.note.context["source_conversation_id"] for item in result.results} == {session}


def test_captions_are_explicit_and_only_replay_representation_changes():
    original = conversation()
    turn = original.turns[0]
    assert "Image caption:" not in _turn_text(turn)
    assert _turn_text(turn, True).endswith("Image caption: a cup on a table")
    assert representation_identity([original]) != representation_identity([original], True)
    changed_gold = replace(original, questions=(replace(original.questions[0], answer="hidden different answer"),))
    assert representation_identity([original], True) == representation_identity([changed_gold], True)
    request = _turn_request(original.id, turn, IngestionPolicy.NORMAL_PRIMA, GenerationConfig(model="fake", provider="test"), True)
    assert "hidden" not in request.input_text and request.metadata["source_session_id"] == "session_1"


class FakeSentenceModel:
    device = "cpu"

    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, text, **kwargs):
        assert kwargs["normalize_embeddings"] and kwargs["batch_size"] > 0
        return [1.0, 0.0, 0.0]


def test_learned_output_is_normalized_without_resizing(monkeypatch):
    fake = FakeSentenceModel()
    fake.encode = lambda *args, **kwargs: [3.0, 4.0, 0.0]
    monkeypatch.setattr(embeddings, "_sentence_transformer_model", lambda *args, **kwargs: fake)
    backend = embeddings.SentenceTransformerEmbeddingBackend(embeddings.EmbeddingBackendConfig("minilm", "fake", 3, revision=REVISION))
    assert backend.embed("source") == pytest.approx([0.6, 0.8, 0.0])


@pytest.mark.parametrize("values", [[float("nan"), 0, 0], [float("inf"), 0, 0], [0, 0, 0], [1, 0]])
def test_learned_vectors_reject_nonfinite_zero_and_wrong_dimensions(monkeypatch, values):
    fake = FakeSentenceModel()
    fake.encode = lambda *args, **kwargs: values
    monkeypatch.setattr(embeddings, "_sentence_transformer_model", lambda *args, **kwargs: fake)
    backend = embeddings.SentenceTransformerEmbeddingBackend(embeddings.EmbeddingBackendConfig("minilm", "fake", 3, revision=REVISION))
    with pytest.raises(ValueError, match="native dimension"):
        backend.embed("source")
    with pytest.raises(ValueError, match="Native embedding dimension"):
        embeddings.SentenceTransformerEmbeddingBackend(embeddings.EmbeddingBackendConfig("minilm", "fake", 64, revision=REVISION))


def test_loaded_model_cache_separates_revisions_devices_and_uses_local_cpu_controls(monkeypatch):
    calls = []
    monkeypatch.setattr(embeddings, "model_snapshot", lambda model, revision, local: (model, {"revision": revision}))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda threads: calls.append(("threads", threads))))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=lambda model, **kwargs: calls.append((model, kwargs)) or SimpleNamespace()))
    embeddings.reset_embedding_backend_cache()
    try:
        first = embeddings._sentence_transformer_model("model", revision=REVISION, device="cpu")
        assert embeddings._sentence_transformer_model("model", revision=REVISION, device="cpu") is first
        embeddings._sentence_transformer_model("model", revision="b" * 40, device="cpu")
        embeddings._sentence_transformer_model("model", revision=REVISION, device="cuda:0")
        loads = [kwargs for model, kwargs in calls if model == "model"]
        assert len(loads) == 3 and all(kwargs["local_files_only"] and not kwargs["trust_remote_code"] for kwargs in loads)
        assert loads[0]["device"] == "cpu"
    finally:
        embeddings.reset_embedding_backend_cache()


def test_required_model_identity_and_local_manifest_integrity(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        embeddings.model_snapshot("model", "main")
    with pytest.raises(FileNotFoundError):
        embeddings.model_snapshot("model", REVISION, str(tmp_path / "absent"))
    content = b'{}'
    (tmp_path / "config.json").write_bytes(content)
    manifest = {"model_identifier": "model", "revision": REVISION, "files": {"config.json": hashlib.sha256(content).hexdigest()}}
    (tmp_path / "prima_model_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert embeddings.model_snapshot("model", REVISION, str(tmp_path))[1]["files"] == manifest["files"]
    (tmp_path / "config.json").write_text('{"changed":true}', encoding="utf-8")
    with pytest.raises(ValueError, match="hashes"):
        embeddings.model_snapshot("model", REVISION, str(tmp_path))


def test_unavailable_required_learned_snapshot_has_no_hash_fallback(monkeypatch):
    def missing(*args, **kwargs):
        assert kwargs["local_files_only"] is True
        raise OSError("Pinned files are not staged")

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=missing))
    embeddings.reset_embedding_backend_cache()
    try:
        with embeddings.use_embedding_backend_config(embeddings.EmbeddingBackendConfig("minilm", "missing", 384, revision=REVISION)):
            with pytest.raises(OSError, match="not staged"):
                embeddings.get_embedding_backend()
    finally:
        embeddings.reset_embedding_backend_cache()


def test_cross_encoder_loader_cache_identity_and_cpu_placement(monkeypatch):
    calls = []
    monkeypatch.setattr(reranker_module, "model_snapshot", lambda model, revision, local: (model, {"revision": revision}))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda threads: None))
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(
        CrossEncoder=lambda model, **kwargs: calls.append(kwargs) or SimpleNamespace(device=kwargs["device"])))
    reranker_module._cross_encoder_model.cache_clear()
    try:
        model = reranker_module._cross_encoder_model("fake", REVISION, "cpu", None, 2)
        assert reranker_module._cross_encoder_model("fake", REVISION, "cpu", None, 2) is model
        reranker_module._cross_encoder_model("fake", "b" * 40, "cpu", None, 2)
        reranker_module._cross_encoder_model("fake", REVISION, "cuda:0", None, 2)
        assert len(calls) == 3 and calls[0]["device"] == "cpu"
        assert all(call["local_files_only"] and not call["trust_remote_code"] for call in calls)
    finally:
        reranker_module._cross_encoder_model.cache_clear()


def test_changed_model_identity_requires_explicit_in_memory_rebuild(monkeypatch):
    monkeypatch.setattr(embeddings, "_sentence_transformer_model", lambda *args, **kwargs: FakeSentenceModel())
    old = embeddings.EmbeddingBackendConfig("minilm", "fake", 3, revision=REVISION)
    repository = InMemoryMemoryRepository()
    pipeline = CanonicalEmbeddingPipeline()
    with use_embedding_pipeline(pipeline, old):
        note = repository.add(MemoryNote.create("Green tea", timestamp=datetime(2023, 5, 8, tzinfo=UTC), context={"source_turn_id": "D1:1"}))
        old_hash = current_embedding_metadata()["backend_fingerprint"]
        assert repository.query((1, 0, 0))
    with use_embedding_pipeline(pipeline, replace(old, revision="b" * 40)):
        assert current_embedding_metadata()["backend_fingerprint"] != old_hash
        with pytest.raises(RuntimeError, match="fingerprint mismatch"):
            repository.query((1, 0, 0))
        assert rebuild_embeddings(repository) == 1
        rebuilt = repository.query((1, 0, 0))[0][0]
        assert rebuilt.timestamp == note.timestamp and rebuilt.context == note.context
        with pytest.raises(ValueError, match="dimension"):
            repository.query((1, 0))


@pytest.mark.parametrize("scores", [[], [1], [float("nan"), 2], [float("inf"), 2]])
def test_cross_encoder_never_falls_back_for_invalid_scores(scores):
    ranker = Reranker(backend="cross_encoder", batch_size=2,
                     model_loader=lambda name: SimpleNamespace(predict=lambda pairs, **kwargs: scores))
    results = [RetrievalResult(MemoryNote.create("text", embedding=(1, 0), note_id=str(index)), 0.5) for index in range(2)]
    with pytest.raises(RuntimeError):
        ranker.rerank(results, RetrievalRequest("query"))


def test_empty_constant_fusion_and_cross_encoder_ties_are_deterministic():
    results = [RetrievalResult(MemoryNote.create("text", embedding=(1, 0), note_id=identity), 0.5) for identity in ("b", "a")]
    fusion = HybridFusion()
    assert fusion.fuse({"dense": []}) == []
    assert [item.note.id for item in fusion.fuse({"dense": results})] == ["a", "b"]
    ranker = Reranker(backend="cross_encoder", model_loader=lambda name: SimpleNamespace(predict=lambda pairs, **kwargs: [3, 3]))
    assert [item.note.id for item in ranker.rerank(results, RetrievalRequest("query"))] == ["a", "b"]
    with pytest.raises(ValueError, match="finite"):
        HybridFusionConfig({"dense": float("nan")})


def test_historical_relative_time_uses_source_anchor_and_missing_anchor_is_neutral():
    analysis = replace(QueryAnalyzer().analyze("What happened last year?"), historical=True, reference_time=datetime(2023, 8, 1, tzinfo=UTC))
    assert temporal_agreement(analysis, datetime(2022, 4, 1, tzinfo=UTC)) == 1
    assert temporal_agreement(analysis, datetime(2023, 4, 1, tzinfo=UTC)) == 0
    assert temporal_agreement(replace(analysis, reference_time=None), datetime(2022, 4, 1, tzinfo=UTC)) == 0.5
    repository = InMemoryMemoryRepository()
    repository.add(MemoryNote.create("Historical source", timestamp=datetime(1990, 1, 1, tzinfo=UTC), embedding=(1, 0)))
    request = RetrievalRequest("What happened last year?", analyzed_query=replace(analysis, reference_time=None), historical=True)
    result = TemporalRetrievalStrategy().retrieve(request, repository)
    assert len(result) == 1 and result[0].explanation["age_hours"] is None


def test_sparse_only_reports_stage_times_without_embedding_a_query():
    repository = InMemoryMemoryRepository()
    repository.add(MemoryNote.create("Green tea", embedding=(1, 0)))
    controller = RetrievalController(repository, strategy_names=("sparse",), reranker=Reranker(backend="disabled"))
    response = controller.retrieve(RetrievalRequest("green tea", profile="simple_rag"))
    assert response.diagnostics["executed_strategies"] == ["sparse"]
    assert response.diagnostics["timings_ms"]["embedding"] == 0.0
    assert all(value >= 0 for value in response.diagnostics["timings_ms"].values())


def test_typed_auxiliary_config_roundtrip_and_identity_changes():
    config = RuntimeConfig(embedding_backend="minilm", embedding_dimensions=384, embedding_revision=REVISION,
                           embedding_device="cpu", embedding_batch_size=8, retrieval_strategies=("dense", "sparse"),
                           reranker_revision="b" * 40, reranker_batch_size=4)
    assert RuntimeConfig.model_validate_json(config.model_dump_json()) == config
    override = embeddings.EmbeddingBackendConfig.from_runtime_config(config)
    with embeddings.use_embedding_backend_config(override):
        changed = embeddings.embedding_backend_config(3)
        assert changed.dimensions == 3 and changed.revision == REVISION and changed.batch_size == 8
    assert config.fingerprint != config.model_copy(update={"embedding_device": "cuda:0"}).fingerprint


def test_development_manifest_is_disjoint_and_ladder_uses_full_histories(tmp_path):
    raw = [{"sample_id": identity, "conversation": {"speaker_a": "A", "speaker_b": "B", "session_1_date_time": "2023-05-08",
            "session_1": [{"speaker": "A", "dia_id": f"D1:{index}", "text": f"Green tea source {index}"} for index in range(1, 4)]},
            "qa": [{"id": "q", "question": "What tea?", "answer": "green", "category": 4, "evidence": ["D1:1"]}]}
           for identity in ("conv-a", "conv-b")]
    dataset = tmp_path / "data.json"
    dataset.write_text(json.dumps(raw), encoding="utf-8")
    manifest = development_manifest(dataset)
    assert set(manifest["development_conversation_ids"]).isdisjoint(manifest["validation_conversation_ids"])
    configs = {name: RuntimeConfig(retrieval_strategies=strategies).model_dump(mode="json")
               for name, strategies in (("sparse", ("sparse",)), ("hash", ("dense", "sparse")))}
    report = run_development(dataset, manifest, configs, tmp_path / "out")
    assert all(row["question_count"] == 1 for row in report["ledger"].values())
    traces = [json.loads(line) for line in (tmp_path / "out/traces.jsonl").read_text().splitlines()]
    assert all(row["replayed_turns"] == 3 for row in traces)
    assert traces[1]["replay_cache_hit"] is True
    assert report["production_promotion"] is False
    with pytest.raises(ValueError, match="identity"):
        run_development(dataset, {**manifest, "selected_question_ids": []}, configs, tmp_path / "other")
