from __future__ import annotations
import importlib
import json
import os
import sys
import types
from pathlib import Path
import pytest
from benchmarks.common.agent import BenchmarkAgent
from benchmarks.common.interfaces import AgentResponse, Conversation, ConversationQuestion
from benchmarks.common.runner import GenericBenchmarkRunner
from benchmarks.hotpotqa.adapter import HotpotQAAdapter
from benchmarks.hotpotqa.checkpoint import append_checkpoint, read_checkpoint, validate_resume
from benchmarks.hotpotqa.evaluate import HotpotQAEvaluator, answer_scores, project_supporting_facts, supporting_fact_scores, validate_predictions
from benchmarks.hotpotqa.experiment import run_hotpotqa_experiment
from benchmarks.hotpotqa.loader import HotpotQADataset

FIXTURE = [
    {"_id": "ok", "question": "Who won?", "answer": "The Alpha", "type": "bridge", "level": "easy", "supporting_facts": [["Doc A", 1]], "context": [["Doc A", ["Intro.", "Alpha won."]], ["Doc B", ["Distractor."]]]},
    {"_id": "fail", "question": "Fail?", "answer": "no", "supporting_facts": [["Fail Doc", 0]], "context": [["Fail Doc", ["FAIL"]]]},
]

def write_fixture(tmp_path: Path, rows=FIXTURE) -> Path:
    path = tmp_path / "hotpot.json"; path.write_text(json.dumps(rows), encoding="utf-8"); return path

def test_loader_validates_schema_and_duplicate_ids(tmp_path: Path) -> None:
    path = write_fixture(tmp_path)
    assert len(HotpotQADataset(path).load()) == 2
    test_path = write_fixture(tmp_path, [{"_id": "test", "question": "Q?", "context": [["T", ["S."]]]}])
    assert HotpotQADataset(test_path).conversations()[0].questions[0].answer is None
    with pytest.raises(ValueError, match="duplicate sample ID"):
        HotpotQADataset(write_fixture(tmp_path, [FIXTURE[0], FIXTURE[0]])).load()
    with pytest.raises(ValueError, match="context entries"):
        HotpotQADataset(write_fixture(tmp_path, [{"_id": "x", "question": "q", "context": [["bad"]]}])).load()

def test_modes_preserve_provenance_and_isolate_gold() -> None:
    distractor = HotpotQAAdapter("distractor").adapt([FIXTURE[0]])[0]
    fullwiki = HotpotQAAdapter("fullwiki").adapt([FIXTURE[0]])[0]
    oracle = HotpotQAAdapter("oracle").adapt([FIXTURE[0]])[0]
    assert len(distractor.turns) == len(fullwiki.turns) == 3
    assert [(turn.metadata["source_title"], turn.metadata["sentence_id"]) for turn in oracle.turns] == [("Doc A", 1)]
    assert fullwiki.metadata["context_source"] == "official_retrieved_context"
    assert oracle.metadata == {"level": "easy", "benchmark_mode": "oracle", "context_source": "oracle_gold_context", "diagnostic": True}
    assert all("gold" not in json.dumps(turn.metadata).lower() and "answer" not in turn.metadata for turn in distractor.turns)
    assert distractor.turns[1].metadata["original_sentence_text"] == "Alpha won."
    with pytest.raises(ValueError, match="requires supporting_facts"):
        HotpotQAAdapter("oracle").adapt([{"_id": "x", "question": "q", "context": [["T", ["S"]]]}])

class InspectingAgent(BenchmarkAgent):
    def __init__(self): self.question = None
    def reset(self): pass
    def process_turn(self, turn): return AgentResponse("")
    def answer_question(self, question): self.question = question; return AgentResponse("Alpha")
    def get_state(self): return {}
    def close(self): pass

def test_generic_runner_hides_gold_until_after_inference() -> None:
    conversation = HotpotQAAdapter().adapt([FIXTURE[0]])[0]; agent = InspectingAgent()
    result = GenericBenchmarkRunner().run(agent, [conversation])[0]
    assert agent.question.answer is None and agent.question.metadata == {} and agent.question.evidence == ()
    assert result.expected_answer == "The Alpha"
    assert result.metadata["question_metadata"]["supporting_facts"] == [["Doc A", 1]]

def test_official_metrics_match_bundled_evaluator(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "ujson", types.SimpleNamespace(load=json.load))
    official = importlib.import_module("benchmarks.hotpotqa.external.hotpot_evaluate_v1")
    for prediction, gold in [("The Alpha!", "alpha"), ("yes", "no"), ("alpha alpha beta", "alpha beta")]:
        em, f1, precision, recall = answer_scores(prediction, gold)
        assert em == float(official.exact_match_score(prediction, gold))
        assert (f1, precision, recall) == official.f1_score(prediction, gold)
    ours = supporting_fact_scores([["A", 0], ["B", 1]], [["A", 0], ["C", 2]])
    metrics = {key: 0 for key in ("sp_em", "sp_f1", "sp_prec", "sp_recall")}
    official_em, official_precision, official_recall = official.update_sp(metrics, [["A", 0], ["B", 1]], [["A", 0], ["C", 2]])
    assert ours == (official_em, metrics["sp_f1"], official_precision, official_recall)
    rows = [{"expected_answer": "alpha", "prediction": "alpha", "supporting_facts": [["A", 0]], "gold_supporting_facts": [["A", 0]]}]
    evaluated = HotpotQAEvaluator().evaluate(rows)
    assert all(evaluated[key] == 1.0 for key in ("em", "f1", "prec", "recall", "sp_em", "sp_f1", "sp_prec", "sp_recall", "joint_em", "joint_f1", "joint_prec", "joint_recall"))

def test_projection_and_prediction_schema() -> None:
    metadata = {"evidence_references": [{"source_id": "m1", "hop": 0, "query": "q", "provenance": {"source_title": "Exact Title", "sentence_id": 2}}]}
    facts, provenance = project_supporting_facts(metadata)
    assert facts == [["Exact Title", 2]] and provenance[0]["source_id"] == "m1"
    validate_predictions({"answer": {"x": "yes"}, "sp": {"x": facts}})
    with pytest.raises(ValueError): validate_predictions({"answer": {}, "sp": {"x": [["T", "1"]]}})

def test_checkpoint_and_manifest_compatibility(tmp_path: Path) -> None:
    path = tmp_path / "check.jsonl"; append_checkpoint(path, {"sample_id": "x"})
    assert read_checkpoint(path) == [{"sample_id": "x"}]
    manifest = {"dataset_fingerprint": "a", "benchmark_mode": "distractor", "provider": "ollama", "model": "m", "reasoning_mode": "adaptive", "top_k": 5, "max_hops": 3, "seed": 13}
    validate_resume(manifest, dict(manifest))
    with pytest.raises(ValueError, match="model"): validate_resume(manifest, manifest | {"model": "other"})

class FakeResult:
    final_response = "alpha"
    def __init__(self, provenance): self.provenance = provenance
    def to_dict(self):
        return {"answer": "alpha", "hop_count": 1, "stop_reason": "sufficient", "errors": [], "trace_summary": [{"type": "RetrievalStarted"}], "evidence_references": [{"source_id": "m", "hop": 0, "query": "q", "provenance": self.provenance}]}
class FakeDocument:
    final_response = ""
    def to_dict(self): return {}
class FakeRuntime:
    instances = []
    def __init__(self, log_path=None): self.docs = []; self.__class__.instances.append(self)
    def ingest_document(self, text, metadata=None): self.docs.append((text, metadata)); return FakeDocument()
    def answer_question(self, question, **options):
        if any(text.endswith(": FAIL") for text, _ in self.docs): raise RuntimeError("synthetic failure")
        return FakeResult(self.docs[-1][1])

def test_synthetic_end_to_end_isolated_and_continues_failures(tmp_path: Path) -> None:
    FakeRuntime.instances = []
    result = run_hotpotqa_experiment(mode="distractor", dataset_path=write_fixture(tmp_path), output_path=tmp_path / "out", max_samples=2, runtime_factory=FakeRuntime)
    assert result["completed"] == 2 and len(FakeRuntime.instances) == 4  # adapter construction plus per-sample reset
    output = tmp_path / "out" / "distractor"
    rows = read_checkpoint(output / "raw" / "hotpot_results.jsonl")
    assert len(rows) == 2 and sum(bool(row["runtime_error"]) for row in rows) == 1
    assert json.loads((output / "raw" / "hotpot_predictions.json").read_text())["sp"]["ok"] == [["Doc B", 0]]
    assert json.loads((output / "metrics" / "run_manifest.json").read_text())["context_source"] == "supplied_distractor_context"
    # A compatible resume skips both completed IDs; an incompatible one is rejected.
    resumed = run_hotpotqa_experiment(mode="distractor", dataset_path=tmp_path / "hotpot.json", output_path=tmp_path / "out", max_samples=2, resume=True, runtime_factory=FakeRuntime)
    assert resumed["completed"] == 2
    with pytest.raises(ValueError, match="model"):
        run_hotpotqa_experiment(mode="distractor", dataset_path=tmp_path / "hotpot.json", output_path=tmp_path / "out", max_samples=2, resume=True, model="other", runtime_factory=FakeRuntime)

def test_hotpot_code_has_no_network_or_direct_retrieval_and_production_has_no_hotpot_imports() -> None:
    root = Path(__file__).parents[2]
    hotpot = "\n".join(path.read_text(encoding="utf-8") for path in (root / "benchmarks" / "hotpotqa").glob("*.py"))
    assert "wikipedia" not in hotpot.lower() and ".retrieve(" not in hotpot and "RetrievalController" not in hotpot
    production = "\n".join(path.read_text(encoding="utf-8") for folder in ("runtime", "reasoning", "memory") for path in (root / folder).glob("*.py"))
    assert "benchmarks.hotpotqa" not in production

@pytest.mark.skipif(os.getenv("HOTPOTQA_OLLAMA_SMOKE") != "1", reason="set HOTPOTQA_OLLAMA_SMOKE=1 for the opt-in local Ollama smoke test")
def test_opt_in_ollama_smoke_is_configured() -> None:
    assert os.getenv("PRIMA_LLM_MODEL", "qwen3.5:4b")
