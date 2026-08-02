import logging
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.common.agent import BenchmarkAgent
from benchmarks.common.interfaces import AgentResponse, BenchmarkResult, Conversation, ConversationQuestion
from benchmarks.common.runner import GenericBenchmarkRunner
from benchmarks.common.utils import configure_benchmark_logger
from benchmarks.locomo.adapter import LoCoMoAdapter
from benchmarks.locomo.evaluate import (
    LoCoMoEvaluator,
    evidence_summary,
    exact_match_score,
    f1_score,
    localized_failure_type,
    rouge_l_score,
)
from llm import provider as provider_module
from llm.llm_types import LLMRequest
from llm.provider import OllamaProvider, ProviderError
from llm.response_parser import extract_answer
from memory.embedding_pipeline import current_embedding_metadata
from memory.experiment_config import EmbeddingExperimentConfig
from memory.maintenance.memory_importance import MemoryImportanceEngine
from memory.memory_note import MemoryNote
from memory.memory_repository import InMemoryMemoryRepository
from runtime.prima_runtime import PrimaRuntime


def test_production_qa_correctness_contract() -> None:
    conversation = LoCoMoAdapter().adapt(
        [{"conversation": {}, "qa": [{"question": "Unknown?", "category": 5, "adversarial_answer": "made up"}]}]
    )[0]
    timestamp = datetime(2022, 1, 21, tzinfo=timezone.utc)
    note = MemoryNote.create("dated memory", embedding=[0.0], timestamp=timestamp)

    assert conversation.questions[0].answer == "No information available"
    assert extract_answer('{"answer": null, "insufficient_information": true}') == "No information available"
    assert extract_answer('{"answer": "null"}') == "No information available"
    assert note.timestamp == timestamp
    assert exact_match_score("The Friday!", "friday") == 1.0
    assert f1_score("cats cats", "cats") == 2 / 3
    assert rouge_l_score("finished screenplay", "she finished screenplay") == 0.8

    result = BenchmarkResult(
        conversation_id="conv",
        question_id="1",
        prompt="What happened?",
        response=AgentResponse(
            "wrong",
            {
                "answer_diagnostics": {
                    "retrieval_stages": {
                        "dense_top30": [{"source_turn_id": "D1:1"}, {"source_turn_id": "D1:2"}],
                        "sparse_top30": [],
                        "fused_top30": [{"source_turn_id": "D1:1"}, {"source_turn_id": "D1:2"}],
                        "reranked_top30": [{"source_turn_id": "D1:1"}, {"source_turn_id": "D1:2"}],
                        "final_candidates": [{"source_turn_id": "D1:1"}],
                    },
                    "stored_source_turn_ids": ["D1:1", "D1:2"],
                    "errors": [],
                }
            },
        ),
        expected_answer="right",
        metadata={"category": "1", "evidence": ["D1:1", "D1:2"]},
    )
    summary = evidence_summary(result)
    assert summary and summary["candidate_evidence_recall"] == 1.0
    assert summary["final_context_evidence_recall"] == 0.5
    assert localized_failure_type(result, summary) == "F_multi_memory_aggregation"

    semantic = current_embedding_metadata(EmbeddingExperimentConfig(representation_mode="semantic", identity_enabled=True))
    raw = current_embedding_metadata(EmbeddingExperimentConfig(representation_mode="raw", identity_enabled=False))
    assert semantic["backend_fingerprint"] != raw["backend_fingerprint"]

    repository = InMemoryMemoryRepository()
    repository.add(MemoryNote.create("Nate likes turtles"))
    importance = MemoryImportanceEngine(repository)
    assert importance.novelty_score("Nate likes turtles") == 0.0
    assert importance.novelty_score("Joanna writes screenplays") == 1.0


def test_bertscore_is_opt_in_and_reports_missing_capability(monkeypatch) -> None:
    result = BenchmarkResult(
        conversation_id="conv",
        question_id="1",
        prompt="question",
        response=AgentResponse("answer"),
        expected_answer="answer",
    )

    disabled = LoCoMoEvaluator().evaluate([result])
    assert disabled["bertscore"] is None
    assert disabled["bertscore_status"] == "disabled"

    monkeypatch.setattr("benchmarks.locomo.evaluate.missing_modules", lambda _modules: ("bert_score",))
    unavailable = LoCoMoEvaluator(include_bertscore=True).evaluate([result])
    assert unavailable["bertscore"] is None
    assert unavailable["bertscore_status"] == "unavailable: bert_score"


def test_benchmark_logger_follows_output_path(tmp_path: Path) -> None:
    logger = configure_benchmark_logger("test.locomo.output", tmp_path / "first")
    logger = configure_benchmark_logger("test.locomo.output", tmp_path / "second")
    files = [Path(handler.baseFilename) for handler in logger.handlers if isinstance(handler, logging.FileHandler)]
    assert files == [tmp_path / "second" / "test_locomo_output.log"]


def test_ollama_timeout_and_retry_contract(monkeypatch) -> None:
    calls: list[int] = []
    payloads: list[dict] = []

    def flaky_post(_url, payload, headers=None, timeout=60):
        calls.append(timeout)
        payloads.append(payload)
        if len(calls) < 3:
            raise ProviderError("temporary failure")
        return {"response": "done"}

    monkeypatch.setenv("PRIMA_LLM_TIMEOUT_SECONDS", "7")
    monkeypatch.setenv("PRIMA_LLM_RETRIES", "2")
    monkeypatch.setenv("PRIMA_ANSWER_SEED", "13")
    monkeypatch.setenv("PRIMA_ANSWER_TEMPERATURE", "0")
    monkeypatch.setattr(provider_module, "post_json", flaky_post)

    response = OllamaProvider(settings=object()).send(LLMRequest(model="qwen3.5:4b", prompt="test"))
    assert response.text == "done"
    assert calls == [7, 7, 7]
    assert payloads[-1]["think"] is False
    assert payloads[-1]["options"]["seed"] == 13
    assert payloads[-1]["options"]["temperature"] == 0.0


def test_production_candidate_pool_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PRIMA_RETRIEVAL_CANDIDATE_POOL_SIZE", "60")
    runtime = PrimaRuntime(log_path=tmp_path / "runtime.log")
    assert runtime.retrieval_controller.candidate_pool_size == 60


def test_runner_reports_each_completed_question() -> None:
    class Agent(BenchmarkAgent):
        def reset(self) -> None:
            pass

        def process_turn(self, turn) -> AgentResponse:
            return AgentResponse("")

        def answer_question(self, question: ConversationQuestion) -> AgentResponse:
            return AgentResponse(question.answer or "")

        def get_state(self) -> dict:
            return {}

        def close(self) -> None:
            pass

    questions = (
        ConversationQuestion(question="one", answer="one", question_id="1"),
        ConversationQuestion(question="two", answer="two", question_id="2"),
    )
    completed: list[BenchmarkResult] = []
    results = GenericBenchmarkRunner().run(Agent(), [Conversation(id="test", turns=(), questions=questions)], completed.append)
    assert [item.question_id for item in completed] == ["1", "2"]
    assert len(results) == 2
