from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from benchmarks.hotpotqa.console import HotpotQATerminalReporter
from benchmarks.hotpotqa.convert_to_json import convert_validation_set
from benchmarks.hotpotqa.data_sources import DATA_SOURCES, choose_dataset_set, get_data_source
from benchmarks.hotpotqa.evaluate import HotpotQAEvaluator, score_hotpot_record
from benchmarks.hotpotqa.experiment import resolve_dataset, resolve_mode, select_conversations
from tests.benchmarks.test_hotpotqa import FIXTURE, write_fixture

ROW = {
    "id": "converted", "question": "Who?", "answer": "Alpha", "type": "bridge",
    "level": "easy", "context": {"title": ["Doc"], "sentences": [["Alpha won."]]},
    "supporting_facts": {"title": ["Doc"], "sent_id": [0]},
}


def install_fake_conversion(monkeypatch, calls: list) -> None:
    monkeypatch.setitem(sys.modules, "pyarrow", types.SimpleNamespace(__version__="19.0.1"))
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(
        load_dataset=lambda *args, **kwargs: calls.append((args, kwargs)) or [ROW],
    ))


def test_registry_contains_only_validation_sets_and_accurate_modes() -> None:
    assert tuple(DATA_SOURCES) == ("distractor", "fullwiki")
    assert get_data_source("distractor").mode == "distractor"
    assert get_data_source("fullwiki").mode == "official_retrieved"
    assert all("train" not in source.parquet_url for source in DATA_SOURCES.values())


def test_conversion_reuse_force_and_dependency_error(tmp_path: Path, monkeypatch) -> None:
    output = write_fixture(tmp_path, [FIXTURE[0]])
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "datasets", None)
    assert convert_validation_set("distractor", output) == output
    with pytest.raises(RuntimeError, match="pip install datasets pyarrow"):
        convert_validation_set("distractor", tmp_path / "missing.json")
    monkeypatch.delitem(sys.modules, "pyarrow", raising=False)
    monkeypatch.delitem(sys.modules, "datasets", raising=False)
    calls = []
    install_fake_conversion(monkeypatch, calls)
    assert convert_validation_set("fullwiki", output, force=True) == output
    assert json.loads(output.read_text(encoding="utf-8"))[0]["_id"] == "converted"
    assert calls[0][1]["split"] == "validation"


def test_interactive_and_noninteractive_dataset_selection(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "2")
    assert choose_dataset_set() == "fullwiki"
    assert "2. Fullwiki" in capsys.readouterr().out
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(ValueError, match="--dataset-set is required"):
        choose_dataset_set()


def test_dataset_context_resolution_and_legacy_alias(tmp_path: Path, monkeypatch) -> None:
    explicit = write_fixture(tmp_path)
    assert resolve_dataset("fullwiki", explicit)[1] == explicit
    prepared = tmp_path / "prepared.json"
    monkeypatch.setattr("benchmarks.hotpotqa.experiment.convert_validation_set", lambda *args, **kwargs: prepared)
    monkeypatch.setattr(type(get_data_source("distractor")), "default_path", property(lambda self: tmp_path / self.filename))
    assert resolve_dataset("distractor", None) == ("distractor", prepared)
    assert resolve_mode(None, "distractor", False) == "distractor"
    assert resolve_mode(None, "fullwiki", False) == "official_retrieved"
    assert resolve_mode("fullwiki", "fullwiki", False) == "official_retrieved"
    assert resolve_mode("oracle", "fullwiki", False) == "oracle"
    with pytest.raises(ValueError, match="incompatible"):
        resolve_mode("distractor", "fullwiki", False)


def test_sampling_is_local_deterministic_and_sliced() -> None:
    items = [types.SimpleNamespace(id=str(index)) for index in range(20)]
    first = [item.id for item in select_conversations(items, "random", 42, 2, 5)]
    assert first == [item.id for item in select_conversations(items, "random", 42, 2, 5)]
    assert first != [item.id for item in select_conversations(items, "random", 43, 2, 5)]
    assert [item.id for item in select_conversations(items, "sequential", None, 2, 5)] == ["2", "3", "4", "5", "6"]


def sample_record(prediction: str = "alpha", error: str | None = None) -> dict:
    return {
        "sample_id": "x", "type": "bridge", "level": "hard", "question": "Question",
        "prediction": prediction, "expected_answer": "alpha", "supporting_facts": [["Doc", 0]],
        "gold_supporting_facts": [["Doc", 0]], "runtime_error": error,
        "ingestion_error": None, "execution_failed": bool(error),
        "failure_stage": "runtime" if error else None,
        "failure_category": "RUNTIME_FAILURE" if error else None,
        "final_stop_reason": "sufficient", "reasoning_hops": 1, "retrieval_calls": 1,
        "reflection_interventions": 0, "evidence_count": 1, "total_latency_ms": 1250.0,
    }


def test_reporter_and_single_score(capsys) -> None:
    reporter = HotpotQATerminalReporter(progress=True)
    config = {
        "dataset_set": "distractor", "mode": "distractor", "runtime_profile": "prima_full",
        "dataset_path": "data.json", "sample_count": 2, "sampling_strategy": "sequential",
        "resolved_seed": 42, "provider": "test", "model": "model", "reasoning_mode": "adaptive",
        "top_k": 5, "max_hops": 3, "output_dir": "out", "resume": False,
    }
    exact, failed = sample_record(), sample_record(error="boom")
    metrics = HotpotQAEvaluator().evaluate([exact, failed])
    reporter.header(config)
    reporter.result(1, 2, exact)
    reporter.result(2, 2, failed)
    reporter.summary([exact, failed], metrics, config, {key: key for key in ("predictions", "metrics", "manifest", "checkpoint", "logs")}, 2.5)
    output = capsys.readouterr().out
    assert "Runtime profile" in output and "Execution failures" in output
    assert score_hotpot_record(exact)["joint_em"] == 1.0


def test_json_summary_cli_and_quiet(monkeypatch, capsys, tmp_path: Path) -> None:
    import benchmarks.hotpotqa.experiment as experiment
    captured = {}
    monkeypatch.setattr(experiment, "run_hotpotqa_experiment", lambda **kwargs: captured.update(kwargs) or {"completed": 1})
    monkeypatch.setattr(sys, "argv", ["experiment", "--mode", "distractor", "--dataset-path", str(tmp_path / "data.json"), "--quiet", "--json-summary"])
    experiment.main()
    assert captured["runtime_profile"] == "prima_full" and captured["quiet"] is True
    assert json.loads(capsys.readouterr().out) == {"completed": 1}
