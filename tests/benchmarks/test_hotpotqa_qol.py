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
from benchmarks.hotpotqa.experiment import resolve_dataset, resolve_mode, run_hotpotqa_experiment, select_conversations
from benchmarks.hotpotqa.checkpoint import read_checkpoint
from tests.benchmarks.test_hotpotqa import FIXTURE, FakeRuntime, write_fixture

ROW = {
    "id": "converted", "question": "Who?", "answer": "Alpha", "type": "bridge", "level": "easy",
    "context": {"title": ["Doc"], "sentences": [["Alpha won."]]},
    "supporting_facts": {"title": ["Doc"], "sent_id": [0]},
}

def install_fake_conversion(monkeypatch, calls: list) -> None:
    monkeypatch.setitem(sys.modules, "pyarrow", types.SimpleNamespace(__version__="19.0.1"))
    monkeypatch.setitem(sys.modules, "datasets", types.SimpleNamespace(load_dataset=lambda *args, **kwargs: calls.append((args, kwargs)) or [ROW]))

def test_registry_contains_only_validation_sets_and_default_names() -> None:
    assert tuple(DATA_SOURCES) == ("distractor", "fullwiki")
    assert get_data_source("distractor").filename == "hotpot_dev_distractor_v1.json"
    assert get_data_source("fullwiki").filename == "hotpot_dev_fullwiki_v1.json"
    assert "validation" in get_data_source("distractor").parquet_url
    assert all("train" not in source.parquet_url for source in DATA_SOURCES.values())
    with pytest.raises(ValueError):
        get_data_source("train")

def test_conversion_custom_path_reuse_force_and_dependency_error(tmp_path: Path, monkeypatch) -> None:
    output = write_fixture(tmp_path, [FIXTURE[0]])
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "datasets", None)
    assert convert_validation_set("distractor", output) == output
    missing = tmp_path / "missing.json"
    with pytest.raises(RuntimeError, match="pip install datasets pyarrow"):
        convert_validation_set("distractor", missing)
    monkeypatch.delitem(sys.modules, "pyarrow", raising=False); monkeypatch.delitem(sys.modules, "datasets", raising=False)
    calls = []; install_fake_conversion(monkeypatch, calls)
    assert convert_validation_set("fullwiki", output, force=True) == output
    converted = json.loads(output.read_text(encoding="utf-8"))[0]
    assert converted == {"_id": "converted", "question": "Who?", "answer": "Alpha", "type": "bridge", "level": "easy", "context": [["Doc", ["Alpha won."]]], "supporting_facts": [["Doc", 0]]}
    assert calls[0][1]["split"] == "validation"

def test_interactive_and_noninteractive_dataset_selection(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys.stdin, "isatty", lambda: True); monkeypatch.setattr("builtins.input", lambda _: "2")
    assert choose_dataset_set() == "fullwiki"
    menu = capsys.readouterr().out
    assert "1. Distractor" in menu and "2. Fullwiki" in menu and "train" not in menu.lower()
    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(ValueError, match="--dataset-set is required"):
        choose_dataset_set()

def test_dataset_and_mode_resolution(tmp_path: Path, monkeypatch) -> None:
    explicit = write_fixture(tmp_path)
    assert resolve_dataset("fullwiki", explicit)[1] == explicit
    called = []
    prepared = tmp_path / "prepared.json"
    monkeypatch.setattr("benchmarks.hotpotqa.experiment.convert_validation_set", lambda dataset_set, output_path=None, force=False: called.append((dataset_set, output_path, force)) or prepared)
    monkeypatch.setattr(type(get_data_source("distractor")), "default_path", property(lambda self: tmp_path / self.filename))
    assert resolve_dataset("distractor", None) == ("distractor", prepared)
    assert called and resolve_mode(None, "distractor", False) == "distractor"
    assert resolve_mode("oracle", "fullwiki", False) == resolve_mode("oracle", "distractor", False) == "oracle"
    with pytest.raises(ValueError, match="incompatible"):
        resolve_mode("distractor", "fullwiki", False)
    assert resolve_mode("distractor", "fullwiki", True) == "distractor"

def test_sampling_is_local_deterministic_and_sliced() -> None:
    items = [types.SimpleNamespace(id=str(index)) for index in range(20)]
    first = [item.id for item in select_conversations(items, "random", 42, 2, 5)]
    assert first == [item.id for item in select_conversations(items, "random", 42, 2, 5)]
    assert first != [item.id for item in select_conversations(items, "random", 43, 2, 5)]
    assert [item.id for item in select_conversations(items, "sequential", None, 2, 5)] == ["2", "3", "4", "5", "6"]

def test_generated_seed_manifest_and_resume_selection(tmp_path: Path) -> None:
    dataset = write_fixture(tmp_path)
    output = tmp_path / "out"
    result = run_hotpotqa_experiment(mode="distractor", dataset_path=dataset, output_path=output, max_samples=2, seed=None, quiet=True, runtime_factory=FakeRuntime)
    manifest = json.loads((output / "distractor" / "metrics" / "run_manifest.json").read_text(encoding="utf-8"))
    assert isinstance(result["resolved_seed"], int) and manifest["configured_seed"] is None
    ids = manifest["selected_sample_ids"]
    resumed = run_hotpotqa_experiment(mode="distractor", dataset_path=dataset, output_path=output, max_samples=2, seed=None, quiet=True, resume=True, runtime_factory=FakeRuntime)
    assert resumed["resolved_seed"] == result["resolved_seed"]
    assert json.loads((output / "distractor" / "metrics" / "run_manifest.json").read_text(encoding="utf-8"))["selected_sample_ids"] == ids
    assert len(read_checkpoint(output / "distractor" / "raw" / "hotpot_results.jsonl")) == 2
    with pytest.raises(ValueError, match="sampling_strategy"):
        run_hotpotqa_experiment(mode="distractor", dataset_path=dataset, output_path=output, max_samples=2, sampling="sequential", seed=None, quiet=True, resume=True, runtime_factory=FakeRuntime)

def sample_record(prediction: str = "alpha", error: str | None = None) -> dict:
    return {"sample_id": "x", "type": "bridge", "level": "hard", "question": "A very long question " * 8, "prediction": prediction, "expected_answer": "alpha", "supporting_facts": [["Doc", 0]], "gold_supporting_facts": [["Doc", 0]], "runtime_error": error, "failure_stage": "runtime" if error else None, "failure_category": "RUNTIME_ERROR" if error else None, "final_stop_reason": "sufficient", "reasoning_hops": 1, "retrieval_calls": 1, "reflection_interventions": 0, "evidence_count": 1, "total_latency": 1.25}

def test_reporter_success_partial_failure_summary_quiet_and_wrapping(capsys, monkeypatch) -> None:
    monkeypatch.setattr("shutil.get_terminal_size", lambda fallback: types.SimpleNamespace(columns=50))
    reporter = HotpotQATerminalReporter(progress=True)
    config = {"dataset_set": "distractor", "mode": "distractor", "dataset_path": "data.json", "sample_count": 3, "sampling_strategy": "random", "resolved_seed": 42, "provider": "ollama", "model": "qwen", "reasoning_mode": "adaptive", "top_k": 5, "max_hops": 3, "output_dir": "out", "resume": False}
    exact, partial, failed = sample_record(), sample_record("alpha beta"), sample_record(error="RuntimeError: boom")
    reporter.header(config); reporter.result(1, 3, exact); reporter.result(2, 3, partial); reporter.result(3, 3, failed)
    metrics = HotpotQAEvaluator().evaluate([exact, partial, failed]); reporter.summary([exact, partial, failed], metrics, config, {key: key for key in ("predictions", "metrics", "manifest", "checkpoint", "logs")}, 2.5)
    output = capsys.readouterr().out
    assert "HotpotQA Result 1/3" in output and "Answer F1" in output and "66.67%" in output
    assert "Runtime status  : failed" in output and "HotpotQA Aggregate Summary" in output
    assert "✓ CORRECT" in output and output.count("✗ WRONG") == 2
    assert "Questions passed: 1/3" in output
    assert "\x1b[" not in output and max(map(len, output.splitlines())) <= 80
    HotpotQATerminalReporter(quiet=True, progress=True).header(config)
    assert capsys.readouterr().out == ""

def test_single_record_score_matches_aggregate() -> None:
    record = sample_record("alpha beta")
    single = score_hotpot_record(record); aggregate = HotpotQAEvaluator().evaluate([record])
    assert all(single[key] == aggregate[key] for key in single)
def test_json_summary_cli_and_quiet(monkeypatch, capsys, tmp_path: Path) -> None:
    import benchmarks.hotpotqa.experiment as experiment
    captured = {}
    monkeypatch.setattr(experiment, "run_hotpotqa_experiment", lambda **kwargs: captured.update(kwargs) or {"completed": 1})
    monkeypatch.setattr(sys, "argv", ["experiment", "--mode", "distractor", "--dataset-path", str(tmp_path / "data.json"), "--quiet", "--json-summary"])
    experiment.main()
    assert captured["quiet"] is True
    assert json.loads(capsys.readouterr().out) == {"completed": 1}
