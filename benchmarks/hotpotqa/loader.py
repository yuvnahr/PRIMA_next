"""Validated HotpotQA JSON loading."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
from benchmarks.common.interfaces import BenchmarkDataset, Conversation
from benchmarks.hotpotqa.adapter import HotpotQAAdapter

class HotpotQADataset(BenchmarkDataset):
    def __init__(self, dataset_path: str | Path, mode: str = "distractor", adapter: HotpotQAAdapter | None = None) -> None:
        self.dataset_path = Path(dataset_path)
        self.adapter = adapter or HotpotQAAdapter(mode)
        self._records: list[dict[str, Any]] | None = None

    def load(self) -> list[dict[str, Any]]:
        path = self.dataset_path
        if not path.exists():
            raise FileNotFoundError(f"HotpotQA dataset not found: {path}")
        if not path.is_file():
            raise ValueError(f"HotpotQA dataset path is not a file: {path}")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid HotpotQA JSON at {path}: {exc}") from exc
        if not isinstance(data, list):
            raise ValueError(f"Invalid HotpotQA schema at {path}: top-level value must be a list")
        seen = set()
        for index, record in enumerate(data):
            self._validate_record(record, index, seen)
        self._records = data
        return data

    def conversations(self) -> list[Conversation]:
        return self.adapter.adapt(self._records if self._records is not None else self.load())

    def _validate_record(self, record: Any, index: int, seen: set[str]) -> None:
        prefix = f"Invalid HotpotQA schema at {self.dataset_path}, sample {index}"
        if not isinstance(record, dict):
            raise ValueError(f"{prefix}: record must be an object")
        for key in ("_id", "question", "context"):
            if key not in record:
                raise ValueError(f"{prefix}: missing {key!r}")
        if not isinstance(record["_id"], str) or not record["_id"]:
            raise ValueError(f"{prefix}: '_id' must be a non-empty string")
        if record["_id"] in seen:
            raise ValueError(f"{prefix}: duplicate sample ID {record['_id']!r}")
        seen.add(record["_id"])
        if not isinstance(record["question"], str) or not record["question"].strip():
            raise ValueError(f"{prefix}: 'question' must be a non-empty string")
        if not isinstance(record["context"], list):
            raise ValueError(f"{prefix}: 'context' must be a list")
        for paragraph in record["context"]:
            if not isinstance(paragraph, list) or len(paragraph) != 2 or not isinstance(paragraph[0], str) or not isinstance(paragraph[1], list) or not all(isinstance(item, str) for item in paragraph[1]):
                raise ValueError(f"{prefix}: context entries must be [title, [sentences]]")
        if "supporting_facts" in record:
            facts = record["supporting_facts"]
            if not isinstance(facts, list) or any(not isinstance(fact, list) or len(fact) != 2 or not isinstance(fact[0], str) or not isinstance(fact[1], int) or isinstance(fact[1], bool) for fact in facts):
                raise ValueError(f"{prefix}: supporting facts must be [title, sentence_id]")
        if "answer" in record and not isinstance(record["answer"], str):
            raise ValueError(f"{prefix}: 'answer' must be a string")
