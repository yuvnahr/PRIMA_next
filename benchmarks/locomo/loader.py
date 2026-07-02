"""Raw LoCoMo dataset loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmarks.common.interfaces import BenchmarkDataset, Conversation
from benchmarks.common.utils import configure_benchmark_logger
from benchmarks.locomo.adapter import LoCoMoAdapter
from benchmarks.locomo.config import DATASET_PATH, LOG_LEVEL, LOG_PATH

logger = configure_benchmark_logger("benchmarks.locomo.loader", LOG_PATH, LOG_LEVEL)


def load_raw_json(dataset_path: Path = DATASET_PATH) -> Any:
    """Validate and load a raw LoCoMo JSON file."""

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"LoCoMo dataset not found: {path}")
    if not path.is_file():
        raise ValueError(f"LoCoMo dataset path is not a file: {path}")

    logger.info("Loading LoCoMo dataset from %s", path)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


class LoCoMoDataset(BenchmarkDataset):
    """Dataset facade that loads LoCoMo JSON and exposes normalized conversations."""

    def __init__(self, dataset_path: Path = DATASET_PATH, adapter: LoCoMoAdapter | None = None) -> None:
        self.dataset_path = Path(dataset_path)
        self.adapter = adapter or LoCoMoAdapter()
        self._raw_data: Any | None = None
        self._conversations: list[Conversation] | None = None

    def load(self) -> Any:
        """Load raw LoCoMo JSON data."""

        self._raw_data = load_raw_json(self.dataset_path)
        self._conversations = None
        return self._raw_data

    def conversations(self) -> list[Conversation]:
        """Return normalized conversations, loading the dataset first if needed."""

        if self._raw_data is None:
            self.load()
        if self._conversations is None:
            self._conversations = self.adapter.adapt(self._raw_data)
        return self._conversations
