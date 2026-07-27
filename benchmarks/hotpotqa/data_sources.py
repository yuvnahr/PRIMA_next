"""Supported HotpotQA validation/evaluation sources."""
from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
from benchmarks.hotpotqa.config import CONTEXT_SOURCES, ROOT

@dataclass(frozen=True)
class HotpotQADataSource:
    dataset_set: str
    label: str
    parquet_url: str
    filename: str
    mode: str
    context_source: str

    @property
    def default_path(self) -> Path:
        return ROOT / "data" / self.filename

DATA_SOURCES = {
    "distractor": HotpotQADataSource(
        "distractor", "Distractor validation/evaluation",
        "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/distractor/validation-00000-of-00001.parquet",
        "hotpot_dev_distractor_v1.json", "distractor", CONTEXT_SOURCES["distractor"],
    ),
    "fullwiki": HotpotQADataSource(
        "fullwiki", "Fullwiki validation/evaluation",
        "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/fullwiki/validation-00000-of-00001.parquet",
        "hotpot_dev_fullwiki_v1.json", "fullwiki", CONTEXT_SOURCES["fullwiki"],
    ),
}

def get_data_source(dataset_set: str) -> HotpotQADataSource:
    try:
        return DATA_SOURCES[dataset_set]
    except KeyError as exc:
        raise ValueError("dataset_set must be 'distractor' or 'fullwiki'") from exc

def choose_dataset_set() -> str:
    if not sys.stdin.isatty():
        raise ValueError("--dataset-set is required when stdin is not interactive")
    print("Select HotpotQA evaluation dataset:")
    for index, source in enumerate(DATA_SOURCES.values(), 1):
        print(f"  {index}. {source.label}")
    choice = input("Selection [1-2]: ").strip()
    if choice not in {"1", "2"}:
        raise ValueError("Selection must be 1 or 2")
    return tuple(DATA_SOURCES)[int(choice) - 1]
