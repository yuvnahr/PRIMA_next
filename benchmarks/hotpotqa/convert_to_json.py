"""Prepare supported HotpotQA validation sets as official-style JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from benchmarks.hotpotqa.data_sources import DATA_SOURCES, choose_dataset_set, get_data_source
from benchmarks.hotpotqa.loader import HotpotQADataset


def _row_to_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "_id": row["id"], "question": row["question"], "answer": row["answer"],
        "type": row["type"], "level": row["level"],
        "context": [[title, sentences] for title, sentences in zip(row["context"]["title"], row["context"]["sentences"], strict=True)],
        "supporting_facts": [[title, int(sentence_id)] for title, sentence_id in zip(row["supporting_facts"]["title"], row["supporting_facts"]["sent_id"], strict=True)],
    }

def convert_validation_set(dataset_set: str, output_path: str | Path | None = None, force: bool = False) -> Path:
    source = get_data_source(dataset_set)
    output = Path(output_path) if output_path is not None else source.default_path
    if output.exists() and not force:
        try:
            HotpotQADataset(output, source.mode).load()
            return output
        except ValueError:
            pass
    try:
        import pyarrow
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Dataset preparation requires: py -m pip install datasets pyarrow==19.0.1") from exc
    if pyarrow.__version__ == "19.0.0":
        raise RuntimeError("PyArrow 19.0.0 cannot read these Parquet files. Run: py -m pip install pyarrow==19.0.1")
    rows = [_row_to_record(row) for row in load_dataset("parquet", data_files={"validation": source.parquet_url}, split="validation", revision="14f0ace3c3fac7bd86149c616b5b05d8282e5c6a")]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    try:
        temporary.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        HotpotQADataset(temporary, source.mode).load()
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a HotpotQA validation/evaluation JSON file.")
    parser.add_argument("--dataset-set", choices=tuple(DATA_SOURCES))
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        dataset_set = args.dataset_set or choose_dataset_set()
        output = convert_validation_set(dataset_set, args.output_path, args.force)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Prepared {dataset_set} validation data: {output}")

if __name__ == "__main__":
    main()
