import json
from pathlib import Path

from datasets import load_dataset
import pyarrow

if pyarrow.__version__ == "19.0.0":
    raise RuntimeError("PyArrow 19.0.0 cannot read these Parquet files. Run: py -m pip install pyarrow==19.0.1")

PARQUET_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet"
)

dataset = load_dataset(
    "parquet",
    data_files={"validation": PARQUET_URL},
    split="validation",
)

records = []
for row in dataset:
    records.append({
        "_id": row["id"],
        "question": row["question"],
        "answer": row["answer"],
        "type": row["type"],
        "level": row["level"],
        "context": [
            [title, sentences]
            for title, sentences in zip(
                row["context"]["title"],
                row["context"]["sentences"],
            )
        ],
        "supporting_facts": [
            [title, int(sentence_id)]
            for title, sentence_id in zip(
                row["supporting_facts"]["title"],
                row["supporting_facts"]["sent_id"],
            )
        ],
    })

output = Path(__file__).parent / "data" / "hotpot_dev_distractor_v1.json"
output.parent.mkdir(parents=True, exist_ok=True)
temporary = output.with_suffix(".json.tmp")
with temporary.open("w", encoding="utf-8") as file:
    json.dump(records, file, ensure_ascii=False)
temporary.replace(output)
print(f"Wrote {len(records)} samples to {output}")

