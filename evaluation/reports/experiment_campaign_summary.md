# PRIMA Experimental Campaign Summary — Subset Validation

## Status

**PASS as preliminary subset validation; not final full-dataset evidence.**

## Methodology

All three campaigns used one LoCoMo conversation with 197 answerable queries. Memories were created through `CanonicalEmbeddingPipeline.create_memory_note()`, queries through `RetrievalRequest`, and candidate generation through the existing dense retrieval strategy. No retrieval algorithm or metric implementation was changed.

## Controlled variables

Dataset subset, memory construction, query set, dense retrieval, `TOP_K=30`, software environment, and hardware were held constant within each campaign. Each campaign changed exactly one configuration variable.

## Independent variables

- Campaign I: embedding backend.
- Campaign II: representation mode.
- Campaign III: identity normalization enabled/disabled.

## Findings

- Best embedding backend on this subset: **Nomic**.
- Best representation on this subset: **Raw**.
- Identity normalization: **mixed impact**; Recall@1 rose slightly, while broader ranking and candidate metrics declined.

## Runtime and reproducibility

Python 3.10.11; Windows; NVIDIA GeForce GTX 1650; CUDA-enabled Torch 2.13.0+cu126; seed 13; `TOP_K=30`. Result files record execution times, configuration, and backend fingerprints.

## Limitations and readiness

The subset contains one of ten LoCoMo conversations. These results are not sufficient for final scientific claims, statistical significance, or final LoCoMo evaluation. Full-dataset reruns are required before publication or model-selection decisions.
