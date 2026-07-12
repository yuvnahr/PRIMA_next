# Campaign I — Embedding Ablation

## Methodology

One LoCoMo conversation was used as a reproducible subset validation: 197 answerable queries, identical memories, identical production pipeline, identical dense retrieval, and identical `TOP_K=30`. Only the embedding backend changed.

## Configuration

Compared backends: Stable, MiniLM, BGE Small, and Nomic. Hardware: NVIDIA GeForce GTX 1650 with CUDA-enabled Torch 2.13.0+cu126. Python: 3.10.11. Backend fingerprints are recorded in the result JSON/CSV.

## Results

Nomic was the best-performing learned backend on this subset: Recall@5 `0.329103`, compared with Stable `0.010152`. BGE Small reached `0.157360`; MiniLM reached `0.157783`.

## Gate

**PASS for subset continuation.** This is preliminary evidence only; it is not a full-dataset publication result.

## Limitations

The result covers one of ten LoCoMo conversations. No claim of statistical generalization to the full dataset is made.
