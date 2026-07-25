# Phase 13 — HotpotQA integration summary

## Implementation

Added `benchmarks/hotpotqa/{config,loader,adapter,runner,evaluate,checkpoint,experiment}.py`, package exports, focused tests, mode-separated output placeholders, and complete usage documentation. Reused shared conversation/result models, `GenericBenchmarkRunner`, `PrimaRuntimeAdapter`, production `PrimaRuntime`, and the unchanged `ReasoningController`.

The only shared/production changes are benchmark-neutral: the runtime now exposes `ingest_document(text, metadata)`, the shared adapter can select document ingestion and records timings, evidence serialization retains generic source provenance, and the generic runner strips answers/evaluator metadata from inference questions before restoring that metadata to results.

## Strategies

- Ingestion: one sentence per semantic document. This preserves exact sentence provenance and avoids a speculative paragraph-to-sentence selector; the trade-off is more retrieval units than paragraph ingestion.
- Provenance: `document_id`, `sample_id`, exact `source_title`, `paragraph_index`, integer `sentence_id`, `original_sentence_text`, and `benchmark_source` travel in generic memory context and returned evidence references. No answer, gold flag, supporting membership, or relevance label is stored.
- Runtime: `PrimaRuntimeAdapter(document_ingestion=True)` calls public `PrimaRuntime.ingest_document`, then public `answer_question(provider, model, top_k, reasoning_mode, max_hops)`.
- Answer prediction: the complete structured `AnswerResult.answer`/`final_response`; no gold-aware rewriting.
- Supporting facts: direct projection only from evidence retained by production reasoning. Each prediction records source evidence ID, hop, and query.
- Isolation: one adapter/runtime per sample plus shared runner reset; failures become checkpoint records and do not stop later samples.

## Validation

Commands run:

```powershell
python -m py_compile benchmarks\common\runtime_adapter.py benchmarks\common\runner.py runtime\prima_runtime.py reasoning\models.py reasoning\evidence_integrator.py benchmarks\hotpotqa\*.py tests\benchmarks\test_hotpotqa.py
python -m pytest -q tests\benchmarks\test_hotpotqa.py
python -m pytest -q tests\reasoning tests\integration\test_runtime_pipeline.py
```

Results: HotpotQA focused suite `8 passed, 1 skipped`; affected reasoning/runtime suite `17 passed`; compile and `git diff --check` passed. Synthetic end-to-end validation uses a fake runtime and requires no Ollama. The opt-in local test is guarded by `HOTPOTQA_OLLAMA_SMOKE=1`.

## Smoke status and limitations

No live Ollama samples were launched: the local check found neither a listed `qwen3.5:4b` model nor official Hotpot JSON files under `external/`. No full benchmark, LoCoMo run, Wikipedia indexing, or unrelated repository campaign was launched.

Known limitation: sentence-level ingestion increases document count. `checkpoint_every` is accepted for compatibility, but records are deliberately flushed after every sample, which is safer than waiting for a larger interval. Retrieval/reflection counts use public trace/result fields only; production adaptive mode may expose only its bounded trace summary.

## Exact local commands

```powershell
python -m benchmarks.hotpotqa.experiment --mode oracle --dataset-path PATH\hotpot_dev_distractor_v1.json --max-samples 1 --provider ollama --model qwen3.5:4b --progress
python -m benchmarks.hotpotqa.experiment --mode distractor --dataset-path PATH\hotpot_dev_distractor_v1.json --max-samples 1 --provider ollama --model qwen3.5:4b --progress
python -m benchmarks.hotpotqa.experiment --mode fullwiki --dataset-path PATH\hotpot_dev_fullwiki_v1.json --max-samples 1 --provider ollama --model qwen3.5:4b --progress
python -m benchmarks.hotpotqa.experiment --mode distractor --dataset-path PATH\hotpot_dev_distractor_v1.json --max-samples 5 --provider ollama --model qwen3.5:4b --progress
```

Deviation: sentence rather than paragraph documents were selected to provide exact supporting-fact provenance. Full trace capture was not added because forcing production diagnostic mode would change the requested adaptive benchmark semantics.

