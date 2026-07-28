# Phase 12 adaptive reasoning summary

Added one benchmark-independent `reasoning` package and integrated it into `PrimaRuntime.answer_question`. Every hop routes through the existing `RetrievalController`; no retrieval, embedding, benchmark, or external-submodule code changed.

The public entry point now returns `AnswerResult`, with `final_response`, `text`, and string conversion compatibility accessors. `single_pass` is a one-call ablation and `PRIMA_REASONING_MODE=single_pass` is the rollback path. Request-local evidence includes source, score, query, hop, and strategy metadata; it is never written through memory admission.

Validation commands: `python -m py_compile runtime/prima_runtime.py reasoning/*.py`, `python -m reasoning.validate_controller`, and `python -m pytest -q tests/reasoning`. Generated validation JSON is ignored. The deterministic checks cover one-pass parity, two-hop bridge acquisition, no-progress detection, contradiction abstention, provenance, and safe traces.

Known limitation: the first controller uses deterministic term support and entity extraction, not an LLM decomposition policy. It intentionally abstains when it cannot derive a grounded follow-up query; add a structured LLM implementation only after a validated schema/retry contract is available.
