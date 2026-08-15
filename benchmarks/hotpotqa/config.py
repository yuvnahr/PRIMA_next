"""HotpotQA benchmark defaults."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = Path(os.getenv("HOTPOTQA_OUTPUT_PATH", ROOT / "outputs"))
PROVIDER = os.getenv("HOTPOTQA_PROVIDER", os.getenv("PRIMA_LLM_PROVIDER", "ollama"))
MODEL = os.getenv("HOTPOTQA_MODEL", os.getenv("PRIMA_LLM_MODEL", "qwen3.5:4b"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
TOP_K = int(os.getenv("HOTPOTQA_TOP_K", "5"))
MAX_HOPS = int(os.getenv("HOTPOTQA_MAX_HOPS", "3"))
REASONING_MODE = os.getenv("HOTPOTQA_REASONING_MODE", "adaptive")
PARALLEL_WORKERS = int(os.getenv("HOTPOTQA_PARALLEL_WORKERS", "1"))
SEED = int(os.environ["HOTPOTQA_SEED"]) if "HOTPOTQA_SEED" in os.environ else None
CONTEXT_SOURCES = {
    "distractor": "supplied_distractor_context",
    "official_retrieved": "official_retrieved_context",
    "oracle": "oracle_gold_context",
}
CONTEXT_ALIASES = {"fullwiki": "official_retrieved"}
