"""Configuration for the LoCoMo benchmark integration."""

from __future__ import annotations

import logging
import os
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = Path(os.getenv("LOCOMO_OUTPUT_PATH", "evaluation/qa"))
DATASET_PATH = Path(os.getenv("LOCOMO_DATASET_PATH", BENCHMARK_ROOT / "external" / "data" / "locomo10.json"))
LOG_PATH = OUTPUT_PATH / "logs"

TOP_K = int(os.getenv("LOCOMO_TOP_K", "5"))
LOG_LEVEL = getattr(logging, os.getenv("LOCOMO_LOG_LEVEL", "INFO").upper(), logging.INFO)
MAX_CONVERSATIONS = int(os.getenv("LOCOMO_MAX_CONVERSATIONS", "1"))
MAX_QUESTIONS = int(os.getenv("LOCOMO_MAX_QUESTIONS", "0"))
SEED = int(os.getenv("LOCOMO_SEED", "13"))
PROVIDER = os.getenv("LOCOMO_PROVIDER", os.getenv("PRIMA_LLM_PROVIDER", "ollama"))
MODEL = os.getenv("LOCOMO_MODEL", os.getenv("PRIMA_LLM_MODEL", ""))
PARALLEL_WORKERS = int(os.getenv("LOCOMO_PARALLEL_WORKERS", "1"))
