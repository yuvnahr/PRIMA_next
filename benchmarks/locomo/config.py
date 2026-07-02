"""Configuration for the LoCoMo benchmark integration."""

from __future__ import annotations

import logging
import os
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).resolve().parent
OUTPUT_PATH = Path(os.getenv("LOCOMO_OUTPUT_PATH", BENCHMARK_ROOT / "outputs"))
DATASET_PATH = Path(os.getenv("LOCOMO_DATASET_PATH", BENCHMARK_ROOT / "external" / "data" / "locomo10.json"))
LOG_PATH = OUTPUT_PATH / "logs"

TOP_K = int(os.getenv("LOCOMO_TOP_K", "5"))
DEVICE = os.getenv("LOCOMO_DEVICE", "cpu")
MAX_CONTEXT = int(os.getenv("LOCOMO_MAX_CONTEXT", "8192"))
LOG_LEVEL = getattr(logging, os.getenv("LOCOMO_LOG_LEVEL", "INFO").upper(), logging.INFO)
