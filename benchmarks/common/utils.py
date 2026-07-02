"""Shared utilities for benchmark integrations."""

from __future__ import annotations

import logging
from pathlib import Path


def ensure_directory(path: Path) -> Path:
    """Create a directory if needed and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_benchmark_logger(name: str, log_dir: Path, level: int = logging.INFO) -> logging.Logger:
    """Create a benchmark logger that writes to a dedicated log directory."""

    ensure_directory(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    log_file = log_dir / f"{name.replace('.', '_')}.log"
    handler_exists = any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_file
        for handler in logger.handlers
    )
    if not handler_exists:
        handler = logging.FileHandler(log_file, encoding="utf-8", delay=True)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)

    return logger
