"""Structured audit logging for LLM calls and sensitive operations.

This module provides a small wrapper over the standard library `logging`
package to emit JSON-like structured audit records. Integrate with your
centralized logging/observability backend in production.
"""
import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger("prima.audit")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AuditLogger:
    @staticmethod
    def log(event: str, data: dict[str, Any] | None = None) -> None:
        payload = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "event": event,
            "data": data or {},
        }
        # Keep the message a single JSON blob for easy ingestion by log collectors
        logger.info(json.dumps(payload))
