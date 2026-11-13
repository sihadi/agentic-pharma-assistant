"""Structured JSON logging (one event per line) that never leaks secrets."""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import datetime, timezone
from typing import Any

# Attributes every LogRecord has; anything else was passed through ``extra``.
_STANDARD_ATTRS = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}
_SECRET_RE = re.compile(r"(sk-[A-Za-z0-9_\-]{8,}|Bearer\s+[A-Za-z0-9._\-]+)")


def _redact(value: str) -> str:
    return _SECRET_RE.sub("[REDACTED]", value)


class JsonFormatter(logging.Formatter):
    """Serialize the message plus *all* ``extra`` fields as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_ATTRS and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return _redact(json.dumps(payload, ensure_ascii=False, default=str))


def configure_logging(level: str = "INFO") -> None:
    """Install one JSON stdout handler; quiet down noisy HTTP libraries."""

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)
    for noisy in ("httpx", "httpcore", "openai", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
