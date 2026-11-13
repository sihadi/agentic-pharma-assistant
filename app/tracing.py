"""Per-request execution trace: every agent step becomes a structured event.

A :class:`Tracer` is created for each question and passed to graph nodes via
LangGraph's ``config`` (thread-safe: no global state). Each event is
1) appended to the in-memory trace returned by the API / streamed to the UI,
2) logged as one JSON line on stdout,
3) optionally persisted to ``data/traces/<trace_id>.json``.

Only synthetic summaries are traced (plans, tool names/arguments, chunk ids,
scores, verdicts) - never the model's private reasoning, never secrets.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("agent.trace")


class Tracer:
    """Collects ordered trace events and LLM usage for one request."""

    def __init__(
        self,
        trace_id: str,
        question: str,
        listener: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.trace_id = trace_id
        self.question = question
        self.listener = listener  # e.g. pushes events to the SSE stream
        self.started = time.perf_counter()
        self.events: list[dict[str, Any]] = []
        self.llm_calls: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def emit(self, event: str, *, status: str = "ok", **data: Any) -> dict[str, Any]:
        record = {
            "seq": len(self.events) + 1,
            "event": event,
            "status": status,
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "elapsed_ms": round((time.perf_counter() - self.started) * 1000, 1),
            **data,
        }
        with self._lock:
            self.events.append(record)
        # Nested under one key: LogRecord forbids extras named "message", "name"...
        LOGGER.info(event, extra={"trace_id": self.trace_id, "trace": record})
        if self.listener is not None:
            self.listener(record)
        return record

    @contextmanager
    def span(self, event: str, **data: Any) -> Iterator[dict[str, Any]]:
        """Time a block; fields added to the yielded dict are traced too."""

        started = time.perf_counter()
        fields: dict[str, Any] = dict(data)
        try:
            yield fields
        except Exception as exc:
            fields["error"] = f"{type(exc).__name__}: {exc}"[:500]
            self.emit(event, status="error", duration_ms=_ms(started), **fields)
            raise
        self.emit(event, duration_ms=_ms(started), **fields)

    def record_llm_call(self, step: str, model: str, duration_ms: float, usage: Any) -> None:
        self.llm_calls.append(
            {
                "step": step,
                "model": model,
                "duration_ms": duration_ms,
                "prompt_tokens": getattr(usage, "prompt_tokens", None),
                "completion_tokens": getattr(usage, "completion_tokens", None),
            }
        )

    def usage_summary(self) -> dict[str, Any]:
        return {
            "duration_ms": _ms(self.started),
            "llm_calls": len(self.llm_calls),
            "prompt_tokens": sum(c["prompt_tokens"] or 0 for c in self.llm_calls),
            "completion_tokens": sum(c["completion_tokens"] or 0 for c in self.llm_calls),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "question": self.question,
            "events": self.events,
            "llm_calls": self.llm_calls,
            "usage": self.usage_summary(),
        }


class TraceStore:
    """Bounded in-memory store of recent traces (+ optional JSON files)."""

    def __init__(self, directory: Path | None, capacity: int = 200) -> None:
        self.directory = directory
        self.capacity = capacity
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def save(self, tracer: Tracer) -> None:
        payload = tracer.to_dict()
        with self._lock:
            self._items[tracer.trace_id] = payload
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        if self.directory is not None:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                path = self.directory / f"{tracer.trace_id}.json"
                path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            except OSError:  # read-only FS must never break a request
                LOGGER.warning("trace_persist_failed", extra={"trace_id": tracer.trace_id})

    def get(self, trace_id: str) -> dict[str, Any] | None:
        with self._lock:
            if trace_id in self._items:
                return self._items[trace_id]
        if self.directory is not None:
            path = self.directory / f"{trace_id}.json"
            if path.is_file():
                return json.loads(path.read_text(encoding="utf-8"))
        return None


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
