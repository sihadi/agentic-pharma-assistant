"""FastAPI entrypoint.

Endpoints
---------
GET  /              minimal frontend (vanilla HTML/CSS/JS)
GET  /health        always 200 - liveness + readiness details
POST /ask           {"question": "..."} -> cited answer, review, plan, trace id
POST /ask/stream    same, as Server-Sent Events (live pipeline steps for the UI)
GET  /trace/{id}    full structured trace of a past request
GET  /corpus        data-lineage manifest summary (raw -> cleaned -> chunks)

Start-up: the HTTP server is available immediately; the corpus is loaded and
the FAISS index built in a background thread (``/health`` reports
``starting`` meanwhile and ``/ask`` waits for readiness, bounded).
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import APIError

from app.config import ConfigurationError, load_settings
from app.llm import LLMError
from app.logging_config import configure_logging
from app.models import AskResponse, QuestionRequest
from app.runtime import Runtime, build_runtime
from app.tracing import Tracer, TraceStore

LOGGER = logging.getLogger("app")
STATIC_DIR = Path(__file__).resolve().parent / "static"
READY_TIMEOUT_S = 180


class AppState:
    """Process-wide runtime holder, filled by the background initialiser."""

    def __init__(self) -> None:
        self.runtime: Runtime | None = None
        self.traces: TraceStore | None = None
        self.ready = threading.Event()
        self.status = "starting"
        self.error: str | None = None
        self.started_at = time.time()

    def initialise(self) -> None:
        for attempt in range(1, 4):
            try:
                settings = load_settings()
                configure_logging(settings.log_level)
                self.runtime = build_runtime(settings)
                self.traces = TraceStore(settings.trace_dir if settings.persist_traces else None)
                self.status, self.error = "ok", None
                self.ready.set()
                LOGGER.info("runtime_ready", extra={"startup_s": round(time.time() - self.started_at, 1),
                                                    "retriever": self.runtime.retriever.stats})
                return
            except ConfigurationError as exc:  # retrying cannot fix a missing variable
                self.status, self.error = "degraded", str(exc)
                LOGGER.error("runtime_configuration_error", extra={"error": str(exc)})
                return
            except Exception as exc:  # noqa: BLE001 - e.g. proxy temporarily unavailable
                self.error = f"{type(exc).__name__}: {str(exc)[:300]}"
                LOGGER.exception("runtime_initialisation_failed", extra={"attempt": attempt})
                time.sleep(5 * attempt)
        self.status = "degraded"


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    state = AppState()
    app.state.app_state = state
    threading.Thread(target=state.initialise, name="runtime-init", daemon=True).start()
    yield


app = FastAPI(
    title="Agentic Pharma Research Assistant",
    version="2.0.0",
    description="Plan → Retrieve → Tools → Critic → Answer over a local FDA/MedlinePlus corpus. Not medical advice.",
    lifespan=lifespan,
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _state(request: Request) -> AppState:
    return request.app.state.app_state


def _runtime_or_503(request: Request) -> tuple[Runtime, TraceStore]:
    state = _state(request)
    if not state.ready.wait(timeout=READY_TIMEOUT_S if state.status == "starting" else 0):
        raise HTTPException(503, detail=f"Assistant not ready ({state.status}). {state.error or 'Retry shortly.'}")
    assert state.runtime is not None and state.traces is not None
    return state.runtime, state.traces


def _validate_length(runtime: Runtime, body: QuestionRequest) -> None:
    if len(body.question) > runtime.settings.max_question_length:
        raise HTTPException(422, detail=f"Question longer than {runtime.settings.max_question_length} characters")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health(request: Request) -> dict[str, Any]:
    """Always HTTP 200; ``status`` is ok | starting | degraded."""

    state = _state(request)
    body: dict[str, Any] = {"status": state.status, "ready": state.ready.is_set()}
    if state.runtime is not None:
        body["corpus"] = state.runtime.corpus.manifest["totals"] | {"index": "faiss.IndexFlatIP + BM25 (RRF)"}
        body["config"] = state.runtime.settings.public_view()
    if state.error:
        body["error"] = state.error
    return body


@app.post("/ask", response_model=AskResponse)
def ask(body: QuestionRequest, request: Request) -> dict[str, Any]:
    runtime, traces = _runtime_or_503(request)
    _validate_length(runtime, body)
    tracer = Tracer(str(uuid4()), body.question)
    try:
        return runtime.agent.run(body.question, tracer)
    except (APIError, LLMError) as exc:
        tracer.emit("request_failed", status="error", error=type(exc).__name__)
        raise HTTPException(502, detail=f"The language model service failed. Trace: {tracer.trace_id}") from exc
    except Exception as exc:
        LOGGER.exception("ask_failed", extra={"trace_id": tracer.trace_id})
        tracer.emit("request_failed", status="error", error=type(exc).__name__)
        raise HTTPException(500, detail=f"The research request could not be completed. Trace: {tracer.trace_id}") from exc
    finally:
        traces.save(tracer)


@app.post("/ask/stream")
def ask_stream(body: QuestionRequest, request: Request) -> StreamingResponse:
    """Server-Sent Events: one ``step`` event per trace event, then ``result`` (or ``error``)."""

    runtime, traces = _runtime_or_503(request)
    _validate_length(runtime, body)
    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    tracer = Tracer(str(uuid4()), body.question, listener=lambda e: events.put(("step", e)))

    def work() -> None:
        try:
            events.put(("result", runtime.agent.run(body.question, tracer)))
        except Exception as exc:  # noqa: BLE001 - reported to the client as an SSE error event
            LOGGER.exception("ask_stream_failed", extra={"trace_id": tracer.trace_id})
            events.put(("error", {"detail": f"Request failed ({type(exc).__name__}).", "trace_id": tracer.trace_id}))
        finally:
            traces.save(tracer)
            events.put(("done", None))

    threading.Thread(target=work, name=f"ask-{tracer.trace_id[:8]}", daemon=True).start()

    def stream() -> Iterator[str]:
        yield _sse("start", {"trace_id": tracer.trace_id})
        while True:
            kind, payload = events.get()
            if kind == "done":
                return
            yield _sse(kind, payload)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/trace/{trace_id}")
def trace(trace_id: str, request: Request) -> dict[str, Any]:
    state = _state(request)
    found = state.traces.get(trace_id) if state.traces else None
    if found is None:
        raise HTTPException(404, detail="Unknown trace id")
    return found


@app.get("/corpus")
def corpus(request: Request) -> dict[str, Any]:
    runtime, _ = _runtime_or_503(request)
    manifest = runtime.corpus.manifest
    return {
        "pipeline_version": manifest["pipeline_version"],
        "fingerprint": manifest["fingerprint"],
        "totals": manifest["totals"],
        "cleaning_rules": manifest["cleaning_rules"],
        "documents": [
            {k: d[k] for k in ("source_file", "drug", "source_type", "title", "source_url", "aliases",
                               "raw_sha256", "cleaned_sha256", "cleaning_ops", "chunks", "sections")}
            for d in manifest["documents"]
        ],
        "retriever": runtime.retriever.stats,
    }


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"
