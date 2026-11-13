"""Single gateway to the Syntax LiteLLM proxy (OpenAI-compatible SDK).

Every model call in the project goes through this class, which gives us one
place for: timeouts/retries, structured outputs, usage accounting in the trace
and a test seam (unit tests inject a fake client).
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

import numpy as np
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from app.config import Settings
from app.tracing import Tracer

LOGGER = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
EMBED_BATCH = 64
EMBED_WORKERS = 4


class LLMError(RuntimeError):
    """Raised when the model cannot produce a usable response."""


class LLMGateway:
    """Thin, typed wrapper over chat completions and embeddings."""

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self.client = client or OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.llm_timeout_s,
            # The LiteLLM proxy answers 429 "deployments in cooldown" under bursts;
            # the SDK retries 408/409/429/5xx with exponential backoff + Retry-After.
            max_retries=6,
        )

    # ------------------------------------------------------------------ embeddings
    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed texts in batches; return an L2-normalised float32 matrix.

        Normalising lets FAISS inner product compute cosine similarity.
        """

        if not texts:
            raise ValueError("Nothing to embed")

        def embed_batch(batch: list[str]) -> list[list[float]]:
            response = self.client.embeddings.create(model=self.settings.embedding_model, input=batch)
            return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]

        # The proxy's latency grows linearly with batch size (measured ~90 ms/text),
        # so a few concurrent batches cut index build time ~4x. Order is preserved.
        batches = [texts[i : i + EMBED_BATCH] for i in range(0, len(texts), EMBED_BATCH)]
        with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
            vectors = [v for batch_vectors in pool.map(embed_batch, batches) for v in batch_vectors]
        matrix = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.clip(norms, 1e-12, None)

    # ------------------------------------------------------------------ chat
    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[T],
        step: str,
        tracer: Tracer | None = None,
    ) -> T:
        """Return a validated Pydantic object.

        First choice: strict JSON-schema structured outputs (the proxy supports
        them - verified during exploration). Fallback: JSON mode + validation,
        so the system degrades gracefully on proxies lacking strict schemas.
        """

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        started = time.perf_counter()
        try:
            response = self.client.beta.chat.completions.parse(
                model=self.settings.chat_model,
                messages=messages,
                response_format=schema,
                temperature=0,
            )
            parsed = response.choices[0].message.parsed
            self._record(tracer, step, started, response)
            if parsed is not None:
                return parsed
            content = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001 - fall back to JSON mode
            LOGGER.warning("structured_output_fallback", extra={"step": step, "error": str(exc)[:200]})
            started = time.perf_counter()
            response = self.client.chat.completions.create(
                model=self.settings.chat_model,
                messages=messages
                + [{"role": "system", "content": "Return only a JSON object matching: "
                    + json.dumps(schema.model_json_schema())}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            self._record(tracer, step, started, response)
            content = response.choices[0].message.content or ""
        try:
            return schema.model_validate(_parse_json(content))
        except (ValueError, ValidationError) as exc:
            raise LLMError(f"{step}: invalid structured response") from exc

    def tool_turn(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str,
        step: str,
        tracer: Tracer | None = None,
    ) -> Any:
        """One native function-calling turn; returns the assistant message."""

        started = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.settings.chat_model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            parallel_tool_calls=True,
            temperature=0,
        )
        self._record(tracer, step, started, response)
        return response.choices[0].message

    def _record(self, tracer: Tracer | None, step: str, started: float, response: Any) -> None:
        if tracer is not None:
            tracer.record_llm_call(
                step,
                self.settings.chat_model,
                round((time.perf_counter() - started) * 1000, 1),
                getattr(response, "usage", None),
            )


def _parse_json(content: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip()).strip()
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object")
    return parsed
