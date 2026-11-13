"""Runtime configuration, read exclusively from environment variables.

Design notes
------------
* Nothing is hard-coded: models, endpoint and key come from the environment
  (``.env`` is loaded for local development only; Docker injects variables).
* Two robustness fixes discovered while auditing the ``.env`` shipped by Syntax:
  - the base URL variable is spelled ``OPENAI_BASE_UR`` in that file, so we
    accept it as a fallback for ``OPENAI_BASE_URL``;
  - values are wrapped in single quotes; ``docker run --env-file`` passes them
    verbatim (quotes included), so one pair of matching quotes is stripped.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ConfigurationError(RuntimeError):
    """Raised when a required runtime configuration value is missing/invalid."""


def _clean(value: str | None) -> str | None:
    """Trim whitespace and one pair of matching surrounding quotes."""

    if value is None:
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value or None


def _env(*names: str) -> str | None:
    """Return the first non-empty variable among ``names``."""

    for name in names:
        value = _clean(os.getenv(name))
        if value:
            return value
    return None


def _required(*names: str) -> str:
    value = _env(*names)
    if value is None:
        raise ConfigurationError(f"Missing required environment variable: {names[0]}")
    return value


def _number(name: str, default: float, minimum: float, cast: type = int):
    raw = _env(name)
    try:
        value = cast(raw) if raw is not None else cast(default)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a valid {cast.__name__}") from exc
    if value < minimum:
        raise ConfigurationError(f"{name} must be >= {minimum}")
    return value


@dataclass(frozen=True)
class Settings:
    """Validated, immutable application settings."""

    openai_api_key: str
    openai_base_url: str | None
    chat_model: str
    embedding_model: str
    data_dir: Path
    # Chunking (see docs/RETRIEVAL_EVALUATION_RESULTS.md for how this was chosen)
    chunk_max_chars: int = 1500
    # Retrieval
    retrieval_top_k: int = 5
    rrf_k: int = 60
    max_evidence: int = 30
    # Agent budgets (bounded loops: no runaway cost / infinite cycles)
    max_agent_turns: int = 4
    max_tool_calls: int = 8
    max_critic_retries: int = 1
    max_question_length: int = 2000
    llm_timeout_s: float = 90.0
    log_level: str = "INFO"
    persist_traces: bool = True

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def cleaned_dir(self) -> Path:
        return self.data_dir / "cleaned"

    @property
    def chunks_path(self) -> Path:
        return self.data_dir / "chunks" / "chunks.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.data_dir / "manifest.json"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def trace_dir(self) -> Path:
        return self.data_dir / "traces"

    def public_view(self) -> dict[str, object]:
        """Settings that are safe to expose (never the API key)."""

        return {
            "chat_model": self.chat_model,
            "embedding_model": self.embedding_model,
            "retrieval_top_k": self.retrieval_top_k,
            "chunk_max_chars": self.chunk_max_chars,
            "max_tool_calls": self.max_tool_calls,
            "max_critic_retries": self.max_critic_retries,
        }


def load_settings(require_llm: bool = True) -> Settings:
    """Load and validate settings without ever logging secret values.

    ``require_llm=False`` lets offline commands (ingestion) run without a key.
    """

    load_dotenv(PROJECT_ROOT / ".env", override=False)
    data_dir = Path(_env("DATA_DIR") or PROJECT_ROOT / "data").expanduser().resolve()

    if require_llm:
        api_key = _required("OPENAI_API_KEY")
        chat_model = _required("OPENAI_MODEL")
        embedding_model = _required("OPENAI_EMBEDDING_MODEL")
    else:
        api_key = _env("OPENAI_API_KEY") or ""
        chat_model = _env("OPENAI_MODEL") or ""
        embedding_model = _env("OPENAI_EMBEDDING_MODEL") or ""

    return Settings(
        openai_api_key=api_key,
        openai_base_url=_env("OPENAI_BASE_URL", "OPENAI_BASE_UR"),
        chat_model=chat_model,
        embedding_model=embedding_model,
        data_dir=data_dir,
        chunk_max_chars=_number("CHUNK_MAX_CHARS", 1500, 300),
        retrieval_top_k=_number("RETRIEVAL_TOP_K", 5, 1),
        rrf_k=_number("RRF_K", 60, 1),
        max_evidence=_number("MAX_EVIDENCE", 30, 5),
        max_agent_turns=_number("MAX_AGENT_TURNS", 4, 1),
        max_tool_calls=_number("MAX_TOOL_CALLS", 8, 1),
        max_critic_retries=_number("MAX_CRITIC_RETRIES", 1, 0),
        max_question_length=_number("MAX_QUESTION_LENGTH", 2000, 10),
        llm_timeout_s=_number("LLM_TIMEOUT_S", 90.0, 5.0, float),
        log_level=(_env("LOG_LEVEL") or "INFO").upper(),
        persist_traces=(_env("PERSIST_TRACES") or "true").lower() in {"1", "true", "yes"},
    )
