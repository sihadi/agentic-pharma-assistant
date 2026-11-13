"""Shared offline fixtures: real corpus, deterministic fake embeddings, scripted LLM.

No test needs network access or an API key.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from app.agent.graph import ResearchAgent
from app.config import PROJECT_ROOT, Settings
from app.ingestion.pipeline import Corpus, build_corpus
from app.llm import LLMGateway
from app.retrieval.hybrid import HybridRetriever

DIM = 512


def fake_embed(texts: list[str]) -> np.ndarray:
    """Hashed bag-of-words vectors: deterministic and lexically meaningful."""

    out = np.zeros((len(texts), DIM), dtype="float32")
    for i, text in enumerate(texts):
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            out[i, int(hashlib.md5(token.encode()).hexdigest(), 16) % DIM] += 1.0
    return out / np.clip(np.linalg.norm(out, axis=1, keepdims=True), 1e-9, None)


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(
        openai_api_key="test-key",
        openai_base_url=None,
        chat_model="fake-chat",
        embedding_model="fake-embedding",
        data_dir=PROJECT_ROOT / "data",
        persist_traces=False,
    )


@pytest.fixture(scope="session")
def corpus_and_cleaned(settings: Settings) -> tuple[Corpus, dict[str, str]]:
    return build_corpus(settings.raw_dir, settings.chunk_max_chars)


@pytest.fixture(scope="session")
def corpus(corpus_and_cleaned) -> Corpus:
    return corpus_and_cleaned[0]


@pytest.fixture(scope="session")
def retriever(corpus: Corpus) -> HybridRetriever:
    r = HybridRetriever(corpus.chunks, corpus.documents, fake_embed)
    r.build()
    return r


# --------------------------------------------------------------------------- scripted LLM
Script = dict | Callable[[str], dict]


def tool_message(*calls: tuple[str, dict[str, Any]], content: str = "") -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        tool_calls=[
            SimpleNamespace(id=f"call_{i}", function=SimpleNamespace(name=name, arguments=json.dumps(args)))
            for i, (name, args) in enumerate(calls)
        ] or None,
    )


class ScriptedLLM(LLMGateway):
    """Replays scripted outputs per step; records every call for assertions."""

    def __init__(self, settings: Settings, structured: dict[str, list[Script]], turns: list[SimpleNamespace]):
        self.settings = settings
        self.client = None
        self.structured_script = {k: list(v) for k, v in structured.items()}
        self.turns = list(turns)
        self.calls: list[dict[str, Any]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        return fake_embed(texts)

    def structured(self, *, system, user, schema, step, tracer=None):
        self.calls.append({"step": step, "user": user})
        item = self.structured_script[step].pop(0)
        return schema.model_validate(item(user) if callable(item) else item)

    def tool_turn(self, *, messages, tools, tool_choice, step, tracer=None):
        self.calls.append({"step": step, "tool_choice": tool_choice, "tools": [t["function"]["name"] for t in tools]})
        return self.turns.pop(0) if self.turns else tool_message(content="READY: done")


def eid_for(user_prompt: str, source_file: str, heading: str) -> str:
    """Find the evidence id the system assigned to a passage, from the prompt text."""

    match = re.search(rf"\[(E\d+)\] {re.escape(source_file)} \| [^|]+ \| [^\n]*{re.escape(heading)}", user_prompt)
    assert match, f"{source_file} / {heading} not in prompt"
    return match.group(1)


def warfarin_plan(intent: str = "lookup") -> dict:
    return {
        "intent": intent,
        "drugs": ["warfarin"],
        "unknown_drugs": [],
        "sub_questions": [{"question": "What are the contraindications of warfarin?",
                           "drug": "warfarin", "sections": ["contraindications"]}],
        "synthesis": "List them.",
    }


def critique(verdict: str = "APPROVE", confidence: float = 0.9, corrected: str = "") -> dict:
    return {"claims": [{"claim": "c", "status": "supported", "evidence_ids": [], "note": ""}],
            "uncovered": [], "individualized_advice": False, "verdict": verdict, "confidence": confidence,
            "revision_instructions": "" if verdict == "APPROVE" else "search more", "corrected_answer": corrected}


@pytest.fixture
def make_agent(settings: Settings, retriever: HybridRetriever):
    def factory(structured: dict[str, list[Script]], turns: list[SimpleNamespace], **overrides):
        s = replace(settings, **overrides)
        llm = ScriptedLLM(s, structured, turns)
        return ResearchAgent(s, llm, retriever), llm

    return factory
