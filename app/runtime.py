"""Composition root: wires settings -> corpus -> LLM -> retriever -> agent.

Used identically by the API, the CLI scripts and the tests, so there is one
place where the object graph is assembled.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from app.agent.graph import ResearchAgent
from app.config import Settings
from app.ingestion.pipeline import Corpus, load_corpus
from app.llm import LLMGateway
from app.retrieval.embedding_cache import EmbeddingCache
from app.retrieval.hybrid import HybridRetriever

LOGGER = logging.getLogger(__name__)


@dataclass
class Runtime:
    settings: Settings
    corpus: Corpus
    llm: LLMGateway
    retriever: HybridRetriever
    agent: ResearchAgent


def build_runtime(settings: Settings, client=None) -> Runtime:
    corpus = load_corpus(settings.raw_dir, settings.cleaned_dir, settings.chunks_path,
                         settings.manifest_path, settings.chunk_max_chars)
    llm = LLMGateway(settings, client)
    cache = EmbeddingCache(settings.cache_dir, settings.embedding_model)
    retriever = HybridRetriever(corpus.chunks, corpus.documents, llm.embed, cache, settings.rrf_k)
    retriever.build()
    return Runtime(settings, corpus, llm, retriever, ResearchAgent(settings, llm, retriever))
