"""Hybrid retriever: FAISS dense search + BM25, fused by Reciprocal Rank Fusion.

* **FAISS ``IndexFlatIP`` on L2-normalised vectors** = exact cosine search.
  With ~600 chunks, brute force is both exact and sub-millisecond; approximate
  indexes (IVF/HNSW) would only add recall loss and tuning for no gain.
* **Metadata pre-filtering** (drug / source type / canonical section) uses a
  FAISS ``IDSelectorBatch`` so filtering happens *inside* the index search
  instead of over-fetching and discarding.
* **RRF** (Cormack et al., 2009): ``score = Σ 1 / (k + rank_i)``. It fuses
  rankings without calibrating incomparable scores (cosine vs BM25), with
  the standard ``k = 60``.

Scores are relevance signals, never probabilities of truth.
"""

from __future__ import annotations

import difflib
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Literal

import faiss
import numpy as np

from app.ingestion.chunker import Chunk, DocumentMeta
from app.retrieval.bm25 import BM25
from app.retrieval.embedding_cache import EmbedFn, EmbeddingCache

LOGGER = logging.getLogger(__name__)
SearchMode = Literal["hybrid", "dense", "lexical"]


class UnknownDrugError(ValueError):
    """The requested drug is not covered by the corpus."""


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # fused RRF score (ranking signal only)
    dense_score: float | None  # cosine similarity in [-1, 1]
    lexical_score: float | None  # BM25 score
    dense_rank: int | None
    lexical_rank: int | None


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        documents: list[DocumentMeta],
        embed_fn: EmbedFn,
        cache: EmbeddingCache | None = None,
        rrf_k: int = 60,
    ) -> None:
        if not chunks:
            raise ValueError("Cannot build a retriever over an empty corpus")
        self.chunks = chunks
        self.documents = documents
        self.embed_fn = embed_fn
        self.cache = cache
        self.rrf_k = rrf_k
        self.index: faiss.Index | None = None
        self.bm25: BM25 | None = None
        self._query_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.aliases = self._build_alias_map(documents)
        self.stats: dict[str, float | int] = {}

    # ------------------------------------------------------------------ build
    def build(self) -> None:
        started = time.perf_counter()
        texts = [c.embedding_text for c in self.chunks]
        if self.cache is not None:
            matrix, computed = self.cache.embed(texts, self.embed_fn)
        else:
            matrix, computed = self.embed_fn(texts), len(texts)
        self.index = faiss.IndexFlatIP(matrix.shape[1])
        self.index.add(matrix)
        self.bm25 = BM25(texts)
        self.stats = {
            "chunks": len(self.chunks),
            "dimensions": int(matrix.shape[1]),
            "embeddings_computed": computed,
            "embeddings_cached": len(texts) - computed,
            "build_ms": round((time.perf_counter() - started) * 1000, 1),
        }
        LOGGER.info("retriever_ready", extra=self.stats)

    @property
    def ready(self) -> bool:
        return self.index is not None and self.bm25 is not None

    # ------------------------------------------------------------------ drugs
    @staticmethod
    def _build_alias_map(documents: list[DocumentMeta]) -> dict[str, str]:
        """Generic names + brand/other names found in the corpus itself."""

        mapping: dict[str, str] = {}
        for doc in documents:
            if doc.drug:
                mapping[doc.drug.lower()] = doc.drug
                for alias in doc.aliases:
                    mapping.setdefault(alias.lower(), doc.drug)
        return mapping

    @property
    def drugs(self) -> list[str]:
        return sorted({d.drug for d in self.documents if d.drug})

    def resolve_drug(self, name: str | None) -> str | None:
        """Map a generic/brand name (typo-tolerant) to a corpus drug, else raise."""

        if not name:
            return None
        key = name.strip().lower()
        if key in self.aliases:
            return self.aliases[key]
        close = difflib.get_close_matches(key, list(self.aliases), n=1, cutoff=0.82)
        if close:
            return self.aliases[close[0]]
        raise UnknownDrugError(
            f"'{name}' is not in the corpus. Available drugs: {', '.join(self.drugs)}"
        )

    # ------------------------------------------------------------------ search
    def _candidate_ids(
        self, drug: str | None, source_type: str | None, sections: list[str] | None
    ) -> np.ndarray | None:
        if not (drug or source_type or sections):
            return None
        wanted = set(sections or [])
        return np.array(
            [
                i
                for i, c in enumerate(self.chunks)
                if (not drug or c.drug == drug)
                and (not source_type or c.source_type == source_type)
                and (not wanted or c.canonical_section in wanted)
            ],
            dtype="int64",
        )

    def _embed_query(self, query: str) -> np.ndarray:
        if query in self._query_cache:
            self._query_cache.move_to_end(query)
            return self._query_cache[query]
        vector = self.embed_fn([query])[:1].astype("float32")
        self._query_cache[query] = vector
        if len(self._query_cache) > 512:
            self._query_cache.popitem(last=False)
        return vector

    def _dense(self, query: str, ids: np.ndarray | None, depth: int) -> list[tuple[int, float]]:
        assert self.index is not None
        params = None
        if ids is not None:
            selector = faiss.IDSelectorBatch(ids)
            params = faiss.SearchParameters(sel=selector)
        scores, idx = self.index.search(self._embed_query(query), depth, params=params)
        return [(int(i), float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]

    def _lexical(self, query: str, ids: np.ndarray | None, depth: int) -> list[tuple[int, float]]:
        assert self.bm25 is not None
        scores = self.bm25.scores(query)
        pool = ids if ids is not None else np.arange(len(self.chunks))
        order = pool[np.argsort(-scores[pool], kind="stable")][:depth]
        return [(int(i), float(scores[i])) for i in order if scores[i] > 0]

    def search(
        self,
        query: str,
        k: int = 5,
        *,
        drug: str | None = None,
        source_type: str | None = None,
        sections: list[str] | None = None,
        mode: SearchMode = "hybrid",
    ) -> list[Hit]:
        """Top-k chunks for ``query`` within the optional metadata filters."""

        if not query.strip():
            raise ValueError("Search query cannot be empty")
        if not self.ready:
            raise RuntimeError("Retriever index is not built")
        drug = self.resolve_drug(drug)
        ids = self._candidate_ids(drug, source_type, sections)
        pool = len(self.chunks) if ids is None else len(ids)
        if pool == 0:
            return []
        depth = min(pool, max(50, k * 10))

        dense = self._dense(query, ids, depth) if mode != "lexical" else []
        lexical = self._lexical(query, ids, depth) if mode != "dense" else []
        fused: dict[int, float] = {}
        for ranking in (dense, lexical):
            for rank, (i, _) in enumerate(ranking, start=1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (self.rrf_k + rank)
        dense_info = {i: (r, s) for r, (i, s) in enumerate(dense, start=1)}
        lexical_info = {i: (r, s) for r, (i, s) in enumerate(lexical, start=1)}

        top = sorted(fused.items(), key=lambda item: -item[1])[:k]
        return [
            Hit(
                chunk=self.chunks[i],
                score=round(score, 5),
                dense_score=round(dense_info[i][1], 4) if i in dense_info else None,
                lexical_score=round(lexical_info[i][1], 3) if i in lexical_info else None,
                dense_rank=dense_info[i][0] if i in dense_info else None,
                lexical_rank=lexical_info[i][0] if i in lexical_info else None,
            )
            for i, score in top
        ]

    # ------------------------------------------------------------------ structure
    def section_chunks(self, drug: str, section: str, source_type: str | None = None) -> list[Chunk]:
        """All chunks of one canonical section of a drug, in document order."""

        drug_name = self.resolve_drug(drug)
        selected = [
            c
            for c in self.chunks
            if c.drug == drug_name
            and c.canonical_section == section
            and (not source_type or c.source_type == source_type)
        ]
        return sorted(selected, key=lambda c: (c.source_type != "dailymed", c.source_file, c.start_char))

    def outline(self, drug: str) -> dict[str, dict[str, int]]:
        """{source_file: {canonical_section: n_chunks}} for one drug."""

        drug_name = self.resolve_drug(drug)
        outline: dict[str, dict[str, int]] = {}
        for c in self.chunks:
            if c.drug == drug_name:
                per_file = outline.setdefault(c.source_file, {})
                per_file[c.canonical_section] = per_file.get(c.canonical_section, 0) + 1
        return outline
