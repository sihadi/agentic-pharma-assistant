"""Okapi BM25 lexical scorer (dependency-free, ~60 lines).

Why lexical at all? Dense embeddings blur exact tokens that matter clinically
(``eGFR``, ``INR``, ``CYP3A4``, ``pimozide``, numeric thresholds). BM25 rewards
exact matches and rare terms (IDF), complementing semantic similarity.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset(
    "a an and are as at be been by can do does for from has have how i if in into is it its "
    "may of on or should that the their there this to was were what when which who will with "
    "you your patient patients drug drugs".split()
)
_SUFFIXES = ("ations", "ation", "ated", "ates", "ions", "ion", "ing", "ed", "es", "s")


def stem(token: str) -> str:
    """Light suffix stripping: contraindicated/contraindications -> contraindic."""

    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


class BM25:
    """Classic BM25 with k1=1.5, b=0.75 (Robertson & Zaragoza defaults)."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        tokenized = [tokenize(doc) for doc in documents]
        self.n_docs = len(tokenized)
        self.doc_len = np.array([len(t) for t in tokenized], dtype="float32")
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for doc_id, tokens in enumerate(tokenized):
            for term, tf in Counter(tokens).items():
                self.postings[term].append((doc_id, tf))
        self.idf = {
            term: math.log(1 + (self.n_docs - len(p) + 0.5) / (len(p) + 0.5))
            for term, p in self.postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document for ``query`` (0 when no term matches)."""

        scores = np.zeros(self.n_docs, dtype="float32")
        norm = self.k1 * (1 - self.b + self.b * self.doc_len / (self.avgdl or 1.0))
        for term in set(tokenize(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, tf in self.postings[term]:
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + norm[doc_id])
        return scores
