"""Hybrid in-memory retrieval: FAISS (dense) + BM25 (lexical) fused with RRF."""

from app.retrieval.hybrid import Hit, HybridRetriever, UnknownDrugError

__all__ = ["Hit", "HybridRetriever", "UnknownDrugError"]
