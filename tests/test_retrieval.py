"""BM25, hybrid fusion, metadata filters and drug resolution."""

from __future__ import annotations

import pytest

from app.retrieval.bm25 import BM25, stem, tokenize
from app.retrieval.hybrid import UnknownDrugError


def test_stemming_unifies_clinical_inflections():
    assert stem("contraindications") == stem("contraindicated")
    assert stem("monitoring") == stem("monitored")
    assert "the" not in tokenize("the INR of the patient")


def test_bm25_ranks_exact_rare_terms_first():
    bm25 = BM25(["warfarin INR monitoring", "metformin eGFR threshold", "general text about drugs"])
    scores = bm25.scores("eGFR threshold")
    assert scores.argmax() == 1 and scores[2] == 0


def test_hybrid_search_returns_scored_metadata(retriever):
    hits = retriever.search("warfarin contraindications pregnancy", k=5)
    assert hits and hits[0].chunk.source_file.endswith("warfarin.md")
    assert all(h.score > 0 for h in hits)
    assert any(h.dense_score is not None and h.lexical_score is not None for h in hits)


def test_filters_restrict_candidates_inside_faiss(retriever):
    hits = retriever.search("bleeding risk", k=8, drug="sertraline", source_type="dailymed",
                            sections=["warnings_precautions"])
    assert hits
    assert {(h.chunk.source_file, h.chunk.canonical_section) for h in hits} == {
        ("dailymed-sertraline.md", "warnings_precautions")}


@pytest.mark.parametrize("name,expected", [("Eliquis", "apixaban"), ("ZOLOFT", "sertraline"),
                                           ("warfarine", "warfarin"), ("Paracetamol", "acetaminophen")])
def test_drug_resolution_uses_corpus_aliases_and_tolerates_typos(retriever, name, expected):
    assert retriever.resolve_drug(name) == expected


def test_unknown_drug_is_reported_not_guessed(retriever):
    with pytest.raises(UnknownDrugError, match="not in the corpus"):
        retriever.resolve_drug("lisinopril")


def test_section_lookup_is_exhaustive_and_ordered(retriever, corpus):
    chunks = retriever.section_chunks("warfarin", "contraindications", "dailymed")
    expected = [c for c in corpus.chunks if c.source_file == "dailymed-warfarin.md"
                and c.canonical_section == "contraindications"]
    assert [c.chunk_id for c in chunks] == [c.chunk_id for c in sorted(expected, key=lambda c: c.start_char)]


def test_public_questions_retrieve_expected_files(retriever):
    cases = {
        "Compare the contraindications of warfarin and apixaban": {"dailymed-warfarin.md", "dailymed-apixaban.md"},
        "eGFR 25 metformin contraindicated threshold": {"dailymed-metformin.md"},
    }
    for query, files in cases.items():
        found = {h.chunk.source_file for h in retriever.search(query, k=10)}
        assert files <= found, query
