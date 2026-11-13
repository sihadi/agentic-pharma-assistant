"""Cleaning, chunking and lineage invariants on the REAL corpus."""

from __future__ import annotations

import json
import re

from app.ingestion.chunker import chunk_document
from app.ingestion.cleaner import clean_text
from app.ingestion.pipeline import corpus_fingerprint
from app.ingestion.taxonomy import canonical_section


def test_cleaning_rules_repair_markup_but_preserve_clinical_text():
    raw = ("eGFR below 30 mL/minute/1.73 m 2is not recommended; mg/m 2 basis. Vitamin B 12Deficiency.\n"
           "Coumadin<sup>®</sup> \\[INR\\] Liver warning:This product.\n• take 2 tablets\nWARNINGS ﻿ and")
    cleaned, ops = clean_text(raw)
    assert "1.73 m² is not recommended" in cleaned
    assert "mg/m² basis" in cleaned
    assert "Vitamin B12 Deficiency" in cleaned
    assert "Coumadin®" in cleaned and "[INR]" in cleaned
    assert "Liver warning: This" in cleaned
    assert "- take 2 tablets" in cleaned
    assert "﻿" not in cleaned and "WARNINGS and" in cleaned
    assert {"unit_m2_superscript", "unit_m2", "vitamin_b12_glued", "html_sup_sub"} <= set(ops)
    # Negations, numbers and words are untouched.
    assert "below 30" in cleaned and "not recommended" in cleaned


def test_cleaning_is_idempotent():
    once, _ = clean_text("Dose: 5 mg/m 2 daily\n\n\n\nB 12 level")
    twice, ops = clean_text(once)
    assert once == twice and ops == {}


def test_every_chunk_is_a_verbatim_slice_of_its_cleaned_document(corpus_and_cleaned):
    corpus, cleaned = corpus_and_cleaned
    for chunk in corpus.chunks:
        assert cleaned[chunk.source_file][chunk.start_char:chunk.end_char] == chunk.text


def test_chunks_carry_citation_metadata_and_respect_size(corpus, settings):
    ids = [c.chunk_id for c in corpus.chunks]
    assert len(ids) == len(set(ids)), "chunk ids must be unique"
    for c in corpus.chunks:
        assert c.source_file and c.section and c.canonical_section and c.text.strip()
        assert len(c.text) <= settings.chunk_max_chars
        assert c.embedding_text.endswith(c.text)
        if c.drug:
            assert c.drug.title() in c.embedding_text.splitlines()[0]


def test_corpus_inventory_matches_syntax_package(corpus):
    totals = corpus.manifest["totals"]
    assert totals["documents"] == 21
    assert len(totals["drugs"]) == 10
    assert not corpus.manifest["warnings"]


def test_list_intro_stays_with_its_items(corpus):
    """'contraindicated in patients with:' must never be separated from its list."""

    for c in corpus.chunks:
        assert not c.text.rstrip().endswith(":"), c.chunk_id


def test_public_question_facts_are_in_well_labelled_chunks(corpus):
    def find(source_file, section, needle):
        return [c for c in corpus.chunks if c.source_file == source_file and c.canonical_section == section
                and needle in c.text]

    assert find("dailymed-warfarin.md", "contraindications", "Pregnancy")
    assert find("dailymed-apixaban.md", "contraindications", "Active pathological bleeding")
    assert find("dailymed-metformin.md", "contraindications", "eGFR below 30 mL/min/1.73 m²")
    assert find("dailymed-sertraline.md", "warnings_precautions", "carefully monitor the international normalized ratio")


def test_taxonomy_maps_all_three_heading_conventions():
    assert canonical_section("4 CONTRAINDICATIONS", "dailymed") == "contraindications"
    assert canonical_section("CONTRAINDICATIONS", "dailymed") == "contraindications"
    assert canonical_section("Directions", "dailymed") == "dosage"
    assert canonical_section("What special precautions should I follow?", "medlineplus") == "warnings_precautions"
    assert canonical_section("WARNING: BLEEDING RISK", "dailymed") == "boxed_warning"
    assert canonical_section("anything", "glossary") == "glossary"


def test_untitled_boxed_warning_is_detected(corpus):
    assert any(c.source_file == "dailymed-ibuprofen.md" and c.canonical_section == "boxed_warning"
               and "Cardiovascular Thrombotic Events" in c.text for c in corpus.chunks)


def test_aliases_are_extracted_from_the_corpus(corpus):
    aliases = {d.drug: d.aliases for d in corpus.documents if d.drug}
    assert "Eliquis" in aliases["apixaban"]
    assert "Coumadin" in aliases["warfarin"]


def test_persisted_artifacts_are_up_to_date(settings):
    """data/chunks + manifest committed in the repo match data/raw (rerun `python -m app.ingestion`)."""

    manifest = json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    assert manifest["fingerprint"] == corpus_fingerprint(settings.raw_dir, settings.chunk_max_chars)
    for doc in manifest["documents"]:
        assert re.fullmatch(r"[0-9a-f]{64}", doc["raw_sha256"])
        assert (settings.cleaned_dir / doc["source_file"]).is_file()


def test_oversized_paragraph_is_split_on_sentence_boundaries():
    text = "# Drug X (DailyMed label)\n\n## 5 WARNINGS AND PRECAUTIONS\n\n" + " ".join(
        f"Sentence number {i} does not end mid-way." for i in range(80))
    chunks = chunk_document("dailymed-drugx.md", text + "\n", max_chars=400)
    assert len(chunks) > 1
    for c in chunks:
        assert c.text.endswith("."), "split must happen at a sentence boundary"
