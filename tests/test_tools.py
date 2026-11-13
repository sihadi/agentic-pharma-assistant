"""Tool contracts: schemas, validation and structured errors."""

from __future__ import annotations

from app.agent.tools import ToolExecutor


def test_tool_specs_are_valid_openai_function_definitions():
    specs = ToolExecutor.specs()
    names = {s["function"]["name"] for s in specs}
    assert names == {"search_corpus", "get_drug_section", "compare_drugs", "list_sections"}
    for spec in specs:
        assert spec["type"] == "function"
        assert spec["function"]["description"]
        assert spec["function"]["parameters"]["type"] == "object"


def test_get_drug_section_returns_full_section(retriever):
    out = ToolExecutor(retriever).execute("get_drug_section", '{"drug": "Coumadin", "section": "contraindications"}')
    assert out.status == "ok"
    assert {c.source_file for c, _, _ in out.chunks} == {"dailymed-warfarin.md"}
    assert any("Pregnancy" in c.text for c, _, _ in out.chunks)


def test_compare_drugs_groups_evidence_per_drug(retriever):
    out = ToolExecutor(retriever).execute(
        "compare_drugs", {"drugs": ["warfarin", "apixaban", "lisinopril"], "section": "contraindications"})
    assert out.status == "ok"
    assert {group for _, _, group in out.chunks} == {"warfarin", "apixaban"}
    assert "lisinopril (not in corpus)" in out.message


def test_invalid_arguments_return_structured_error(retriever):
    out = ToolExecutor(retriever).execute("get_drug_section", {"drug": "warfarin", "section": "astrology"})
    assert out.status == "error" and out.message.startswith("ERROR: Invalid arguments")


def test_unknown_drug_and_unknown_tool_are_reported(retriever):
    executor = ToolExecutor(retriever)
    assert "not in the corpus" in executor.execute("list_sections", {"drug": "lisinopril"}).message
    assert "Unknown tool" in executor.execute("delete_everything", {}).message
    assert "not valid JSON" in executor.execute("search_corpus", "{oops").message


def test_missing_section_suggests_alternatives(retriever):
    out = ToolExecutor(retriever).execute("get_drug_section", {"drug": "acetaminophen", "section": "contraindications"})
    assert out.status == "empty"
    assert "Available sections" in out.message and "warnings_precautions" in out.message
