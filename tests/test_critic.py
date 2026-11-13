"""Deterministic layer of the critic (no LLM)."""

from __future__ import annotations

from app.agent.critic import audit_answer

EVIDENCE = {
    "E1": {"text": "Metformin is contraindicated in patients with an eGFR below 30 mL/min/1.73 m²."},
    "E2": {"text": "Initiation is not recommended in patients with an eGFR between 30 to 45 mL/min."},
}
QUESTION = "A patient with an eGFR of 25 mL/min: is metformin contraindicated?"


def test_fully_grounded_answer_scores_one():
    answer = ("Yes. An eGFR of 25 mL/min is below the contraindication threshold of 30 mL/min/1.73 m² [E1].\n\n"
              "**Threshold**\n- Initiation is not recommended between 30 and 45 mL/min [E2].")
    audit = audit_answer(answer, EVIDENCE, QUESTION)
    assert audit["citation_coverage"] == 1.0
    assert audit["numeric_issues"] == [] and audit["invalid_citations"] == []
    assert audit["grounding_score"] == 1.0


def test_invented_threshold_is_caught():
    audit = audit_answer("Metformin is contraindicated when eGFR is below 35 mL/min [E1].", EVIDENCE, QUESTION)
    assert [i["number"] for i in audit["numeric_issues"]] == ["35"]
    assert audit["grounding_score"] < 1.0


def test_numbers_from_the_question_are_exempt():
    audit = audit_answer("An eGFR of 25 mL/min falls below the 30 mL/min threshold [E1].", EVIDENCE, QUESTION)
    assert audit["numeric_issues"] == []


def test_invalid_and_missing_citations_are_reported():
    answer = ("Metformin causes lactic acidosis in every patient taking it [E9].\n"
              "It is also contraindicated with severe renal impairment of the kidney.")
    audit = audit_answer(answer, EVIDENCE, QUESTION)
    assert audit["invalid_citations"] == ["E9"]
    assert len(audit["uncited_statements"]) == 1
    assert audit["citation_coverage"] == 0.5


def test_explicit_gaps_do_not_require_citations():
    audit = audit_answer("The provided documents do not state a dose adjustment for dialysis patients.", EVIDENCE,
                         QUESTION)
    assert audit["statements"] == 0 and audit["citation_coverage"] == 1.0
