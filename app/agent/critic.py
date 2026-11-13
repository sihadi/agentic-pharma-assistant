"""Two-layer critic: deterministic citation audit + LLM claim-level review.

Layer 1 - deterministic (cheap, cannot hallucinate):
  * citation validity: every [E#] exists in the evidence registry;
  * citation coverage: share of factual sentences carrying >= 1 citation;
  * numeric grounding: every number in a cited sentence appears in the text
    of the passages it cites (numbers from the question are exempt). This
    directly targets the riskiest failure in this domain: an invented dose
    or threshold.
Layer 2 - LLM review: claim-by-claim status (supported / unsupported /
  misattributed / ambiguous), plan coverage, individualized-advice check,
  verdict, confidence and a corrected answer.

Layer 1 can VETO an LLM "APPROVE" (forces REVISE) - the model cannot talk
its way past a broken citation or an unsupported number. The final
confidence = LLM confidence x deterministic grounding score, both reported.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.agent.evidence import CITATION_RE, render
from app.agent.prompts import CRITIC_SYSTEM
from app.llm import LLMGateway
from app.tracing import Tracer

Verdict = Literal["APPROVE", "REVISE", "INSUFFICIENT_EVIDENCE"]
NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:[.,]\d+)*")
NON_CLAIM_RE = re.compile(
    r"(do(?:es)? not (?:state|mention|provide|specify|address|contain)|not (?:stated|found|available|covered)|"
    r"no (?:information|data|evidence))",
    re.IGNORECASE,
)


class ClaimCheck(BaseModel):
    claim: str
    status: Literal["supported", "unsupported", "misattributed", "ambiguous"]
    evidence_ids: list[str]
    note: str


class Critique(BaseModel):
    claims: list[ClaimCheck]
    uncovered: list[str]
    individualized_advice: bool
    verdict: Verdict
    confidence: float = Field(description="0 to 1")
    revision_instructions: str
    corrected_answer: str


# --------------------------------------------------------------------------- layer 1
def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER_RE.findall(text)}


def _statements(answer: str) -> list[str]:
    """Factual statements = prose sentences and bullet items (headings/table rules skipped)."""

    statements: list[str] = []
    for raw_line in answer.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or re.fullmatch(r"\|?[\s:\-|]+\|?", line):
            continue
        line = re.sub(r"^(?:[-*+]|\d+[.)])\s+", "", line)
        if re.fullmatch(r"\*\*[^*]+\*\*:?", line):
            continue  # a bold group label ("**Threshold**") is structure, not a claim
        if line.startswith("|"):
            statements.append(line)  # a table row is one statement
            continue
        statements.extend(s for s in re.split(r"(?<=[.!?])\s+(?=[A-Z*])", line) if s.strip())
    # Bold group labels such as "**Shared:**" are structure, not claims.
    return [s for s in statements if len(re.sub(r"[*_`:|\s]", "", s)) > 25]


def audit_answer(answer: str, evidence_by_eid: dict[str, dict[str, Any]], question: str) -> dict[str, Any]:
    question_numbers = _numbers(question)
    cited_ids = CITATION_RE.findall(answer)
    invalid = sorted({eid for eid in cited_ids if eid not in evidence_by_eid})
    claims = [s for s in _statements(answer) if not NON_CLAIM_RE.search(s)]
    uncited = [s for s in claims if not CITATION_RE.search(s)]
    numeric_issues = []
    for sentence in claims:
        ids = [e for e in CITATION_RE.findall(sentence) if e in evidence_by_eid]
        if not ids:
            continue
        evidence_numbers = set().union(*(_numbers(evidence_by_eid[e]["text"]) for e in ids))
        for number in _numbers(CITATION_RE.sub("", sentence)) - question_numbers:
            if number not in evidence_numbers:
                numeric_issues.append({"number": number, "sentence": sentence[:200], "cited": ids})
    coverage = 1.0 if not claims else (len(claims) - len(uncited)) / len(claims)
    score = coverage * (0.5 if invalid else 1.0) * (0.85 ** len(numeric_issues))
    return {
        "citations": len(cited_ids),
        "invalid_citations": invalid,
        "statements": len(claims),
        "uncited_statements": [s[:200] for s in uncited],
        "citation_coverage": round(coverage, 3),
        "numeric_issues": numeric_issues,
        "grounding_score": round(score, 3),
    }


# --------------------------------------------------------------------------- critic
class Critic:
    def __init__(self, llm: LLMGateway) -> None:
        self.llm = llm

    def review(
        self,
        *,
        question: str,
        plan: dict[str, Any],
        answer: str,
        evidence: list[dict[str, Any]],
        tracer: Tracer | None = None,
    ) -> dict[str, Any]:
        evidence_by_eid = {e["eid"]: e for e in evidence}
        audit = audit_answer(answer, evidence_by_eid, question)
        user = (
            f"QUESTION:\n{question}\n\nPLAN:\n{json.dumps(plan.get('sub_questions', []), ensure_ascii=False)}"
            f"\n\nEVIDENCE:\n{render(evidence)}\n\nDRAFT ANSWER:\n{answer}"
        )
        critique = self.llm.structured(system=CRITIC_SYSTEM, user=user, schema=Critique, step="critic", tracer=tracer)

        verdict = critique.verdict
        vetoes = []
        if audit["invalid_citations"]:
            vetoes.append("invalid citation ids")
        if audit["numeric_issues"]:
            vetoes.append("numbers not found in cited evidence")
        if audit["citation_coverage"] < 0.6:
            vetoes.append("too many uncited statements")
        if verdict == "APPROVE" and vetoes:
            verdict = "REVISE"

        llm_confidence = max(0.0, min(1.0, critique.confidence))
        flagged = [c.model_dump() for c in critique.claims if c.status != "supported"]
        instructions = critique.revision_instructions
        if vetoes:
            instructions = (instructions + " " if instructions else "") + (
                "Deterministic audit: " + "; ".join(vetoes) + ". "
                + "; ".join(f"'{i['number']}' not in {i['cited']}" for i in audit["numeric_issues"][:5])
            )
        return {
            "verdict": verdict,
            "llm_verdict": critique.verdict,
            "vetoes": vetoes,
            "confidence": round(llm_confidence * audit["grounding_score"], 3),
            "llm_confidence": round(llm_confidence, 3),
            "audit": audit,
            "claims_checked": len(critique.claims),
            "claims_supported": sum(c.status == "supported" for c in critique.claims),
            "flagged_claims": flagged,
            "uncovered": critique.uncovered,
            "individualized_advice": critique.individualized_advice,
            "revision_instructions": instructions.strip(),
            "corrected_answer": critique.corrected_answer.strip(),
        }
