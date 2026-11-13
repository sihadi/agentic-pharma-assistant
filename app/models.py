"""HTTP contracts (Pydantic v2) - documented automatically in OpenAPI (/docs)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class QuestionRequest(BaseModel):
    """Body of POST /ask."""

    question: str = Field(min_length=3, max_length=2000, examples=["What are the main contraindications for warfarin?"])

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 3:
            raise ValueError("Question is empty")
        return value


class Citation(BaseModel):
    ref: int = Field(description="Number used in the answer, e.g. [1]")
    evidence_id: str
    source_file: str
    drug: str | None
    source_type: str
    section: str
    canonical_section: str
    chunk_id: str
    excerpt: str = Field(description="Verbatim cited passage from data/cleaned/<source_file>")


class Review(BaseModel):
    model_config = ConfigDict(extra="allow")

    verdict: str = Field(description="APPROVE | REVISE | INSUFFICIENT_EVIDENCE | N/A")
    confidence: float | None = Field(description="LLM confidence x deterministic grounding score")


class AskResponse(BaseModel):
    answer: str = Field(description="Markdown answer with numbered citations, sources and disclaimer")
    answer_body: str
    status: Literal["answered", "answered_with_caveats", "insufficient_evidence", "out_of_scope"]
    sources: list[str]
    citations: list[Citation]
    limitations: list[str]
    review: Review
    plan: dict[str, Any]
    tools_used: list[dict[str, Any]]
    disclaimer: str
    trace_id: str
    usage: dict[str, Any]
