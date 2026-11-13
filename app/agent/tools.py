"""Local tools exposed to the agent through native OpenAI function calling.

The LLM sees the four tool schemas below and decides *which* to call, with
*which* arguments, and *how many times* - there is no keyword routing in the
code. Each tool is complementary, so the choice is meaningful:

=================  ==========================================================
search_corpus      ranked hybrid search (semantic + keyword), optional filters
get_drug_section   *exhaustive* canonical section of one label (not ranked):
                   the right tool when a complete list matters (all
                   contraindications) - top-k search could miss items
compare_drugs      the same section for 2-4 drugs side by side, for
                   comparisons; organises evidence, never concludes
list_sections      outline of a drug's documents, to discover where info is
=================  ==========================================================

Arguments are validated with Pydantic *before* execution; any failure (bad
arguments, unknown drug, empty result) is returned to the agent as a
structured, actionable message instead of crashing the workflow.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from app.ingestion.chunker import Chunk
from app.ingestion.taxonomy import SECTION_DESCRIPTIONS, CanonicalSection
from app.retrieval.hybrid import HybridRetriever, UnknownDrugError

SourceType = Literal["dailymed", "medlineplus", "glossary"]
MAX_SECTION_CHUNKS = 8


class SearchCorpusArgs(BaseModel):
    query: str = Field(min_length=2, max_length=500, description="Focused natural-language search query.")
    drug: str | None = Field(None, description="Restrict to one drug (generic or brand name).")
    source_type: SourceType | None = Field(None, description="Restrict to one source type.")
    section: CanonicalSection | None = Field(None, description="Restrict to one canonical section.")
    k: int = Field(5, ge=1, le=10, description="Number of passages to return.")


class GetDrugSectionArgs(BaseModel):
    drug: str = Field(min_length=2, max_length=80, description="Drug name (generic or brand).")
    section: CanonicalSection = Field(description="Canonical section to return in full.")
    source_type: Literal["dailymed", "medlineplus"] | None = Field(
        "dailymed", description="Label to read; null = both. DailyMed is the authoritative label."
    )


class CompareDrugsArgs(BaseModel):
    drugs: list[str] = Field(min_length=2, max_length=4, description="2 to 4 drug names.")
    section: CanonicalSection = Field(description="Section to align across the drugs.")
    focus: str | None = Field(
        None, max_length=300, description="Optional aspect to rank passages within each drug's section."
    )


class ListSectionsArgs(BaseModel):
    drug: str = Field(min_length=2, max_length=80, description="Drug name (generic or brand).")


TOOL_DESCRIPTIONS: dict[str, str] = {
    "search_corpus": (
        "Ranked hybrid (semantic + keyword) search over the local corpus. Best for specific facts: "
        "thresholds, monitoring, mechanisms, interactions mentioned in another drug's label, or when "
        "you do not know which section holds the answer. Optional filters: drug, source_type, section."
    ),
    "get_drug_section": (
        "Return the COMPLETE text of one canonical section of a drug's document (exhaustive, not "
        "ranked). Use when the question needs a full list, e.g. all contraindications or the boxed "
        "warning, where ranked search could miss items."
    ),
    "compare_drugs": (
        "Retrieve the same canonical section for 2-4 drugs, grouped per drug, to compare them. "
        "With 'focus', passages are ranked against it inside each drug's section. It only organises "
        "evidence; you draw no conclusion beyond what the passages state."
    ),
    "list_sections": (
        "List which documents and canonical sections exist for a drug, with passage counts. Use it to "
        "discover where information may live before searching."
    ),
}

_ARG_MODELS: dict[str, type[BaseModel]] = {
    "search_corpus": SearchCorpusArgs,
    "get_drug_section": GetDrugSectionArgs,
    "compare_drugs": CompareDrugsArgs,
    "list_sections": ListSectionsArgs,
}


@dataclass
class ToolOutcome:
    """Result of one tool call, before evidence ids are assigned by the graph."""

    name: str
    arguments: dict[str, Any]
    status: Literal["ok", "error", "empty"]
    chunks: list[tuple[Chunk, float | None, str | None]] = field(default_factory=list)  # chunk, score, group
    message: str = ""
    duration_ms: float = 0.0


def _schema(model: type[BaseModel]) -> dict[str, Any]:
    schema = model.model_json_schema()
    schema.pop("title", None)
    for prop in schema.get("properties", {}).values():
        prop.pop("title", None)
    return schema


class ToolExecutor:
    def __init__(self, retriever: HybridRetriever) -> None:
        self.retriever = retriever

    @staticmethod
    def specs() -> list[dict[str, Any]]:
        """OpenAI ``tools=[...]`` definitions generated from the Pydantic models."""

        return [
            {
                "type": "function",
                "function": {"name": name, "description": TOOL_DESCRIPTIONS[name], "parameters": _schema(model)},
            }
            for name, model in _ARG_MODELS.items()
        ]

    def execute(self, name: str, raw_arguments: str | dict[str, Any]) -> ToolOutcome:
        started = time.perf_counter()
        try:
            args = json.loads(raw_arguments) if isinstance(raw_arguments, str) else dict(raw_arguments)
        except json.JSONDecodeError:
            return self._error(name, {}, "Arguments are not valid JSON.", started)
        model = _ARG_MODELS.get(name)
        if model is None:
            return self._error(name, args, f"Unknown tool '{name}'. Available: {', '.join(_ARG_MODELS)}.", started)
        try:
            validated = model.model_validate(args)
            outcome = getattr(self, f"_{name}")(validated)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors())
            return self._error(name, args, f"Invalid arguments - {problems}", started)
        except UnknownDrugError as exc:
            return self._error(name, args, str(exc), started)
        outcome.duration_ms = round((time.perf_counter() - started) * 1000, 1)
        return outcome

    @staticmethod
    def _error(name: str, args: dict[str, Any], message: str, started: float) -> ToolOutcome:
        return ToolOutcome(name, args, "error", message=f"ERROR: {message}",
                           duration_ms=round((time.perf_counter() - started) * 1000, 1))

    # ------------------------------------------------------------------ tools
    def _search_corpus(self, a: SearchCorpusArgs) -> ToolOutcome:
        hits = self.retriever.search(
            a.query, a.k, drug=a.drug, source_type=a.source_type, sections=[a.section] if a.section else None
        )
        if not hits:
            return ToolOutcome("search_corpus", a.model_dump(), "empty",
                               message="No passage matched these filters. Try removing a filter.")
        return ToolOutcome("search_corpus", a.model_dump(), "ok", [(h.chunk, h.score, None) for h in hits])

    def _get_drug_section(self, a: GetDrugSectionArgs) -> ToolOutcome:
        chunks = self.retriever.section_chunks(a.drug, a.section, a.source_type)
        if not chunks:
            outline = self.retriever.outline(a.drug)
            available = sorted({s for per_file in outline.values() for s in per_file})
            return ToolOutcome("get_drug_section", a.model_dump(), "empty",
                               message=f"No '{a.section}' section for {a.drug} in that source. "
                                       f"Available sections: {', '.join(available)}. "
                                       "Consider search_corpus (e.g. OTC labels list 'Do not use' under warnings).")
        note = ""
        if len(chunks) > MAX_SECTION_CHUNKS:
            note = (f"Section has {len(chunks)} passages; showing the first {MAX_SECTION_CHUNKS}. "
                    "Use search_corpus with section filter to target the rest.")
            chunks = chunks[:MAX_SECTION_CHUNKS]
        return ToolOutcome("get_drug_section", a.model_dump(), "ok", [(c, None, None) for c in chunks], note)

    def _compare_drugs(self, a: CompareDrugsArgs) -> ToolOutcome:
        collected: list[tuple[Chunk, float | None, str | None]] = []
        missing: list[str] = []
        for drug in a.drugs:
            try:
                name = self.retriever.resolve_drug(drug) or drug
            except UnknownDrugError:
                missing.append(f"{drug} (not in corpus)")
                continue
            if a.focus:
                hits = self.retriever.search(a.focus, 4, drug=name, source_type="dailymed", sections=[a.section])
                items = [(h.chunk, h.score, name) for h in hits]
            else:
                items = [(c, None, name) for c in
                         self.retriever.section_chunks(name, a.section, "dailymed")[:MAX_SECTION_CHUNKS // 2]]
            if not items:
                missing.append(name)
            collected.extend(items)
        message = f"No '{a.section}' passages found for: {', '.join(missing)}." if missing else ""
        status = "ok" if collected else "empty"
        return ToolOutcome("compare_drugs", a.model_dump(), status, collected, message)

    def _list_sections(self, a: ListSectionsArgs) -> ToolOutcome:
        outline = self.retriever.outline(a.drug)
        lines = [f"{self.retriever.resolve_drug(a.drug)}:"]
        for source_file, sections in outline.items():
            listed = ", ".join(f"{s} ({n})" for s, n in sorted(sections.items()))
            lines.append(f"- {source_file}: {listed}")
        return ToolOutcome("list_sections", a.model_dump(), "ok", message="\n".join(lines))


def section_catalog() -> str:
    return "\n".join(f"- {key}: {desc}" for key, desc in SECTION_DESCRIPTIONS.items())
