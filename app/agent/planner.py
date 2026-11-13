"""Planner: decompose a question into atomic, retrievable sub-questions."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.agent.prompts import PLANNER_SYSTEM
from app.agent.tools import section_catalog
from app.ingestion.taxonomy import CanonicalSection
from app.llm import LLMGateway
from app.retrieval.hybrid import HybridRetriever, UnknownDrugError
from app.tracing import Tracer

Intent = Literal[
    "lookup", "comparison", "interaction", "population_or_threshold", "mechanism", "general", "out_of_scope"
]


# No default values: these models are sent as strict JSON schemas.
class SubQuestion(BaseModel):
    question: str = Field(description="Atomic sub-question answerable from one document.")
    drug: str | None = Field(description="Generic drug name from the catalogue, or null for general concepts.")
    sections: list[CanonicalSection] = Field(description="1-3 most likely canonical sections.")


class Plan(BaseModel):
    intent: Intent
    drugs: list[str] = Field(description="Catalogue drugs involved (generic names).")
    unknown_drugs: list[str] = Field(description="Drugs mentioned but absent from the catalogue.")
    sub_questions: list[SubQuestion]
    synthesis: str = Field(description="How sub-answers combine into the final answer.")


class Planner:
    def __init__(self, llm: LLMGateway, retriever: HybridRetriever) -> None:
        self.llm = llm
        self.retriever = retriever

    def _drug_catalogue(self) -> str:
        names: dict[str, list[str]] = {}
        for alias, drug in self.retriever.aliases.items():
            if alias != drug:
                names.setdefault(drug, []).append(alias)
        return "; ".join(
            f"{drug}" + (f" ({', '.join(sorted(names.get(drug, []))[:6])})" if names.get(drug) else "")
            for drug in self.retriever.drugs
        )

    def plan(self, question: str, tracer: Tracer | None = None) -> Plan:
        system = PLANNER_SYSTEM.format(drugs=self._drug_catalogue(), sections=section_catalog())
        plan = self.llm.structured(system=system, user=question, schema=Plan, step="plan", tracer=tracer)
        return self._normalise(plan, question)

    def _normalise(self, plan: Plan, question: str) -> Plan:
        """Post-validate LLM output against the real corpus (never trust names blindly)."""

        known, unknown = [], list(plan.unknown_drugs)
        for name in plan.drugs:
            try:
                resolved = self.retriever.resolve_drug(name)
                if resolved and resolved not in known:
                    known.append(resolved)
            except UnknownDrugError:
                unknown.append(name)
        subs = []
        for sub in plan.sub_questions[:5]:
            drug = sub.drug
            try:
                drug = self.retriever.resolve_drug(drug)
            except UnknownDrugError:
                drug = None
            subs.append(SubQuestion(question=sub.question, drug=drug, sections=sub.sections[:3]))
        if not subs and plan.intent != "out_of_scope":
            subs = [SubQuestion(question=question, drug=known[0] if len(known) == 1 else None, sections=[])]
        return Plan(
            intent=plan.intent,
            drugs=known,
            unknown_drugs=sorted(set(unknown)),
            sub_questions=subs,
            synthesis=plan.synthesis,
        )
