"""LangGraph orchestration: Plan -> Retrieve -> Tools (agent loop) -> Draft -> Critic -> Answer.

                         ┌──────────── out_of_scope ────────────┐
    START ─► plan ─► retrieve ─► agent ⇄ tools                   │
                                   │ (no more tool calls / budget)│
                                   ▼                              ▼
                                generate ─► critic ─► finalize ─► END
                                   ▲           │ REVISE & budget left
                                   └─ agent ◄─ prepare_revision

* ``agent`` is a native function-calling turn: the model decides which tools
  to call (or none). The first turn uses ``tool_choice="required"`` so the
  "use tools" step always happens, but WHICH tools and arguments remain the
  model's decision. Loops are bounded (turns, total tool calls, revisions).
* ``critic`` routes: on REVISE the agent receives the reviewer's feedback and
  can gather the missing evidence before a new draft (a real correction
  loop, not a re-run of the same retrieval).
* Every node emits trace events through the per-request ``Tracer`` passed in
  LangGraph's ``config`` (thread-safe; the compiled graph is shared).
"""

from __future__ import annotations

import json
from typing import Any, Literal
from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel

from app.agent import evidence as ev
from app.agent.critic import Critic
from app.agent.planner import Plan, Planner
from app.agent.prompts import AGENT_SYSTEM, DISCLAIMER, GENERATOR_SYSTEM
from app.agent.state import AgentState
from app.agent.tools import ToolExecutor, section_catalog
from app.config import Settings
from app.llm import LLMGateway
from app.retrieval.hybrid import HybridRetriever, UnknownDrugError
from app.tracing import Tracer


class GeneratedAnswer(BaseModel):
    answer: str
    coverage: Literal["complete", "partial", "none"]
    gaps: list[str]


def _tracer(config: RunnableConfig) -> Tracer:
    return config["configurable"]["tracer"]


class ResearchAgent:
    def __init__(self, settings: Settings, llm: LLMGateway, retriever: HybridRetriever) -> None:
        self.settings = settings
        self.llm = llm
        self.retriever = retriever
        self.tools = ToolExecutor(retriever)
        self.planner = Planner(llm, retriever)
        self.critic = Critic(llm)
        self.graph = self._build()

    # ================================================================== graph
    def _build(self):
        g = StateGraph(AgentState)
        g.add_node("plan", self._plan)
        g.add_node("retrieve", self._retrieve)
        g.add_node("agent", self._agent)
        g.add_node("tools", self._tools)
        g.add_node("generate", self._generate)
        g.add_node("critic", self._critic)
        g.add_node("prepare_revision", self._prepare_revision)
        g.add_node("finalize", self._finalize)

        g.add_edge(START, "plan")
        g.add_conditional_edges("plan", self._route_after_plan, {"retrieve": "retrieve", "finalize": "finalize"})
        g.add_edge("retrieve", "agent")
        g.add_conditional_edges("agent", self._route_after_agent, {"tools": "tools", "generate": "generate"})
        g.add_edge("tools", "agent")
        g.add_edge("generate", "critic")
        g.add_conditional_edges("critic", self._route_after_critic,
                                {"revise": "prepare_revision", "finalize": "finalize"})
        g.add_edge("prepare_revision", "agent")
        g.add_edge("finalize", END)
        return g.compile()

    def run(self, question: str, tracer: Tracer | None = None) -> dict[str, Any]:
        tracer = tracer or Tracer(str(uuid4()), question)
        tracer.emit("request_received", question=question)
        state = self.graph.invoke(
            {"question": question, "trace_id": tracer.trace_id, "evidence": {}, "messages": [],
             "tool_log": [], "tool_calls_used": 0, "agent_turns": 0, "revisions": 0},
            config={"configurable": {"tracer": tracer}, "recursion_limit": 60},
        )
        result = state["result"]
        result["usage"] = tracer.usage_summary()
        tracer.emit("answer_emitted", answer_status=result["status"], sources=result["sources"],
                    confidence=result["review"].get("confidence"), **tracer.usage_summary())
        return result

    # ================================================================== nodes
    def _plan(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        with _tracer(config).span("plan_created") as t:
            plan = self.planner.plan(state["question"], tracer=_tracer(config))
            t.update(intent=plan.intent, drugs=plan.drugs, unknown_drugs=plan.unknown_drugs,
                     sub_questions=[s.model_dump() for s in plan.sub_questions])
        return {"plan": plan.model_dump()}

    def _retrieve(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        """Initial in-memory similarity retrieval: one search per sub-question + the full question."""

        evidence = dict(state["evidence"])
        plan = state["plan"]
        queries = [(s["question"], s["drug"]) for s in plan["sub_questions"]] + [(state["question"], None)]
        with _tracer(config).span("retrieval_completed", top_k=self.settings.retrieval_top_k) as t:
            per_query = []
            for query, drug in queries:
                try:
                    hits = self.retriever.search(query, self.settings.retrieval_top_k, drug=drug)
                except UnknownDrugError:
                    hits = self.retriever.search(query, self.settings.retrieval_top_k)
                eids = [ev.register(evidence, h.chunk, found_by="retrieve", score=h.score) for h in hits]
                per_query.append({
                    "query": query, "drug_filter": drug,
                    "results": [{"eid": e, "chunk_id": h.chunk.chunk_id, "rrf": h.score,
                                 "cosine": h.dense_score, "bm25": h.lexical_score} for e, h in zip(eids, hits)],
                })
            t.update(queries=per_query, unique_chunks=len(evidence))
        return {"evidence": evidence}

    def _agent(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        tracer = _tracer(config)
        budget = self.settings.max_tool_calls * (1 + state["revisions"]) - state["tool_calls_used"]
        if state["agent_turns"] >= self.settings.max_agent_turns or budget <= 0:
            tracer.emit("agent_stopped", reason="budget exhausted", turns=state["agent_turns"],
                        tool_calls_used=state["tool_calls_used"])
            return {}

        messages = list(state["messages"])
        system = AGENT_SYSTEM.format(drugs=", ".join(self.retriever.drugs), sections=section_catalog(),
                                     budget=budget)
        if not messages:
            plan = state["plan"]
            messages = [{"role": "user", "content": (
                f"QUESTION: {state['question']}\n\nPLAN ({plan['intent']}):\n"
                + "\n".join(f"{i}. {s['question']} [drug={s['drug']}, sections={s['sections']}]"
                            for i, s in enumerate(plan["sub_questions"], 1))
                + f"\nSynthesis: {plan['synthesis']}\n\nEVIDENCE ALREADY RETRIEVED (excerpts):\n"
                + ev.render(list(state["evidence"].values()), max_chars=300)
            )}]
        first_turn = state["agent_turns"] == 0 and state["revisions"] == 0
        message = self.llm.tool_turn(
            messages=[{"role": "system", "content": system}, *messages],
            tools=self.tools.specs(),
            tool_choice="required" if first_turn else "auto",
            step="agent",
            tracer=tracer,
        )
        assistant: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.function.name, "arguments": c.function.arguments}}
                for c in message.tool_calls
            ]
        tracer.emit(
            "tool_selected" if message.tool_calls else "agent_ready",
            turn=state["agent_turns"] + 1,
            calls=[{"tool": c.function.name, "arguments": _safe_json(c.function.arguments)}
                   for c in (message.tool_calls or [])],
            note=(message.content or "")[:300],
        )
        return {"messages": messages + [assistant], "agent_turns": state["agent_turns"] + 1}

    def _tools(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        tracer = _tracer(config)
        evidence = dict(state["evidence"])
        messages = list(state["messages"])
        tool_log = list(state["tool_log"])
        used = state["tool_calls_used"]
        budget = self.settings.max_tool_calls * (1 + state["revisions"])
        seen = {(t["tool"], json.dumps(t["arguments"], sort_keys=True)) for t in tool_log}

        for call in messages[-1].get("tool_calls", []):
            name, raw = call["function"]["name"], call["function"]["arguments"]
            signature = (name, json.dumps(_safe_json(raw), sort_keys=True))
            if used >= budget:
                content = "SKIPPED: tool-call budget exhausted. Work with the evidence you have."
                record = {"tool": name, "arguments": _safe_json(raw), "status": "skipped_budget"}
            elif signature in seen:
                content = "SKIPPED: identical call already executed; its evidence is already available."
                record = {"tool": name, "arguments": _safe_json(raw), "status": "skipped_duplicate"}
            else:
                used += 1
                seen.add(signature)
                outcome = self.tools.execute(name, raw)
                blocks, eids, last_group = [], [], None
                for chunk, score, group in outcome.chunks:
                    eid = ev.register(evidence, chunk, found_by=name, score=score)
                    eids.append(eid)
                    if group and group != last_group:
                        blocks.append(f"### {group.title()}")
                        last_group = group
                    blocks.append(f"{ev.header(evidence[chunk.chunk_id])}\n{chunk.text}")
                content = "\n\n".join(filter(None, [outcome.message, *blocks])) or "No result."
                record = {"tool": name, "arguments": outcome.arguments, "status": outcome.status,
                          "evidence_ids": eids, "chunk_ids": [c.chunk_id for c, _, _ in outcome.chunks],
                          "message": outcome.message[:300], "duration_ms": outcome.duration_ms}
            tool_log.append(record)
            tracer.emit("tool_executed", **record)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": content})
        return {"messages": messages, "evidence": evidence, "tool_log": tool_log, "tool_calls_used": used}

    def _generate(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        tracer = _tracer(config)
        selected = ev.select_for_answer(state["evidence"], self.settings.max_evidence)
        feedback = ""
        if state.get("critique"):
            c = state["critique"]
            feedback = (
                "\n\nREVIEWER FEEDBACK ON THE PREVIOUS DRAFT (fix these issues):\n"
                f"{c['revision_instructions']}\nFlagged claims: "
                + json.dumps([{k: f[k] for k in ('claim', 'status', 'note')} for f in c["flagged_claims"]],
                             ensure_ascii=False)
                + f"\nPrevious draft:\n{state['draft']['answer']}"
            )
        user = (f"QUESTION:\n{state['question']}\n\nPLAN:\n"
                + "\n".join(f"- {s['question']}" for s in state["plan"]["sub_questions"])
                + f"\n\nEVIDENCE:\n{ev.render(selected)}{feedback}")
        with tracer.span("draft_created", version=state["revisions"] + 1, evidence_in_prompt=len(selected)) as t:
            draft = self.llm.structured(system=GENERATOR_SYSTEM, user=user, schema=GeneratedAnswer,
                                        step="generate", tracer=tracer)
            t.update(coverage=draft.coverage, gaps=draft.gaps,
                     cited=sorted(set(ev.CITATION_RE.findall(draft.answer)), key=lambda e: int(e[1:])))
        return {"draft": draft.model_dump()}

    def _critic(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        tracer = _tracer(config)
        selected = ev.select_for_answer(state["evidence"], self.settings.max_evidence)
        with tracer.span("critic_completed", attempt=state["revisions"] + 1) as t:
            report = self.critic.review(question=state["question"], plan=state["plan"],
                                        answer=state["draft"]["answer"], evidence=selected, tracer=tracer)
            t.update(verdict=report["verdict"], llm_verdict=report["llm_verdict"], vetoes=report["vetoes"],
                     confidence=report["confidence"], audit=report["audit"],
                     flagged_claims=report["flagged_claims"], uncovered=report["uncovered"])
        return {"critique": report}

    def _prepare_revision(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        c = state["critique"]
        note = (f"REVIEWER FEEDBACK (revision {state['revisions'] + 1}): {c['revision_instructions']}\n"
                f"Uncovered: {c['uncovered']}\nIf the corpus may contain the missing evidence, call tools to "
                "fetch it; otherwise reply READY.")
        _tracer(config).emit("revision_requested", revision=state["revisions"] + 1,
                             instructions=c["revision_instructions"][:500])
        return {"messages": [*state["messages"], {"role": "user", "content": note}],
                "revisions": state["revisions"] + 1, "agent_turns": 0}

    def _finalize(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        plan, critique = state["plan"], state.get("critique") or {}
        limitations: list[str] = []
        if plan["unknown_drugs"]:
            limitations.append(f"Not covered by the corpus: {', '.join(plan['unknown_drugs'])}.")

        if plan["intent"] == "out_of_scope" or "draft" not in state:
            body = ("This question is outside the scope of the local corpus, so no grounded answer can be given. "
                    f"The corpus covers: {', '.join(self.retriever.drugs)} (FDA labels and MedlinePlus "
                    "monographs) plus a pharmacology glossary.")
            status, citations, review = "out_of_scope", [], {"verdict": "N/A", "confidence": None}
        else:
            evidence_by_eid = ev.by_eid(state["evidence"])
            body = state["draft"]["answer"].strip()
            used_correction = False
            if critique["verdict"] != "APPROVE" and critique.get("corrected_answer"):
                corrected_ids = ev.CITATION_RE.findall(critique["corrected_answer"])
                if corrected_ids and all(e in evidence_by_eid for e in corrected_ids):
                    body, used_correction = critique["corrected_answer"], True
            body, citations = _number_citations(body, evidence_by_eid)
            limitations += state["draft"]["gaps"] + [f"Not fully covered: {u}" for u in critique["uncovered"]]
            status = {"APPROVE": "answered", "REVISE": "answered_with_caveats",
                      "INSUFFICIENT_EVIDENCE": "insufficient_evidence"}[critique["verdict"]]
            review = {k: critique[k] for k in ("verdict", "llm_verdict", "vetoes", "confidence", "llm_confidence",
                                               "claims_checked", "claims_supported", "flagged_claims",
                                               "individualized_advice", "audit")}
            review["revisions"] = state["revisions"]
            review["used_corrected_answer"] = used_correction

        sources = sorted({c["source_file"] for c in citations})
        answer_md = (f"{body}\n\n**Sources:** {', '.join(sources) if sources else 'none'}"
                     f"\n\n> **Disclaimer:** {DISCLAIMER}")
        result = {
            "answer": answer_md,
            "answer_body": body,
            "status": status,
            "sources": sources,
            "citations": citations,
            "limitations": _dedupe(limitations),
            "review": review,
            "plan": plan,
            "tools_used": [{"tool": t["tool"], "arguments": t["arguments"], "status": t["status"]}
                           for t in state.get("tool_log", [])],
            "disclaimer": DISCLAIMER,
            "trace_id": state["trace_id"],
        }
        return {"result": result}

    # ================================================================== routing
    @staticmethod
    def _route_after_plan(state: AgentState) -> str:
        return "finalize" if state["plan"]["intent"] == "out_of_scope" else "retrieve"

    @staticmethod
    def _route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1] if state["messages"] else {}
        return "tools" if last.get("role") == "assistant" and last.get("tool_calls") else "generate"

    def _route_after_critic(self, state: AgentState) -> str:
        if state["critique"]["verdict"] == "REVISE" and state["revisions"] < self.settings.max_critic_retries:
            return "revise"
        return "finalize"


# ====================================================================== helpers
def _safe_json(raw: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"_raw": raw}
    except json.JSONDecodeError:
        return {"_raw": raw[:200]}


def _number_citations(body: str, evidence_by_eid: dict[str, dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Replace [E#] by [n] in order of first use; drop ids absent from the registry."""

    order: dict[str, int] = {}

    def replace(match) -> str:
        eid = match.group(1)
        if eid not in evidence_by_eid:
            return ""
        order.setdefault(eid, len(order) + 1)
        return f"[{order[eid]}]"

    numbered = ev.CITATION_RE.sub(replace, body)
    numbered = numbered.replace("][", "] [")
    citations = []
    for eid, ref in order.items():
        e = evidence_by_eid[eid]
        citations.append({
            "ref": ref, "evidence_id": eid, "source_file": e["source_file"], "drug": e["drug"],
            "source_type": e["source_type"], "section": " › ".join(e["section_path"]),
            "canonical_section": e["canonical_section"], "chunk_id": e["chunk_id"], "excerpt": e["text"],
        })
    return numbered, citations


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(i.strip() for i in items if i and i.strip()))
