"""End-to-end LangGraph behaviour with a scripted LLM (real corpus + retriever)."""

from __future__ import annotations

from app.agent.prompts import DISCLAIMER
from app.tracing import Tracer

from .conftest import critique, eid_for, tool_message, warfarin_plan

CONTRA = ("dailymed-warfarin.md", "4 CONTRAINDICATIONS")


def answer_citing_contraindications(text: str = "Warfarin is contraindicated in pregnancy"):
    return lambda user: {"answer": f"{text} [{eid_for(user, *CONTRA)}].", "coverage": "complete", "gaps": []}


def events(tracer: Tracer) -> list[str]:
    return [e["event"] for e in tracer.events]


def test_full_pipeline_plan_retrieve_tools_critic_answer(make_agent):
    agent, llm = make_agent(
        {"plan": [warfarin_plan()], "generate": [answer_citing_contraindications()], "critic": [critique()]},
        [tool_message(("get_drug_section", {"drug": "warfarin", "section": "contraindications"})),
         tool_message(content="READY: complete")],
    )
    tracer = Tracer("t1", "What are the main contraindications for warfarin?")
    result = agent.run(tracer.question, tracer)

    assert result["status"] == "answered"
    assert result["sources"] == ["dailymed-warfarin.md"]
    assert result["answer_body"].endswith("[1].")
    assert DISCLAIMER in result["answer"] and "not medical advice" in result["answer"]
    assert result["citations"][0]["chunk_id"].startswith("dailymed-warfarin.md::contraindications::")
    assert [t["tool"] for t in result["tools_used"]] == ["get_drug_section"]
    # The tool step is mandatory on the first turn, but the model chooses which tool.
    agent_calls = [c for c in llm.calls if c["step"] == "agent"]
    assert agent_calls[0]["tool_choice"] == "required" and agent_calls[1]["tool_choice"] == "auto"
    assert len(agent_calls[0]["tools"]) == 4
    for expected in ["request_received", "plan_created", "retrieval_completed", "tool_selected",
                     "tool_executed", "agent_ready", "draft_created", "critic_completed", "answer_emitted"]:
        assert expected in events(tracer)


def test_critic_revise_triggers_targeted_new_evidence_and_redraft(make_agent):
    agent, llm = make_agent(
        {"plan": [warfarin_plan()],
         "generate": [answer_citing_contraindications(), answer_citing_contraindications("Pregnancy is listed")],
         "critic": [critique("REVISE", 0.4), critique("APPROVE", 0.95)]},
        [tool_message(("get_drug_section", {"drug": "warfarin", "section": "contraindications"})),
         tool_message(content="READY"),
         tool_message(("search_corpus", {"query": "mechanical heart valves pregnancy", "drug": "warfarin"})),
         tool_message(content="READY")],
    )
    tracer = Tracer("t2", "warfarin contraindications?")
    result = agent.run(tracer.question, tracer)

    assert result["review"]["revisions"] == 1 and result["status"] == "answered"
    assert events(tracer).count("critic_completed") == 2 and "revision_requested" in events(tracer)
    assert [t["tool"] for t in result["tools_used"]] == ["get_drug_section", "search_corpus"]
    # The second draft receives the reviewer's feedback.
    second_generate = [c for c in llm.calls if c["step"] == "generate"][1]
    assert "REVIEWER FEEDBACK" in second_generate["user"]


def test_deterministic_audit_vetoes_an_llm_approval(make_agent):
    agent, _ = make_agent(
        {"plan": [warfarin_plan()],
         "generate": [answer_citing_contraindications("Warfarin is contraindicated above 97 years of age")],
         "critic": [critique("APPROVE", 0.99)]},
        [tool_message(("list_sections", {"drug": "warfarin"}))],
        max_critic_retries=0,
    )
    result = agent.run("warfarin contraindications?")
    review = result["review"]
    assert review["llm_verdict"] == "APPROVE" and review["verdict"] == "REVISE"
    assert "numbers not found in cited evidence" in review["vetoes"]
    assert result["status"] == "answered_with_caveats"
    assert review["confidence"] < 0.99


def test_unknown_citations_are_stripped_from_the_final_answer(make_agent):
    agent, _ = make_agent(
        {"plan": [warfarin_plan()],
         "generate": [lambda user: {"answer": f"Pregnancy [{eid_for(user, *CONTRA)}][E999].",
                                    "coverage": "complete", "gaps": []}],
         "critic": [critique()]},
        [tool_message(("get_drug_section", {"drug": "warfarin", "section": "contraindications"}))],
        max_critic_retries=0,
    )
    result = agent.run("warfarin contraindications?")
    assert "E999" not in result["answer"] and len(result["citations"]) == 1


def test_out_of_scope_question_short_circuits_without_tools(make_agent):
    plan = warfarin_plan("out_of_scope") | {"drugs": [], "unknown_drugs": [], "sub_questions": []}
    agent, llm = make_agent({"plan": [plan]}, [])
    result = agent.run("Who won the World Cup?")
    assert result["status"] == "out_of_scope" and result["citations"] == []
    assert [c["step"] for c in llm.calls] == ["plan"]
    assert DISCLAIMER in result["answer"]


def test_agent_loop_stops_when_tool_budget_is_spent(make_agent):
    endless = [tool_message(("search_corpus", {"query": f"warfarin q{i}"})) for i in range(20)]
    agent, llm = make_agent(
        {"plan": [warfarin_plan()], "generate": [answer_citing_contraindications()], "critic": [critique()]},
        endless, max_agent_turns=3, max_tool_calls=2,
    )
    tracer = Tracer("t-budget", "warfarin?")
    result = agent.run("warfarin?", tracer)
    assert len([c for c in llm.calls if c["step"] == "agent"]) == 2
    assert len(result["tools_used"]) == 2
    assert "agent_stopped" in events(tracer) and result["status"] == "answered"


def test_parallel_calls_beyond_budget_and_duplicates_are_skipped(make_agent):
    same = ("get_drug_section", {"drug": "warfarin", "section": "contraindications"})
    agent, _ = make_agent(
        {"plan": [warfarin_plan()], "generate": [answer_citing_contraindications()], "critic": [critique()]},
        [tool_message(same, same, ("list_sections", {"drug": "warfarin"}), ("search_corpus", {"query": "INR"}))],
        max_tool_calls=2,
    )
    statuses = [t["status"] for t in agent.run("warfarin?")["tools_used"]]
    assert statuses == ["ok", "skipped_duplicate", "ok", "skipped_budget"]
