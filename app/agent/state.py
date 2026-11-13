"""Typed, JSON-serialisable state carried between LangGraph nodes."""

from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    trace_id: str
    plan: dict[str, Any]  # Plan.model_dump()
    evidence: dict[str, dict[str, Any]]  # chunk_id -> evidence entry (see evidence.py)
    messages: list[dict[str, Any]]  # OpenAI-format tool-calling conversation of the agent
    tool_log: list[dict[str, Any]]  # one record per executed tool call
    tool_calls_used: int
    agent_turns: int
    draft: dict[str, Any]  # {"answer", "coverage", "gaps"}
    critique: dict[str, Any]  # critic report (see critic.py)
    revisions: int
    result: dict[str, Any]  # final API payload
