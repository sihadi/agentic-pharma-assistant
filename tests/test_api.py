"""HTTP contract: /health always 200, /ask and /ask/stream behaviour."""

from __future__ import annotations

import json
import time

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import ConfigurationError
from app.runtime import Runtime

from .conftest import critique, eid_for, tool_message, warfarin_plan


def _wait_ready(client: TestClient) -> dict:
    for _ in range(100):
        body = client.get("/health").json()
        if body["status"] != "starting":
            return body
        time.sleep(0.05)
    raise AssertionError("runtime never became ready")


@pytest.fixture
def client(monkeypatch, make_agent, corpus, retriever, settings):
    def fake_runtime(_settings):
        agent, llm = make_agent(
            {"plan": [warfarin_plan()] * 2,
             "generate": [lambda u: {"answer": f"Pregnancy [{eid_for(u, 'dailymed-warfarin.md', '4 CONTRAINDICATIONS')}].",
                                     "coverage": "complete", "gaps": []}] * 2,
             "critic": [critique()] * 2},
            [tool_message(("get_drug_section", {"drug": "warfarin", "section": "contraindications"}))],
        )
        return Runtime(settings, corpus, llm, retriever, agent)

    monkeypatch.setattr(main, "load_settings", lambda: settings)
    monkeypatch.setattr(main, "build_runtime", fake_runtime)
    with TestClient(main.app) as test_client:
        yield test_client


def test_health_is_200_and_reports_readiness(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = _wait_ready(client)
    assert body["status"] == "ok" and body["ready"] is True
    assert body["corpus"]["documents"] == 21
    assert "openai_api_key" not in json.dumps(body) and "test-key" not in json.dumps(body)


def test_ask_returns_cited_answer_with_disclaimer(client):
    _wait_ready(client)
    response = client.post("/ask", json={"question": "What are the main contraindications for warfarin?"})
    assert response.status_code == 200
    body = response.json()
    assert body["sources"] == ["dailymed-warfarin.md"]
    assert "not medical advice" in body["answer"] and body["disclaimer"]
    assert body["citations"][0]["source_file"] == "dailymed-warfarin.md"
    assert client.get(f"/trace/{body['trace_id']}").status_code == 200


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 2001}])
def test_ask_rejects_invalid_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_ask_stream_emits_steps_then_result(client):
    _wait_ready(client)
    with client.stream("POST", "/ask/stream", json={"question": "warfarin contraindications?"}) as response:
        text = "".join(response.iter_text())
    names = [line[7:] for line in text.splitlines() if line.startswith("event: ")]
    assert names[0] == "start" and names[-1] == "result"
    assert names.count("step") >= 8


def test_health_stays_200_when_configuration_is_missing(monkeypatch):
    def broken():
        raise ConfigurationError("Missing required environment variable: OPENAI_API_KEY")

    monkeypatch.setattr(main, "load_settings", broken)
    with TestClient(main.app) as client:
        body = _wait_ready(client)
        assert client.get("/health").status_code == 200
        assert body["status"] == "degraded" and "OPENAI_API_KEY" in body["error"]
        assert client.post("/ask", json={"question": "warfarin?"}).status_code == 503
