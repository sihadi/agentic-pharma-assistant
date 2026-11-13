# AI coding configuration (Claude Code)

This file is the project guidance given to the AI coding assistant (Claude
Code) used to build this repository, published per the assignment's
"AI Assistance — Policy & Transparency" section. No MCP servers or custom
skills were used.

## Project rules given to the assistant
- Stack is fixed: LangGraph, OpenAI SDK through the Syntax LiteLLM proxy
  (env vars only), FAISS in-memory, FastAPI + Pydantic v2, vanilla
  HTML/CSS/JS, multi-stage Dockerfile. Python 3.11+ (built on 3.14).
- Never hard-code keys, models or paths; never log or commit secrets.
- Keep `data/raw` immutable; regenerate `data/cleaned`, `data/chunks` and
  `data/manifest.json` with `python -m app.ingestion` after any change to
  cleaning or chunking, and keep `tests/test_ingestion.py` green.
- Every design choice must be justified; retrieval changes must be measured
  with `python -m scripts.eval_retrieval`.
- Answers must stay grounded in the corpus, cite passages, and always carry
  the "not medical advice" disclaimer.

## Commands
- Tests (offline): `python -m pytest -q`
- Ingestion: `python -m app.ingestion`
- Retrieval evaluation: `python -m scripts.eval_retrieval`
- End-to-end evaluation: `python -m scripts.run_eval_questions`
- Run: `uvicorn app.main:app --port 8000`
