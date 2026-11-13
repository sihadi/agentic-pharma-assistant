# Agentic Pharma Research Assistant

An agentic research assistant that answers single-document and multi-hop
pharmacology questions over a local corpus of FDA DailyMed labels and
MedlinePlus monographs. Every answer is **planned, retrieved, tool-augmented,
reviewed by a critic and cited claim by claim**, and it always carries a
*not medical advice* disclaimer.

```text
Question ─► Plan ─► Retrieve ─► Agent ⇄ Tools ─► Draft ─► Critic ─► Answer
           (LLM)   (FAISS+BM25)  (native function     (cited)  (audit+LLM, │  [n] citations
                                  calling, 4 tools)            veto)      │  sources, disclaimer
                                        ▲                                 │
                                        └──── REVISE (bounded) ◄──────────┘
```

Stack: **LangGraph** · **OpenAI SDK via the Syntax LiteLLM proxy** · **FAISS
(in-memory)** + BM25 · **FastAPI / Pydantic v2** · vanilla **HTML/CSS/JS** ·
multi-stage **Docker** (Python 3.14, which satisfies the "Python 3.11+"
requirement).

---

## 1. Run it (≤ 5 minutes)

### Docker (as required)

```bash
docker build -t agent .
docker run -p 8000:8000 --env-file .env agent
```

`.env` must provide `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` and
`OPENAI_EMBEDDING_MODEL` (see `.env.example`). The `.env` shipped by Syntax
works **as is**: its `OPENAI_BASE_UR` spelling is accepted, and its quoted
values are unquoted (`docker --env-file` passes quotes verbatim).

Then open **http://localhost:8000** (UI), or:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "What are the main contraindications for warfarin?"}'
```

`/health` answers `200` immediately. Its `status` is `starting` while the
index is built (about 10–20 s, the time to embed 587 chunks), then `ok`.
`/ask` waits for readiness automatically.

### Local (Python ≥ 3.11)

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                               # fill in the 4 variables
uvicorn app.main:app --port 8000
python -m pytest -q                                # 55 offline tests, no key needed
```

---

## 2. API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Always 200. `status`: `starting` / `ok` / `degraded` (+ cause, never secrets), corpus size, models |
| POST | `/ask` | `{"question": "..."}` → answer JSON (below) |
| POST | `/ask/stream` | Same, as Server-Sent Events: one `step` event per pipeline stage, then `result` |
| GET | `/trace/{trace_id}` | Full structured trace of a request |
| GET | `/corpus` | Data-lineage manifest (hashes, cleaning operations, chunk counts) |
| GET | `/docs` | OpenAPI documentation |

`POST /ask` response (abridged):

```json
{
  "answer": "Markdown answer with [1] citations …\n\n**Sources:** dailymed-metformin.md\n\n> **Disclaimer:** …",
  "status": "answered | answered_with_caveats | insufficient_evidence | out_of_scope",
  "sources": ["dailymed-metformin.md"],
  "citations": [{"ref": 1, "source_file": "dailymed-metformin.md", "section": "4 CONTRAINDICATIONS",
                 "chunk_id": "dailymed-metformin.md::contraindications::000", "excerpt": "…verbatim…"}],
  "review": {"verdict": "APPROVE", "confidence": 0.99, "flagged_claims": [], "audit": {"citation_coverage": 1.0, "numeric_issues": []}},
  "plan": {"intent": "population_or_threshold", "sub_questions": ["…"]},
  "tools_used": [{"tool": "get_drug_section", "arguments": {"drug": "metformin", "section": "dosage"}}],
  "limitations": [], "disclaimer": "…not medical advice…", "trace_id": "…", "usage": {"llm_calls": 5}
}
```

Errors: `422` invalid question, `503` not ready / misconfigured, `502` LLM
proxy failure, `500` other. A `trace_id` is always returned; stack traces and
secrets never are.

---

## 3. How it works (and why)

Design rationale: [ARCHITECTURE.md](ARCHITECTURE.md). Corpus audit:
[EXPLORATION.md](EXPLORATION.md). Module-by-module walkthrough:
[docs/TECHNICAL_WALKTHROUGH.md](docs/TECHNICAL_WALKTHROUGH.md). The design
documents are written in French.

### Data lineage: raw → cleaned → chunks

| Stage | Location | Content |
|---|---|---|
| Raw | `data/raw/` | The 21 Syntax files, byte-identical, never modified |
| Cleaned | `data/cleaned/` | 11 conservative, documented rules (lost superscripts `m 2`→`m²`, HTML `<sup>`, escapes, zero-width chars…), each substitution **counted** per document. Words, numbers and negations are never altered |
| Chunks | `data/chunks/chunks.jsonl` | 587 chunks with `chunk_id`, drug, source type, section path, canonical section, char offsets (`cleaned[start:end] == text`, tested) |
| Manifest | `data/manifest.json` | sha256 of raw and cleaned files, rules applied, parameters, fingerprint. The API rebuilds stale artifacts automatically |

Rebuild offline with `python -m app.ingestion`. The output is deterministic.

### Chunking and retrieval: chosen by measurement

- **Structure-aware chunking**: a heading tree, blocks (a list stays with its
  intro line, a table with its header), sentence-safe splits, 1500 chars max.
- **Contextual header** embedded with each chunk (`Warfarin — FDA DailyMed
  label / Section: 4 CONTRAINDICATIONS`); the cited text stays verbatim.
- **Hybrid search**: FAISS `IndexFlatIP` (exact cosine) + BM25, fused by
  **Reciprocal Rank Fusion**; metadata filters run inside FAISS
  (`IDSelectorBatch`).
- Drug names are resolved through **aliases extracted from the corpus**
  (Eliquis → apixaban, Coumadin → warfarin) and tolerate typos.

`python -m scripts.eval_retrieval`, on 22 labelled queries
([full table](docs/RETRIEVAL_EVALUATION_RESULTS.md)):

| Configuration | File recall@5 | Section hit@5 | MRR |
|---|---:|---:|---:|
| Fixed 800/100 windows, dense (common baseline) | .875 | .864 | .818 |
| + structure-aware chunks | .938 | .909 | .864 |
| + contextual header | .938 | .955 | .955 |
| **+ BM25 hybrid / RRF (runtime config)** | **1.000** | **1.000** | **.970** |
| BM25 only | .875 | .864 | .795 |

### Agent-driven tools (native OpenAI function calling)

| Tool | Purpose |
|---|---|
| `search_corpus` | Ranked hybrid search with optional drug / source / section filters |
| `get_drug_section` | The **complete** canonical section of a label (exhaustive lists such as contraindications) |
| `compare_drugs` | The same section for 2–4 drugs, grouped, for comparisons (organizes evidence, draws no conclusion) |
| `list_sections` | A drug's document outline, to find where information lives |

The model receives the four schemas and chooses which tools to call, with
which arguments and how many times. There is no keyword routing. Arguments are
validated with Pydantic, and errors come back to the agent as actionable
messages. Tool turns, total calls and revisions are budgeted, and duplicate
calls are skipped.

### Critic: two layers, with a veto

1. **Deterministic audit**: cited ids must exist, every factual sentence must
   be cited, and **every number in a cited sentence must appear in the cited
   passage** (catches invented doses and thresholds).
2. **LLM review** (strict structured output): each claim is classified as
   supported / unsupported / misattributed / ambiguous; the review also checks
   plan coverage and flags individualized advice, then returns a verdict, a
   confidence and a corrected answer.

The audit can overturn an LLM `APPROVE`. A `REVISE` verdict sends the
feedback back to the agent, which fetches the missing evidence before
redrafting. Citations use evidence ids from a registry of retrieved passages,
so **a citation cannot point to something that was not retrieved**.

### Traceability

Each request has a `trace_id`. Every step (plan, retrieval queries with chunk
ids and cosine / BM25 / RRF scores, tool choices and arguments, drafts, critic
verdicts and vetoes, tokens and latency per LLM call) is logged as JSON on
stdout, streamed live to the UI, served at `/trace/{id}` and saved to
`data/traces/`. The model's private reasoning and secrets are never logged.

---

## 4. Evaluation questions (end-to-end, real proxy)

`python -m scripts.run_eval_questions` ([full answers](docs/EVAL_RESULTS.md)):

| Id | Question | Status | Critic | Cited sources | Time |
|---|---|---|---|---|---:|
| P1 | Main contraindications for warfarin | answered | APPROVE 0.99 | dailymed-warfarin.md | 16.5 s |
| P2 | Compare contraindications warfarin vs apixaban | answered | APPROVE 0.80 | dailymed-warfarin.md, dailymed-apixaban.md | 21.7 s |
| P3 | Warfarin + sertraline: bleeding risk and monitoring | answered | APPROVE 0.99 | dailymed-warfarin.md, dailymed-sertraline.md | 22.3 s |
| P4 | Metformin with eGFR 25: contraindicated? threshold? | answered | APPROVE 0.99 | dailymed-metformin.md | 12.6 s |
| U1 | Zoloft with an MAO inhibitor (brand name) | answered | APPROVE 0.99 | dailymed-sertraline.md, medlineplus-sertraline.md | 18.2 s |
| U2 | Acetaminophen maximum daily dose | answered | APPROVE 0.99 | dailymed-/medlineplus-acetaminophen.md | 20.2 s |
| U3 | Contraindications of lisinopril (not in corpus) | out_of_scope | — | — | 2.0 s |
| U4 | 2022 World Cup winner (off-topic) | out_of_scope | — | — | 1.8 s |

All expected source files are cited and every answer carries the disclaimer.
Confidence is the LLM critic's confidence × the deterministic grounding score;
in P2, 2 of 11 sentences carry their citation one sentence late, and the
audit reports it.

Run-to-run variation is expected. In a later P4 run inside the Docker
container, the critic flagged a **genuine clinical ambiguity**: the question
gives "25 mL/min", while the label threshold is indexed "mL/min/1.73 m²". It
also rejected a "confirm the lab value" sentence as advice not stated in the
label. The result was `answered_with_caveats` with the flags shown, which is
the behavior the assignment asks for.

---

## 5. Project structure

```text
app/
  main.py, models.py      FastAPI endpoints + Pydantic contracts
  config.py               environment-only settings (+ Syntax .env quirks)
  llm.py                  single gateway: embeddings, structured outputs, tool calling, retries
  tracing.py              per-request tracer + trace store
  runtime.py              composition root
  ingestion/              taxonomy, cleaner, chunker, pipeline (python -m app.ingestion)
  retrieval/              bm25, embedding cache, hybrid FAISS+BM25 retriever
  agent/                  state, prompts, planner, tools, evidence registry, critic, LangGraph graph
  static/                 frontend (index.html, style.css, app.js)
data/                     raw/ cleaned/ chunks/ manifest.json   (cache/ and traces/ are git-ignored)
eval/                     labelled retrieval queries
scripts/                  eval_retrieval.py, run_eval_questions.py
tests/                    55 offline tests (scripted LLM, fake embeddings, real corpus)
docs/                     evaluation results + technical walkthrough
```

## 6. Configuration

| Variable | Default | Role |
|---|---|---|
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`, `OPENAI_EMBEDDING_MODEL` | required | LiteLLM proxy access |
| `RETRIEVAL_TOP_K` | 5 | passages per initial search |
| `CHUNK_MAX_CHARS` | 1500 | chunk budget (rebuilds artifacts if changed) |
| `MAX_AGENT_TURNS` / `MAX_TOOL_CALLS` / `MAX_CRITIC_RETRIES` | 4 / 8 / 1 | loop budgets |
| `LOG_LEVEL`, `PERSIST_TRACES`, `LLM_TIMEOUT_S` | INFO, true, 90 | operations |

## 7. Limitations

- Answers are only as complete as the 21 provided documents. Absent
  information is reported as such, never filled in from general knowledge.
- Retrieval metrics come from a small labelled set (22 queries). The critic's
  confidence is a ranking signal, not a calibrated probability.
- The FAISS index is rebuilt in memory at start-up (the embedding cache only
  speeds up restarts). Each question costs about 5 LLM calls (≈ 12–25 s).

---

> **Disclaimer:** This information is generated from publicly available drug
> labels and is for educational/research purposes only. It is not medical
> advice. Always consult a qualified healthcare professional before making any
> medical decision.
