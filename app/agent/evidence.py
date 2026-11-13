"""Evidence registry: the single source of truth for what may be cited.

Every chunk that reaches the agent (initial retrieval or any tool) is
registered once under a stable short id (``E1``, ``E2``...). The generator must
cite those ids inline; the finalizer converts them to numbered references and
silently drops any id that is not in the registry - so a citation can never
point to a passage that was not actually retrieved.
"""

from __future__ import annotations

import re
from typing import Any

from app.ingestion.chunker import Chunk

CITATION_RE = re.compile(r"\[(E\d+)\]")


def register(
    evidence: dict[str, dict[str, Any]], chunk: Chunk, *, found_by: str, score: float | None = None
) -> str:
    """Add ``chunk`` to ``evidence`` (in place) and return its evidence id."""

    entry = evidence.get(chunk.chunk_id)
    if entry is None:
        entry = {
            "eid": f"E{len(evidence) + 1}",
            "chunk_id": chunk.chunk_id,
            "source_file": chunk.source_file,
            "drug": chunk.drug,
            "source_type": chunk.source_type,
            "section": chunk.section,
            "section_path": chunk.section_path,
            "canonical_section": chunk.canonical_section,
            "text": chunk.text,
            "found_by": [],
            "best_score": None,
        }
        evidence[chunk.chunk_id] = entry
    if found_by not in entry["found_by"]:
        entry["found_by"].append(found_by)
    if score is not None and (entry["best_score"] is None or score > entry["best_score"]):
        entry["best_score"] = score
    return entry["eid"]


def by_eid(evidence: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {entry["eid"]: entry for entry in evidence.values()}


def header(entry: dict[str, Any]) -> str:
    drug = (entry["drug"] or "general").title()
    return f"[{entry['eid']}] {entry['source_file']} | {drug} | {' > '.join(entry['section_path'])}"


def render(entries: list[dict[str, Any]], max_chars: int | None = None) -> str:
    """Render evidence for a prompt: header line + (optionally truncated) text."""

    blocks = []
    for entry in entries:
        text = entry["text"]
        if max_chars and len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " …"
        blocks.append(f"{header(entry)}\n{text}")
    return "\n\n".join(blocks)


def select_for_answer(evidence: dict[str, dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """Keep the most useful evidence within the prompt budget.

    Tool results are targeted by the agent, so they rank before passages that
    only came from the broad initial retrieval; ties break on fused score.
    """

    def priority(entry: dict[str, Any]) -> tuple[int, float]:
        targeted = any(source != "retrieve" for source in entry["found_by"])
        return (0 if targeted else 1, -(entry["best_score"] or 0.0))

    chosen = sorted(evidence.values(), key=priority)[:limit]
    return sorted(chosen, key=lambda e: int(e["eid"][1:]))
