"""Ingestion pipeline with full data lineage.

    data/raw/*.md            immutable copy of the Syntax corpus (never modified)
        │  clean_text()      conservative, rule-based, every substitution counted
        ▼
    data/cleaned/*.md        what is chunked and cited (offsets refer to it)
        │  chunk_document()  structure-aware chunks + contextual embedding text
        ▼
    data/chunks/chunks.jsonl one JSON object per chunk
    data/manifest.json       sha256 of raw + cleaned files, rules applied,
                             parameters, per-document counts, fingerprint

The output is deterministic (no timestamps): re-running on the same input
produces byte-identical artifacts, and the ``fingerprint`` lets the API detect
a stale chunk file and rebuild it automatically at start-up.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.ingestion.chunker import Chunk, DocumentMeta, chunk_document, parse_document_meta
from app.ingestion.cleaner import RULES, clean_text

LOGGER = logging.getLogger(__name__)
PIPELINE_VERSION = "2.0"
SUPPORTED_SUFFIXES = (".md", ".txt")


@dataclass
class Corpus:
    documents: list[DocumentMeta]
    chunks: list[Chunk]
    manifest: dict[str, Any]


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _raw_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"Raw corpus directory not found: {raw_dir}")
    # Sort on the *string* name: Path ordering is case-insensitive on Windows but
    # case-sensitive on Linux, which made the fingerprint OS-dependent.
    files = sorted((p for p in raw_dir.iterdir() if p.suffix.lower() in SUPPORTED_SUFFIXES), key=lambda p: p.name)
    if not files:
        raise FileNotFoundError(f"No .md/.txt documents in {raw_dir}")
    return files


def corpus_fingerprint(raw_dir: Path, max_chars: int) -> str:
    """Hash of raw contents + pipeline parameters: changes iff output would."""

    digest = hashlib.sha256(f"{PIPELINE_VERSION}|{max_chars}".encode())
    for path in _raw_files(raw_dir):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_corpus(raw_dir: Path, max_chars: int) -> tuple[Corpus, dict[str, str]]:
    """Run raw -> cleaned -> chunks in memory. Returns corpus + cleaned texts."""

    documents: list[DocumentMeta] = []
    chunks: list[Chunk] = []
    cleaned_texts: dict[str, str] = {}
    doc_entries: list[dict[str, Any]] = []
    warnings: list[str] = []

    for path in _raw_files(raw_dir):
        try:
            raw = path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError:
            warnings.append(f"{path.name}: not valid UTF-8, skipped")
            continue
        if not raw.strip():
            warnings.append(f"{path.name}: empty file, skipped")
            continue
        cleaned, ops = clean_text(raw)
        meta = parse_document_meta(path.name, cleaned)
        doc_chunks = chunk_document(path.name, cleaned, max_chars=max_chars)
        documents.append(meta)
        chunks.extend(doc_chunks)
        cleaned_texts[path.name] = cleaned
        doc_entries.append(
            {
                **asdict(meta),
                "raw_sha256": _sha256(raw),
                "cleaned_sha256": _sha256(cleaned),
                "raw_chars": len(raw),
                "cleaned_chars": len(cleaned),
                "cleaning_ops": ops,
                "chunks": len(doc_chunks),
                "sections": sorted({c.canonical_section for c in doc_chunks}),
            }
        )

    sizes = sorted(len(c.text) for c in chunks)
    manifest = {
        "pipeline_version": PIPELINE_VERSION,
        "fingerprint": corpus_fingerprint(raw_dir, max_chars),
        "parameters": {"chunk_max_chars": max_chars, "strategy": "structure-aware + contextual header"},
        "cleaning_rules": [{"name": r.name, "rationale": r.rationale} for r in RULES],
        "totals": {
            "documents": len(documents),
            "chunks": len(chunks),
            "chunk_chars_min": sizes[0] if sizes else 0,
            "chunk_chars_median": sizes[len(sizes) // 2] if sizes else 0,
            "chunk_chars_max": sizes[-1] if sizes else 0,
            "drugs": sorted({d.drug for d in documents if d.drug}),
        },
        "warnings": warnings,
        "documents": doc_entries,
    }
    return Corpus(documents, chunks, manifest), cleaned_texts


def write_artifacts(corpus: Corpus, cleaned: dict[str, str], cleaned_dir: Path,
                    chunks_path: Path, manifest_path: Path) -> None:
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    for name, text in cleaned.items():
        (cleaned_dir / name).write_text(text, encoding="utf-8", newline="\n")
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    with chunks_path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in corpus.chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    manifest_path.write_text(
        json.dumps(corpus.manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"
    )


def load_corpus(raw_dir: Path, cleaned_dir: Path, chunks_path: Path, manifest_path: Path,
                max_chars: int) -> Corpus:
    """Load persisted chunks if they match the raw corpus, else rebuild them."""

    fingerprint = corpus_fingerprint(raw_dir, max_chars)
    if manifest_path.is_file() and chunks_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("fingerprint") == fingerprint:
            chunks = [
                Chunk.from_dict(json.loads(line))
                for line in chunks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            documents = [
                DocumentMeta(**{k: d[k] for k in DocumentMeta.__dataclass_fields__})
                for d in manifest["documents"]
            ]
            LOGGER.info("corpus_loaded", extra={"source": "artifacts", "chunks": len(chunks)})
            return Corpus(documents, chunks, manifest)
        LOGGER.warning("corpus_artifacts_stale", extra={"action": "rebuild"})

    corpus, cleaned = build_corpus(raw_dir, max_chars)
    try:
        write_artifacts(corpus, cleaned, cleaned_dir, chunks_path, manifest_path)
    except OSError:
        LOGGER.warning("corpus_artifacts_not_written", extra={"reason": "read-only filesystem"})
    LOGGER.info("corpus_loaded", extra={"source": "rebuilt", "chunks": len(corpus.chunks)})
    return corpus
