"""Structure-aware chunking of cleaned Markdown documents.

Strategy (justified + measured in docs/RETRIEVAL_EVALUATION_RESULTS.md):

1. **Heading tree** - every chunk belongs to exactly one leaf section and
   carries its full breadcrumb (``5 WARNINGS AND PRECAUTIONS > 5.3 Increased
   Risk of Bleeding``) and a canonical section key (see ``taxonomy.py``).
2. **Semantic blocks** - inside a section, paragraphs / lists / tables are
   blocks. A line ending with ``:`` is glued to the list that follows, so
   "contraindicated in patients with:" never gets separated from its items.
3. **Greedy packing** up to ``max_chars``; oversized blocks are split on
   line, then sentence boundaries (never inside a sentence unless a single
   sentence exceeds the budget) - negations, conditions and units stay whole.
4. **Contextual embedding text** - ``text`` is a *verbatim* slice of the
   cleaned file (``cleaned[start_char:end_char] == text``), used for display
   and citation. ``embedding_text`` prepends drug, source and section path
   (+ the list lead-in for continuation chunks): a chunk that only says
   "Pregnancy" still embeds as a *warfarin contraindication*.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

from app.ingestion.taxonomy import canonical_section

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$", re.MULTILINE)
SOURCE_LINE_RE = re.compile(r"^Source:\s*(\S+)\s*$", re.MULTILINE)
BLANK_SPLIT_RE = re.compile(r"\n[ \t]*\n")
SENTENCE_END_RE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\[])")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")

SOURCE_LABELS = {
    "dailymed": "FDA DailyMed prescribing label",
    "medlineplus": "MedlinePlus consumer drug monograph",
    "glossary": "Pharmacology glossary",
}


@dataclass(frozen=True)
class DocumentMeta:
    source_file: str
    drug: str | None
    source_type: str
    title: str
    source_url: str | None
    aliases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source_file: str
    drug: str | None
    source_type: str
    section: str
    section_path: list[str]
    canonical_section: str
    text: str
    embedding_text: str
    start_char: int
    end_char: int
    part: int
    parts: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        return cls(**data)

    def citation_label(self) -> str:
        return f"{self.source_file} › {' › '.join(self.section_path) or self.section}"


# --------------------------------------------------------------------------- metadata
def parse_document_meta(source_file: str, text: str) -> DocumentMeta:
    """Derive drug/source metadata from the file-naming convention and header."""

    stem = source_file.rsplit(".", 1)[0].lower()
    if stem == "glossary":
        drug, source_type = None, "glossary"
    else:
        source_type, _, drug_part = stem.partition("-")
        drug = drug_part.replace("-", " ") or None
        if source_type not in SOURCE_LABELS:
            source_type = "other"
    title_match = re.search(r"^#[ \t]+(.+)$", text, re.MULTILINE)
    url_match = SOURCE_LINE_RE.search(text)
    return DocumentMeta(
        source_file=source_file,
        drug=drug,
        source_type=source_type,
        title=title_match.group(1).strip() if title_match else source_file,
        source_url=url_match.group(1) if url_match else None,
        aliases=_extract_aliases(text),
    )


def _extract_aliases(text: str) -> list[str]:
    """Brand / other names listed by MedlinePlus (data-derived alias map).

    Combination products are deliberately ignored: "Percocet" must not be
    resolved to a single ingredient.
    """

    aliases: list[str] = []
    for match in re.finditer(r"^##[ \t]+(Brand names|Other names)[ \t]*$", text, re.MULTILINE):
        body = text[match.end():]
        next_heading = HEADING_RE.search(body)
        body = body[: next_heading.start()] if next_heading else body
        for line in body.splitlines():
            name = re.sub(r"[®¶™*]", "", LIST_ITEM_RE.sub("", line)).strip()
            if name and len(name) <= 40 and not name.lower().startswith("this "):
                aliases.append(name)
    return sorted(set(aliases))


# --------------------------------------------------------------------------- sections
@dataclass(frozen=True)
class _Section:
    title: str
    path: list[str]
    canonical: str
    body_start: int
    body_end: int


def _sections(text: str, source_type: str) -> list[_Section]:
    headings = list(HEADING_RE.finditer(text))
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []
    for i, match in enumerate(headings):
        level, title = len(match.group(1)), match.group(2).strip()
        body_end = headings[i + 1].start() if i + 1 < len(headings) else len(text)
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        path = [t for lvl, t in stack if lvl > 1]  # level 1 = document title
        if level == 1:
            # Text between the title and the first section. In FDA labels this
            # position holds the boxed warning (it precedes section 1), and one
            # label (ibuprofen) ships it without a heading.
            if source_type == "dailymed":
                title, canonical = "Boxed warning (untitled)", "boxed_warning"
            else:
                title = "Document header"
                canonical = canonical_section(title, source_type)  # "glossary" or "other"
            sections.append(_Section(title, [title], canonical, match.end(), body_end))
            continue
        top = path[0] if path else title
        sections.append(
            _Section(title, path, canonical_section(top, source_type), match.end(), body_end)
        )
    return sections


# --------------------------------------------------------------------------- blocks
@dataclass
class _Span:
    start: int
    end: int
    lead_in: str | None = None  # context line for continuation pieces


def _blocks(text: str, start: int, end: int) -> list[_Span]:
    """Paragraph/list/table blocks; intro lines ending with ':' join the next block."""

    blocks: list[_Span] = []
    cursor = start
    for sep in list(BLANK_SPLIT_RE.finditer(text, start, end)) + [None]:
        block_end = sep.start() if sep else end
        segment = text[cursor:block_end]
        if segment.strip():
            lead = len(segment) - len(segment.lstrip())
            blocks.append(_Span(cursor + lead, cursor + len(segment.rstrip())))
        cursor = sep.end() if sep else end

    merged: list[_Span] = []
    for block in blocks:
        prev = merged[-1] if merged else None
        prev_text = text[prev.start:prev.end] if prev else ""
        is_list = bool(LIST_ITEM_RE.match(text[block.start:block.end]))
        prev_is_list = bool(prev) and bool(LIST_ITEM_RE.match(prev_text.splitlines()[-1]))
        if prev and (prev_text.rstrip().endswith(":") or (is_list and prev_is_list)):
            prev.end = block.end
        else:
            merged.append(block)
    return merged


def _units(text: str, span: _Span, max_chars: int) -> list[_Span]:
    """Split an oversized block on lines, then sentences, then whitespace."""

    body = text[span.start:span.end]
    first_line = body.split("\n", 1)[0].strip()
    lead_in = first_line if (first_line.endswith(":") or "|" in first_line) else None
    lead_in = lead_in[:200] if lead_in else None

    pieces = [_Span(span.start, span.end)]
    for splitter in (re.compile(r"\n"), SENTENCE_END_RE, re.compile(r"\s+")):
        refined: list[_Span] = []
        for piece in pieces:
            if piece.end - piece.start <= max_chars:
                refined.append(piece)
                continue
            cursor = piece.start
            for m in splitter.finditer(text, piece.start, piece.end):
                refined.append(_Span(cursor, m.start()))
                cursor = m.end()
            refined.append(_Span(cursor, piece.end))
        pieces = [p for p in refined if text[p.start:p.end].strip()]
    for piece in pieces[1:]:
        piece.lead_in = lead_in
    return pieces


def _pack(text: str, spans: list[_Span], max_chars: int) -> list[_Span]:
    packed: list[_Span] = []
    for span in spans:
        if packed and span.end - packed[-1].start <= max_chars:
            packed[-1].end = span.end
        else:
            packed.append(_Span(span.start, span.end, span.lead_in))
    return packed


# --------------------------------------------------------------------------- public API
def chunk_document(source_file: str, text: str, max_chars: int = 1500) -> list[Chunk]:
    """Chunk one *cleaned* document into citation-safe, context-rich chunks."""

    if max_chars < 100:
        raise ValueError("max_chars is too small to hold a meaningful passage")
    meta = parse_document_meta(source_file, text)
    drug_label = meta.drug.title() if meta.drug else "General"
    chunks: list[Chunk] = []
    counters: dict[str, int] = {}

    for section in _sections(text, meta.source_type):
        units: list[_Span] = []
        for block in _blocks(text, section.body_start, section.body_end):
            block_text = text[block.start:block.end]
            if SOURCE_LINE_RE.fullmatch(block_text.strip()):
                continue  # the URL is kept as document metadata, not as evidence
            units.extend(
                _units(text, block, max_chars) if len(block_text) > max_chars else [block]
            )
        pieces = _pack(text, units, max_chars)
        for part, piece in enumerate(pieces, start=1):
            body = text[piece.start:piece.end]
            n = counters.get(section.canonical, 0)
            counters[section.canonical] = n + 1
            header = (
                f"{drug_label} — {SOURCE_LABELS.get(meta.source_type, meta.source_type)}\n"
                f"Section: {' > '.join(section.path)}\n"
            )
            if piece.lead_in and part > 1:
                header += f"(continued: {piece.lead_in})\n"
            chunks.append(
                Chunk(
                    chunk_id=f"{source_file}::{section.canonical}::{n:03d}",
                    source_file=source_file,
                    drug=meta.drug,
                    source_type=meta.source_type,
                    section=section.title,
                    section_path=section.path,
                    canonical_section=section.canonical,
                    text=body,
                    embedding_text=f"{header}\n{body}",
                    start_char=piece.start,
                    end_char=piece.end,
                    part=part,
                    parts=len(pieces),
                )
            )
    return chunks


def chunk_fixed_size(source_file: str, text: str, size: int = 800, overlap: int = 100) -> list[Chunk]:
    """Naive fixed-window baseline (the 800/100 recommendation of the spec).

    Used only by ``scripts/eval_retrieval.py`` to *measure* what structure-aware
    chunking buys us; it is not used at runtime.
    """

    meta = parse_document_meta(source_file, text)
    chunks: list[Chunk] = []
    for n, start in enumerate(range(0, len(text), size - overlap)):
        body = text[start:start + size]
        if not body.strip():
            continue
        chunks.append(
            Chunk(
                chunk_id=f"{source_file}::fixed::{n:03d}",
                source_file=source_file,
                drug=meta.drug,
                source_type=meta.source_type,
                section="unknown",
                section_path=["unknown"],
                canonical_section="other",
                text=body,
                embedding_text=body,
                start_char=start,
                end_char=start + len(body),
                part=n + 1,
                parts=0,
            )
        )
    return chunks
