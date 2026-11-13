"""Conservative, auditable text cleaning (raw -> cleaned).

Principle: repair *typography and markup* only - never words, numbers,
negations or clinical statements (the spec requires the original text to be
preserved). Every rule was derived from a measured noise pattern in the raw
corpus (see EXPLORATION.md) and every substitution is counted so the manifest
records exactly what changed in each document.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


@dataclass(frozen=True)
class CleaningRule:
    name: str
    pattern: re.Pattern[str]
    replacement: str
    rationale: str


RULES: tuple[CleaningRule, ...] = (
    CleaningRule(
        "zero_width_chars",
        re.compile(r"[ \t]*[﻿​‌‍⁠]"),
        "",
        "Invisible BOM / zero-width characters embedded mid-text ('WARNINGS \\ufeff and').",
    ),
    CleaningRule(
        "html_sup_sub",
        re.compile(r"<su[pb]>(.*?)</su[pb]>"),
        r"\1",
        "Inline HTML left by the HTML->Markdown export (Coumadin<sup>®</sup>, B<sub>12</sub>).",
    ),
    CleaningRule(
        "markdown_unescape",
        re.compile(r"\\([\[\]()*_#.!\-])"),
        r"\1",
        "Escaped Markdown punctuation (\\[prothrombin test\\]) pollutes tokens.",
    ),
    CleaningRule(
        "unit_m2_superscript",
        re.compile(r"(?<=/)m 2(?=[a-z])|(?<=1\.73 )m 2(?=[a-z])"),
        "m² ",
        "Lost superscript glued to next word: '1.73 m 2is' -> '1.73 m² is'.",
    ),
    CleaningRule(
        "unit_m2",
        re.compile(r"(?<=/)m 2\b|(?<=1\.73 )m 2\b"),
        "m²",
        "Lost superscript in body-surface units: 'mL/min/1.73 m 2' -> 'm²'.",
    ),
    CleaningRule(
        "vitamin_b12_glued",
        re.compile(r"\bB 12(?=[A-Za-z])"),
        "B12 ",
        "Lost subscript glued to next word: 'B 12Deficiency' -> 'B12 Deficiency'.",
    ),
    CleaningRule(
        "vitamin_b12",
        re.compile(r"\bB 12\b"),
        "B12",
        "Lost subscript: 'vitamin B 12' -> 'vitamin B12'.",
    ),
    CleaningRule(
        "colon_spacing",
        re.compile(r"(?<=[a-z]):(?=[A-Z])"),
        ": ",
        "Label glued to its value: 'Liver warning:This' -> 'Liver warning: This'.",
    ),
    CleaningRule(
        "bullet_normalization",
        re.compile(r"^[ \t]*•[ \t]*", re.MULTILINE),
        "- ",
        "Unicode bullets -> Markdown list items so the chunker keeps lists intact.",
    ),
    CleaningRule(
        "trailing_whitespace",
        re.compile(r"[ \t]+$", re.MULTILINE),
        "",
        "Invisible trailing spaces.",
    ),
    CleaningRule(
        "blank_line_runs",
        re.compile(r"\n{3,}"),
        "\n\n",
        "Collapse runs of blank lines (paragraph boundaries are preserved).",
    ),
)


def clean_text(raw: str) -> tuple[str, dict[str, int]]:
    """Return the cleaned text and the number of substitutions per rule."""

    text = unicodedata.normalize("NFC", raw.replace("\r\n", "\n").replace("\r", "\n"))
    counts: dict[str, int] = {}
    for rule in RULES:
        text, n = rule.pattern.subn(rule.replacement, text)
        if n:
            counts[rule.name] = n
    return text.strip() + "\n", counts
