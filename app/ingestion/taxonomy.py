"""Canonical section taxonomy shared by DailyMed labels and MedlinePlus pages.

Why: the corpus mixes three heading conventions -
  * FDA PLR labels with numbered headings ("4 CONTRAINDICATIONS"),
  * older/OTC labels ("CONTRAINDICATIONS", "Uses", "Directions"),
  * MedlinePlus questions ("What special precautions should I follow?").
Mapping all of them to one small vocabulary lets the agent ask for
``get_drug_section("warfarin", "contraindications")`` and lets retrieval
filter on a stable key, whatever the source format.

The numbered mapping follows the FDA Physician Labeling Rule section order
(21 CFR 201.57(c)), so it is a documented standard, not a guess.
"""

from __future__ import annotations

import re
from typing import Literal

CanonicalSection = Literal[
    "boxed_warning",
    "indications",
    "dosage",
    "dosage_forms",
    "contraindications",
    "warnings_precautions",
    "adverse_reactions",
    "drug_interactions",
    "specific_populations",
    "overdosage",
    "description",
    "clinical_pharmacology",
    "nonclinical_toxicology",
    "clinical_studies",
    "how_supplied_storage",
    "patient_counseling",
    "missed_dose",
    "diet",
    "brand_names",
    "other",
    "glossary",
]

SECTION_DESCRIPTIONS: dict[str, str] = {
    "boxed_warning": "Boxed / important warning (most serious risks)",
    "indications": "Approved uses / why the drug is prescribed",
    "dosage": "Dosage and administration, incl. renal/hepatic dose adjustments",
    "dosage_forms": "Dosage forms and strengths",
    "contraindications": "Conditions in which the drug must not be used",
    "warnings_precautions": "Warnings and precautions (incl. patient precautions)",
    "adverse_reactions": "Side effects / adverse reactions",
    "drug_interactions": "Drug-drug and drug-food interactions",
    "specific_populations": "Pregnancy, lactation, pediatric, geriatric, renal, hepatic",
    "overdosage": "Overdose signs and management",
    "description": "Chemical/product description",
    "clinical_pharmacology": "Mechanism of action, pharmacodynamics, pharmacokinetics",
    "nonclinical_toxicology": "Carcinogenesis, mutagenesis, animal toxicology",
    "clinical_studies": "Clinical trial results",
    "how_supplied_storage": "How supplied, storage and disposal",
    "patient_counseling": "Patient counseling / patient information",
    "missed_dose": "What to do if a dose is missed",
    "diet": "Dietary instructions",
    "brand_names": "Brand and other names",
    "other": "Other information",
    "glossary": "Pharmacology glossary definitions",
}

# FDA PLR section numbers -> canonical key.
_PLR_NUMBERS: dict[int, str] = {
    1: "indications",
    2: "dosage",
    3: "dosage_forms",
    4: "contraindications",
    5: "warnings_precautions",
    6: "adverse_reactions",
    7: "drug_interactions",
    8: "specific_populations",
    9: "other",  # drug abuse and dependence
    10: "overdosage",
    11: "description",
    12: "clinical_pharmacology",
    13: "nonclinical_toxicology",
    14: "clinical_studies",
    15: "other",  # references
    16: "how_supplied_storage",
    17: "patient_counseling",
}

# Ordered (pattern, key) rules for un-numbered headings. First match wins.
_TITLE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(p, re.IGNORECASE), key)
    for p, key in [
        (r"^(important )?warning\b", "boxed_warning"),
        (r"^notice", "other"),
        (r"why is this medication prescribed|^indications|^uses$", "indications"),
        (r"other uses for this medicine", "indications"),
        (r"how should this medicine be used|^dosage|^directions", "dosage"),
        (r"^contraindications", "contraindications"),
        (r"special precautions|^warnings|^precautions", "warnings_precautions"),
        (r"side effects|^adverse reactions", "adverse_reactions"),
        (r"drug interactions", "drug_interactions"),
        (r"pregnancy|nursing mothers|labor and delivery|pediatric use|geriatric use", "specific_populations"),
        (r"overdos", "overdosage"),
        (r"clinical pharmacology|pharmacokinetics|pharmacodynamics|mechanism of action", "clinical_pharmacology"),
        (r"storage and disposal|how supplied", "how_supplied_storage"),
        (r"information for patients|patient information|patient counseling", "patient_counseling"),
        (r"forget a dose|missed dose", "missed_dose"),
        (r"dietary", "diet"),
        (r"brand names|other names", "brand_names"),
        (r"other information", "other"),
    ]
]

_PLR_RE = re.compile(r"^(\d{1,2})\s+[A-Z]")


def canonical_section(top_heading: str, source_type: str) -> str:
    """Map a document's top-level (``##``) heading to a canonical key."""

    if source_type == "glossary":
        return "glossary"
    title = top_heading.strip()
    match = _PLR_RE.match(title)
    if match:
        return _PLR_NUMBERS.get(int(match.group(1)), "other")
    for pattern, key in _TITLE_RULES:
        if pattern.search(title):
            return key
    return "other"
