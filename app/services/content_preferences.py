from __future__ import annotations

import re
from typing import Any


LGBTQ_STRONG_KEYWORDS = {
    "gay relationship",
    "lesbian relationship",
    "gay romance",
    "lesbian romance",
    "same sex attraction",
    "same-sex attraction",
    "same sex relationship",
    "same-sex relationship",
    "gay couple",
    "lesbian couple",
    "boys love",
    "girls love",
}

LGBTQ_BROAD_KEYWORDS = {
    "lgbt",
    "lgbtq",
    "lgbtq+",
    "gay theme",
    "lesbian",
    "queer",
    "bisexuality",
    "homosexuality",
    "male homosexuality",
    "female homosexuality",
    "coming out",
    "transgender",
    "transsexuality",
    "non-binary",
    "nonbinary",
    "gender transition",
    "gay friend",
    "lesbian friend",
    "gay character",
    "lesbian character",
    "bisexual character",
    "queer character",
    "transgender character",
    "gay protagonist",
    "lesbian protagonist",
    "gay man",
    "gay woman",
    "same sex marriage",
    "same-sex marriage",
    "gay marriage",
}

LGBTQ_EXPLICIT_TEXT_PATTERNS = (
    r"\bgay\b",
    r"\blesbian\b",
    r"\bqueer\b",
    r"\bbisexual\b",
    r"\bsame[- ]sex\b",
    r"\bhomosexual",
    r"\btransgender\b",
    r"\btrans woman\b",
    r"\btrans man\b",
    r"\bcoming out\b",
)

LGBTQ_PROFILE_KEYWORDS = LGBTQ_STRONG_KEYWORDS | LGBTQ_BROAD_KEYWORDS


def normalize_content_preference_value(value: Any) -> str:
    return " ".join(
        str(value or "")
        .strip()
        .lower()
        .replace("_", " ")
        .split()
    )


def extract_tmdb_keyword_entries(keywords: Any) -> list[dict[str, Any]]:
    if isinstance(keywords, dict):
        entries = keywords.get("results") or keywords.get("keywords") or []
    elif isinstance(keywords, list):
        entries = keywords
    else:
        entries = []
    return [entry for entry in entries if isinstance(entry, dict)]


def extract_tmdb_keyword_names(keywords: Any) -> list[str]:
    return [
        str(entry.get("name"))
        for entry in extract_tmdb_keyword_entries(keywords)
        if entry.get("name")
    ]


def filter_profile_keyword_ids(keywords: Any) -> list[int]:
    """Prevent LGBTQ subject tags from becoming positive taste axes.

    The source title itself remains part of the taste profile through its other
    features. This only prevents these metadata tags from seeding future rows.
    """
    result: list[int] = []
    for entry in extract_tmdb_keyword_entries(keywords):
        keyword_id = entry.get("id")
        if not keyword_id:
            continue
        keyword_name = normalize_content_preference_value(entry.get("name"))
        if keyword_name in LGBTQ_PROFILE_KEYWORDS:
            continue
        result.append(keyword_id)
    return result


def lgbtq_content_preference_reason_values(
    *,
    title: str = "",
    overview: str = "",
    keywords: Any = None,
) -> str | None:
    """Return an exclusion reason only when explicit metadata is corroborated.

    Mirrors Seasonal Spotlight:
    - one strong relationship/romance keyword is sufficient;
    - otherwise two broad keywords are required;
    - or one broad keyword plus explicit title/overview evidence.
    A lone broad keyword is insufficient.
    """
    normalized_keywords = {
        normalize_content_preference_value(keyword)
        for keyword in extract_tmdb_keyword_names(keywords)
        if normalize_content_preference_value(keyword)
    }

    strong = sorted(normalized_keywords & LGBTQ_STRONG_KEYWORDS)
    if strong:
        return "strong_keyword:" + ",".join(strong)

    broad = sorted(normalized_keywords & LGBTQ_BROAD_KEYWORDS)
    if len(broad) >= 2:
        return "corroborating_keywords:" + ",".join(broad)

    if broad:
        normalized_overview = normalize_content_preference_value(overview)
        if any(
            re.search(pattern, normalized_overview)
            for pattern in LGBTQ_EXPLICIT_TEXT_PATTERNS
        ):
            return "keyword_plus_explicit_overview:" + ",".join(broad)

        normalized_title = normalize_content_preference_value(title)
        if any(
            re.search(pattern, normalized_title)
            for pattern in LGBTQ_EXPLICIT_TEXT_PATTERNS
        ):
            return "keyword_plus_explicit_title:" + ",".join(broad)

    return None
