"""Zero-LLM pre-filter applied to every scraper's output.

Free filtering before the paid stages is the single biggest cost lever in a
digest pipeline: a niche feed can hand you 200 items of which 5 matter, and the
five are visible from the text alone. Everything here runs in microseconds and
never calls a model.

Order matters and is intentional:

1. ``min_content_chars`` — nothing to summarise, nothing to score.
2. ``must_match`` — the topical gate (``"МИРЭА"``, ``"44-ФЗ"``, ``"AI"``).
3. ``exclude_patterns`` — the noise gate (spam words, off-topic sections).
4. ``max_items_to_analyze`` — the budget gate, applied newest-first so a cap
   never silently drops fresh material in favour of old.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, List, Sequence

import re


@dataclass
class PrefilterResult:
    """What survived the free stage, plus per-reason counters."""

    items: List[object]
    dropped_min_content: int = 0
    dropped_must_match: int = 0
    dropped_exclude: int = 0
    dropped_cap: int = 0

    @property
    def dropped_total(self) -> int:
        return (
            self.dropped_min_content
            + self.dropped_must_match
            + self.dropped_exclude
            + self.dropped_cap
        )

    def to_dict(self) -> dict:
        return {
            "kept": len(self.items),
            "dropped_min_content": self.dropped_min_content,
            "dropped_must_match": self.dropped_must_match,
            "dropped_exclude": self.dropped_exclude,
            "dropped_cap": self.dropped_cap,
            "dropped_total": self.dropped_total,
        }


def _text_of(item: object) -> str:
    title = str(getattr(item, "title", "") or "")
    content = str(getattr(item, "content", "") or "")
    return f"{title}\n{content}"


def _published(item: object) -> datetime:
    value = getattr(item, "published_at", None)
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.min.replace(tzinfo=timezone.utc)


def _compiled(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        text = str(pattern or "").strip()
        if not text:
            continue
        try:
            compiled.append(re.compile(text, re.IGNORECASE))
        except re.error:
            # A typo in config must not kill the run; treat it as a literal.
            compiled.append(re.compile(re.escape(text), re.IGNORECASE))
    return compiled


def apply_prefilter(
    items: Sequence[object],
    *,
    must_match: Sequence[str] | None = None,
    exclude_patterns: Sequence[str] | None = None,
    min_content_chars: int = 0,
    max_items_to_analyze: int = 0,
) -> PrefilterResult:
    """Apply the free filters and return survivors plus counters."""
    result = PrefilterResult(items=[])
    needles = [str(n).casefold() for n in (must_match or []) if str(n).strip()]
    excluded = _compiled(exclude_patterns or [])

    for item in items:
        text = _text_of(item)
        if min_content_chars > 0:
            usable = (getattr(item, "content", "") or "").strip()
            if len(usable) < min_content_chars and len(str(getattr(item, "title", "") or "")) < min_content_chars:
                result.dropped_min_content += 1
                continue

        if needles:
            haystack = text.casefold()
            if not any(needle in haystack for needle in needles):
                result.dropped_must_match += 1
                continue

        if excluded and any(pattern.search(text) for pattern in excluded):
            result.dropped_exclude += 1
            continue

        result.items.append(item)

    if max_items_to_analyze and len(result.items) > max_items_to_analyze:
        ordered = sorted(result.items, key=_published, reverse=True)
        result.dropped_cap = len(ordered) - max_items_to_analyze
        result.items = ordered[:max_items_to_analyze]

    return result
