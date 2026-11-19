"""Remember what a channel already published so digests do not spam reprints.

URL identity alone is not enough: the same MIREA story often reappears next day
under another outlet URL (Life.ru vs SamaraOnline, Techcult vs Mail.ru, …).
This module keeps a lightweight memory from prior Horizon Markdown digests and
matches new candidates by URL and near-duplicate titles.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Iterable
from urllib.parse import urldefrag

from .history import HistoryEntry, _read_entries, _url_key

_NOISE = re.compile(r"[^\w\sа-яё]+", re.IGNORECASE)
_SPACE = re.compile(r"\s+")

# Near-reprint titles (same story, lightly rewritten headline).
_TITLE_RATIO = 0.78
_TITLE_JACCARD = 0.55


@dataclass(frozen=True)
class PublishedRef:
    title: str
    url: str
    fingerprint: str
    digest_date: date


def title_fingerprint(title: str) -> str:
    """Normalize a headline for durable cross-URL identity."""
    text = _NOISE.sub(" ", (title or "").casefold())
    return _SPACE.sub(" ", text).strip()


def titles_similar(a: str, b: str) -> bool:
    """True when two headlines likely describe the same published material."""
    fa, fb = title_fingerprint(a), title_fingerprint(b)
    if not fa or not fb:
        return False
    if fa == fb:
        return True
    if SequenceMatcher(None, fa, fb).ratio() >= _TITLE_RATIO:
        return True
    ta, tb = set(fa.split()), set(fb.split())
    if not ta or not tb:
        return False
    return len(ta & tb) / len(ta | tb) >= _TITLE_JACCARD


def _from_history(entries: Iterable[HistoryEntry], *, since: date) -> list[PublishedRef]:
    refs: list[PublishedRef] = []
    seen: set[str] = set()
    for entry in entries:
        if entry.digest_date < since:
            continue
        fp = title_fingerprint(entry.title)
        url = _url_key(entry.url)
        key = fp or url
        if not key or key in seen:
            continue
        seen.add(key)
        refs.append(
            PublishedRef(
                title=entry.title,
                url=url,
                fingerprint=fp,
                digest_date=entry.digest_date,
            )
        )
    return refs


def load_published_memory(
    summaries_dir: Path | None,
    *,
    days: int = 30,
    before: date | None = None,
) -> list[PublishedRef]:
    """Load already-digested materials from on-disk Horizon summaries."""
    if summaries_dir is None or not summaries_dir.exists():
        return []
    today = before or datetime.now(timezone.utc).date()
    since = today - timedelta(days=max(1, days))
    entries = _read_entries(summaries_dir)
    return _from_history(entries, since=since)


def is_already_published(
    *,
    title: str,
    url: str,
    memory: list[PublishedRef],
    extra_urls: Iterable[str] | None = None,
    extra_fingerprints: Iterable[str] | None = None,
) -> bool:
    """Return True if this candidate was already covered for the channel."""
    url_key = _url_key(urldefrag(url or "")[0])
    blocked_urls = {_url_key(u) for u in (extra_urls or []) if u}
    if url_key and url_key in blocked_urls:
        return True
    fp = title_fingerprint(title)
    blocked_fps = {title_fingerprint(x) for x in (extra_fingerprints or []) if x}
    if fp and fp in blocked_fps:
        return True
    for ref in memory:
        if url_key and url_key == ref.url:
            return True
        if fp and ref.fingerprint and titles_similar(title, ref.title):
            return True
    return False


def filter_new_items(
    items: list,
    summaries_dir: Path | None,
    *,
    days: int = 30,
    exclude_urls: Iterable[str] | None = None,
    exclude_fingerprints: Iterable[str] | None = None,
) -> tuple[list, list]:
    """Split items into (fresh, already_published) using digest memory + extras."""
    memory = load_published_memory(summaries_dir, days=days)
    fresh: list = []
    dropped: list = []
    for item in items:
        title = str(getattr(item, "title", "") or "")
        url = str(getattr(item, "url", "") or "")
        if is_already_published(
            title=title,
            url=url,
            memory=memory,
            extra_urls=exclude_urls,
            extra_fingerprints=exclude_fingerprints,
        ):
            dropped.append(item)
        else:
            fresh.append(item)
    return fresh, dropped


def history_titles(summaries_dir: Path | None, *, days: int = 30) -> list[str]:
    """Titles already present in digests — for AI topic-dedup phantoms."""
    return [ref.title for ref in load_published_memory(summaries_dir, days=days)]
