"""Remember what a channel already published so digests do not spam reprints.

URL identity alone is not enough: the same story often reappears next day under
another outlet URL (Life.ru vs SamaraOnline, Techcult vs Mail.ru, …). Matching
happens on three levels, cheapest first:

1. identity — ``canonical_key`` (tender number, lot GUID) or normalized URL;
2. headline — normalized fingerprint plus fuzzy similarity;
3. event — the AI-produced ``event_key`` filled during analysis.

Levels 1 and 2 run before any LLM call. Level 3 runs after analysis, because the
event key only exists once the model has read the material.

History comes from :class:`~src.storage.state.StateStore` when available, and
from the Markdown digests on disk otherwise, so a fresh data directory still
behaves sensibly on its first run.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
import re
from typing import Iterable, Optional
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
    digest_date: date | None = None
    event_key: str | None = None
    canonical_key: str | None = None


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


def canonical_key_for(item: object) -> str | None:
    """Identity of the material: explicit key first, normalized URL second."""
    explicit = getattr(item, "canonical_key", None)
    if explicit:
        return str(explicit).strip() or None
    url = str(getattr(item, "url", "") or "")
    if not url:
        return None
    return _url_key(urldefrag(url)[0]) or None


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
    store: object | None = None,
) -> list[PublishedRef]:
    """Load already-covered materials from the state store or on-disk digests."""
    if store is not None and getattr(store, "enabled", False):
        refs: list[PublishedRef] = []
        for event in store.iter_events(days=days):  # type: ignore[attr-defined]
            refs.append(
                PublishedRef(
                    title=event.title,
                    url=event.url_key or "",
                    fingerprint=event.title_fp or title_fingerprint(event.title),
                    digest_date=event.digest_date,
                    event_key=event.event_key,
                    canonical_key=event.canonical_key,
                )
            )
        if refs:
            return refs

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
    canonical_key: str | None = None,
    event_key: str | None = None,
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
        if canonical_key and ref.canonical_key and canonical_key == ref.canonical_key:
            return True
        if url_key and url_key == ref.url:
            return True
        if event_key and ref.event_key and event_key == ref.event_key:
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
    store: object | None = None,
    check_event_keys: bool = False,
) -> tuple[list, list]:
    """Split items into (fresh, already_published) using memory + extras.

    ``check_event_keys`` enables level 3 — call it after analysis so reprints
    that share no vocabulary with the original headline are still caught.
    """
    memory = load_published_memory(summaries_dir, days=days, store=store)
    fresh: list = []
    dropped: list = []
    for item in items:
        title = str(getattr(item, "title", "") or "")
        url = str(getattr(item, "url", "") or "")
        event_key = None
        if check_event_keys:
            processing = getattr(item, "processing", None)
            analysis = getattr(processing, "analysis", None) if processing else None
            event_key = getattr(analysis, "event_key", None) if analysis else None
        if is_already_published(
            title=title,
            url=url,
            memory=memory,
            extra_urls=exclude_urls,
            extra_fingerprints=exclude_fingerprints,
            canonical_key=canonical_key_for(item),
            event_key=event_key,
        ):
            dropped.append(item)
        else:
            fresh.append(item)
    return fresh, dropped


def history_titles(
    summaries_dir: Path | None,
    *,
    days: int = 30,
    store: object | None = None,
) -> list[str]:
    """Titles already present in digests — for AI topic-dedup phantoms."""
    return [
        ref.title
        for ref in load_published_memory(summaries_dir, days=days, store=store)
    ]


def remember_items(
    store: object | None,
    items: Iterable[object],
    *,
    digest_date: date | str | None = None,
) -> int:
    """Persist delivered items into the state store. Returns rows written."""
    if store is None or not getattr(store, "enabled", False):
        return 0
    from ..storage.state import content_hash as _content_hash

    written = 0
    for item in items:
        title = str(getattr(item, "title", "") or "")
        if not title:
            continue
        processing = getattr(item, "processing", None)
        analysis = getattr(processing, "analysis", None) if processing else None
        url = str(getattr(item, "url", "") or "")
        store.remember(  # type: ignore[attr-defined]
            title=title,
            canonical_key=canonical_key_for(item),
            url_key=_url_key(urldefrag(url)[0]) if url else None,
            title_fp=title_fingerprint(title),
            event_key=getattr(analysis, "event_key", None) if analysis else None,
            content_hash_value=_content_hash(title, getattr(item, "content", None)),
            summary=getattr(analysis, "summary", None) if analysis else None,
            digest_date=digest_date,
        )
        written += 1
    return written


@dataclass(frozen=True)
class MemorySplit:
    """Result of the three-level memory pass."""

    fresh: list
    updates: list
    duplicates: list

    @property
    def dropped_count(self) -> int:
        return len(self.duplicates)


def split_by_memory(
    items: list,
    summaries_dir: Path | None,
    *,
    days: int = 30,
    exclude_urls: Iterable[str] | None = None,
    exclude_fingerprints: Iterable[str] | None = None,
    store: object | None = None,
    check_event_keys: bool = False,
) -> MemorySplit:
    """Classify items as fresh, updated or duplicate.

    Uses the durable store when present (which can tell an amendment apart from
    a reprint) and falls back to Markdown history otherwise.
    """
    from ..storage.state import content_hash as _content_hash

    store_ready = store is not None and getattr(store, "enabled", False)
    memory = load_published_memory(summaries_dir, days=days, store=store)
    fresh: list = []
    updates: list = []
    duplicates: list = []

    for item in items:
        title = str(getattr(item, "title", "") or "")
        url = str(getattr(item, "url", "") or "")
        event_key = None
        if check_event_keys:
            processing = getattr(item, "processing", None)
            analysis = getattr(processing, "analysis", None) if processing else None
            event_key = getattr(analysis, "event_key", None) if analysis else None

        if store_ready:
            verdict = store.classify(  # type: ignore[attr-defined]
                canonical_key=canonical_key_for(item),
                url_key=_url_key(urldefrag(url)[0]) if url else None,
                title_fp=title_fingerprint(title),
                event_key=event_key,
                content_hash_value=_content_hash(title, getattr(item, "content", None)),
            )
            if verdict.is_duplicate:
                duplicates.append(item)
                continue
            if verdict.is_update:
                metadata = getattr(item, "metadata", None)
                if isinstance(metadata, dict):
                    metadata["is_update"] = True
                    metadata["memory_reason"] = verdict.reason
                    if verdict.previous is not None:
                        metadata["previous_title"] = verdict.previous.title
                updates.append(item)
                continue

        if is_already_published(
            title=title,
            url=url,
            memory=memory,
            extra_urls=exclude_urls,
            extra_fingerprints=exclude_fingerprints,
            canonical_key=canonical_key_for(item),
            event_key=event_key,
        ):
            duplicates.append(item)
            continue

        fresh.append(item)

    return MemorySplit(fresh=fresh, updates=updates, duplicates=duplicates)


def state_path_for(storage: object) -> Optional[Path]:
    """Resolve ``state.db`` from a StorageManager without importing it."""
    data_dir = getattr(storage, "data_dir", None)
    if data_dir is None:
        return None
    return Path(data_dir) / "state.db"
