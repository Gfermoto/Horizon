"""Durable local state for Horizon runs.

A small SQLite (WAL) store that survives across runs and keeps three things
that the Markdown digests alone cannot express:

* ``published_events`` — what the channel already covered, so a reprint under a
  different outlet URL or a lightly rewritten headline never ships twice.
  It also records a content hash, which lets a *material change* (a tender
  amendment, a corrected price) surface as an update instead of a duplicate.
* ``analysis_cache`` — LLM analysis keyed by content, so re-fetching the same
  article within a few days does not burn tokens.
* ``source_health`` — per-source success/failure history backing the circuit
  breaker and the ``run_report``.

The module intentionally depends on nothing but the standard library: tenants
deploy Horizon as a CLI in places where adding a database dependency would be a
hard sell, and SQLite ships with Python.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

__all__ = [
    "EventVerdict",
    "PublishedEvent",
    "SourceHealth",
    "StateStore",
    "content_hash",
    "default_state_path",
]

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS published_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_key TEXT,
    url_key TEXT,
    title_fp TEXT,
    event_key TEXT,
    content_hash TEXT,
    title TEXT NOT NULL,
    summary TEXT,
    digest_date TEXT,
    first_seen_at REAL NOT NULL,
    last_seen_at REAL NOT NULL,
    published_at REAL,
    version INTEGER NOT NULL DEFAULT 1
);

CREATE INDEX IF NOT EXISTS ix_published_canonical
    ON published_events(canonical_key);
CREATE INDEX IF NOT EXISTS ix_published_url
    ON published_events(url_key);
CREATE INDEX IF NOT EXISTS ix_published_title_fp
    ON published_events(title_fp);
CREATE INDEX IF NOT EXISTS ix_published_event
    ON published_events(event_key);

CREATE TABLE IF NOT EXISTS analysis_cache (
    cache_key TEXT PRIMARY KEY,
    profile_id TEXT,
    analysis_json TEXT NOT NULL,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_analysis_created
    ON analysis_cache(created_at);

CREATE TABLE IF NOT EXISTS source_health (
    source_key TEXT PRIMARY KEY,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    cooldown_until REAL,
    last_ok_at REAL,
    last_error TEXT,
    total_fetched INTEGER NOT NULL DEFAULT 0,
    avg_latency_ms REAL,
    updated_at REAL NOT NULL
);
"""


def default_state_path(data_dir: str | Path) -> Path:
    """Return the canonical ``state.db`` path for a Horizon data directory."""
    return Path(data_dir) / "state.db"


def content_hash(title: str, content: str | None) -> str:
    """Stable hash of the *substance* of an item, used to detect amendments."""
    payload = f"{(title or '').strip()}\n{(content or '')[:4000].strip()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PublishedEvent:
    """A single material already covered by the channel."""

    id: int
    title: str
    canonical_key: Optional[str]
    url_key: Optional[str]
    title_fp: Optional[str]
    event_key: Optional[str]
    content_hash: Optional[str]
    digest_date: Optional[date]
    first_seen_at: float
    last_seen_at: float
    version: int


@dataclass(frozen=True)
class EventVerdict:
    """Outcome of asking the store whether an item is fresh."""

    status: str  # "new" | "duplicate" | "update"
    reason: str = ""
    previous: Optional[PublishedEvent] = None

    @property
    def is_new(self) -> bool:
        return self.status == "new"

    @property
    def is_duplicate(self) -> bool:
        return self.status == "duplicate"

    @property
    def is_update(self) -> bool:
        return self.status == "update"


@dataclass
class SourceHealth:
    """Health counters for one configured source."""

    source_key: str
    consecutive_failures: int = 0
    cooldown_until: Optional[float] = None
    last_ok_at: Optional[float] = None
    last_error: Optional[str] = None
    total_fetched: int = 0
    avg_latency_ms: Optional[float] = None
    updated_at: float = field(default_factory=time.time)

    @property
    def in_cooldown(self) -> bool:
        return self.cooldown_until is not None and self.cooldown_until > time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_key": self.source_key,
            "consecutive_failures": self.consecutive_failures,
            "cooldown_until": self.cooldown_until,
            "in_cooldown": self.in_cooldown,
            "last_ok_at": self.last_ok_at,
            "last_error": self.last_error,
            "total_fetched": self.total_fetched,
            "avg_latency_ms": self.avg_latency_ms,
        }


class StateStore:
    """Thin SQLite wrapper; safe to open per run and cheap to close."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self._conn: Optional[sqlite3.Connection] = None

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.path is not None

    def __enter__(self) -> "StateStore":
        self.connect()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def connect(self) -> sqlite3.Connection:
        if self._conn is not None:
            return self._conn
        if self.path is None:
            raise RuntimeError("StateStore has no path configured")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.path), timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=15000")
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            finally:
                self._conn = None

    def __iter__(self) -> Iterator[PublishedEvent]:
        """Iterate every remembered event (newest first)."""
        for row in self.iter_events():
            yield row

    # ------------------------------------------------------------------
    # published memory
    # ------------------------------------------------------------------
    def iter_events(self, *, days: int | None = None, limit: int | None = None) -> List[PublishedEvent]:
        conn = self.connect()
        query = "SELECT * FROM published_events"
        params: list[Any] = []
        if days is not None:
            cutoff = time.time() - days * 86400
            query += " WHERE last_seen_at >= ?"
            params.append(cutoff)
        query += " ORDER BY last_seen_at DESC"
        if limit is not None:
            query += " LIMIT ?"
            params.append(int(limit))
        return [self._row_to_event(row) for row in conn.execute(query, params)]

    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> PublishedEvent:
        digest_date = None
        if row["digest_date"]:
            try:
                digest_date = date.fromisoformat(str(row["digest_date"]))
            except ValueError:
                digest_date = None
        return PublishedEvent(
            id=int(row["id"]),
            title=str(row["title"] or ""),
            canonical_key=row["canonical_key"],
            url_key=row["url_key"],
            title_fp=row["title_fp"],
            event_key=row["event_key"],
            content_hash=row["content_hash"],
            digest_date=digest_date,
            first_seen_at=float(row["first_seen_at"]),
            last_seen_at=float(row["last_seen_at"]),
            version=int(row["version"] or 1),
        )

    def find(
        self,
        *,
        canonical_key: str | None = None,
        url_key: str | None = None,
        title_fp: str | None = None,
        event_key: str | None = None,
    ) -> Optional[PublishedEvent]:
        """Return the most recent remembered event matching any supplied key."""
        clauses: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("canonical_key", canonical_key),
            ("url_key", url_key),
            ("title_fp", title_fp),
            ("event_key", event_key),
        ):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if not clauses:
            return None
        conn = self.connect()
        row = conn.execute(
            "SELECT * FROM published_events WHERE "
            + " OR ".join(clauses)
            + " ORDER BY last_seen_at DESC LIMIT 1",
            params,
        ).fetchone()
        return self._row_to_event(row) if row else None

    def classify(
        self,
        *,
        canonical_key: str | None = None,
        url_key: str | None = None,
        title_fp: str | None = None,
        event_key: str | None = None,
        content_hash_value: str | None = None,
    ) -> EventVerdict:
        """Decide whether an item is new, a duplicate, or a real update.

        A changed ``content_hash`` on an already-known identity means the source
        edited the material (tender amendment, corrected price): that is worth
        publishing, but it must be labelled as an update, not as fresh news.
        """
        previous = self.find(
            canonical_key=canonical_key,
            url_key=url_key,
            title_fp=title_fp,
            event_key=event_key,
        )
        if previous is None:
            return EventVerdict(status="new")
        if content_hash_value and previous.content_hash:
            if content_hash_value != previous.content_hash:
                return EventVerdict(
                    status="update", reason="content changed", previous=previous
                )
            return EventVerdict(
                status="duplicate", reason="identical content", previous=previous
            )
        return EventVerdict(status="duplicate", reason="known identity", previous=previous)

    def remember(
        self,
        *,
        title: str,
        canonical_key: str | None = None,
        url_key: str | None = None,
        title_fp: str | None = None,
        event_key: str | None = None,
        content_hash_value: str | None = None,
        summary: str | None = None,
        digest_date: date | str | None = None,
        published_at: float | None = None,
        now: float | None = None,
    ) -> None:
        """Insert or refresh the memory row for one material."""
        conn = self.connect()
        stamp = time.time() if now is None else now
        day = None
        if isinstance(digest_date, date):
            day = digest_date.isoformat()
        elif digest_date:
            day = str(digest_date)

        previous = self.find(
            canonical_key=canonical_key,
            url_key=url_key,
            title_fp=title_fp,
            event_key=event_key,
        )
        if previous is None:
            conn.execute(
                "INSERT INTO published_events ("
                "canonical_key, url_key, title_fp, event_key, content_hash, title, "
                "summary, digest_date, first_seen_at, last_seen_at, published_at, version"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                (
                    canonical_key,
                    url_key,
                    title_fp,
                    event_key,
                    content_hash_value,
                    title,
                    summary,
                    day,
                    stamp,
                    stamp,
                    published_at,
                ),
            )
            return

        changed = bool(
            content_hash_value
            and previous.content_hash
            and content_hash_value != previous.content_hash
        )
        conn.execute(
            "UPDATE published_events SET "
            "canonical_key = COALESCE(?, canonical_key), "
            "url_key = COALESCE(?, url_key), "
            "title_fp = COALESCE(?, title_fp), "
            "event_key = COALESCE(?, event_key), "
            "content_hash = COALESCE(?, content_hash), "
            "title = ?, summary = COALESCE(?, summary), "
            "digest_date = COALESCE(?, digest_date), "
            "last_seen_at = ?, published_at = COALESCE(?, published_at), "
            "version = version + ? "
            "WHERE id = ?",
            (
                canonical_key,
                url_key,
                title_fp,
                event_key,
                content_hash_value,
                title,
                summary,
                day,
                stamp,
                published_at,
                1 if changed else 0,
                previous.id,
            ),
        )

    def count(self) -> int:
        conn = self.connect()
        return int(conn.execute("SELECT COUNT(*) FROM published_events").fetchone()[0])

    # ------------------------------------------------------------------
    # analysis cache
    # ------------------------------------------------------------------
    def cache_get(self, cache_key: str, *, ttl_hours: float = 72.0) -> Optional[Dict[str, Any]]:
        conn = self.connect()
        row = conn.execute(
            "SELECT analysis_json, created_at FROM analysis_cache WHERE cache_key = ?",
            (cache_key,),
        ).fetchone()
        if row is None:
            return None
        if time.time() - float(row["created_at"]) > ttl_hours * 3600:
            return None
        try:
            return json.loads(row["analysis_json"])
        except (ValueError, TypeError):
            return None

    def cache_put(
        self,
        cache_key: str,
        payload: Dict[str, Any],
        *,
        profile_id: str | None = None,
    ) -> None:
        conn = self.connect()
        conn.execute(
            "INSERT OR REPLACE INTO analysis_cache "
            "(cache_key, profile_id, analysis_json, created_at) VALUES (?, ?, ?, ?)",
            (cache_key, profile_id, json.dumps(payload, ensure_ascii=False), time.time()),
        )

    def cache_prune(self, *, ttl_hours: float = 168.0) -> int:
        conn = self.connect()
        cutoff = time.time() - ttl_hours * 3600
        cur = conn.execute("DELETE FROM analysis_cache WHERE created_at < ?", (cutoff,))
        return int(cur.rowcount or 0)

    # ------------------------------------------------------------------
    # source health
    # ------------------------------------------------------------------
    def health_get(self, source_key: str) -> SourceHealth:
        conn = self.connect()
        row = conn.execute(
            "SELECT * FROM source_health WHERE source_key = ?", (source_key,)
        ).fetchone()
        if row is None:
            return SourceHealth(source_key=source_key)
        return SourceHealth(
            source_key=str(row["source_key"]),
            consecutive_failures=int(row["consecutive_failures"] or 0),
            cooldown_until=row["cooldown_until"],
            last_ok_at=row["last_ok_at"],
            last_error=row["last_error"],
            total_fetched=int(row["total_fetched"] or 0),
            avg_latency_ms=row["avg_latency_ms"],
            updated_at=float(row["updated_at"] or time.time()),
        )

    def health_all(self) -> List[SourceHealth]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM source_health ORDER BY source_key"
        ).fetchall()
        return [
            SourceHealth(
                source_key=str(row["source_key"]),
                consecutive_failures=int(row["consecutive_failures"] or 0),
                cooldown_until=row["cooldown_until"],
                last_ok_at=row["last_ok_at"],
                last_error=row["last_error"],
                total_fetched=int(row["total_fetched"] or 0),
                avg_latency_ms=row["avg_latency_ms"],
                updated_at=float(row["updated_at"] or time.time()),
            )
            for row in rows
        ]

    def health_record(
        self,
        source_key: str,
        *,
        ok: bool,
        latency_ms: float | None = None,
        fetched: int = 0,
        error: str | None = None,
        cooldown_seconds: float = 0.0,
    ) -> SourceHealth:
        """Record one fetch outcome and return the updated health row."""
        current = self.health_get(source_key)
        now = time.time()
        if ok:
            failures = 0
            cooldown_until = None
            last_ok_at = now
            last_error = None
        else:
            failures = current.consecutive_failures + 1
            cooldown_until = now + cooldown_seconds if cooldown_seconds > 0 else None
            last_ok_at = current.last_ok_at
            last_error = (error or "unknown error")[:500]

        if latency_ms is not None:
            if current.avg_latency_ms is None:
                avg = float(latency_ms)
            else:
                # Exponential moving average keeps the row cheap to maintain.
                avg = current.avg_latency_ms * 0.7 + float(latency_ms) * 0.3
        else:
            avg = current.avg_latency_ms

        conn = self.connect()
        conn.execute(
            "INSERT INTO source_health ("
            "source_key, consecutive_failures, cooldown_until, last_ok_at, last_error, "
            "total_fetched, avg_latency_ms, updated_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(source_key) DO UPDATE SET "
            "consecutive_failures = excluded.consecutive_failures, "
            "cooldown_until = excluded.cooldown_until, "
            "last_ok_at = excluded.last_ok_at, "
            "last_error = excluded.last_error, "
            "total_fetched = excluded.total_fetched, "
            "avg_latency_ms = excluded.avg_latency_ms, "
            "updated_at = excluded.updated_at",
            (
                source_key,
                failures,
                cooldown_until,
                last_ok_at,
                last_error,
                current.total_fetched + max(0, int(fetched)),
                avg,
                now,
            ),
        )
        return self.health_get(source_key)


def cooldown_seconds_for(failures: int, base_hours: float = 6.0) -> float:
    """Exponential cooldown: 6h, 24h, 72h, then capped at 72h."""
    if failures <= 0:
        return 0.0
    ladder = [base_hours, base_hours * 4, base_hours * 12]
    hours = ladder[min(failures - 1, len(ladder) - 1)]
    return hours * 3600


def _utc_from_epoch(value: float | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromtimestamp(value, tz=timezone.utc)


def bootstrap_from_summaries(
    store: StateStore,
    summaries_dir: str | Path | None,
    *,
    days: int = 180,
    force: bool = False,
) -> int:
    """One-time import of ``horizon-*.md`` history into ``published_events``.

    Returns the number of events imported. Runs only when the store is empty
    unless ``force`` is set, so repeated runs stay cheap.
    """
    if summaries_dir is None:
        return 0
    directory = Path(summaries_dir)
    if not directory.exists():
        return 0
    if not force and store.count() > 0:
        return 0

    from ..processing.history import _read_entries
    from ..processing.published_memory import title_fingerprint

    entries = _read_entries(directory)
    if not entries:
        return 0
    cutoff = date.fromordinal(date.today().toordinal() - max(1, days))
    imported = 0
    for entry in entries:
        if entry.digest_date < cutoff:
            continue
        url_key = entry.url.rstrip("/")
        store.remember(
            title=entry.title,
            url_key=url_key,
            title_fp=title_fingerprint(entry.title),
            summary=entry.summary[:1000] if entry.summary else None,
            digest_date=entry.digest_date,
        )
        imported += 1
    return imported
