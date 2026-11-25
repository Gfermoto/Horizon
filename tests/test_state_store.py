"""Durable state: three-level memory, analysis cache and source health."""

from datetime import date, timezone
from datetime import datetime

import pytest

from src.models import ContentItem, SourceType
from src.processing.published_memory import (
    canonical_key_for,
    remember_items,
    split_by_memory,
    title_fingerprint,
)
from src.storage.state import StateStore, cooldown_seconds_for, default_state_path


def make_item(title: str, url: str, content: str = "body", **kwargs) -> ContentItem:
    return ContentItem(
        id=f"rss:test:{abs(hash(title)) % 10_000}",
        source_type=SourceType.RSS,
        title=title,
        url=url,
        content=content,
        published_at=datetime(2026, 11, 18, tzinfo=timezone.utc),
        **kwargs,
    )


@pytest.fixture()
def store(tmp_path):
    state = StateStore(default_state_path(tmp_path))
    yield state
    state.close()


def test_new_then_duplicate_then_update(store):
    payload = dict(
        title="РТУ МИРЭА синтезировал МОКП",
        url_key="https://example.com/mokp",
        title_fp=title_fingerprint("РТУ МИРЭА синтезировал МОКП"),
        content_hash_value="hash-a",
    )
    assert store.classify(
        url_key=payload["url_key"],
        title_fp=payload["title_fp"],
        content_hash_value="hash-a",
    ).is_new

    store.remember(**payload)
    assert store.classify(
        url_key=payload["url_key"],
        title_fp=payload["title_fp"],
        content_hash_value="hash-a",
    ).is_duplicate

    verdict = store.classify(
        url_key=payload["url_key"],
        title_fp=payload["title_fp"],
        content_hash_value="hash-b",
    )
    assert verdict.is_update
    assert verdict.previous is not None


def test_canonical_key_beats_url(store):
    store.remember(
        title="Тендер 12345 капремонт",
        canonical_key="tender:12345",
        url_key="https://etp.example/lot/1",
        content_hash_value="a",
    )
    verdict = store.classify(
        canonical_key="tender:12345",
        url_key="https://etp.example/lot/1-mirror",
        content_hash_value="a",
    )
    assert verdict.is_duplicate
    assert verdict.previous is not None
    assert verdict.previous.canonical_key == "tender:12345"


def test_split_by_memory_marks_update_not_dropped(store):
    first = make_item(
        "Тендер 777 капремонт школы",
        "https://etp.example/lot/777",
        content="НМЦК 1 000 000",
        canonical_key="tender:777",
    )
    remember_items(store, [first], digest_date=date(2026, 11, 18))

    amended = make_item(
        "Тендер 777 капремонт школы",
        "https://etp.example/lot/777",
        content="НМЦК 1 200 000",
        canonical_key="tender:777",
    )
    reprint = make_item(
        "Тендер 777 капремонт школы",
        "https://mirror.example/lot/777",
        content="НМЦК 1 000 000",
        canonical_key="tender:777",
    )
    split = split_by_memory(
        [amended, reprint], None, store=store, days=30
    )
    assert split.updates == [amended]
    assert amended.metadata.get("is_update") is True
    assert split.duplicates == [reprint]


def test_remember_items_writes_event_keys(store):
    item = make_item("Story", "https://example.com/story")
    assert remember_items(store, [item], digest_date=date(2026, 11, 19)) == 1
    events = store.iter_events(days=30)
    assert len(events) == 1
    assert events[0].title == "Story"


def test_canonical_key_falls_back_to_url():
    item = make_item("Story", "https://example.com/story/")
    assert canonical_key_for(item) == "https://example.com/story"


def test_cooldown_ladder():
    assert cooldown_seconds_for(0) == 0
    assert cooldown_seconds_for(1) == 6 * 3600
    assert cooldown_seconds_for(2) == 24 * 3600
    assert cooldown_seconds_for(9) == 72 * 3600


def test_source_health_tracks_failures_and_recovery(store):
    failed = store.health_record("rss:feed", ok=False, error="timeout", cooldown_seconds=3600)
    assert failed.consecutive_failures == 1
    assert failed.in_cooldown

    recovered = store.health_record("rss:feed", ok=True, latency_ms=120, fetched=5)
    assert recovered.consecutive_failures == 0
    assert not recovered.in_cooldown
    assert recovered.total_fetched == 5


def test_analysis_cache_respects_ttl(store):
    store.cache_put("k1", {"score": 9}, profile_id="ru-digest")
    assert store.cache_get("k1", ttl_hours=72) == {"score": 9}
    assert store.cache_get("k1", ttl_hours=-1) is None
    assert store.cache_prune(ttl_hours=0) == 1


def test_source_yield_is_tracked_per_run(store):
    store.health_record("rss:feed", ok=True, fetched=100)
    store.health_record_yield("rss:feed", fetched=100, kept=40)

    health = store.health_get("rss:feed")
    assert health.total_fetched == 100
    assert health.total_kept == 40
    assert health.yield_ratio == 0.4
    assert health.yield_runs == 1
    assert not health.muted


def test_yield_ratio_is_unmeasured_without_a_fetch(store):
    """A survivor with no fetch behind it is not a ratio and must read as unknown."""
    store.health_record_yield("rss:mystery", fetched=0, kept=3)

    health = store.health_get("rss:mystery")
    assert health.yield_ratio is None
    assert health.total_kept == 3
    # A source with nothing to fetch must not be muted for it.
    assert not health.muted


def test_low_yield_streak_mutes_after_threshold(store):
    for _ in range(2):
        store.health_record_yield("rss:noisy", fetched=200, kept=1, min_yield=0.05, mute_after=3)
    assert not store.health_get("rss:noisy").muted

    store.health_record_yield("rss:noisy", fetched=200, kept=1, min_yield=0.05, mute_after=3)
    muted = store.health_get("rss:noisy")
    assert muted.muted
    assert muted.low_yield_streak == 3
    assert muted.muted_reason.startswith("low_yield:")


def test_healthy_yield_resets_the_streak(store):
    store.health_record_yield("rss:mixed", fetched=100, kept=0, min_yield=0.05, mute_after=3)
    store.health_record_yield("rss:mixed", fetched=100, kept=0, min_yield=0.05, mute_after=3)
    store.health_record_yield("rss:mixed", fetched=100, kept=50, min_yield=0.05, mute_after=3)

    health = store.health_get("rss:mixed")
    assert health.low_yield_streak == 0
    assert not health.muted


def test_unmute_clears_the_mark(store):
    for _ in range(3):
        store.health_record_yield("rss:dead", fetched=50, kept=0, min_yield=0.05, mute_after=3)
    assert store.health_get("rss:dead").muted

    store.health_unmute("rss:dead")
    cleared = store.health_get("rss:dead")
    assert not cleared.muted
    assert cleared.muted_reason is None
    assert cleared.low_yield_streak == 0


def test_health_rows_survive_a_pre_existing_database(tmp_path):
    """An existing state.db predates the yield columns; it must still open."""
    import sqlite3

    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        "CREATE TABLE source_health (source_key TEXT PRIMARY KEY,"
        " consecutive_failures INTEGER NOT NULL DEFAULT 0, cooldown_until REAL,"
        " last_ok_at REAL, last_error TEXT, total_fetched INTEGER NOT NULL DEFAULT 0,"
        " avg_latency_ms REAL, updated_at REAL NOT NULL);"
        "INSERT INTO source_health VALUES ('rss:old', 2, NULL, NULL, 'boom', 42, 10.0, 1.0);"
    )
    conn.commit()
    conn.close()

    with StateStore(path) as legacy:
        health = legacy.health_get("rss:old")
        assert health.consecutive_failures == 2
        assert health.total_fetched == 42
        # New columns appear with neutral defaults rather than breaking the row.
        assert health.total_kept == 0
        assert health.yield_ratio is None
        assert not health.muted


def test_run_report_keeps_only_sources_this_run_touched(store):
    """A switched-off connector must stop reporting its old error.

    ``health_all()`` returns every source ever recorded. VK kept "no token" in
    the report long after the addon was turned off, which held the snapshot at
    ``partial_failure`` and made the cabinet look broken for a source nobody was
    collecting. Sources still relevant - muted for low yield, or inside a cooldown
    - stay, because the operator needs to see those.
    """
    import time

    from src.orchestrator import FetchReport, HorizonOrchestrator, SourceFetchOutcome

    store.health_record("Сообщества VK", ok=False, error="нет токена")
    store.health_record("Посты vc.ru", ok=True, fetched=10)
    store.health_record("Старый фид", ok=False, error="timeout", cooldown_seconds=3600)

    orchestrator = object.__new__(HorizonOrchestrator)
    orchestrator._state_store = store
    orchestrator.last_fetch_report = FetchReport(
        outcomes=[
            SourceFetchOutcome(source_name="Посты vc.ru", status="success"),
        ]
    )

    keys = {row["source_key"] for row in orchestrator._sources_health_rows()}

    assert "Сообщества VK" not in keys
    assert "Посты vc.ru" in keys
    # Still inside its cooldown, so it is news the operator can act on.
    assert "Старый фид" in keys
