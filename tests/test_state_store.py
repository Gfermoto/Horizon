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
