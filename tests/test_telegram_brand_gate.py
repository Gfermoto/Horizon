"""Brand gate for Telegram ContentItems (PR radar coverage)."""

from __future__ import annotations

from datetime import datetime, timezone

from src.models import ContentItem, SourceType
from src.processing.prefilter import apply_prefilter


def test_prefilter_keeps_telegram_post_with_brand_and_drops_others():
    hit = ContentItem(
        id="telegram:sbernews:1",
        source_type=SourceType.TELEGRAM,
        title="Сбер запустил нового робота",
        content="В канале рассказывают про Сбер и его продукт.",
        url="https://t.me/sbernews/1",
        published_at=datetime(2026, 11, 24, 10, 0, tzinfo=timezone.utc),
    )
    miss = ContentItem(
        id="telegram:sbernews:2",
        source_type=SourceType.TELEGRAM,
        title="Погода на выходные",
        content="Без упоминания бренда, просто прогноз.",
        url="https://t.me/sbernews/2",
        published_at=datetime(2026, 11, 24, 11, 0, tzinfo=timezone.utc),
    )

    result = apply_prefilter([hit, miss], must_match=["Сбер"])
    assert result.dropped_must_match == 1
    assert len(result.items) == 1
    assert result.items[0].id == hit.id
    assert result.items[0].source_type == SourceType.TELEGRAM
