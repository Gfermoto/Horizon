"""Zero-LLM pre-filter and per-run budget guard."""

from datetime import datetime, timedelta, timezone

from src.ai.tokens import BudgetGuard, reset_usage, record_usage
from src.models import ContentItem, SourceType
from src.processing.prefilter import apply_prefilter

NOW = datetime(2026, 11, 19, 12, tzinfo=timezone.utc)


def item(title, content="body text", hours_ago=0):
    return ContentItem(
        id=f"rss:test:{title[:12]}",
        source_type=SourceType.RSS,
        title=title,
        url=f"https://example.com/{abs(hash(title)) % 10000}",
        content=content,
        published_at=NOW - timedelta(hours=hours_ago),
    )


def test_prefilter_must_match_drops_off_topic():
    items = [item("РТУ МИРЭА открыл лабораторию"), item("Погода в Москве")]
    result = apply_prefilter(items, must_match=["мирэа"])
    assert [i.title for i in result.items] == ["РТУ МИРЭА открыл лабораторию"]
    assert result.dropped_must_match == 1


def test_prefilter_exclude_patterns_and_min_content():
    items = [
        item("Реклама курсов", "купите курс"),
        item("Новость", ""),
        item("Нормальная новость", "достаточно текста для анализа"),
    ]
    result = apply_prefilter(
        items, exclude_patterns=["реклама"], min_content_chars=10
    )
    assert [i.title for i in result.items] == ["Нормальная новость"]
    assert result.dropped_exclude == 1
    assert result.dropped_min_content == 1


def test_prefilter_cap_keeps_newest():
    items = [item(f"Story {n}", hours_ago=n) for n in range(5)]
    result = apply_prefilter(items, max_items_to_analyze=2)
    assert len(result.items) == 2
    assert result.dropped_cap == 3
    published = [i.published_at for i in result.items]
    assert published == sorted(published, reverse=True)


def test_prefilter_keeps_everything_by_default():
    items = [item("Anything"), item("Else")]
    result = apply_prefilter(items)
    assert len(result.items) == 2
    assert result.dropped_total == 0


def test_budget_guard_allows_when_unlimited():
    guard = BudgetGuard()
    assert guard.allow(stage="analyze")
    assert not guard.truncated


def test_budget_guard_blocks_after_limit():
    reset_usage()
    guard = BudgetGuard(max_total_tokens=1_000)
    assert guard.allow(stage="analyze")
    record_usage("deepseek", input_tokens=900, output_tokens=200)
    assert guard.exhausted()
    assert guard.truncated
    assert "token budget" in guard.reason
    reset_usage()


def test_budget_guard_projects_cost():
    reset_usage()
    guard = BudgetGuard(max_cost_usd=0.0001, usd_per_million_tokens=100.0)
    # 0.0001 USD at $100/M tokens = 1 token budget; any real stage exceeds it.
    assert not guard.allow(stage="analyze", estimated_tokens=10)
    assert "cost budget" in guard.reason
    reset_usage()


def test_budget_report_shape():
    guard = BudgetGuard(max_total_tokens=5000, max_cost_usd=1.0)
    report = guard.report()
    assert report["max_total_tokens"] == 5000
    assert report["truncated"] is False
    assert set(report) >= {"tokens_spent", "cost_usd", "reason"}
