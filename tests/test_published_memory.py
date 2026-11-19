from datetime import date
from pathlib import Path

from src.processing.published_memory import (
    filter_new_items,
    is_already_published,
    load_published_memory,
    title_fingerprint,
    titles_similar,
)


def test_titles_similar_catches_light_rewrites() -> None:
    a = "Ректор РТУ МИРЭА Станислав Кудж: важно уметь спорить с ИИ и сохранять ответственность за решения"
    b = "Ректор РТУ МИРЭА Станислав Кудж: человек должен уметь спорить с ИИ и отвечать за решения"
    assert titles_similar(a, b)
    assert title_fingerprint(a)


def test_filter_new_items_uses_digest_history(tmp_path: Path) -> None:
    summaries = tmp_path / "summaries"
    summaries.mkdir()
    (summaries / "horizon-2026-11-18-ru.md").write_text(
        "# Digests\n\n"
        "## [Ректор РТУ МИРЭА Станислав Кудж: важно уметь спорить с ИИ]"
        "(https://samaraonline24.ru/old)\n\n"
        "Summary one.\n",
        encoding="utf-8",
    )
    class Item:
        def __init__(self, title: str, url: str) -> None:
            self.title = title
            self.url = url

    fresh_title = "Студенты РТУ МИРЭА посетили НПП Исток"
    reprint = Item(
        "Ректор РТУ МИРЭА Станислав Кудж: человек должен уметь спорить с ИИ",
        "https://life.ru/new-url",
    )
    fresh = Item(fresh_title, "https://regions.ru/fresh")
    memory = load_published_memory(summaries, days=30, before=date(2026, 11, 19))
    assert memory
    assert is_already_published(
        title=reprint.title, url=reprint.url, memory=memory
    )
    assert not is_already_published(
        title=fresh.title, url=fresh.url, memory=memory
    )
    kept, dropped = filter_new_items([reprint, fresh], summaries, days=30)
    assert reprint in dropped
    assert fresh in kept
