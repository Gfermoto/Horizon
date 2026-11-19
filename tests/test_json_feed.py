"""Declarative/plugin sources: registry discovery and GenericJSONScraper."""

import asyncio
from datetime import datetime, timezone

import httpx
import pytest

from src.scrapers.json_feed import GenericJSONScraper, _dig, _parse_date, scraper_factory
from src.scrapers.registry import (
    ScraperPlugin,
    discover_plugins,
    get_scraper_plugin,
    register_plugin,
)

SINCE = datetime(2026, 11, 1, tzinfo=timezone.utc)

PAYLOAD = {
    "data": {
        "items": [
            {
                "regNumber": "0173100000123000001",
                "purchaseName": "Капитальный ремонт школы № 12",
                "detailUrl": "https://tenders.example/notice/1",
                "publishDate": "2026-11-18T09:30:00+03:00",
                "description": "Работы по капремонту",
                "maxPrice": 12500000,
                "customer": {"fullName": "ГБОУ Школа № 12"},
            },
            {
                "regNumber": "0173100000123000002",
                "purchaseName": "Поставка компьютеров",
                "detailUrl": "https://tenders.example/notice/2",
                "publishDate": "2026-10-01T09:30:00+03:00",
                "maxPrice": 900000,
                "customer": {"fullName": "ГБОУ Школа № 7"},
            },
        ]
    }
}


def test_dig_resolves_nested_paths():
    assert _dig(PAYLOAD, "data.items.0.regNumber") == "0173100000123000001"
    assert _dig(PAYLOAD, "data.items.5.regNumber") is None
    assert _dig(PAYLOAD, "missing.path") is None


def test_parse_date_handles_iso_epoch_and_rfc822():
    assert _parse_date("2026-11-18T09:30:00+03:00", None).year == 2026
    assert _parse_date(1795084153, None).year >= 2026
    assert _parse_date("Wed, 18 Nov 2026 09:30:00 +0300", None).month == 11


def test_json_feed_maps_declarative_config():
    def handler(request):
        return httpx.Response(200, json=PAYLOAD)

    config = {
        "url": "https://tenders.example/api/list",
        "name": "Tenders",
        "items_path": "data.items",
        "id_field": "regNumber",
        "title_field": "purchaseName",
        "url_field": "detailUrl",
        "date_field": "publishDate",
        "content_field": "description",
        "canonical_key_field": "regNumber",
        "entity_kind": "tender",
        "metadata_fields": {"nmck": "maxPrice", "customer": "customer.fullName"},
    }

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await GenericJSONScraper(config, client).fetch(SINCE)

    result = asyncio.run(run())
    assert result.status == "success"
    assert len(result.items) == 1  # the October item is outside the window
    item = result.items[0]
    assert item.entity_kind == "tender"
    assert item.canonical_key == "0173100000123000001"
    assert item.metadata["nmck"] == 12500000
    assert item.metadata["customer"] == "ГБОУ Школа № 12"


def test_json_feed_reports_missing_url():
    async def run():
        async with httpx.AsyncClient() as client:
            return await GenericJSONScraper({"name": "broken"}, client).fetch(SINCE)

    result = asyncio.run(run())
    assert result.status == "failure"
    assert result.failures


def test_json_feed_rejects_private_hosts():
    async def run():
        async with httpx.AsyncClient() as client:
            return await GenericJSONScraper(
                {"url": "http://localhost:8000/list"}, client
            ).fetch(SINCE)

    result = asyncio.run(run())
    assert result.status == "failure"
    assert "insecure url" in "; ".join(result.failures.values())


def test_registry_exposes_builtin_and_plugins():
    plugins = discover_plugins(refresh=True)
    assert {"rss", "json_feed", "github"} <= set(plugins)
    assert get_scraper_plugin("json_feed").builtin


def test_registry_rejects_unknown_type():
    discover_plugins(refresh=True)
    with pytest.raises(KeyError):
        get_scraper_plugin("does-not-exist")


def test_register_plugin_overrides_lookup():
    def factory(config, client):  # pragma: no cover - never called
        raise AssertionError("not built in this test")

    register_plugin(ScraperPlugin(name="acme", factory=factory, builtin=False))
    assert get_scraper_plugin("acme").factory is factory


def test_scraper_factory_accepts_models_and_dicts():
    from src.models import CustomSourceConfig

    config = CustomSourceConfig(type="json_feed", config={"url": "https://example.com/x"})
    scraper = scraper_factory(config, httpx.AsyncClient())
    assert isinstance(scraper, GenericJSONScraper)
