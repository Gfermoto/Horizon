"""Declarative scraper for any public JSON feed.

Most "new source" requests are not new code — they are a new URL plus a mapping
from that document to ``ContentItem``. This scraper covers those cases (open
REST APIs, government registries, tender aggregators that publish JSON, internal
dashboards) without anyone writing a connector.

Configuration lives in ``sources.custom``:

.. code-block:: json

    {
      "type": "json_feed",
      "name": "Tenders 44-FZ",
      "url": "https://api.example.ru/tenders?limit=50",
      "items_path": "data.items",
      "id_field": "regNumber",
      "title_field": "purchaseName",
      "url_field": "detailUrl",
      "date_field": "publishDate",
      "content_field": "description",
      "canonical_key_field": "regNumber",
      "entity_kind": "tender",
      "metadata_fields": {"nmck": "maxPrice", "customer": "customer.fullName"}
    }

Field paths support dotted access (``customer.fullName``) and list indices
(``items.0.name``). Nothing here is domain-specific: the same scraper serves
news feeds, tender registries and release notes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

import httpx

from .base import BaseScraper, FetchResult
from ..models import ContentItem, SourceType
from ..url_security import UnsafeURLError, validate_http_url

DEFAULT_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _dig(data: Any, path: str | None) -> Any:
    """Resolve a dotted path against nested dicts/lists."""
    if not path:
        return data
    current = data
    for part in path.split("."):
        if isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        elif isinstance(current, dict):
            current = current.get(part)
        else:
            return None
        if current is None:
            return None
    return current


def _parse_date(value: Any, date_format: str | None) -> datetime:
    """Parse the common date shapes, always returning an aware UTC datetime."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, (int, float)):
        stamp = float(value)
        if stamp > 1e11:  # milliseconds
            stamp /= 1000.0
        return datetime.fromtimestamp(stamp, tz=timezone.utc)
    text = str(value or "").strip()
    if not text:
        return datetime.now(timezone.utc)

    if date_format:
        try:
            parsed = datetime.strptime(text, date_format)
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            pass

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    try:
        parsed = parsedate_to_datetime(text)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except (TypeError, ValueError):
        return datetime.now(timezone.utc)


class GenericJSONScraper(BaseScraper):
    """Fetch a public JSON endpoint and map it onto ``ContentItem``."""

    def __init__(self, config: Dict[str, Any], http_client: httpx.AsyncClient):
        super().__init__(config, http_client)
        self.config = config or {}

    async def fetch(self, since: datetime) -> FetchResult:
        result = FetchResult()
        url = str(self.config.get("url") or "").strip()
        if not url:
            result.fail("json_feed", ValueError("url is required"))
            return result

        try:
            validate_http_url(url)
        except UnsafeURLError as exc:
            result.fail("json_feed", ValueError(f"insecure url: {exc}"))
            return result

        await result.collect(
            self.config.get("name") or url, self._fetch_source(url, since)
        )
        return result

    async def _fetch_source(self, url: str, since: datetime) -> List[ContentItem]:
        headers = {"Accept": "application/json"}
        for header in self.config.get("headers") or []:
            if isinstance(header, str) and ":" in header:
                key, value = header.split(":", 1)
                headers[key.strip()] = value.strip()

        response = await self.client.get(url, headers=headers)
        response.raise_for_status()
        payload = response.json()

        rows = _dig(payload, self.config.get("items_path"))
        if rows is None:
            rows = payload if isinstance(payload, list) else []
        if not isinstance(rows, list):
            raise ValueError(
                f"items_path resolved to {type(rows).__name__}, expected a list"
            )

        max_items = int(self.config.get("max_items") or 0)
        if max_items > 0:
            rows = rows[:max_items]

        entity_kind = str(self.config.get("entity_kind") or "article")
        date_format = self.config.get("date_format")
        metadata_fields: Dict[str, str] = dict(self.config.get("metadata_fields") or {})
        default_profile = self.config.get("profile")

        items: List[ContentItem] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = _dig(row, self.config.get("title_field") or "title")
            link = _dig(row, self.config.get("url_field") or "url")
            if not title or not link:
                continue
            raw_date = _dig(row, self.config.get("date_field") or "published_at")
            published = _parse_date(raw_date, date_format)
            if published < since:
                continue

            native_id = _dig(row, self.config.get("id_field") or "id")
            canonical = _dig(row, self.config.get("canonical_key_field"))
            content = _dig(row, self.config.get("content_field") or "content")

            metadata: Dict[str, Any] = {
                "source_name": self.config.get("name") or "",
                "custom_source_type": self.config.get("custom_type") or "json_feed",
            }
            for key, path in metadata_fields.items():
                value = _dig(row, path)
                if value is not None:
                    metadata[key] = value

            items.append(
                ContentItem(
                    id=self._generate_id(
                        "custom",
                        entity_kind,
                        str(native_id or canonical or link)[:120],
                    ),
                    source_type=SourceType.RSS,
                    title=str(title),
                    url=str(link),
                    content=str(content) if content is not None else None,
                    published_at=published,
                    metadata=metadata,
                    profile=default_profile,
                    canonical_key=str(canonical) if canonical else None,
                    entity_kind=entity_kind,
                )
            )
        return items


def scraper_factory(config: Any, http_client: httpx.AsyncClient) -> BaseScraper:
    """Entry point used by the plugin registry for ``type: json_feed``."""
    payload = config if isinstance(config, dict) else getattr(config, "config", {}) or {}
    return GenericJSONScraper(payload, http_client)
