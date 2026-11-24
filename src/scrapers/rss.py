"""RSS feed scraper implementation."""

import asyncio
import calendar
import hashlib
import logging
import math
import os
import re
from datetime import datetime, timezone
from typing import List, Optional
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qs, urlsplit
import httpx
import feedparser

from .base import BaseScraper, FetchResult
from ..extractors import ExtractorRegistry
from ..models import ContentItem, SourceType, RSSSourceConfig

logger = logging.getLogger(__name__)

# Identify feed requests with a browser-compatible UA while keeping Horizon token
# (publishers such as Nature return empty pages to bare bot/httpx User-Agents).
RSS_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36 "
    "Horizon/0.1.0 (+https://github.com/Thysrael/Horizon)"
)
RSS_ACCEPT = (
    "application/rss+xml, application/atom+xml, application/rdf+xml, "
    "application/xml, text/xml;q=0.9, */*;q=0.8"
)
MAX_CONCURRENT_FEEDS = 4
EXTRACT_CONCURRENCY = 6
EXTRACT_TIMEOUT_SEC = 10.0

_STOPWORDS = frozenset({
    "в", "во", "на", "по", "из", "за", "для", "от", "до", "про", "об", "о",
    "к", "ко", "с", "со", "у", "и", "или", "не", "ни", "а", "но",
    "and", "the", "for", "of", "in", "on", "to", "with", "at", "by",
})
_ORG_PREFIXES = frozenset({
    "рту", "ниу", "мгту", "фгбоу", "фгаоу", "пао", "ооо", "зао", "оао", "ао",
    "гк", "нпо", "ано", "гуп", "фгуп", "мбоу", "гаоу",
})
_RU_ENDINGS = (
    "ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими", "ах", "ях",
    "ов", "ев", "ей", "ам", "ям", "ом", "ем", "ой", "ую", "юю",
    "ая", "яя", "ое", "ее", "ые", "ие", "ый", "ий",
    "а", "я", "у", "ю", "е", "и", "ы", "о", "ь", "s",
)


def _unwrap_entry_url(raw_url: str) -> str:
    """Unwrap search-engine redirect wrappers (e.g. Bing News apiclick.aspx?url=...)."""
    url = (raw_url or "").strip()
    if not url:
        return url
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    host = (parts.hostname or "").lower()
    if host.endswith("bing.com") and "apiclick" in parts.path.lower():
        target = (parse_qs(parts.query).get("url") or [""])[0].strip()
        if target.startswith(("http://", "https://")):
            return target
    return url


def _is_search_feed_url(feed_url: str) -> bool:
    lower = feed_url.lower()
    return "bing.com/news/search" in lower or "news.google.com/rss/search" in lower


def _publisher_label(entry: dict, resolved_url: str, fallback: str) -> str:
    """Prefer the real publisher name/domain over a synthetic search-feed title."""
    for key in ("news_source", "source"):
        val = entry.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
        if isinstance(val, dict) and isinstance(val.get("title"), str) and val["title"].strip():
            return val["title"].strip()
    try:
        host = (urlsplit(resolved_url).hostname or "").lower()
        if host.startswith("www."):
            host = host[4:]
        if host and not host.endswith(("bing.com", "google.com")):
            return host
    except ValueError:
        pass
    return fallback


def _stem_token(raw_token: str) -> tuple[str, bool]:
    """Return (normalized_pattern_stem, exact_word_end) for a query token."""
    is_acronym = len(raw_token) >= 2 and raw_token.isupper()
    norm = raw_token.lower().replace("ё", "е")
    if len(norm) <= 3 or (is_acronym and len(norm) <= 4):
        return norm, True
    if is_acronym:
        return norm, False
    for ending in _RU_ENDINGS:
        if norm.endswith(ending) and len(norm) - len(ending) >= 4:
            return norm[: -len(ending)], False
    return norm, False


def _matches_topic_query(text: str, query: str) -> bool:
    """Verify that ``text`` genuinely mentions the entity/topic in ``query``."""
    cleaned_q = (query or "").strip()
    if not cleaned_q:
        return True
    haystack = (text or "").lower().replace("ё", "е")
    if not haystack:
        return False

    clauses = re.split(r"\s+(?:OR|ИЛИ)\s+|\|", cleaned_q)
    matched_any_clause = False
    has_valid_clause = False

    for clause in clauses:
        raw_tokens = [
            tok for tok in re.findall(r"[A-Za-zА-Яа-яЁё0-9]+(?:-[A-Za-zА-Яа-яЁё0-9]+)*", clause)
            if len(tok) >= 2 and tok.lower().replace("ё", "е") not in _STOPWORDS
        ]
        if not raw_tokens:
            continue
        has_valid_clause = True
        non_prefix = [
            tok for tok in raw_tokens
            if tok.lower().replace("ё", "е") not in _ORG_PREFIXES
        ]
        required = non_prefix if non_prefix else raw_tokens

        clause_ok = True
        for tok in required:
            stem, exact_end = _stem_token(tok)
            escaped = re.escape(stem)
            pattern = (
                rf"(?<![A-Za-zА-Яа-яЁё0-9]){escaped}(?![A-Za-zА-Яа-яЁё0-9])"
                if exact_end
                else rf"(?<![A-Za-zА-Яа-яЁё0-9]){escaped}"
            )
            if not re.search(pattern, haystack):
                clause_ok = False
                break
        if clause_ok:
            matched_any_clause = True
            break

    return matched_any_clause if has_valid_clause else True

# Rate-limited feeds (e.g. Reddit RSS answers bursts with 429) are retried with
# linear backoff, honouring numeric or HTTP-date Retry-After values up to the cap.
RATE_LIMIT_RETRIES = 3
RATE_LIMIT_BACKOFF_SEC = 15.0
RATE_LIMIT_MAX_WAIT_SEC = 60.0
# A host that is unreachable (blocked, no route) will not recover inside one run,
# so connection failures get far fewer, shorter attempts. Without this a blocked
# host such as Google News from a Russian network burned minutes of every run
# waiting on retries that could never succeed.
TRANSPORT_RETRIES = 1
TRANSPORT_BACKOFF_SEC = 3.0


def _rate_limit_delay(response: httpx.Response, attempt: int) -> float:
    """Seconds to wait before retry number ``attempt`` (1-based)."""
    retry_after = response.headers.get("Retry-After", "")
    try:
        delay = float(retry_after)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(retry_after)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            delay = math.ceil(
                (retry_at - datetime.now(timezone.utc)).total_seconds()
            )
        except (TypeError, ValueError, OverflowError):
            delay = RATE_LIMIT_BACKOFF_SEC * attempt
    return max(0.0, min(delay, RATE_LIMIT_MAX_WAIT_SEC))


class RSSScraper(BaseScraper):
    """Scraper for RSS/Atom feeds."""

    def __init__(
        self,
        sources: List[RSSSourceConfig],
        http_client: httpx.AsyncClient,
        extractors: Optional[ExtractorRegistry] = None,
    ):
        """Initialize RSS scraper.

        Args:
            sources: List of RSS feed configurations
            http_client: Shared async HTTP client
            extractors: Optional registry of content extractors for full article fetching
        """
        super().__init__({"sources": sources}, http_client)
        self._extractors = extractors

    async def fetch(self, since: datetime) -> FetchResult:
        """Fetch RSS feed items.

        Args:
            since: Only fetch items published after this time

        Returns:
            FetchResult: Items and per-feed failures
        """
        result = FetchResult()
        sources = [source for source in self.config["sources"] if source.enabled]
        slots = asyncio.Semaphore(MAX_CONCURRENT_FEEDS)
        host_locks = {source.url.host: asyncio.Lock() for source in sources}

        async def fetch_source(source: RSSSourceConfig) -> FetchResult:
            # Wait for the host before occupying a global slot, so queued feeds
            # from one host cannot prevent other hosts from making progress.
            async with host_locks[source.url.host], slots:
                try:
                    return await self._fetch_feed(source, since)
                except Exception as exc:
                    failure = FetchResult()
                    failure.fail(source.name, exc)
                    return failure

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(fetch_source(source)) for source in sources]

        # Preserve configured feed order regardless of completion order.
        for task in tasks:
            feed_result = task.result()
            result.items.extend(feed_result.items)
            result.failures.update(feed_result.failures)
            result.successful += feed_result.successful

        return result

    async def _fetch_feed(
        self, source: RSSSourceConfig, since: datetime
    ) -> FetchResult:
        """Fetch items from a single RSS feed.

        Args:
            source: RSS feed configuration
            since: Only fetch items after this time

        Returns:
            FetchResult: Feed content items and entry failures
        """
        result = FetchResult()

        # Expand environment variables in URL (e.g. ${LWN_TOKEN})
        feed_url = re.sub(
            r"\$\{(\w+)\}",
            lambda m: os.environ.get(m.group(1), m.group(0)).strip(),
            str(source.url),
        )
        is_search_feed = _is_search_feed_url(feed_url)
        response = None
        last_transport_error: Exception | None = None
        for attempt in range(RATE_LIMIT_RETRIES + 1):
            try:
                response = await self.client.get(
                    feed_url,
                    follow_redirects=True,
                    headers={"User-Agent": RSS_USER_AGENT, "Accept": RSS_ACCEPT},
                )
                last_transport_error = None
            except httpx.TransportError as exc:
                last_transport_error = exc
                # Unreachable host: retrying inside one run rarely helps, so the
                # transport budget is deliberately short.
                if attempt >= TRANSPORT_RETRIES:
                    raise
                delay = TRANSPORT_BACKOFF_SEC * (attempt + 1)
                logger.info(
                    "RSS feed %s transport error (%s); retrying in %.0fs (%d/%d)",
                    source.name,
                    type(exc).__name__,
                    delay,
                    attempt + 1,
                    TRANSPORT_RETRIES,
                )
                await asyncio.sleep(delay)
                continue
            status_code = response.status_code if isinstance(response.status_code, int) else None
            if status_code == 429 and attempt < RATE_LIMIT_RETRIES:
                delay = _rate_limit_delay(response, attempt + 1)
                logger.info(
                    "RSS feed %s rate-limited (429); retrying in %.0fs (%d/%d)",
                    source.name, delay, attempt + 1, RATE_LIMIT_RETRIES,
                )
                await asyncio.sleep(delay)
                continue
            if (
                status_code is not None
                and 500 <= status_code < 600
                and attempt < RATE_LIMIT_RETRIES
            ):
                delay = RATE_LIMIT_BACKOFF_SEC * (attempt + 1)
                logger.info(
                    "RSS feed %s server error (%s); retrying in %.0fs (%d/%d)",
                    source.name, status_code, delay, attempt + 1, RATE_LIMIT_RETRIES,
                )
                await asyncio.sleep(delay)
                continue
            break
        if response is None:
            assert last_transport_error is not None
            raise last_transport_error
        response.raise_for_status()
        feed = feedparser.parse(response.content, response_headers=dict(response.headers))
        if feed.bozo and not feed.entries and not isinstance(
            feed.bozo_exception, feedparser.NonXMLContentType
        ):
            raise ValueError("Invalid RSS/Atom feed")

        for ordinal, entry in enumerate(feed.entries, start=1):
            if source.max_entries is not None and len(result.items) >= source.max_entries:
                break
            try:
                published_at = self._parse_date(entry)
                if not published_at or published_at < since:
                    continue

                raw_link = entry.get("link", str(source.url))
                resolved_url = _unwrap_entry_url(raw_link) or str(source.url)
                title = entry.get("title", "Untitled")
                rss_content = self._extract_content(entry)

                # For regular feeds with must_match, skip unrelated items before
                # making external article-extraction HTTP calls.
                if (
                    source.must_match
                    and not is_search_feed
                    and not _matches_topic_query(f"{title}\n{rss_content}", source.must_match)
                ):
                    continue

                feed_id = str(source.url).split("//")[1].replace("/", "_")
                entry_id = (
                    resolved_url
                    if resolved_url != raw_link
                    else entry.get("id", resolved_url)
                )
                entry_hash = hashlib.sha256(str(entry_id).encode("utf-8")).hexdigest()[:16]
                content = rss_content

                if source.content_extractor and self._extractors:
                    extractor = self._extractors.get(source.content_extractor)
                    if extractor and resolved_url and (
                        is_search_feed or source.must_match or len(rss_content) < 500
                    ):
                        full = await extractor.extract(resolved_url, self.client)
                        if full:
                            content = full

                if source.must_match and not _matches_topic_query(
                    f"{title}\n{rss_content}\n{content}", source.must_match
                ):
                    continue

                feed_label = (
                    _publisher_label(entry, resolved_url, source.name)
                    if is_search_feed
                    else source.name
                )
                result.items.append(ContentItem(
                    id=self._generate_id("rss", feed_id, entry_hash),
                    source_type=SourceType.RSS,
                    title=title,
                    url=resolved_url,
                    content=content,
                    author=entry.get("author", feed_label),
                    published_at=published_at,
                    profile=source.profile,
                    metadata={
                        "feed_name": feed_label,
                        "category": source.category,
                        "tags": [tag.term for tag in entry.get("tags", [])],
                    },
                ))
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                label = f"{source.name}/entry:{ordinal}"
                # Avoid multiline validation details that include entry values.
                reason = str(exc).partition("\n")[0][:200]
                error = f"{type(exc).__name__}: {reason}"
                result.failures[label] = error
                logger.warning("Skipping RSS entry %s: %s", label, error)

        # A feed with only rejected entries is not a successful empty response.
        result.successful = int(bool(result.items) or not result.failures)
        return result

    def _parse_date(self, entry: dict) -> datetime:
        """Parse publication date from feed entry.

        Args:
            entry: Feed entry data

        Returns:
            datetime: Parsed publication date or None
        """
        # Try different date fields
        for field in ["published", "updated", "created"]:
            if field in entry:
                try:
                    # Try parsing structured time first
                    if f"{field}_parsed" in entry and entry[f"{field}_parsed"]:
                        return datetime.fromtimestamp(
                            calendar.timegm(entry[f"{field}_parsed"]), tz=timezone.utc
                        )
                    # Fallback to string parsing
                    date_str = entry[field]
                    parsed_date = parsedate_to_datetime(date_str)
                    if parsed_date.tzinfo is None:
                        parsed_date = parsed_date.replace(tzinfo=timezone.utc)
                    return parsed_date
                except Exception:
                    continue

        return None

    def _extract_content(self, entry: dict) -> str:
        """Extract text content from feed entry.

        Args:
            entry: Feed entry data

        Returns:
            str: Extracted text content
        """
        content = entry.get("content", [])
        if not isinstance(content, (list, tuple)):
            content = []
        candidates = []
        for part in content:
            if not isinstance(part, dict):
                continue
            media_type = part.get("type", "")
            if not isinstance(media_type, str):
                continue
            media_type = media_type.split(";", 1)[0].strip().lower()
            if media_type and not (
                media_type.startswith("text/")
                or media_type in ("html", "xhtml", "application/xhtml+xml")
            ):
                continue
            candidates.append(part.get("value"))
        # max keeps the first candidate on ties, so prefer content alternatives.
        candidates.extend([entry.get("summary"), entry.get("description")])
        return max(
            (value for value in candidates if isinstance(value, str) and value.strip()),
            key=len,
            default="",
        )
