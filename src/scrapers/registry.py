"""Scraper plugin registry.

Third-party packages can contribute connectors without patching Horizon:

.. code-block:: toml

    [project.entry-points."horizon.scrapers"]
    tenders = "dawnwire_connectors_tenders:scraper_factory"

The factory receives ``(config, http_client)`` and returns a
:class:`~src.scrapers.base.BaseScraper`. That keeps a commercial connector (an
authenticated tender platform, a paid API) in its own repository while the open
engine stays generic.

Built-in scrapers are registered here too, so the orchestrator has one place to
ask "what can I fetch?" instead of hardcoding every import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any, Callable, Dict, List, Optional

import httpx

from .base import BaseScraper
from .github import GitHubScraper
from .gdelt import GDELTScraper
from .google_news import GoogleNewsScraper
from .hackernews import HackerNewsScraper
from .json_feed import GenericJSONScraper
from .openbb import OpenBBScraper
from .ossinsight import OSSInsightScraper
from .reddit import RedditScraper
from .rss import RSSScraper
from .telegram import TelegramScraper
from .twitter import TwitterScraper
from .twitter_playwright import TwitterPlaywrightScraper

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "horizon.scrapers"

ScraperFactory = Callable[[Any, httpx.AsyncClient], BaseScraper]


@dataclass(frozen=True)
class ScraperPlugin:
    """One discoverable scraper implementation."""

    name: str
    factory: ScraperFactory
    description: str = ""
    builtin: bool = True

    def build(self, config: Any, client: httpx.AsyncClient) -> BaseScraper:
        return self.factory(config, client)


@dataclass
class PluginLoadReport:
    """What plugin discovery found, including failures."""

    loaded: List[str] = field(default_factory=list)
    failed: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"loaded": list(self.loaded), "failed": dict(self.failed)}


_BUILTIN: Dict[str, ScraperPlugin] = {
    "github": ScraperPlugin("github", lambda cfg, client: GitHubScraper(cfg, client)),
    "hackernews": ScraperPlugin(
        "hackernews", lambda cfg, client: HackerNewsScraper(cfg, client)
    ),
    "rss": ScraperPlugin("rss", lambda cfg, client: RSSScraper(cfg, client)),
    "reddit": ScraperPlugin("reddit", lambda cfg, client: RedditScraper(cfg, client)),
    "telegram": ScraperPlugin(
        "telegram", lambda cfg, client: TelegramScraper(cfg, client)
    ),
    "twitter": ScraperPlugin("twitter", lambda cfg, client: TwitterScraper(cfg, client)),
    "twitter_playwright": ScraperPlugin(
        "twitter_playwright", lambda cfg, client: TwitterPlaywrightScraper(cfg)
    ),
    "openbb": ScraperPlugin("openbb", lambda cfg, client: OpenBBScraper(cfg, client)),
    "ossinsight": ScraperPlugin(
        "ossinsight", lambda cfg, client: OSSInsightScraper(cfg, client)
    ),
    "gdelt": ScraperPlugin("gdelt", lambda cfg, client: GDELTScraper(cfg, client)),
    "google_news": ScraperPlugin(
        "google_news", lambda cfg, client: GoogleNewsScraper(cfg, client)
    ),
    "json_feed": ScraperPlugin(
        "json_feed", lambda cfg, client: GenericJSONScraper(cfg, client)
    ),
}

_installed: Optional[Dict[str, ScraperPlugin]] = None
_last_report: PluginLoadReport = PluginLoadReport()


def _iter_entry_points() -> List[Any]:
    """Yield entry points across importlib.metadata API generations."""
    try:
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - Python < 3.8 fallback
        return []

    try:
        found = entry_points(group=ENTRY_POINT_GROUP)
        return list(found)
    except TypeError:  # pragma: no cover - legacy API
        all_points = entry_points()
        return list(getattr(all_points, "select", lambda **_: [])(group=ENTRY_POINT_GROUP))


def discover_plugins(*, refresh: bool = False) -> Dict[str, ScraperPlugin]:
    """Return built-ins plus every installed ``horizon.scrapers`` plugin."""
    global _installed, _last_report
    if _installed is not None and not refresh:
        return _installed

    report = PluginLoadReport(loaded=sorted(_BUILTIN))
    plugins: Dict[str, ScraperPlugin] = dict(_BUILTIN)

    for entry_point in _iter_entry_points():
        name = entry_point.name
        try:
            factory = entry_point.load()
        except Exception as exc:  # a broken plugin must not break the run
            logger.warning("Horizon scraper plugin %r failed to load: %s", name, exc)
            report.failed[name] = f"{type(exc).__name__}: {exc}"
            continue
        if not callable(factory):
            report.failed[name] = "entry point is not callable"
            continue
        plugins[name] = ScraperPlugin(
            name=name, factory=factory, description="external plugin", builtin=False
        )
        report.loaded.append(name)

    _installed = plugins
    _last_report = report
    return plugins


def plugin_report() -> PluginLoadReport:
    """Last discovery result (useful for diagnostics and the run report)."""
    discover_plugins()
    return _last_report


def get_scraper_plugin(name: str) -> ScraperPlugin:
    """Look up one scraper by plugin name."""
    plugins = discover_plugins()
    try:
        return plugins[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown scraper type {name!r}; known: {', '.join(sorted(plugins))}"
        ) from exc


def register_plugin(plugin: ScraperPlugin) -> None:
    """Register a plugin programmatically (tests, embedded usage)."""
    plugins = discover_plugins()
    plugins[plugin.name] = plugin
    global _installed
    _installed = plugins
