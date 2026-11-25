"""Main orchestrator coordinating the entire workflow."""

import asyncio
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from functools import cached_property
from pathlib import Path
from typing import Dict, List, Literal, Optional
from urllib.parse import unquote_plus, urlsplit
import httpx
from rich.console import Console

from .console_icons import get_icons
from .models import Config, ContentItem
from .storage.manager import StorageManager, safe_output_path
from .storage.state import StateStore, bootstrap_from_summaries, default_state_path
from .services.email import EmailManager
from .services.webhook import WebhookNotifier
from .services.wechat import WeChatNotifier
from .scrapers.base import BaseScraper
from .scrapers.github import GitHubScraper
from .scrapers.hackernews import HackerNewsScraper
from .scrapers.rss import RSSScraper
from .scrapers.reddit import RedditScraper
from .scrapers.telegram import TelegramScraper
from .scrapers.twitter import TwitterScraper
from .scrapers.twitter_playwright import TwitterPlaywrightScraper
from .scrapers.openbb import OpenBBScraper
from .scrapers.ossinsight import OSSInsightScraper
from .scrapers.gdelt import GDELTScraper
from .scrapers.google_news import GoogleNewsScraper
from .scrapers.registry import get_scraper_plugin
from .ai.client import AIClient, create_ai_client
from .ai.errors import AIProviderUnavailableError
from .ai.tokens import BudgetGuard
from .ai.analyzer import ContentAnalyzer
from .ai.summarizer import DailySummarizer
from .ai.enricher import ContentEnricher, EnrichmentBatchResult
from .ai.tokens import get_usage_snapshot, write_usage_snapshot
from .processing import ProfileRegistry
from .processing.content import COMMENTS_MARKER, split_content
from .processing.tools import ToolRegistry


_TRACKING_QUERY_PARAMETERS = {
    "_ga",
    "dclid",
    "fbclid",
    "gclid",
    "igshid",
    "li_fat_id",
    "mc_cid",
    "mc_eid",
    "msclkid",
    "ttclid",
    "twclid",
    "vero_id",
}


def _deduplication_url_key(url: str) -> tuple[str, str, str, str, Optional[int], str, str]:
    """Return a conservative URL identity key for cross-source deduplication."""
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    port = parsed.port
    if (scheme, port) in {("http", 80), ("https", 443)}:
        port = None

    path = parsed.path.rstrip("/") or "/"
    query_parts = []
    for part in parsed.query.split("&") if parsed.query else []:
        name = unquote_plus(part.partition("=")[0]).lower()
        if name.startswith("utm_") or name in _TRACKING_QUERY_PARAMETERS:
            continue
        query_parts.append(part)

    return (
        scheme,
        parsed.username or "",
        parsed.password or "",
        host,
        port,
        path,
        "&".join(query_parts),
    )


@dataclass
class BalancedDigestResult:
    """Items and selection statistics from balanced digest filtering."""

    items: List[ContentItem]
    enabled: bool = False
    group_counts: Dict[str, int] = field(default_factory=dict)
    group_limits: Dict[str, Optional[int]] = field(default_factory=dict)
    duplicate_categories: List[str] = field(default_factory=list)


@dataclass
class FilteringPipelineResult:
    """Items and statistics from score, topic, and digest filtering."""

    items: List[ContentItem]
    threshold_count: int
    topic_dedup_count: int
    topic_dedup_removed: int
    balanced_digest: BalancedDigestResult
    eligible_count: Optional[int] = None


@dataclass
class SourceFetchOutcome:
    """Result of fetching one configured source."""

    source_name: str
    status: Literal["success", "empty", "partial_failure", "failure", "skipped_cooldown"]
    items: List[ContentItem] = field(default_factory=list)
    error: Optional[str] = None
    failures: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        result: Dict[str, object] = {
            "source": self.source_name,
            "status": self.status,
            "item_count": len(self.items),
        }
        if self.error is not None:
            result["error"] = self.error
        if self.failures:
            result["failures"] = self.failures
        return result


@dataclass
class FetchReport:
    """Aggregate diagnostics for one fetch across configured sources."""

    outcomes: List[SourceFetchOutcome] = field(default_factory=list)

    @property
    def status(self) -> Literal["not_attempted", "success", "partial_failure", "failure"]:
        if not self.outcomes:
            return "not_attempted"
        if self.all_failed:
            return "failure"
        if self.failed_count:
            return "partial_failure"
        return "success"

    @property
    def failed_count(self) -> int:
        return sum(outcome.status in {"failure", "partial_failure"} for outcome in self.outcomes)

    @property
    def skipped_count(self) -> int:
        return sum(1 for outcome in self.outcomes if outcome.status == "skipped_cooldown")

    @property
    def all_failed(self) -> bool:
        attempted = [o for o in self.outcomes if o.status != "skipped_cooldown"]
        return bool(attempted) and all(outcome.status == "failure" for outcome in attempted)

    def failure_message(self) -> str:
        failures = "; ".join(
            f"{outcome.source_name}: {outcome.error or 'unknown error'}"
            for outcome in self.outcomes
            if outcome.status == "failure"
        )
        return f"All {len(self.outcomes)} attempted sources failed ({failures})"

    def all_items(self) -> List[ContentItem]:
        """Every item any source returned, before any dedup or filtering."""
        return [item for outcome in self.outcomes for item in outcome.items]

    def to_dict(self) -> Dict[str, object]:
        return {
            "status": self.status,
            "attempted": len(self.outcomes),
            "successful": len(self.outcomes) - self.failed_count,
            "empty": sum(outcome.status == "empty" for outcome in self.outcomes),
            "failed": self.failed_count,
            "partial_failure": sum(outcome.status == "partial_failure" for outcome in self.outcomes),
            "skipped_cooldown": sum(
                1 for outcome in self.outcomes if outcome.status == "skipped_cooldown"
            ),
            "item_count": sum(len(outcome.items) for outcome in self.outcomes),
            "sources": [outcome.to_dict() for outcome in self.outcomes],
        }


class HorizonOrchestrator:
    """Orchestrates the complete workflow for content aggregation and analysis."""

    icons = get_icons()

    def __init__(
        self,
        config: Config,
        storage: StorageManager,
        console: Optional[Console] = None,
        profiles: Optional[ProfileRegistry] = None,
    ):
        """Initialize orchestrator.

        Args:
            config: Application configuration
            storage: Storage manager
            console: Shared Rich Console instance
        """
        self.config = config
        self.storage = storage
        self.console = console or Console(stderr=True)
        self.icons = get_icons(config.display.icon_style)
        self.profiles = profiles or ProfileRegistry.load(
            Path(config.processing.profiles_dir), config.processing.default_profile
        )
        self.profiles.validate_source_references(
            config.sources.model_dump(mode="json")
        )
        for profile_id in config.processing.profile_settings:
            self.profiles.get(profile_id)
        if config.digest.profile_order:
            configured_profiles = set(config.digest.profile_order)
            unknown_profiles = configured_profiles - self.profiles.ids
            if unknown_profiles:
                raise ValueError(
                    "digest.profile_order contains unknown profiles "
                    f"({', '.join(sorted(unknown_profiles))})"
                )
            config.digest.profile_order.extend(
                profile.id
                for profile in self.profiles.profiles
                if profile.id not in configured_profiles
            )
        self.profiles.validate_digest_references(config.digest)
        self.email_manager = EmailManager(config.email, console=self.console) if config.email else None
        self.webhook_notifier = (
            WebhookNotifier(config.webhook, console=self.console, icons=self.icons)
            if config.webhook and config.webhook.enabled
            else None
        )
        self.wechat_notifier = (
            WeChatNotifier(config.wechat, self.storage, console=self.console, icons=self.icons)
            if config.wechat and config.wechat.enabled
            else None
        )
        self.last_fetch_report: Optional[FetchReport] = None
        self.run_counters: Dict[str, int] = {
            "fetched": 0,
            "memory_skipped": 0,
            "memory_updates": 0,
            "analyzed": 0,
            "event_skipped": 0,
        }
        self._state_store: Optional[StateStore] = None
        self._budget = BudgetGuard(
            max_total_tokens=getattr(config.ai, "max_total_tokens", None),
            max_cost_usd=getattr(config.ai, "max_cost_usd", None),
            usd_per_million_tokens=float(
                getattr(config.ai, "usd_per_million_tokens", 1.0) or 1.0
            ),
        )
        self._budget_report: Dict[str, object] = {}

    @property
    def state_store(self) -> Optional[StateStore]:
        """Durable SQLite state for this data directory (None when disabled)."""
        config = getattr(self, "config", None)
        if not getattr(getattr(config, "collection", None), "state_enabled", True):
            return None
        if getattr(self, "_state_store", None) is None:
            data_dir = getattr(getattr(self, "storage", None), "data_dir", None)
            if data_dir is None:
                return None
            self._state_store = StateStore(default_state_path(data_dir))
        return self._state_store

    def close_state(self) -> None:
        """Flush and close the state store if it was opened during this run."""
        if getattr(self, "_state_store", None) is not None:
            self._state_store.close()
            self._state_store = None

    @cached_property
    def ai_client(self) -> AIClient:
        """Share one client and account selection across the run's AI stages."""
        return create_ai_client(self.config.ai)

    async def run(self, force_hours: Optional[int] = None) -> None:
        """Execute the complete workflow.

        Args:
            force_hours: Hours already returned by collection_hours(). This
                method does not check the window again.
        """
        self.console.print(
            f"[bold cyan]{self.icons['start']} Horizon - Starting aggregation...[/bold cyan]\n"
        )

        # Check email subscriptions if configured
        if (
            self.email_manager
            and self.config.email
            and self.config.email.enabled
            and self.config.email.imap_enabled
        ):
            self.console.print(f"{self.icons['email']} Checking for new email subscriptions...")
            self.email_manager.check_subscriptions(self.storage)

        try:
            # 0. One-time import of existing Markdown digests into state.db so a
            # fresh install keeps its history instead of starting empty.
            state = self.state_store
            if state is not None:
                summaries_dir_for_bootstrap = getattr(self.storage, "summaries_dir", None)
                imported = bootstrap_from_summaries(
                    state, summaries_dir_for_bootstrap
                )
                if imported:
                    self.console.print(
                        f"   Imported {imported} historical items into state.db\n"
                    )

            # 1. Determine time window
            since = self._determine_time_window(force_hours)
            self.console.print(
                f"{self.icons['date']} Fetching content since: "
                f"{since.strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            budget = getattr(self, "_budget", None)
            if budget is not None:
                budget.reset()

            # 2. Fetch content from all sources
            all_items = await self.fetch_all_sources(since)
            self.console.print(
                f"{self.icons['fetched']} Fetched {len(all_items)} items from all sources\n"
            )

            if self.last_fetch_report and self.last_fetch_report.all_failed:
                message = self.last_fetch_report.failure_message()
                if getattr(self.config.collection, "tolerate_total_failure", False):
                    # Degraded but not fatal: keep scheduling stable and let the
                    # run_report explain what happened.
                    self.console.print(
                        f"[yellow]{self.icons['warning']} {message}; "
                        "continuing with whatever was fetched[/yellow]"
                    )
                    self.run_counters["all_sources_failed"] = 1
                else:
                    raise RuntimeError(message)

            if getattr(self.config.collection, "exclude_urls", None):
                excluded_keys = {
                    _deduplication_url_key(u)
                    for u in self.config.collection.exclude_urls
                    if u
                }
                before_exclude = len(all_items)
                all_items = [
                    item
                    for item in all_items
                    if _deduplication_url_key(str(item.url)) not in excluded_keys
                ]
                if len(all_items) < before_exclude:
                    self.console.print(
                        f"   Skipped {before_exclude - len(all_items)} already-published items\n"
                    )

            # Cross-day memory: same story under a new outlet URL must not re-enter.
            # Level 1+2 (identity + headline) run before any LLM call; level 3
            # (event key) runs after analysis below.
            from .processing.published_memory import split_by_memory

            exclude_fps = list(
                getattr(self.config.collection, "exclude_title_fingerprints", None) or []
            )
            memory_days = int(getattr(self.config.collection, "memory_days", 30) or 30)
            state = self.state_store
            summaries_dir = getattr(getattr(self, "storage", None), "summaries_dir", None)
            before_memory = len(all_items)
            split = split_by_memory(
                all_items,
                summaries_dir,
                days=memory_days,
                exclude_urls=getattr(self.config.collection, "exclude_urls", None) or [],
                exclude_fingerprints=exclude_fps,
                store=state,
            )
            all_items = split.fresh + split.updates
            if split.updates:
                self.console.print(
                    f"   {len(split.updates)} items are updates of published material\n"
                )
            if split.duplicates:
                self.console.print(
                    f"   Skipped {before_memory - len(all_items)} items already in digest history\n"
                )
            counters = getattr(self, "run_counters", None)
            if counters is None:
                counters = self.run_counters = {
                    "fetched": 0,
                    "memory_skipped": 0,
                    "memory_updates": 0,
                    "analyzed": 0,
                    "event_skipped": 0,
                }
            counters["memory_skipped"] = len(split.duplicates)
            counters["memory_updates"] = len(split.updates)

            if not all_items:
                # Nothing new, but the sources did answer: record that so a healthy
                # radar never looks like one that was never measured. This must not
                # count as a yield sample: every item here was already seen, so the
                # ratio would be 0% and three quiet days would mute a healthy feed.
                self._record_source_yields(
                    [],
                    self.last_fetch_report.all_items() if self.last_fetch_report else [],
                    count_yield=False,
                )
                self.console.print("[yellow]No new content found. Exiting.[/yellow]")
                return

            # 3. Merge cross-source duplicates (same URL from different sources)
            merged_items = self.merge_cross_source_duplicates(all_items)
            if len(merged_items) < len(all_items):
                self.console.print(
                    f"{self.icons['merge']} Merged "
                    f"{len(all_items) - len(merged_items)} cross-source duplicates "
                    f"→ {len(merged_items)} unique items\n"
                )

            # 3b. Free pre-filter: drop noise before spending a single token.
            from .processing.prefilter import apply_prefilter

            prefiltered = apply_prefilter(
                merged_items,
                must_match=getattr(self.config.collection, "must_match", None) or [],
                exclude_patterns=getattr(self.config.collection, "exclude_patterns", None) or [],
                min_content_chars=int(
                    getattr(self.config.collection, "min_content_chars", 0) or 0
                ),
                max_items_to_analyze=int(
                    getattr(self.config.collection, "max_items_to_analyze", 0) or 0
                ),
            )
            if prefiltered.dropped_total:
                self.console.print(
                    f"   Pre-filter kept {len(prefiltered.items)} of "
                    f"{len(prefiltered.items) + prefiltered.dropped_total} "
                    f"(off-topic {prefiltered.dropped_must_match}, "
                    f"noise {prefiltered.dropped_exclude}, "
                    f"short {prefiltered.dropped_min_content}, "
                    f"cap {prefiltered.dropped_cap})\n"
                )
            merged_items = prefiltered.items
            counters = getattr(self, "run_counters", None)
            if counters is not None:
                counters["prefiltered"] = prefiltered.dropped_total
            self._record_source_yields(prefiltered.items, all_items)

            if not merged_items:
                self.console.print("[yellow]Nothing passed the pre-filter. Exiting.[/yellow]")
                return

            # 4. Analyze with AI
            analyzed_items = await self.analyze_items(merged_items)
            self.console.print(
                f"{self.icons['ai']} Analyzed {len(analyzed_items)} items with AI\n"
            )
            self.run_counters["analyzed"] = len(analyzed_items)

            # 4b. Level 3 memory: event keys now exist, so reprints that share no
            # vocabulary with the original headline can still be dropped.
            if state is not None and analyzed_items:
                event_split = split_by_memory(
                    analyzed_items,
                    summaries_dir,
                    days=memory_days,
                    store=state,
                    check_event_keys=True,
                )
                if event_split.duplicates:
                    mode = getattr(
                        getattr(self.config, "collection", None),
                        "event_dedup_mode",
                        "drop",
                    ) or "drop"
                    if mode == "witness":
                        for item in event_split.duplicates:
                            metadata = getattr(item, "metadata", None)
                            if not isinstance(metadata, dict):
                                try:
                                    item.metadata = {}
                                    metadata = item.metadata
                                except Exception:
                                    metadata = None
                            if isinstance(metadata, dict):
                                metadata["is_witness"] = True
                        analyzed_items = (
                            event_split.fresh
                            + event_split.updates
                            + event_split.duplicates
                        )
                        self.run_counters["event_witnesses"] = len(
                            event_split.duplicates
                        )
                        self.console.print(
                            f"   Kept {len(event_split.duplicates)} event reprints "
                            "as witnesses\n"
                        )
                    else:
                        self.console.print(
                            f"   Skipped {len(event_split.duplicates)} items matching "
                            "already-published events\n"
                        )
                        analyzed_items = event_split.fresh + event_split.updates
                        self.run_counters["event_skipped"] = len(
                            event_split.duplicates
                        )

            if not analyzed_items:
                self.console.print("[yellow]No new content found. Exiting.[/yellow]")
                return

            # 5. Filter, deduplicate, and balance the digest
            filtering_result = await self.select_digest_items(
                analyzed_items,
            )
            important_items = filtering_result.items

            # Show per-sub-source selection breakdown
            selected_counts: Dict[str, int] = defaultdict(int)
            for item in important_items:
                key = f"{item.source_type.value}/{self._sub_source_label(item)}"
                selected_counts[key] += 1
            for source_key, count in sorted(selected_counts.items()):
                self.console.print(f"      {self.icons['detail']} {source_key}: {count}")
            self.console.print("")

            # 6. Search related stories + enrich with background knowledge (2nd AI pass)
            enrichment_result = await self.enrich_items(important_items)
            if enrichment_result.status == "failure":
                details = "; ".join(
                    f"{item_id}: {reason}"
                    for item_id, reason in enrichment_result.failures.items()
                )
                raise RuntimeError(
                    f"Enrichment failed for all {len(important_items)} selected items: "
                    + details
                )

            # 7. Generate and save daily summaries for each configured language
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            for lang in self.config.ai.languages:
                language_items = [
                    item for item in important_items
                    if item.processing and lang in item.processing.artifacts
                ]
                summarizer = DailySummarizer(
                    profile_names=self.profiles.names,
                    profile_order=self.config.digest.profile_order,
                )
                summary = await summarizer.generate_summary(language_items, today, len(all_items), language=lang)

                # Save to data/summaries/
                summary_path = self.storage.save_daily_summary(today, summary, language=lang)
                self.console.print(
                    f"{self.icons['save']} Saved {lang.upper()} summary to: {summary_path}\n"
                )

                # Machine-readable companion: publishers and dashboards read
                # this instead of re-parsing Markdown with regexes.
                if getattr(self.config.digest, "export_json", True):
                    json_path = self._write_digest_json(
                        today, lang, language_items, len(all_items)
                    )
                    if json_path is not None:
                        self.console.print(
                            f"{self.icons['document']} Saved {lang.upper()} JSON digest to: "
                            f"{json_path}\n"
                        )

                # Mark the delivered material as published so tomorrow's run
                # cannot ship it again under another outlet URL.
                if state is not None and lang == self.config.ai.languages[0]:
                    from .processing.published_memory import remember_items

                    remembered = remember_items(
                        state, language_items, digest_date=today
                    )
                    if remembered:
                        self.console.print(
                            f"   Remembered {remembered} published items in state.db\n"
                        )

                # Copy to docs/ for GitHub Pages
                try:
                    from pathlib import Path

                    post_filename = f"{today}-summary-{lang}.md"
                    posts_dir = Path("docs/_posts")
                    posts_dir.mkdir(parents=True, exist_ok=True)

                    dest_path = safe_output_path(posts_dir, post_filename)

                    # Add Jekyll front matter
                    front_matter = (
                        "---\n"
                        "layout: default\n"
                        f"title: \"Horizon Summary: {today} ({lang.upper()})\"\n"
                        f"date: {today}\n"
                        f"lang: {lang}\n"
                        "---\n\n"
                    )

                    # Strip leading H1 header to avoid duplication with Jekyll title
                    summary_content = summary
                    first_line = summary_content.strip().split("\n")[0]
                    if first_line.startswith("# "):
                        parts = summary_content.split("\n", 1)
                        if len(parts) > 1:
                            summary_content = parts[1].strip()

                    with open(dest_path, "w", encoding="utf-8") as f:
                        f.write(front_matter + summary_content)

                    self.console.print(
                        f"{self.icons['document']} Copied {lang.upper()} summary "
                        f"to GitHub Pages: {dest_path}\n"
                    )
                except Exception as e:
                    self.console.print(
                        f"[yellow]{self.icons['warning']} Failed to copy "
                        f"{lang.upper()} summary to docs/: {e}[/yellow]\n"
                    )

                # Send email if configured
                if self.email_manager and self.config.email and self.config.email.enabled:
                    self.console.print(
                        f"{self.icons['email']} Sending {lang.upper()} email summary..."
                    )
                    subscribers = self.storage.load_subscribers()
                    subject = f"Horizon Summary ({lang.upper()}) - {today}"
                    self.email_manager.send_daily_summary(summary, subject, subscribers)

                # Send webhook notification if configured
                if self.webhook_notifier:
                    await self.webhook_notifier.send_daily_summary(
                        summary=summary,
                        important_items=language_items,
                        all_items_count=len(all_items),
                        date=today,
                        lang=lang,
                        summarizer=summarizer,
                    )
                if self.wechat_notifier:
                    await self.wechat_notifier.send_daily_summary(summary, lang)

            self.console.print(
                f"[bold green]{self.icons['success']} "
                "Horizon completed successfully![/bold green]"
            )
            usage = get_usage_snapshot()
            if usage.total_tokens > 0:
                self.console.print(
                    f"\n{self.icons['tokens']} Token usage this run: "
                    f"{usage.total_tokens} tokens "
                    f"(input: {usage.total_input_tokens}, output: {usage.total_output_tokens})"
                )
                for provider, u in sorted(usage.per_provider.items()):
                    if u.total <= 0:
                        continue
                    self.console.print(
                        f"   {self.icons['detail']} {provider}: {u.total} tokens "
                        f"(in: {u.input_tokens}, out: {u.output_tokens})"
                    )

            usage_path = os.environ.get("HORIZON_USAGE_JSON")
            if usage_path:
                try:
                    write_usage_snapshot(usage_path, usage)
                except OSError as exc:
                    self.console.print(
                        f"[yellow]Could not write usage JSON: {exc}[/yellow]"
                    )

            self._write_run_report(
                date=datetime.now(timezone.utc).strftime("%Y-%m-%d"), status="success"
            )
            if budget is not None:
                self._budget_report = budget.report()
                if budget.truncated:
                    self.console.print(
                        f"[yellow]{self.icons['warning']} Budget truncated this run: "
                        f"{budget.reason}[/yellow]"
                    )

        except Exception as e:
            self.console.print(
                f"[bold red]{self.icons['error']} Error: {e}[/bold red]"
            )

            # Send webhook failure notification if configured
            if self.webhook_notifier:
                await self.webhook_notifier.send_failure(
                    date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    error_message=str(e),
                )
            if self.wechat_notifier:
                await self.wechat_notifier.send_failure(
                    date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                    error_message=str(e),
                )

            self._write_run_report(
                date=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                status="failure",
                error=str(e),
            )
            raise

    def _determine_time_window(self, force_hours: Optional[int] = None) -> datetime:
        hours = force_hours if force_hours is not None else self.config.collection.time_window_hours
        return datetime.now(timezone.utc) - timedelta(hours=hours)

    def _write_digest_json(
        self,
        date: str,
        language: str,
        items: List[ContentItem],
        total_fetched: int,
    ) -> Optional[Path]:
        """Write the versioned JSON digest next to the Markdown one."""
        from .export.json_digest import build_digest_payload, write_digest_json

        try:
            payload = build_digest_payload(
                date=date,
                language=language,
                items=items,
                total_fetched=total_fetched,
                window_hours=int(self.config.collection.time_window_hours),
                counters=self.run_counters,
                sources_health=self._sources_health_rows(),
                usage=self._usage_payload(),
                budget=getattr(self, "_budget_report", {}),
            )
            return write_digest_json(
                Path(self.storage.summaries_dir), date, language, payload
            )
        except Exception as exc:  # never fail a run over the optional export
            self.console.print(
                f"[yellow]{self.icons['warning']} JSON digest export failed: {exc}[/yellow]"
            )
            return None

    def _usage_payload(self) -> Dict[str, object]:
        try:
            usage = get_usage_snapshot()
        except Exception:
            return {}
        return {
            "total_tokens": usage.total_tokens,
            "input_tokens": usage.total_input_tokens,
            "output_tokens": usage.total_output_tokens,
            "per_provider": {
                provider: {
                    "total": record.total,
                    "input_tokens": record.input_tokens,
                    "output_tokens": record.output_tokens,
                }
                for provider, record in usage.per_provider.items()
            },
        }

    def _sources_health_rows(self) -> List[Dict[str, object]]:
        state = self.state_store
        if state is None:
            return [
                outcome.to_dict()
                for outcome in (self.last_fetch_report.outcomes if self.last_fetch_report else [])
            ]
        try:
            return [health.to_dict() for health in state.health_all()]
        except Exception:
            return []

    def _write_run_report(self, *, date: str, status: str, error: str | None = None) -> None:
        """Persist per-run diagnostics for operators and downstream schedulers."""
        digest_cfg = getattr(self.config, "digest", None)
        if digest_cfg is not None and not getattr(digest_cfg, "write_run_report", True):
            return
        storage = getattr(self, "storage", None)
        summaries_dir = getattr(storage, "summaries_dir", None) if storage else None
        if summaries_dir is None:
            return
        from .export.json_digest import write_run_report

        window_hours = int(
            getattr(getattr(self.config, "collection", None), "time_window_hours", 24) or 24
        )
        payload: Dict[str, object] = {
            "schema_version": 1,
            "date": date,
            "status": status,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window_hours": window_hours,
            "counters": dict(self.run_counters),
            "fetch": self.last_fetch_report.to_dict() if self.last_fetch_report else {},
            "sources_health": self._sources_health_rows(),
            "usage": self._usage_payload(),
            "budget": getattr(self, "_budget_report", {}),
        }
        if error:
            payload["error"] = error
        try:
            write_run_report(Path(summaries_dir), payload)
        except Exception as exc:
            self.console.print(
                f"[yellow]{self.icons['warning']} run_report.json write failed: {exc}[/yellow]"
            )

    async def fetch_all_sources(self, since: datetime) -> List[ContentItem]:
        """Fetch content from all configured sources.

        This is a stable stage entry point for integrations such as MCP.

        Args:
            since: Fetch items published after this time

        Returns:
            List[ContentItem]: All fetched items
        """
        self.last_fetch_report = None
        parked: List[SourceFetchOutcome] = []
        # A blocked or unroutable host must fail fast: a 30s connect timeout
        # multiplied by retries stalled a whole run when one feed was unreachable.
        http_timeout = self._http_timeout()
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(http_timeout, connect=min(10.0, http_timeout))
        ) as client:
            tasks = []

            def _skip_if_parked(name: str) -> bool:
                if not self._source_in_cooldown(name):
                    return False
                self.console.print(
                    f"[yellow]{self.icons['warning']} Skipping {name}: in cooldown[/yellow]"
                )
                parked.append(
                    SourceFetchOutcome(
                        source_name=name,
                        status="skipped_cooldown",
                        error="circuit breaker cooldown",
                    )
                )
                return True

            # GitHub sources
            if self.config.sources.github and not _skip_if_parked("GitHub"):
                github_scraper = GitHubScraper(self.config.sources.github, client)
                tasks.append(self._fetch_with_progress("GitHub", github_scraper, since))

            # Hacker News
            if self.config.sources.hackernews.enabled and not _skip_if_parked("Hacker News"):
                hn_scraper = HackerNewsScraper(self.config.sources.hackernews, client)
                tasks.append(self._fetch_with_progress("Hacker News", hn_scraper, since))

            # RSS feeds
            if self.config.sources.rss and not _skip_if_parked("RSS Feeds"):
                from .extractors import ExtractorRegistry
                rss_scraper = RSSScraper(
                    self.config.sources.rss,
                    client,
                    ExtractorRegistry(self.config.extractors),
                )
                tasks.append(self._fetch_with_progress("RSS Feeds", rss_scraper, since))

            # Reddit
            if self.config.sources.reddit.enabled and not _skip_if_parked("Reddit"):
                reddit_scraper = RedditScraper(self.config.sources.reddit, client)
                tasks.append(self._fetch_with_progress("Reddit", reddit_scraper, since))

            # Telegram
            if self.config.sources.telegram.enabled and not _skip_if_parked("Telegram"):
                telegram_scraper = TelegramScraper(self.config.sources.telegram, client)
                tasks.append(self._fetch_with_progress("Telegram", telegram_scraper, since))

            # Twitter (Apify or Playwright mode)
            if (
                self.config.sources.twitter
                and self.config.sources.twitter.enabled
                and not _skip_if_parked("Twitter")
            ):
                tw_cfg = self.config.sources.twitter
                if tw_cfg.mode == "playwright":
                    twitter_scraper = TwitterPlaywrightScraper(tw_cfg)
                else:
                    twitter_scraper = TwitterScraper(tw_cfg, client)
                tasks.append(self._fetch_with_progress("Twitter", twitter_scraper, since))

            # OpenBB (financial news / filings via the OpenBB Platform SDK)
            if (
                self.config.sources.openbb
                and self.config.sources.openbb.enabled
                and not _skip_if_parked("OpenBB")
            ):
                openbb_scraper = OpenBBScraper(self.config.sources.openbb, client)
                tasks.append(self._fetch_with_progress("OpenBB", openbb_scraper, since))

            # OSS Insight trending repos
            if (
                self.config.sources.ossinsight
                and self.config.sources.ossinsight.enabled
                and not _skip_if_parked("OSS Insight")
            ):
                oss_scraper = OSSInsightScraper(self.config.sources.ossinsight, client)
                tasks.append(self._fetch_with_progress("OSS Insight", oss_scraper, since))

            # GDELT 2.0 DOC API (key-less global news)
            if (
                self.config.sources.gdelt
                and self.config.sources.gdelt.enabled
                and not _skip_if_parked("GDELT")
            ):
                gdelt_scraper = GDELTScraper(self.config.sources.gdelt, client)
                tasks.append(self._fetch_with_progress("GDELT", gdelt_scraper, since))

            # Google News RSS (key-less news search)
            if (
                self.config.sources.google_news
                and self.config.sources.google_news.enabled
                and not _skip_if_parked("Google News")
            ):
                gn_scraper = GoogleNewsScraper(self.config.sources.google_news, client)
                tasks.append(self._fetch_with_progress("Google News", gn_scraper, since))

            # Custom sources: declarative json_feed and installed plugins.
            for custom in getattr(self.config.sources, "custom", None) or []:
                if not getattr(custom, "enabled", True):
                    continue
                label = custom.name or custom.type
                if _skip_if_parked(label):
                    continue
                plugin = get_scraper_plugin(custom.type)
                payload = dict(custom.config or {})
                payload.setdefault("type", custom.type)
                payload.setdefault("name", label)
                payload.setdefault("profile", custom.profile)
                payload.setdefault("custom_type", custom.type)
                if custom.category and "category" not in payload:
                    payload["category"] = custom.category
                scraper = plugin.build(payload, client)
                tasks.append(self._fetch_with_progress(label, scraper, since))

            # Fetch all concurrently
            outcomes = await asyncio.gather(*tasks)
            self.last_fetch_report = FetchReport(outcomes=[*outcomes, *parked])

            # Flatten successful and empty outcomes; failures remain in the report.
            all_items: List[ContentItem] = []
            for outcome in outcomes:
                all_items.extend(outcome.items)

            return all_items

    async def _fetch_with_progress(
        self, name: str, scraper: BaseScraper, since: datetime
    ) -> SourceFetchOutcome:
        """Fetch from a scraper with progress indication.

        Args:
            name: Source name for display
            scraper: Scraper instance
            since: Fetch items after this time

        Returns:
            SourceFetchOutcome: Named fetch result and diagnostics
        """
        self.console.print(f"{self.icons['fetch']} Fetching from {name}...")
        started = time.monotonic()
        try:
            result = await scraper.fetch(since)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            self.console.print(f"[red]   Failed to fetch {name}: {error}[/red]")
            self._record_source_health(
                name, ok=False, error=error, latency_ms=(time.monotonic() - started) * 1000
            )
            return SourceFetchOutcome(
                source_name=name,
                status="failure",
                error=error,
            )

        latency_ms = (time.monotonic() - started) * 1000
        items = result.items
        self.console.print(f"   Found {len(items)} items from {name}")
        ok = result.status != "failure"
        # Tag each item with the task that produced it. The pre-filter runs on the
        # merged pool, so without this marker a surviving item cannot be credited
        # back to its source and yield is measured against the wrong row.
        for item in items:
            try:
                item.metadata.setdefault("_source_task", name)
            except AttributeError:
                pass
        self._record_source_health(
            name,
            ok=ok,
            latency_ms=latency_ms,
            fetched=len(items),
            error=(
                "; ".join(f"{source}: {error}" for source, error in result.failures.items())
                if not ok and result.failures
                else None
            ),
        )
        for source, error in result.failures.items():
            self.console.print(f"[yellow]   {name}/{source}: {error}[/yellow]")

        # Show per-sub-source breakdown when there are multiple sub-sources
        sub_counts: Dict[str, int] = defaultdict(int)
        for item in items:
            sub_counts[self._sub_source_label(item)] += 1
        if len(sub_counts) > 1:
            for sub, count in sorted(sub_counts.items()):
                self.console.print(f"      {self.icons['detail']} {sub}: {count}")

        return SourceFetchOutcome(
            source_name=name,
            status=result.status,
            items=items,
            error="; ".join(f"{source}: {error}" for source, error in result.failures.items()) or None,
            failures=result.failures,
        )

    def _http_timeout(self) -> float:
        """Per-request timeout for source fetching."""
        collection = getattr(getattr(self, "config", None), "collection", None)
        raw = getattr(collection, "http_timeout_sec", None)
        try:
            value = float(raw) if raw is not None else 30.0
        except (TypeError, ValueError):
            value = 30.0
        return max(5.0, min(value, 120.0))

    @staticmethod
    def _task_of(item) -> str:
        """Which fetch task produced this item.

        Yield is recorded per task row (``RSS Feeds``, ``Telegram``), because that
        is the row the breaker and the cabinet already read.
        """
        meta = getattr(item, "metadata", None) or {}
        task = str(meta.get("_source_task") or "").strip()
        if task:
            return task
        return "unknown"

    def _record_source_yields(self, kept_items, raw_items, *, count_yield: bool = True) -> None:
        """Attribute survivors back to their sources.

        Called twice in a run: right after fetching (so a run that exits early on
        "nothing new" still records that every source answered) and again after
        the pre-filter (which is what actually measures usefulness). Counting only
        the second call made a healthy radar look like it never measured anything.

        ``count_yield=False`` records only liveness. The first call must not count
        a sample: on the "nothing new" path every item was already seen, so the
        ratio is 0% by construction and three quiet runs would mute a good feed.
        """
        state = self.state_store
        if state is None:
            return
        collection = getattr(getattr(self, "config", None), "collection", None)
        min_yield = getattr(collection, "min_source_yield", None)
        mute_after = int(getattr(collection, "mute_after_low_yield_runs", 0) or 0)
        effective_min = float(min_yield) if (min_yield is not None and count_yield) else None

        fetched_by: dict[str, int] = defaultdict(int)
        for item in raw_items or []:
            fetched_by[self._task_of(item)] += 1
        kept_by: dict[str, int] = defaultdict(int)
        for item in kept_items or []:
            kept_by[self._task_of(item)] += 1

        for label, fetched in fetched_by.items():
            try:
                state.health_record_yield(
                    label,
                    fetched=fetched,
                    kept=kept_by.get(label, 0),
                    min_yield=effective_min,
                    mute_after=mute_after,
                    record_sample=count_yield,
                )
            except Exception:
                # Yield bookkeeping must never break a run.
                pass
        # Survivors whose task never reported a fetch are items already in memory
        # from an earlier run, not a measurement. Writing them would create rows
        # with kept > 0 and fetched == 0, which is not a ratio at all.

    def _record_source_health(
        self,
        name: str,
        *,
        ok: bool,
        latency_ms: float | None = None,
        fetched: int = 0,
        error: str | None = None,
    ) -> None:
        """Persist the fetch outcome so the breaker and run report see it."""
        state = self.state_store
        if state is None:
            return
        collection = getattr(getattr(self, "config", None), "collection", None)
        try:
            failures_before = state.health_get(name).consecutive_failures
            cooldown = 0.0
            max_failures = int(getattr(collection, "max_consecutive_failures", 0) or 0)
            if not ok and max_failures > 0 and failures_before + 1 >= max_failures:
                from .storage.state import cooldown_seconds_for

                cooldown = cooldown_seconds_for(
                    failures_before + 1,
                    float(getattr(collection, "cooldown_base_hours", 6.0) or 6.0),
                )
            state.health_record(
                name,
                ok=ok,
                latency_ms=latency_ms,
                fetched=fetched,
                error=error,
                cooldown_seconds=cooldown,
            )
            if cooldown:
                self.console.print(
                    f"[yellow]   {name} parked for {cooldown / 3600:.0f}h after "
                    f"{failures_before + 1} consecutive failures[/yellow]"
                )
        except Exception:
            # Health bookkeeping must never break a run.
            pass

    def _source_in_cooldown(self, name: str) -> bool:
        state = self.state_store
        if state is None:
            return False
        try:
            return state.health_get(name).in_cooldown
        except Exception:
            return False

    @staticmethod
    def _sub_source_label(item: ContentItem) -> str:
        """Return a human-readable sub-source label for an item."""
        meta = item.metadata
        if meta.get("subreddit"):
            return f"r/{meta['subreddit']}"
        if meta.get("feed_name"):
            return meta["feed_name"]
        if meta.get("channel"):
            return f"@{meta['channel']}"
        if meta.get("period") and meta.get("repo"):
            return f"ossinsight:{meta.get('primary_language', 'all')}"
        if meta.get("repo"):
            return meta["repo"]
        if meta.get("watchlist"):
            return meta["watchlist"]
        if meta.get("source_name"):
            return meta["source_name"]
        if meta.get("gn_query"):
            return f"google_news:{meta['gn_query']}"
        if meta.get("domain"):
            return meta["domain"]
        return item.author or "unknown"

    def merge_cross_source_duplicates(self, items: List[ContentItem]) -> List[ContentItem]:
        """Merge items that point to the same URL from different sources.

        This is a stable stage helper for integrations such as MCP.

        Keeps the item with the richest content and combines metadata.

        Args:
            items: Items to deduplicate

        Returns:
            List[ContentItem]: Deduplicated items
        """
        # Group by normalized URL
        url_groups: Dict[tuple[object, ...], List[ContentItem]] = {}
        for item in items:
            if isinstance(item.profile, list):
                requested_profile: object = tuple(
                    profile_id.strip() for profile_id in item.profile
                )
            else:
                requested_profile = (item.profile or "auto").strip() or "auto"
            key = (*_deduplication_url_key(str(item.url)), requested_profile)
            url_groups.setdefault(key, []).append(item)

        merged = []
        for group in url_groups.values():
            group_copies = [item.model_copy(deep=True) for item in group]
            if len(group) == 1:
                merged.append(group_copies[0])
                continue

            # Pick the item with the richest content as primary
            primary = max(group_copies, key=lambda x: len(x.content or ""))

            # Merge metadata and source info from other items
            all_sources = []
            for item in group_copies:
                if item.source_type.value not in all_sources:
                    all_sources.append(item.source_type.value)
                # Merge metadata (engagement, discussion, etc.)
                for mk, mv in item.metadata.items():
                    if mk not in primary.metadata or not primary.metadata[mk]:
                        primary.metadata[mk] = mv

                # Append content (e.g., comments from another source)
                if item is not primary and item.content:
                    if primary.content and item.content not in primary.content:
                        primary.content = (primary.content or "") + f"\n\n--- From {item.source_type.value} ---\n" + item.content

            primary.metadata["merged_sources"] = all_sources
            merged.append(primary)

        return merged

    async def merge_topic_duplicates(
        self,
        items: List[ContentItem],
        *,
        log: bool = True,
    ) -> List[ContentItem]:
        """Merge items covering the same topic using AI semantic deduplication.

        This is a stable stage helper for integrations such as MCP.

        Sends all item titles, tags, and summaries to AI in a single call.
        The highest-scored item of each duplicate group is kept as the primary.
        Source text and community comments from duplicate items are merged into
        the primary's source text and comments respectively.

        Falls back to returning items unchanged if the AI call fails.
        """
        if len(items) <= 1:
            return items

        from .ai.prompting.deduplication import TOPIC_DEDUP_SYSTEM, TOPIC_DEDUP_USER
        from .ai.utils import parse_json_response

        # Build the item list for the prompt
        lines = []
        for i, item in enumerate(items):
            analysis = item.processing.analysis if item.processing else None
            tags = ", ".join(analysis.tags) if analysis and analysis.tags else "—"
            summary = analysis.summary if analysis else "—"
            lines.append(f"[{i}] {item.title}\n    Tags: {tags}\n    Summary: {summary}")
        items_text = "\n\n".join(lines)

        try:
            ai_client = self.ai_client
            response = await ai_client.complete(
                system=TOPIC_DEDUP_SYSTEM,
                user=TOPIC_DEDUP_USER.format(items=items_text),
            )
            result = parse_json_response(response)
            if result is None:
                if log:
                    self.console.print("[yellow]  dedup: could not parse AI response, skipping[/yellow]")
                return items

            duplicate_groups = result.get("duplicates", [])
        except AIProviderUnavailableError:
            raise
        except Exception as e:
            if log:
                self.console.print(f"[yellow]  dedup: AI call failed ({e}), skipping[/yellow]")
            return items

        if not duplicate_groups:
            return items

        def score_of(idx: int) -> float:
            analysis = items[idx].processing.analysis if items[idx].processing else None
            return analysis.score if analysis and analysis.score is not None else float("-inf")

        # Keep valid indices only; groups that share an item describe one topic
        if not isinstance(duplicate_groups, list):
            duplicate_groups = []
        parent = list(range(len(items)))

        def find(idx: int) -> int:
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        grouped: set[int] = set()
        for group in duplicate_groups:
            if not isinstance(group, list):
                continue
            members = sorted({
                idx
                for idx in group
                if isinstance(idx, int) and not isinstance(idx, bool) and 0 <= idx < len(items)
            })
            if len(members) < 2:
                continue
            grouped.update(members)
            for idx in members[1:]:
                parent[find(idx)] = find(members[0])
        clusters: Dict[int, List[int]] = defaultdict(list)
        for idx in sorted(grouped):
            clusters[find(idx)].append(idx)

        # Build a set of indices to drop (all non-primary duplicates)
        drop_indices: set[int] = set()
        for members in clusters.values():
            # Highest analysis score wins; ties keep the earlier (higher-ranked) item
            primary_idx = min(members, key=lambda idx: (-score_of(idx), idx))
            primary = items[primary_idx]
            for dup_idx in members:
                if dup_idx == primary_idx:
                    continue
                dup = items[dup_idx]
                # Merge the duplicate's source text before the primary's community
                # comments so later stages still read it as source content
                primary_parts = split_content(primary.content)
                dup_parts = split_content(dup.content)
                header = f"--- From {dup.source_type.value} ---"
                main = primary_parts.main
                if dup_parts.main and dup_parts.main not in main:
                    main = f"{main}\n\n{header}\n{dup_parts.main}" if main else f"{header}\n{dup_parts.main}"
                comments = primary_parts.comments
                if dup_parts.comments and dup_parts.comments not in comments:
                    labeled = f"{header}\n{dup_parts.comments}"
                    comments = f"{comments}\n\n{labeled}" if comments else labeled
                merged = main
                if comments:
                    merged = f"{main}\n\n{COMMENTS_MARKER}\n{comments}"
                primary.content = merged or primary.content
                if log:
                    self.console.print(
                        f"   [dim]dedup: keep [{primary_idx}] {primary.title}[/dim]\n"
                        f"   [dim]       drop [{dup_idx}] {dup.title}[/dim]"
                    )
                drop_indices.add(dup_idx)

        return [item for i, item in enumerate(items) if i not in drop_indices]

    async def drop_already_published(
        self,
        items: List[ContentItem],
        *,
        log: bool = True,
        days: int = 30,
    ) -> List[ContentItem]:
        """Drop new items that AI judges to be reprints of prior digest stories."""
        if not items:
            return items
        from .models import (
            ClassificationResult,
            ContentAnalysis,
            ProcessingResult,
            SourceType,
        )
        from .processing.published_memory import history_titles

        storage = getattr(self, "storage", None)
        summaries_dir = getattr(storage, "summaries_dir", None) if storage else None
        state = None
        try:
            state = self.state_store
        except Exception:
            state = None
        prior_titles = history_titles(summaries_dir, days=days, store=state)
        extra_fps = list(
            (getattr(getattr(self.config, "collection", None), "exclude_title_fingerprints", None) or [])
        )
        # Fingerprints may be raw titles or already-normalized; keep as phantom titles.
        prior_titles = list(dict.fromkeys([*prior_titles, *[fp for fp in extra_fps if fp]]))
        if not prior_titles:
            return items

        phantoms: List[ContentItem] = []
        for index, title in enumerate(prior_titles[:80]):
            phantoms.append(
                ContentItem(
                    id=f"history:published:{index}",
                    source_type=SourceType.RSS,
                    title=title,
                    url=f"https://dawnwire.local/already-published/{index}",
                    published_at=datetime.now(timezone.utc),
                    metadata={"already_published": True},
                    processing=ProcessingResult(
                        classification=ClassificationResult(
                            profile=self.profiles.default_profile,
                            method="source_override",
                        ),
                        analysis=ContentAnalysis(
                            score=10.0,
                            reason="already published",
                            summary=title,
                        ),
                    ),
                )
            )

        combined = phantoms + items
        merged = await self.merge_topic_duplicates(combined, log=log)
        kept = [
            item
            for item in merged
            if not (item.metadata or {}).get("already_published")
            and not str(item.url).startswith("https://dawnwire.local/already-published/")
        ]
        return kept

    async def filter_items(
        self,
        items: List[ContentItem],
        *,
        threshold: Optional[float] = None,
        topic_dedup: bool = True,
        apply_balance: bool = True,
        log: bool = True,
    ) -> FilteringPipelineResult:
        """Apply score thresholding, optional topic dedup, and digest balancing."""
        threshold_items = []
        for item in items:
            if self.passes_profile_filter(item, threshold):
                threshold_items.append(item)
        threshold_items.sort(
            key=lambda item: (
                item.processing.analysis.score
                if item.processing and item.processing.analysis and item.processing.analysis.score is not None
                else -1
            ),
            reverse=True,
        )

        if log:
            self.console.print(
                f"{self.icons['filter']} Selected {len(threshold_items)} items "
                "with profile filters\n"
            )

        deduped_items = threshold_items
        if topic_dedup and deduped_items:
            profile_groups: Dict[str, List[ContentItem]] = defaultdict(list)
            for item in deduped_items:
                profile_id = (
                    item.processing.classification.profile
                    if item.processing
                    else self.profiles.default_profile
                )
                profile_groups[profile_id].append(item)
            deduped_items = []
            for profile_id, profile_items in profile_groups.items():
                settings = self.config.processing.profile_settings.get(profile_id)
                if settings is None or settings.topic_dedup:
                    deduped_items.extend(
                        await self.merge_topic_duplicates(profile_items, log=log)
                    )
                else:
                    deduped_items.extend(profile_items)
            deduped_items.sort(
                key=lambda item: (
                    item.processing.analysis.score
                    if item.processing
                    and item.processing.analysis
                    and item.processing.analysis.score is not None
                    else -1
                ),
                reverse=True,
            )
        topic_dedup_removed = len(threshold_items) - len(deduped_items)

        if log and topic_dedup_removed:
            self.console.print(
                f"{self.icons['cleanup']} Removed {topic_dedup_removed} topic duplicates "
                f"→ {len(deduped_items)} unique items\n"
            )

        if topic_dedup and deduped_items:
            before_history = len(deduped_items)
            deduped_items = await self.drop_already_published(deduped_items, log=log)
            history_removed = before_history - len(deduped_items)
            if log and history_removed:
                self.console.print(
                    f"{self.icons['cleanup']} Dropped {history_removed} reprints of "
                    f"already-published stories → {len(deduped_items)} unique items\n"
                )

        balanced_digest = (
            self.apply_balanced_digest(deduped_items, log=log)
            if apply_balance
            else BalancedDigestResult(items=deduped_items)
        )
        return FilteringPipelineResult(
            items=balanced_digest.items,
            threshold_count=len(threshold_items),
            topic_dedup_count=len(deduped_items),
            topic_dedup_removed=topic_dedup_removed,
            balanced_digest=balanced_digest,
        )

    async def select_digest_items(
        self,
        items: List[ContentItem],
        *,
        threshold: Optional[float] = None,
        topic_dedup: bool = True,
        log: bool = True,
    ) -> FilteringPipelineResult:
        """Select final digest items using the same stages for every entry point."""
        initial = await self.filter_items(
            items,
            threshold=threshold,
            topic_dedup=topic_dedup,
            apply_balance=False,
            log=log,
        )
        candidates = initial.items
        await self._expand_twitter_discussion(candidates)

        # Targeted re-analysis can lower a score, so reapply profile filters.
        eligible = [
            item
            for item in candidates
            if self.passes_profile_filter(item, threshold)
        ]
        eligible.sort(
            key=lambda item: (
                item.processing.analysis.score
                if item.processing
                and item.processing.analysis
                and item.processing.analysis.score is not None
                else -1
            ),
            reverse=True,
        )
        balanced = self.apply_balanced_digest(eligible, log=log)
        return FilteringPipelineResult(
            items=balanced.items,
            threshold_count=initial.threshold_count,
            topic_dedup_count=initial.topic_dedup_count,
            topic_dedup_removed=initial.topic_dedup_removed,
            balanced_digest=balanced,
            eligible_count=len(eligible),
        )

    def passes_profile_filter(
        self,
        item: ContentItem,
        threshold: Optional[float] = None,
    ) -> bool:
        if not item.processing or not item.processing.analysis:
            return False
        profile_id = item.processing.classification.profile
        settings = self.config.processing.profile_settings.get(profile_id)
        effective_threshold = threshold
        if effective_threshold is None and settings is not None:
            effective_threshold = settings.threshold
        if effective_threshold is None:
            return True
        score = item.processing.analysis.score
        return score is not None and score >= effective_threshold

    def apply_balanced_digest(
        self,
        items: List[ContentItem],
        *,
        log: bool = True,
    ) -> BalancedDigestResult:
        """Apply configured category quotas and the final item cap.

        Categories are read from ``item.metadata["category"]``, or from the
        resolved ``item.processing.classification.profile`` when
        ``digest.group_by`` is ``"profile"``. If a category appears in more than
        one configured group, the first group in config order wins.
        """
        digest = self.config.digest
        groups = digest.category_groups
        max_items = digest.max_items

        if not groups and max_items is None and digest.default_group_limit is None:
            return BalancedDigestResult(items=items)

        sorted_items = sorted(
            items,
            key=lambda item: (
                item.processing.analysis.score
                if item.processing and item.processing.analysis and item.processing.analysis.score is not None
                else -1
            ),
            reverse=True,
        )

        category_to_group: Dict[str, str] = {}
        duplicate_categories: List[str] = []
        for group_key, group in groups.items():
            for category in group.categories:
                if category in category_to_group:
                    if category_to_group[category] != group_key:
                        duplicate_categories.append(category)
                    continue
                category_to_group[category] = group_key

        if log:
            for category in sorted(set(duplicate_categories)):
                first_group = category_to_group[category]
                self.console.print(
                    f"[yellow]Warning: category '{category}' is configured in multiple "
                    f"groups; using '{first_group}'.[/yellow]"
                )

        selected: List[tuple[ContentItem, str]] = []
        group_counts: Dict[str, int] = defaultdict(int)
        default_group = digest.default_group

        for item in sorted_items:
            if digest.group_by == "profile":
                category = (
                    item.processing.classification.profile if item.processing else None
                )
            else:
                category = item.metadata.get("category")
            group_key = (
                category_to_group.get(category, default_group)
                if isinstance(category, str)
                else default_group
            )

            if group_key in groups:
                limit = groups[group_key].limit
            else:
                limit = digest.default_group_limit

            if limit is not None and group_counts[group_key] >= limit:
                continue

            selected.append((item, group_key))
            group_counts[group_key] += 1

        if max_items is not None:
            selected = selected[:max_items]

        final_counts: Dict[str, int] = defaultdict(int)
        for _, group_key in selected:
            final_counts[group_key] += 1

        group_limits: Dict[str, Optional[int]] = {
            group_key: group.limit for group_key, group in groups.items()
        }
        group_limits.setdefault(default_group, digest.default_group_limit)

        if log:
            self.console.print(
                f"{self.icons['balance']} Balanced digest selected "
                f"{len(selected)}/{len(items)} items"
            )
            for group_key, group in groups.items():
                label = group.name or group_key
                self.console.print(
                    f"      {self.icons['detail']} {label}: "
                    f"{final_counts.get(group_key, 0)}/{group.limit}"
                )
            if (
                final_counts.get(default_group, 0)
                or digest.default_group_limit is not None
            ):
                limit_label = (
                    str(digest.default_group_limit)
                    if digest.default_group_limit is not None
                    else "unlimited"
                )
                self.console.print(
                    f"      {self.icons['detail']} {default_group}: "
                    f"{final_counts.get(default_group, 0)}/{limit_label}"
                )
            self.console.print("")

        return BalancedDigestResult(
            items=[item for item, _ in selected],
            enabled=True,
            group_counts=dict(final_counts),
            group_limits=group_limits,
            duplicate_categories=sorted(set(duplicate_categories)),
        )

    async def _expand_twitter_discussion(self, items: List[ContentItem]) -> None:
        """Second-stage: fetch reply text for important Twitter items and re-analyze.

        Only runs when sources.twitter.fetch_reply_text is True.
        Bounded by max_tweets_to_expand to control cost.
        """
        tw_cfg = self.config.sources.twitter
        if not tw_cfg or not tw_cfg.enabled or not tw_cfg.fetch_reply_text:
            return

        from .models import SourceType

        twitter_items = [
            item for item in items
            if item.source_type == SourceType.TWITTER
        ][:tw_cfg.max_tweets_to_expand]

        if not twitter_items:
            return

        self.console.print(
            f"{self.icons['discussion']} Fetching reply text for "
            f"{len(twitter_items)} Twitter items..."
        )

        async with httpx.AsyncClient(timeout=30.0) as client:
            if tw_cfg.mode == "playwright":
                self.console.print(
                    "   [yellow]Reply expansion not yet supported in Playwright mode.[/yellow]"
                )
                return
            scraper = TwitterScraper(tw_cfg, client)
            expanded = []
            for item in twitter_items:
                try:
                    reply_lines = await scraper.fetch_replies_for_item(item)
                    if TwitterScraper.append_discussion_content(item, reply_lines):
                        expanded.append(item)
                        self.console.print(
                            f"   {self.icons['discussion']} {len(reply_lines)} replies "
                            f"added to: {item.title[:60]}"
                        )
                except Exception as exc:
                    self.console.print(
                        f"   [yellow]{self.icons['warning']} Reply fetch failed for "
                        f"{item.id}: {exc}[/yellow]"
                    )

        if not expanded:
            return

        self.console.print(
            f"   Re-analyzing {len(expanded)} Twitter items with reply context...\n"
        )
        ai_client = self.ai_client
        analyzer = ContentAnalyzer(
            ai_client, self.profiles, console=self.console,
            concurrency=self.config.ai.analysis_concurrency,
            throttle_sec=self.config.ai.throttle_sec,
        )
        await analyzer.analyze_batch(expanded)

    async def enrich_items(self, items: List[ContentItem]) -> EnrichmentBatchResult:
        """Enrich items with background knowledge (2nd AI pass).

        For each item that passed the score threshold, call AI to generate
        background knowledge based on the item's actual content.

        Args:
            items: Important items to enrich (modified in-place)
        """
        if not items:
            return EnrichmentBatchResult()

        budget = getattr(self, "_budget", None)
        if budget is not None and budget.exhausted():
            self.console.print(
                f"[yellow]{self.icons['warning']} Skipping enrichment: "
                f"{budget.reason}[/yellow]"
            )
            self._budget_report = budget.report()
            return EnrichmentBatchResult()

        self.console.print(
            f"{self.icons['enrich']} Enriching with background knowledge..."
        )
        ai_client = self.ai_client
        enricher = ContentEnricher(
            ai_client,
            self.profiles,
            self.config.ai.languages,
            console=self.console,
            tools=ToolRegistry(self.storage.summaries_dir),
            concurrency=self.config.ai.enrichment_concurrency,
        )
        result = await enricher.enrich_batch(items)
        self.console.print(
            f"   Enriched {result.succeeded_count}/{len(items)} items"
        )
        if result.failed_count:
            self.console.print(
                f"   [yellow]Skipped {result.failed_count} items after enrichment "
                f"failed: {', '.join(result.failed_ids)}[/yellow]"
            )
        self.console.print("")
        return result

    async def analyze_items(self, items: List[ContentItem]) -> List[ContentItem]:
        """Analyze content items with AI.

        Args:
            items: Items to analyze

        Returns:
            List[ContentItem]: Analyzed items
        """
        budget = getattr(self, "_budget", None)
        if budget is not None and not budget.allow(stage="analyze", estimated_tokens=0):
            if budget.exhausted():
                self.console.print(
                    f"[yellow]{self.icons['warning']} Skipping AI analysis: "
                    f"{budget.reason}[/yellow]"
                )
                self._budget_report = budget.report()
                return []

        self.console.print(f"{self.icons['ai']} Analyzing content with AI...")

        ai_client = self.ai_client
        analyzer = ContentAnalyzer(
            ai_client, self.profiles, console=self.console,
            concurrency=self.config.ai.analysis_concurrency,
            throttle_sec=self.config.ai.throttle_sec,
            cache=self.state_store,
            cache_ttl_hours=float(
                getattr(self.config.collection, "analysis_cache_ttl_hours", 72.0) or 0.0
            ),
        )

        analyzed = await analyzer.analyze_batch(items)
        if analyzer.cache_hits:
            self.console.print(
                f"   Reused {analyzer.cache_hits} cached analyses (no tokens spent)"
            )
            counters = getattr(self, "run_counters", None)
            if counters is not None:
                counters["cache_hits"] = analyzer.cache_hits
        return analyzed
