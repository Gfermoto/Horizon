"""Core data models for Horizon."""

from datetime import datetime, timezone
from enum import Enum
import re
from typing import Annotated, Literal, Optional, List, Dict, Any, NamedTuple, Union
from pydantic import BaseModel, ConfigDict, HttpUrl, Field, field_validator, model_validator


class SourceType(str, Enum):
    """Supported information source types."""

    GITHUB = "github"
    HACKERNEWS = "hackernews"
    RSS = "rss"
    REDDIT = "reddit"
    TELEGRAM = "telegram"
    TWITTER = "twitter"
    OPENBB = "openbb"
    OSSINSIGHT = "ossinsight"
    GDELT = "gdelt"
    GOOGLE_NEWS = "google_news"


class SourceDefinition(NamedTuple):
    """How a top-level source is represented in SourcesConfig."""

    config_field: str
    config_is_list: bool = False
    item_fields: tuple[str, ...] = ()


SOURCE_REGISTRY = {
    SourceType.GITHUB.value: SourceDefinition("github", config_is_list=True),
    SourceType.HACKERNEWS.value: SourceDefinition("hackernews"),
    SourceType.RSS.value: SourceDefinition("rss", config_is_list=True),
    SourceType.REDDIT.value: SourceDefinition("reddit", item_fields=("subreddits", "users")),
    SourceType.TELEGRAM.value: SourceDefinition("telegram", item_fields=("channels",)),
    SourceType.TWITTER.value: SourceDefinition("twitter", item_fields=("users", "keywords")),
    SourceType.OPENBB.value: SourceDefinition("openbb", item_fields=("watchlists",)),
    SourceType.OSSINSIGHT.value: SourceDefinition("ossinsight"),
    SourceType.GDELT.value: SourceDefinition("gdelt"),
    SourceType.GOOGLE_NEWS.value: SourceDefinition("google_news"),
}

ProfileRoute = Optional[Union[str, List[str]]]


class ClassificationResult(BaseModel):
    """Resolved processing profile for a content item."""

    profile: str
    method: Literal["source_override", "ai_match"]
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    reason: Optional[str] = None


class ContentAnalysis(BaseModel):
    """Profile-driven first-pass analysis."""

    score: Optional[float] = Field(default=None, ge=0, le=10, allow_inf_nan=False)
    reason: str
    summary: str
    tags: List[str] = Field(default_factory=list)
    # Short normalized identity of the underlying event (3-6 tokens). The AI
    # emits it in the same analysis pass so cross-day dedup costs no extra call.
    event_key: Optional[str] = None
    # Optional PR-radar fields (ignored by news profiles).
    sentiment: Optional[str] = None
    mention_type: Optional[str] = None
    risk_summary: Optional[str] = None
    recommended_action: Optional[str] = None
    brand_matched: Optional[str] = None


class ArtifactSource(BaseModel):
    """External source used while producing an artifact."""

    id: str
    title: str
    url: str


class ContentBlock(BaseModel):
    """A renderable section produced by an enrichment profile."""

    id: str
    type: Literal["section"] = "section"
    title: str
    content: str
    source_refs: List[str] = Field(default_factory=list)
    primary: bool = False


class ContentArtifact(BaseModel):
    """Localized, profile-defined enriched content."""

    language: str
    title: str
    blocks: List[ContentBlock] = Field(default_factory=list)
    sources: List[ArtifactSource] = Field(default_factory=list)


class ProcessingResult(BaseModel):
    """All AI processing state for a content item."""

    classification: ClassificationResult
    analysis: Optional[ContentAnalysis] = None
    artifacts: Dict[str, ContentArtifact] = Field(default_factory=dict)


class ContentItem(BaseModel):
    """Unified content item model from any source."""

    model_config = ConfigDict(extra="forbid")

    id: str  # Format: {source}:{subtype}:{native_id}
    source_type: SourceType
    title: str
    url: HttpUrl
    content: Optional[str] = None
    author: Optional[str] = None
    published_at: datetime
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    profile: ProfileRoute = None
    processing: Optional[ProcessingResult] = None
    # Stable identity of the material at the source (tender notice number, lot
    # GUID, article slug). Falls back to the normalized URL when absent, so a
    # plugin connector can dedup on the platform id instead of the page URL.
    canonical_key: Optional[str] = None
    # What the item represents. Keeps one pipeline usable for news, tenders,
    # releases and regulatory notices without per-vertical code.
    entity_kind: str = "article"


class AIProvider(str, Enum):
    """Supported AI providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OPENAI_CHATGPT = "openai-chatgpt"
    AZURE = "azure"
    ALI = "ali"
    GEMINI = "gemini"
    DOUBAO = "doubao"
    MINIMAX = "minimax"
    DEEPSEEK = "deepseek"
    GIGACHAT = "gigachat"
    OLLAMA = "ollama"


# Provider-specific defaults used by setup and provider-chain expansion.
AI_PROVIDER_DEFAULTS = {
    AIProvider.ANTHROPIC: {
        "model": "claude-sonnet-5-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": None,
    },
    AIProvider.OPENAI: {
        "model": "gpt-4",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": None,
    },
    AIProvider.OPENAI_CHATGPT: {
        "model": "",  # Selected from the signed-in account's model catalog.
        "api_key_env": "",
        "base_url": None,
    },
    AIProvider.AZURE: {
        "model": "gpt-4",
        "api_key_env": "AZURE_OPENAI_API_KEY",
        "base_url": None,
        "azure_endpoint_env": "AZURE_OPENAI_ENDPOINT",
        "api_version": "2024-10-21",
    },
    AIProvider.ALI: {
        "model": "qwen-plus",
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    },
    AIProvider.GEMINI: {
        "model": "gemini-1.5-flash",
        "api_key_env": "GOOGLE_API_KEY",
        "base_url": None,
    },
    AIProvider.DOUBAO: {
        "model": "doubao-pro-32k",
        "api_key_env": "DOUBAO_API_KEY",
        "base_url": "https://ark.cn-beijing.volces.com/api/v3",
    },
    AIProvider.MINIMAX: {
        "model": "MiniMax-M3.1-Flash",
        "api_key_env": "MINIMAX_API_KEY",
        "base_url": "https://api.minimax.io/v1",
    },
    AIProvider.DEEPSEEK: {
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
    },
    AIProvider.GIGACHAT: {
        "model": "GigaChat-3-Ultra",
        "api_key_env": "GIGACHAT_API_KEY",
        "base_url": "https://api.giga.chat/v1",
    },
    AIProvider.OLLAMA: {
        "model": "llama3.1",
        "api_key_env": "",
        "base_url": "http://localhost:11434/v1",
    },
}


class AIConfig(BaseModel):
    """AI client configuration."""

    provider: AIProvider
    provider_chain: Optional[str] = None
    model: str
    base_url: Optional[str] = None
    api_key_env: str
    chatgpt_account: Optional[str] = None
    extra_body: Optional[Dict[str, Any]] = None
    temperature: float = 0.3
    max_tokens: int = 4096
    throttle_sec: float = 0.0
    analysis_concurrency: int = 1
    enrichment_concurrency: int = 1
    languages: List[str] = Field(default_factory=lambda: ["en"])
    # Per-run spend ceiling. None means unlimited (the historical behaviour).
    max_total_tokens: Optional[int] = Field(default=None, ge=1)
    max_cost_usd: Optional[float] = Field(default=None, gt=0)
    # Blended $/1M tokens used to translate the token ceiling into money.
    usd_per_million_tokens: float = Field(default=1.0, ge=0)
    # Azure OpenAI specific; required when provider == AZURE
    azure_endpoint_env: Optional[str] = None
    api_version: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def default_chatgpt_credentials(cls, value: Any) -> Any:
        if isinstance(value, dict) and value.get("provider") == AIProvider.OPENAI_CHATGPT:
            value = dict(value)
            value.setdefault("api_key_env", "")
        return value

    @model_validator(mode="after")
    def validate_chatgpt_settings(self) -> "AIConfig":
        chain = [name.strip() for name in (self.provider_chain or "").split(",")]
        if "openai-chatgpt" in chain and self.provider != AIProvider.OPENAI_CHATGPT:
            raise ValueError("ChatGPT in provider_chain requires ai.provider=openai-chatgpt and its account-specific model")
        if self.provider == AIProvider.OPENAI_CHATGPT:
            if not self.model.strip():
                raise ValueError("Choose a ChatGPT model using horizon-chatgpt models")
            if self.base_url and self.base_url.rstrip("/") != "https://api.openai.com/v1":
                raise ValueError("openai-chatgpt uses https://api.openai.com/v1; custom base_url is not supported")
            if self.extra_body:
                raise ValueError("ai.extra_body is not supported for openai-chatgpt preview requests")
            if self.api_key_env:
                raise ValueError("openai-chatgpt uses OAuth; leave api_key_env empty")
        return self

    @field_validator("languages")
    @classmethod
    def validate_languages(cls, languages: List[str]) -> List[str]:
        """Allow conventional language tags while excluding path syntax."""
        language_tag = re.compile(r"^[A-Za-z]{2,8}(?:[-_][A-Za-z0-9]{1,8})*$")
        invalid = [language for language in languages if not language_tag.fullmatch(language)]
        if invalid:
            raise ValueError(f"invalid language code: {invalid[0]!r}")
        language_keys = [language.replace("_", "-").casefold() for language in languages]
        if len(language_keys) != len(set(language_keys)):
            raise ValueError("output languages must be unique")
        return languages


class GitHubSourceConfig(BaseModel):
    """GitHub source configuration."""

    type: str  # "user_events", "repo_releases", etc.
    username: Optional[str] = None
    owner: Optional[str] = None
    repo: Optional[str] = None
    enabled: bool = True
    category: Optional[str] = None
    profile: ProfileRoute = None


class HackerNewsConfig(BaseModel):
    """Hacker News configuration."""

    enabled: bool = True
    fetch_top_stories: int = 30
    min_score: int = 100
    category: Optional[str] = None
    profile: ProfileRoute = None


class ExtractorType(str, Enum):
    TRAFILATURA = "trafilatura"
    FIRECRAWL = "firecrawl"
    OUTCRAWL = "outcrawl"
    SPECSTREAM = "specstream"


class TrafilaturaExtractorConfig(BaseModel):
    type: Literal[ExtractorType.TRAFILATURA] = ExtractorType.TRAFILATURA
    favor_precision: bool = False
    favor_recall: bool = False


class FirecrawlExtractorConfig(BaseModel):
    type: Literal[ExtractorType.FIRECRAWL] = ExtractorType.FIRECRAWL
    api_key_env: str = "FIRECRAWL_API_KEY"
    only_main_content: bool = True


class OutcrawlExtractorConfig(BaseModel):
    type: Literal[ExtractorType.OUTCRAWL] = ExtractorType.OUTCRAWL
    api_key_env: str = "OUTCRAWL_API_KEY"
    only_main_content: bool = True


class SpecStreamExtractorConfig(BaseModel):
    type: Literal[ExtractorType.SPECSTREAM] = ExtractorType.SPECSTREAM
    api_key_env: str = "SPECSTREAM_API_KEY"


ExtractorConfig = Annotated[
    Union[
        TrafilaturaExtractorConfig,
        FirecrawlExtractorConfig,
        OutcrawlExtractorConfig,
        SpecStreamExtractorConfig,
    ],
    Field(discriminator="type"),
]


class RSSSourceConfig(BaseModel):
    """RSS feed source configuration."""

    name: str
    url: HttpUrl
    enabled: bool = True
    category: Optional[str] = None
    content_extractor: Optional[str] = None
    profile: ProfileRoute = None
    max_entries: Optional[int] = Field(default=None, ge=1)
    must_match: Optional[str] = None


class RedditSubredditConfig(BaseModel):
    """Configuration for monitoring a specific subreddit."""

    subreddit: str
    enabled: bool = True
    sort: str = "hot"  # hot, new, top, rising
    time_filter: str = (
        "day"  # hour, day, week, month, year, all (only for top/controversial)
    )
    fetch_limit: int = 25
    min_score: int = 10
    category: Optional[str] = None
    profile: ProfileRoute = None


class RedditUserConfig(BaseModel):
    """Configuration for monitoring a specific Reddit user."""

    username: str  # without u/ prefix
    enabled: bool = True
    sort: str = "new"
    fetch_limit: int = 10
    category: Optional[str] = None
    profile: ProfileRoute = None


class RedditConfig(BaseModel):
    """Reddit source configuration."""

    enabled: bool = True
    subreddits: List[RedditSubredditConfig] = Field(default_factory=list)
    users: List[RedditUserConfig] = Field(default_factory=list)
    fetch_comments: int = 5  # top comments per post, 0 to disable


class TelegramChannelConfig(BaseModel):
    """Configuration for monitoring a specific Telegram channel."""

    channel: str  # channel username, e.g. "zaihuapd"
    enabled: bool = True
    fetch_limit: int = 20
    category: Optional[str] = None
    profile: ProfileRoute = None


class TelegramConfig(BaseModel):
    """Telegram source configuration."""

    enabled: bool = True
    channels: List[TelegramChannelConfig] = Field(default_factory=list)


class TwitterConfig(BaseModel):
    """Twitter source configuration.

    Two modes are supported:
    - "apify": Use Apify scweet actor (requires APIFY_TOKEN, more reliable)
    - "playwright": Use Playwright + browser cookies (free, no token needed)

    `keywords` uses Apify search mode. Playwright logs a warning and skips them.
    """

    enabled: bool = True
    mode: str = "apify"  # "apify" or "playwright"
    users: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    fetch_limit: int = 10
    category: Optional[str] = None
    profile: ProfileRoute = None
    fetch_reply_text: bool = False
    max_replies_per_tweet: int = 3
    max_tweets_to_expand: int = 10
    reply_min_likes: int = 0
    # Apify settings (used when mode == "apify")
    apify_token_env: str = "APIFY_TOKEN"
    actor_id: str = "altimis~scweet"
    # Playwright settings (used when mode == "playwright")
    cookie_dir: str = "data"
    cookie_file_pattern: str = "x_cookies_*.json"


class OpenBBWatchlist(BaseModel):
    """A named watchlist of tickers fetched from one OpenBB provider.

    Each watchlist produces one news.company() call per run, so group
    symbols by provider rather than creating one watchlist per symbol.
    """

    name: str
    symbols: List[str] = Field(default_factory=list)
    enabled: bool = True
    provider: str = "yfinance"
    fetch_limit: int = 20
    category: Optional[str] = None
    profile: ProfileRoute = None


class OpenBBConfig(BaseModel):
    """OpenBB Platform source configuration.

    Uses the installed `openbb` SDK to fetch news and filings for a set of
    tickers. The SDK is an optional dependency; if it is not installed the
    scraper reports a source failure while other sources can still run.

    Provider credentials (FMP, Benzinga, Polygon, Intrinio, Tiingo, etc.)
    are resolved by openbb from environment variables / its own user
    settings file, so Horizon does not need to pass them explicitly.
    """

    enabled: bool = True
    watchlists: List[OpenBBWatchlist] = Field(default_factory=list)
    fetch_filings: bool = False
    filings_provider: str = "sec"


class OSSInsightConfig(BaseModel):
    """OSS Insight trending repos source configuration.

    Pulls top star-gain repositories from the OSS Insight public API and
    emits them as ContentItems. Optional `keywords` filter limits results
    to repos whose description, repo name, or collection names contain at
    least one of the listed substrings (case-insensitive). Leave
    `keywords` empty to ingest everything trending in the configured
    languages.
    """

    enabled: bool = False
    period: str = "past_24_hours"  # past_24_hours, past_28_days
    languages: List[str] = Field(
        default_factory=lambda: ["All", "Python", "TypeScript"]
    )
    keywords: List[str] = Field(default_factory=list)
    min_stars: int = 5
    max_items: int = 30
    category: Optional[str] = None
    profile: ProfileRoute = None


class GDELTConfig(BaseModel):
    """GDELT 2.0 DOC API source configuration.

    Queries the key-less GDELT DOC API
    (https://api.gdeltproject.org/api/v2/doc/doc) for recent news articles
    matching a search query and emits them as ContentItems. No API key is
    required. The DOC API caps results at 250 records per request, so keep
    `max_records` modest.
    """

    enabled: bool = False
    query: str = "artificial intelligence"
    mode: str = "ArtList"
    max_records: int = 75  # GDELT DOC API caps at 250; keep modest
    timespan: Optional[str] = None  # e.g. "24h"; overrides since-derived window
    language: Optional[str] = None  # sourcelang filter, e.g. "english"; None = no filter
    country: Optional[str] = None  # sourcecountry filter; None = no filter
    category: Optional[str] = None  # Horizon category label for downstream grouping
    profile: ProfileRoute = None


class GoogleNewsConfig(BaseModel):
    """Google News RSS search source configuration.

    Builds Google News RSS search URLs
    (https://news.google.com/rss/search) for a query and parses the
    resulting feed via feedparser. No API key is required.
    """

    enabled: bool = False
    query: str = "artificial intelligence"
    language: str = "en"  # hl
    country: str = "US"  # gl
    ceid: Optional[str] = None  # when None scraper derives it as "{country}:{language}"
    max_results: int = 100  # cap ~100
    category: Optional[str] = None
    profile: ProfileRoute = None


class CustomSourceConfig(BaseModel):
    """A declarative or plugin-provided source, configured without new code.

    ``type`` names either a built-in plugin (``json_feed``) or an entry point
    registered under ``horizon.scrapers``. ``config`` is passed through to the
    plugin untouched, so a vertical connector owns its own schema.
    """

    model_config = ConfigDict(extra="forbid")

    type: str
    name: Optional[str] = None
    enabled: bool = True
    category: Optional[str] = None
    profile: ProfileRoute = None
    config: Dict[str, Any] = Field(default_factory=dict)


class SourcesConfig(BaseModel):
    """All sources configuration."""

    github: List[GitHubSourceConfig] = Field(default_factory=list)
    hackernews: HackerNewsConfig = Field(default_factory=HackerNewsConfig)
    rss: List[RSSSourceConfig] = Field(default_factory=list)
    reddit: RedditConfig = Field(default_factory=RedditConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    twitter: Optional[TwitterConfig] = None
    openbb: Optional[OpenBBConfig] = None
    ossinsight: OSSInsightConfig = Field(default_factory=OSSInsightConfig)
    gdelt: Optional[GDELTConfig] = None
    google_news: Optional[GoogleNewsConfig] = None
    # Declarative / plugin sources: json_feed or any installed entry point.
    custom: List[CustomSourceConfig] = Field(default_factory=list)


class WebhookConfig(BaseModel):
    """Webhook notification configuration."""

    url_env: Optional[str] = (
        None  # Environment variable name containing the webhook URL
    )
    request_body: Optional[Union[str, dict, list]] = (
        None  # POST body: real JSON object or string with #{key} placeholders; if empty, will use GET
    )
    headers: Optional[str] = None  # Custom headers, "Key: Value" per line
    delivery: Literal["summary", "summary_and_items"] = "summary"
    overview_position: Literal["first", "last"] = "first"
    platform: Literal["generic", "feishu", "lark", "dingtalk", "slack", "discord"] = "generic"
    layout: Literal["markdown", "collapsible"] = "markdown"
    fallback_layout: Literal["markdown", "collapsible"] = (
        "markdown"  # Layout to use when the requested layout is unsupported
    )
    languages: Optional[List[str]] = (
        None  # Optional language filter for webhook delivery; defaults to all AI languages
    )
    enabled: bool = False


class WeChatConfig(BaseModel):
    """Optional iLink delivery; credentials live in the data directory's session file."""

    enabled: bool = False
    languages: Optional[List[str]] = None
    chunk_size: int = Field(default=4000, gt=0, le=4000)


class EmailConfig(BaseModel):
    """Email configuration for updates/subscriptions."""

    imap_server: str
    imap_port: int = 993
    imap_enabled: bool = True
    smtp_server: str
    smtp_port: int = 465
    smtp_username: Optional[str] = None
    email_address: str
    password_env: str = "EMAIL_PASSWORD"
    sender_name: str = "Horizon Daily"
    subscribe_keyword: str = "SUBSCRIBE"
    unsubscribe_keyword: str = "UNSUBSCRIBE"
    enabled: bool = False


class CategoryGroupConfig(BaseModel):
    """A quota group containing one or more source categories."""

    name: Optional[str] = None
    limit: int = Field(gt=0)
    categories: List[str] = Field(min_length=1)


class ProfileSettingsConfig(BaseModel):
    """User preferences applied to a processing profile at runtime."""

    model_config = ConfigDict(extra="forbid")

    threshold: Optional[float] = Field(default=None, ge=0, le=10)
    topic_dedup: bool = True


class ProcessingConfig(BaseModel):
    """Profile discovery and fallback settings."""

    model_config = ConfigDict(extra="forbid")

    profiles_dir: str = "profiles"
    default_profile: str = "tech-news"
    profile_settings: Dict[str, ProfileSettingsConfig] = Field(default_factory=dict)


class DisplayConfig(BaseModel):
    """Controls terminal output presentation."""

    model_config = ConfigDict(extra="forbid")

    icon_style: Literal["emoji", "nerd", "ascii"] = "emoji"


class CollectionConfig(BaseModel):
    """Controls which source items are fetched."""

    model_config = ConfigDict(extra="forbid")

    # Positivity is collection_hours(), applied once before a run. This field
    # stores the configured number and does not check it again.
    time_window_hours: int = 24
    exclude_urls: List[str] = Field(default_factory=list)
    # Normalized headlines already delivered (cross-URL reprints of the same story).
    exclude_title_fingerprints: List[str] = Field(default_factory=list)
    # How far back the durable event memory looks when blocking reprints.
    memory_days: int = Field(default=30, ge=1, le=3650)
    # Set false to ignore state.db entirely (Markdown history is still used).
    state_enabled: bool = True
    # Circuit breaker: after N consecutive failures a source is parked.
    max_consecutive_failures: int = Field(default=0, ge=0)
    # Base cooldown for the breaker; escalates 6h -> 24h -> 72h when > 0.
    cooldown_base_hours: float = Field(default=6.0, ge=0)
    # A source that fetches but never contributes is muted after this many
    # consecutive runs below ``min_source_yield``. Advisory, not destructive.
    min_source_yield: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    mute_after_low_yield_runs: int = Field(default=0, ge=0)
    # Connector bookkeeping written by the control plane: which external sources
    # are active for this radar and which are present but disabled. The engine
    # does not act on it, but rejecting it would fail every run that has any.
    connectors: Optional[Dict[str, List[str]]] = None
    # Per-request timeout for source fetching. A blocked host fails fast instead
    # of stalling the run through long retries.
    http_timeout_sec: float = Field(default=30.0, ge=5.0, le=120.0)
    # When every source fails, still finish the run instead of raising.
    tolerate_total_failure: bool = False

    # --- zero-LLM pre-filter -------------------------------------------------
    # Drop items whose text contains none of these substrings (case-insensitive).
    must_match: List[str] = Field(default_factory=list)
    # Drop items whose title/text matches any of these regular expressions.
    exclude_patterns: List[str] = Field(default_factory=list)
    # Drop items with less usable text than this (0 disables the check).
    min_content_chars: int = Field(default=0, ge=0)
    # Cap how many items reach the paid analysis stage (0 = unlimited).
    max_items_to_analyze: int = Field(default=0, ge=0)
    # Skip re-analysing identical content seen within this window (0 disables).
    analysis_cache_ttl_hours: float = Field(default=72.0, ge=0)
    # drop — suppress reprints of already-published event_key (news digests).
    # witness — keep reprints in the digest as is_witness metadata (PR radar).
    event_dedup_mode: Literal["drop", "witness"] = "drop"

    @field_validator("time_window_hours", mode="before")
    @classmethod
    def reject_boolean_time_window(cls, value: Any) -> Any:
        # Pydantic's lax int conversion accepts bool as 0/1, so reject it before
        # collection_hours() can only see the coerced integer.
        if isinstance(value, bool):
            raise ValueError("time_window_hours must be an integer, not a boolean")
        return value


class DigestConfig(BaseModel):
    """Controls grouping and limits in the final digest."""

    model_config = ConfigDict(extra="forbid")

    max_items: Optional[int] = Field(default=None, gt=0)
    category_groups: Dict[str, CategoryGroupConfig] = Field(default_factory=dict)
    group_by: Literal["category", "profile"] = "category"
    default_group: str = "other"
    default_group_limit: Optional[int] = Field(default=None, gt=0)
    profile_order: List[str] = Field(default_factory=list)
    # Emit a versioned machine-readable digest next to the Markdown one.
    export_json: bool = True
    # Write run_report.json (counters, source health, usage) for observability.
    write_run_report: bool = True

    @field_validator("profile_order")
    @classmethod
    def validate_profile_order(cls, value: List[str]) -> List[str]:
        if any(not profile_id.strip() for profile_id in value):
            raise ValueError("digest.profile_order entries must be non-empty strings")
        if len(value) != len(set(value)):
            raise ValueError("digest.profile_order entries must be unique")
        return value


class Config(BaseModel):
    """Main configuration model."""

    model_config = ConfigDict(extra="forbid")

    ai: AIConfig
    sources: SourcesConfig
    collection: CollectionConfig = Field(default_factory=CollectionConfig)
    digest: DigestConfig = Field(default_factory=DigestConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    extractors: Dict[str, ExtractorConfig] = Field(default_factory=dict)
    email: Optional[EmailConfig] = None
    webhook: Optional[WebhookConfig] = None
    wechat: Optional[WeChatConfig] = None
