"""Machine-readable digest export.

Markdown is for humans; this is for everything downstream — a publisher, a CRM,
a tender dashboard. The contract is versioned so consumers can pin
``schema_version`` and ignore fields they do not understand.

Design rules:

* Every field the engine knows about is present, even when empty, so consumers
  never need ``if key in item`` gymnastics.
* ``metadata`` is passed through untouched: that is where a scraper plugin puts
  domain fields (НМЦК, deadline, customer, OKPD2) without touching this module.
* Items carry their memory verdict (``is_update``) so a consumer can render an
  amendment differently from a fresh story.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .._file_utils import _atomic_write_text

__all__ = [
    "SCHEMA_VERSION",
    "build_digest_payload",
    "item_to_dict",
    "write_digest_json",
    "write_run_report",
]

SCHEMA_VERSION = 1


def _iso(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


def item_to_dict(item: Any, *, language: str, index: int) -> Dict[str, Any]:
    """Serialize one ``ContentItem`` for the JSON contract."""
    processing = getattr(item, "processing", None)
    analysis = getattr(processing, "analysis", None) if processing else None
    artifact = None
    if processing is not None:
        artifacts = getattr(processing, "artifacts", None) or {}
        artifact = artifacts.get(language)
    metadata = dict(getattr(item, "metadata", None) or {})
    if analysis is not None:
        for key in (
            "sentiment",
            "mention_type",
            "risk_summary",
            "recommended_action",
            "brand_matched",
        ):
            value = getattr(analysis, key, None)
            if value:
                metadata.setdefault(key, value)

    blocks: List[Dict[str, Any]] = []
    sources: List[Dict[str, Any]] = []
    artifact_title = None
    if artifact is not None:
        artifact_title = getattr(artifact, "title", None)
        for block in getattr(artifact, "blocks", None) or []:
            blocks.append(
                {
                    "id": getattr(block, "id", ""),
                    "title": getattr(block, "title", ""),
                    "content": getattr(block, "content", ""),
                    "primary": bool(getattr(block, "primary", False)),
                }
            )
        for source in getattr(artifact, "sources", None) or []:
            sources.append(
                {
                    "id": getattr(source, "id", ""),
                    "title": getattr(source, "title", ""),
                    "url": getattr(source, "url", ""),
                }
            )

    url = getattr(item, "url", None)
    return {
        "index": index,
        "id": getattr(item, "id", ""),
        "entity_kind": getattr(item, "entity_kind", "article") or "article",
        "is_update": bool(metadata.get("is_update")),
        "is_witness": bool(metadata.get("is_witness")),
        "canonical_key": getattr(item, "canonical_key", None),
        "title": artifact_title or getattr(item, "title", ""),
        "source_title": getattr(item, "title", ""),
        "url": str(url) if url is not None else "",
        "author": getattr(item, "author", None),
        "published_at": _iso(getattr(item, "published_at", None)),
        "updated_at": _iso(getattr(item, "updated_at", None)),
        "fetched_at": _iso(getattr(item, "fetched_at", None)),
        "source_type": (
            getattr(item.source_type, "value", str(getattr(item, "source_type", "")))
        ),
        "profile": (
            getattr(processing.classification, "profile", None) if processing else None
        ),
        "score": getattr(analysis, "score", None) if analysis else None,
        "reason": getattr(analysis, "reason", None) if analysis else None,
        "summary": getattr(analysis, "summary", None) if analysis else None,
        "event_key": getattr(analysis, "event_key", None) if analysis else None,
        "sentiment": getattr(analysis, "sentiment", None) if analysis else None,
        "tags": list(getattr(analysis, "tags", None) or []) if analysis else [],
        "blocks": blocks,
        "sources": sources,
        "metadata": metadata,
    }


def build_digest_payload(
    *,
    date: str,
    language: str,
    items: List[Any],
    total_fetched: int,
    window_hours: int,
    counters: Optional[Dict[str, int]] = None,
    sources_health: Optional[List[Dict[str, Any]]] = None,
    usage: Optional[Dict[str, Any]] = None,
    budget: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble the versioned digest document."""
    counters = dict(counters or {})
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "date": date,
        "language": language,
        "window_hours": int(window_hours),
        "totals": {
            "fetched": int(total_fetched),
            "selected": len(items),
            "prefiltered": int(counters.get("prefiltered", 0)),
            "memory_skipped": int(counters.get("memory_skipped", 0)),
            "memory_updates": int(counters.get("memory_updates", 0)),
            "analyzed": int(counters.get("analyzed", 0)),
            "event_skipped": int(counters.get("event_skipped", 0)),
            "cache_hits": int(counters.get("cache_hits", 0)),
            "dropped_by_balance": int(counters.get("dropped_by_balance", 0)),
        },
        "budget": budget or {},
        "usage": usage or {},
        "sources_health": sources_health or [],
        "items": [
            item_to_dict(item, language=language, index=index)
            for index, item in enumerate(items, start=1)
        ],
    }


def write_digest_json(directory: Path, date: str, language: str, payload: Dict[str, Any]) -> Path:
    """Write ``horizon-{date}-{lang}.json`` next to the Markdown digest."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"horizon-{date}-{language}.json"
    import json

    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path


def write_run_report(directory: Path, payload: Dict[str, Any]) -> Path:
    """Write ``run_report.json`` — one row per run, overwritten each time."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "run_report.json"
    import json

    _atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return path
