"""Lightweight token usage tracker shared across AI clients.

This module keeps a simple in-memory counter of tokens used during a single
Horizon run, so the orchestrator can print a summary at the end.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict


@dataclass
class ProviderUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass
class TokenUsageSnapshot:
    total_input_tokens: int
    total_output_tokens: int
    per_provider: Dict[str, ProviderUsage] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens


_provider_usage: Dict[str, ProviderUsage] = {}


def record_usage(provider: str, input_tokens: int = 0, output_tokens: int = 0) -> None:
    """Accumulate token usage for a given provider.

    Args:
        provider: Provider identifier, e.g. "openai", "anthropic".
        input_tokens: Prompt / input tokens used.
        output_tokens: Completion / output tokens used.
    """
    if input_tokens <= 0 and output_tokens <= 0:
        return

    usage = _provider_usage.setdefault(provider, ProviderUsage())
    usage.input_tokens += max(0, input_tokens)
    usage.output_tokens += max(0, output_tokens)


def get_usage_snapshot() -> TokenUsageSnapshot:
    """Return a snapshot of accumulated token usage."""
    total_in = sum(u.input_tokens for u in _provider_usage.values())
    total_out = sum(u.output_tokens for u in _provider_usage.values())
    return TokenUsageSnapshot(
        total_input_tokens=total_in,
        total_output_tokens=total_out,
        per_provider={provider: replace(usage) for provider, usage in _provider_usage.items()},
    )


def reset_usage() -> None:
    """Reset all accumulated usage (useful for tests)."""
    _provider_usage.clear()


class BudgetExceeded(RuntimeError):
    """Raised when a run would spend more than the configured budget."""


@dataclass
class BudgetGuard:
    """Hard ceiling on tokens/cost for one run, with soft degradation.

    The point is not to abort mid-run — a half-written digest is worse than a
    shorter one. When the ceiling approaches, cheap optional stages (extra
    enrichment) are skipped and the digest is marked ``budget_truncated``.
    """

    max_total_tokens: int | None = None
    max_cost_usd: float | None = None
    # Approximate blended price per 1M tokens; override per deployment.
    usd_per_million_tokens: float = 1.0
    _spent_tokens: int = 0
    _truncated: bool = False
    _reason: str = ""

    def reset(self) -> None:
        self._spent_tokens = 0
        self._truncated = False
        self._reason = ""

    def sync_from_usage(self) -> int:
        """Copy the process-wide counter into this guard; returns tokens spent."""
        self._spent_tokens = get_usage_snapshot().total_tokens
        return self._spent_tokens

    @property
    def tokens_spent(self) -> int:
        return self._spent_tokens

    @property
    def cost_usd(self) -> float:
        return self._spent_tokens / 1_000_000 * self.usd_per_million_tokens

    @property
    def truncated(self) -> bool:
        return self._truncated

    @property
    def reason(self) -> str:
        return self._reason

    def remaining_tokens(self) -> int | None:
        if self.max_total_tokens is None:
            return None
        return max(0, self.max_total_tokens - self._spent_tokens)

    def exhausted(self) -> bool:
        """True when no further paid stage should run."""
        self.sync_from_usage()
        if self.max_total_tokens is not None and self._spent_tokens >= self.max_total_tokens:
            self._truncated = True
            self._reason = (
                f"token budget reached ({self._spent_tokens}/{self.max_total_tokens})"
            )
            return True
        if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            self._truncated = True
            self._reason = (
                f"cost budget reached (${self.cost_usd:.2f}/${self.max_cost_usd:.2f})"
            )
            return True
        return False

    def allow(self, *, stage: str, estimated_tokens: int = 0) -> bool:
        """Ask before a paid stage. Marks the run truncated when refused."""
        self.sync_from_usage()
        projected = self._spent_tokens + max(0, estimated_tokens)
        if self.max_total_tokens is not None and projected > self.max_total_tokens:
            self._truncated = True
            self._reason = f"{stage} would exceed token budget"
            return False
        if self.max_cost_usd is not None:
            projected_cost = projected / 1_000_000 * self.usd_per_million_tokens
            if projected_cost > self.max_cost_usd:
                self._truncated = True
                self._reason = f"{stage} would exceed cost budget"
                return False
        return True

    def report(self) -> dict:
        return {
            "max_total_tokens": self.max_total_tokens,
            "max_cost_usd": self.max_cost_usd,
            "tokens_spent": self._spent_tokens,
            "cost_usd": round(self.cost_usd, 4),
            "truncated": self._truncated,
            "reason": self._reason,
        }


def write_usage_snapshot(path: str | Path, snapshot: TokenUsageSnapshot) -> None:
    """Write a usage snapshot as JSON for billing and per-run accounting."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            {
                "total_input_tokens": snapshot.total_input_tokens,
                "total_output_tokens": snapshot.total_output_tokens,
                "total_tokens": snapshot.total_tokens,
                "per_provider": {
                    provider: {
                        "input_tokens": usage.input_tokens,
                        "output_tokens": usage.output_tokens,
                        "total_tokens": usage.total,
                    }
                    for provider, usage in snapshot.per_provider.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
