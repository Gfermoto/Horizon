"""GigaChat OAuth: exchange a long-lived auth key for short-lived access tokens.

The operator saves one GigaChat ``Authorization`` key, but that key is not a
bearer token: it has to be exchanged for an access token, and that token lives
30 minutes. A radar run takes far longer than that — the first version minted a
token once before the run and every step after minute 30 came back 401, which
looked exactly like an expired operator key.

This provider refreshes ahead of expiry, and once more on a 401, so a long run
never dies halfway through. It is an ``httpx.Auth``, so the OpenAI SDK keeps
using its normal transport.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from typing import AsyncGenerator, Generator, Iterable

import httpx

log = logging.getLogger(__name__)

OAUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
SCOPES: tuple[str, ...] = ("GIGACHAT_API_PERS", "GIGACHAT_API_B2B", "GIGACHAT_API_CORP")
# Renew this many seconds before the server-side expiry.
SKEW_SEC = 180
DEFAULT_TTL_SEC = 1800


class GigaChatAuthError(RuntimeError):
    """The auth key itself was rejected; the run cannot continue."""


def _expiry_of(payload: dict) -> float:
    raw = payload.get("expires_at")
    try:
        return float(raw) / 1000.0
    except (TypeError, ValueError):
        return time.time() + DEFAULT_TTL_SEC


def auth_key_from_env() -> str:
    """The long-lived key, read from whichever variable the caller set."""
    for name in ("GIGACHAT_AUTH_KEY", "GW_GIGACHAT_AUTH_KEY"):
        value = (os.environ.get(name) or "").strip()
        if value:
            return value
    return ""


class GigaChatAuth(httpx.Auth):
    """Bearer auth that keeps a GigaChat access token fresh for the whole run."""

    # A retry re-sends the body, so httpx has to buffer it.
    requires_request_body = True

    def __init__(self, auth_key: str, *, scopes: Iterable[str] = SCOPES):
        key = (auth_key or "").strip()
        if key.lower().startswith("basic "):
            key = key[6:].strip()
        if not key:
            raise GigaChatAuthError("GigaChat: пустой ключ авторизации")
        self._key = key
        self._scopes = tuple(scopes)
        self._token: str | None = None
        self._expires_at = 0.0
        self._lock = threading.Lock()

    # --- token state -----------------------------------------------------
    def _fresh(self) -> bool:
        return bool(self._token) and time.time() < (self._expires_at - SKEW_SEC)

    def _store(self, payload: dict) -> str | None:
        token = payload.get("access_token")
        if not token:
            return None
        self._token = str(token)
        self._expires_at = _expiry_of(payload)
        return self._token

    def _invalidate(self) -> None:
        self._token = None
        self._expires_at = 0.0

    @staticmethod
    def _headers(key: str) -> dict[str, str]:
        return {
            "Authorization": f"Basic {key}",
            "RqUID": str(uuid.uuid4()),
            "Content-Type": "application/x-www-form-urlencoded",
        }

    # --- sync ------------------------------------------------------------
    def _mint(self) -> str:
        with self._lock:
            if self._fresh():
                return str(self._token)
            last_status = 0
            last_body = ""
            for scope in self._scopes:
                response = httpx.post(
                    OAUTH_URL,
                    headers=self._headers(self._key),
                    data={"scope": scope},
                    timeout=20,
                    verify=False,
                )
                last_status = response.status_code
                if last_status < 300:
                    token = self._store(response.json())
                    if token:
                        log.info("gigachat: access token refreshed (scope %s)", scope)
                        return token
                last_body = response.text[:180]
            raise GigaChatAuthError(
                f"GigaChat не принял ключ авторизации ({last_status}): {last_body}"
            )

    # --- async -----------------------------------------------------------
    async def _amint(self) -> str:
        if self._fresh():
            return str(self._token)
        last_status = 0
        last_body = ""
        async with httpx.AsyncClient(verify=False, timeout=20.0) as client:
            for scope in self._scopes:
                response = await client.post(
                    OAUTH_URL,
                    headers=self._headers(self._key),
                    data={"scope": scope},
                )
                last_status = response.status_code
                if last_status < 300:
                    token = self._store(response.json())
                    if token:
                        log.info("gigachat: access token refreshed (scope %s)", scope)
                        return token
                last_body = response.text[:180]
        raise GigaChatAuthError(
            f"GigaChat не принял ключ авторизации ({last_status}): {last_body}"
        )

    # --- httpx hooks -----------------------------------------------------
    def auth_flow(self, request: httpx.Request) -> Generator[httpx.Request, httpx.Response, None]:
        request.headers["Authorization"] = f"Bearer {self._mint()}"
        response = yield request
        if response.status_code == 401:
            # Token died earlier than announced (revoked, clock skew): mint a new
            # one and replay the same request once.
            log.warning("gigachat: 401 after refresh, minting a new token")
            self._invalidate()
            request.headers["Authorization"] = f"Bearer {self._mint()}"
            yield request

    async def async_auth_flow(
        self, request: httpx.Request
    ) -> AsyncGenerator[httpx.Request, httpx.Response]:
        request.headers["Authorization"] = f"Bearer {await self._amint()}"
        response = yield request
        if response.status_code == 401:
            log.warning("gigachat: 401 after refresh, minting a new token")
            self._invalidate()
            request.headers["Authorization"] = f"Bearer {await self._amint()}"
            yield request
