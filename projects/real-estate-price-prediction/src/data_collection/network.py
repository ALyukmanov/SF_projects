"""
Network fetching component — isolated from parsing and persistence.

Extracted from ``cian_scraper.py`` (Phase 1 parser-architecture refactor) so
that retry/backoff/rate-limiting policy is a standalone, independently
testable unit instead of being interleaved with HTML-parsing code. This
module makes real HTTP requests — it is NOT covered by the offline fixture
tests in ``tests/data_collection/`` (those exercise pure parsing functions
only); it is instead tested with a mocked ``requests`` session so retry/
backoff/give-up decisions can be verified without touching the network.

Design notes (why this shape):
- A thin ``FetchResult`` return type instead of raising on non-200 responses
  lets callers decide what "no listings on this page" vs. "give up
  entirely" means, without every caller needing its own try/except around
  requests' exception hierarchy.
- Retry policy: exponential backoff with jitter on timeouts/connection
  errors and HTTP 429; immediate give-up (no retry budget wasted) on 403/404,
  since those are not transient.
- No CAPTCHA/anti-bot bypass logic anywhere in this module, by design —
  see ``src/data_collection/README.md`` "What this module intentionally
  does NOT do."
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Optional

import requests

from src.data_collection.parsing_utils import random_user_agent
from src.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;"
    "q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.5,en;q=0.3",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Cache-Control": "max-age=0",
}


@dataclass
class FetchResult:
    """Outcome of a single fetch attempt.

    Attributes:
        ok: True if a 200 response with a body was obtained.
        status_code: HTTP status code, if a response was received at all
            (None for a network-level failure with no response object).
        text: Response body text, only set when ``ok`` is True.
        gave_up_reason: Human-readable reason fetching stopped, only set
            when ``ok`` is False (e.g. ``"blocked_403"``, ``"exhausted_retries"``).
    """

    ok: bool
    status_code: Optional[int] = None
    text: Optional[str] = None
    gave_up_reason: Optional[str] = None


def build_session() -> requests.Session:
    """Return a ``requests.Session`` with a randomised User-Agent and
    browser-like headers (see ``_DEFAULT_HEADERS``)."""
    session = requests.Session()
    session.headers.update(_DEFAULT_HEADERS)
    session.headers["User-Agent"] = random_user_agent()
    return session


def rotate_user_agent(session: requests.Session) -> None:
    """Assign a fresh random User-Agent to *session* in place."""
    session.headers["User-Agent"] = random_user_agent()


def fetch_with_retry(
    session: requests.Session,
    url: str,
    *,
    max_retries: int = 3,
    base_delay: float = 10.0,
    timeout: float = 30.0,
    sleep_fn=time.sleep,
) -> FetchResult:
    """GET *url* through *session* with exponential backoff + jitter.

    Args:
        session: A ``requests.Session`` (see :func:`build_session`).
        url: Full URL to fetch.
        max_retries: Maximum number of attempts before giving up.
        base_delay: Base seconds for exponential backoff on HTTP 429
            (``base_delay * 2**attempt``) and on timeouts/connection errors
            (``base_delay * 2**(attempt-1)``), both plus random jitter.
        timeout: Per-request timeout in seconds.
        sleep_fn: Injectable sleep function — tests pass a no-op so retry
            logic can be verified without actually waiting in real time.

    Returns:
        A :class:`FetchResult`. Never raises for ordinary HTTP/network
        failures — those are reported via ``ok=False`` / ``gave_up_reason``,
        so callers don't need a try/except around this call.
    """
    last_status: Optional[int] = None
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, timeout=timeout)
        except requests.exceptions.Timeout:
            logger.warning("Timeout on attempt %d/%d for %s", attempt, max_retries, url)
            last_status = None
        except requests.exceptions.ConnectionError as exc:
            logger.warning(
                "Connection error on attempt %d/%d for %s: %s", attempt, max_retries, url, exc
            )
            last_status = None
        else:
            last_status = response.status_code
            if response.status_code == 200:
                return FetchResult(ok=True, status_code=200, text=response.text)

            if response.status_code == 429:
                wait = base_delay * (2**attempt) + random.uniform(0, 5)
                logger.warning(
                    "HTTP 429 on attempt %d/%d for %s. Backing off %.0f s.",
                    attempt,
                    max_retries,
                    url,
                    wait,
                )
                sleep_fn(wait)
                rotate_user_agent(session)
                continue

            if response.status_code in (403, 404):
                # Not transient — retrying wastes a request budget for no
                # benefit and, for 403 specifically, looks more bot-like.
                logger.error("HTTP %d for %s. Not retrying.", response.status_code, url)
                return FetchResult(
                    ok=False,
                    status_code=response.status_code,
                    gave_up_reason=f"blocked_{response.status_code}",
                )

            logger.warning(
                "HTTP %d on attempt %d/%d for %s", response.status_code, attempt, max_retries, url
            )

        if attempt < max_retries:
            wait = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 3)
            logger.debug("Retrying %s in %.0f s.", url, wait)
            sleep_fn(wait)

    logger.error("All %d attempts failed for %s (last_status=%s)", max_retries, url, last_status)
    return FetchResult(ok=False, status_code=last_status, gave_up_reason="exhausted_retries")
