"""
Tests for src/data_collection/network.py — retry/backoff policy in isolation
from parsing. Uses a mocked ``requests.Session`` (via a tiny fake object),
never touches the network, and injects a no-op ``sleep_fn`` so retry tests
run instantly instead of actually waiting through backoff delays.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest
import requests

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_collection.network import FetchResult, fetch_with_retry  # noqa: E402


class _FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class _FakeSession:
    """Returns a scripted sequence of responses/exceptions, one per .get() call."""

    def __init__(self, script: List) -> None:
        self._script = list(script)
        self.calls = 0
        self.headers: dict = {}

    def get(self, url: str, timeout: float = 30.0):
        self.calls += 1
        item = self._script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def no_sleep():
    """A sleep_fn that records calls but doesn't actually block."""
    calls = []

    def _sleep(seconds: float) -> None:
        calls.append(seconds)

    _sleep.calls = calls  # type: ignore[attr-defined]
    return _sleep


class TestFetchWithRetrySuccess:
    def test_200_on_first_try_returns_ok(self, no_sleep):
        session = _FakeSession([_FakeResponse(200, "hello")])
        result = fetch_with_retry(session, "http://example.com", sleep_fn=no_sleep)
        assert result.ok is True
        assert result.status_code == 200
        assert result.text == "hello"
        assert session.calls == 1
        assert no_sleep.calls == [], "Should not sleep when the first attempt succeeds."


class TestFetchWithRetryGiveUp:
    def test_403_gives_up_immediately_no_retry(self, no_sleep):
        session = _FakeSession([_FakeResponse(403)])
        result = fetch_with_retry(session, "http://example.com", max_retries=3, sleep_fn=no_sleep)
        assert result.ok is False
        assert result.status_code == 403
        assert result.gave_up_reason == "blocked_403"
        assert session.calls == 1, "403 must not be retried — it's not transient."

    def test_404_gives_up_immediately_no_retry(self, no_sleep):
        session = _FakeSession([_FakeResponse(404)])
        result = fetch_with_retry(session, "http://example.com", max_retries=3, sleep_fn=no_sleep)
        assert result.ok is False
        assert result.status_code == 404
        assert session.calls == 1

    def test_exhausts_retries_on_repeated_500(self, no_sleep):
        session = _FakeSession([_FakeResponse(500)] * 3)
        result = fetch_with_retry(session, "http://example.com", max_retries=3, sleep_fn=no_sleep)
        assert result.ok is False
        assert result.gave_up_reason == "exhausted_retries"
        assert session.calls == 3


class TestFetchWithRetryBackoff:
    def test_429_retries_and_eventually_succeeds(self, no_sleep):
        session = _FakeSession([_FakeResponse(429), _FakeResponse(200, "ok")])
        result = fetch_with_retry(session, "http://example.com", max_retries=3, sleep_fn=no_sleep)
        assert result.ok is True
        assert result.text == "ok"
        assert session.calls == 2
        assert len(no_sleep.calls) == 1, "Should sleep once between the 429 and the retry."

    def test_timeout_then_success(self, no_sleep):
        session = _FakeSession([requests.exceptions.Timeout(), _FakeResponse(200, "ok")])
        result = fetch_with_retry(session, "http://example.com", max_retries=3, sleep_fn=no_sleep)
        assert result.ok is True
        assert session.calls == 2

    def test_connection_error_then_success(self, no_sleep):
        session = _FakeSession(
            [requests.exceptions.ConnectionError("refused"), _FakeResponse(200, "ok")]
        )
        result = fetch_with_retry(session, "http://example.com", max_retries=3, sleep_fn=no_sleep)
        assert result.ok is True
        assert session.calls == 2


def test_fetch_result_is_a_plain_dataclass():
    """Sanity check the return type has the documented shape."""
    r = FetchResult(ok=True, status_code=200, text="x")
    assert r.ok and r.status_code == 200 and r.text == "x" and r.gave_up_reason is None
