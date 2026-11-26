from __future__ import annotations

import types
from typing import Any

import pytest

from supernav.runtime.support.llm_client import LLMRetryExhaustedError, is_retryable_llm_error, resilient_chat_completion, resilient_responses_completion


class _StatusError(Exception):
    def __init__(self, status_code: int, *, headers: dict[str, str] | None = None):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code
        self.response = types.SimpleNamespace(
            status_code=status_code,
            headers=headers or {},
        )


class _FakeClient:
    def __init__(self, outcomes: list[Any]):
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=self._create)
        )

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class _FakeResponsesClient:
    def __init__(self, outcomes: list[Any]):
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.responses = types.SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_resilient_chat_completion_retries_transient_statuses() -> None:
    client = _FakeClient([_StatusError(502), _StatusError(503), "ok"])
    sleeps: list[float] = []

    result = resilient_chat_completion(
        client,
        request_kwargs={"model": "gpt", "messages": [], "timeout": 30},
        max_attempts=5,
        sleep_func=sleeps.append,
        random_uniform=lambda _low, high: high,
    )

    assert result.response == "ok"
    assert result.telemetry.retry_count == 2
    assert result.telemetry.final == "ok_after_retry"
    assert sleeps == [1.0, 2.0]
    assert len(client.calls) == 3


def test_resilient_chat_completion_retries_connection_reset() -> None:
    client = _FakeClient([ConnectionResetError("connection reset by peer"), "ok"])

    result = resilient_chat_completion(
        client,
        request_kwargs={"model": "gpt", "messages": []},
        sleep_func=lambda _seconds: None,
        random_uniform=lambda _low, _high: 0.0,
    )

    assert result.response == "ok"
    assert result.telemetry.retry_count == 1
    assert is_retryable_llm_error(ConnectionResetError("connection reset")) is True


@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_resilient_chat_completion_does_not_retry_non_429_4xx(
    status_code: int,
) -> None:
    client = _FakeClient([_StatusError(status_code)])
    sleeps: list[float] = []

    with pytest.raises(_StatusError):
        resilient_chat_completion(
            client,
            request_kwargs={"model": "gpt", "messages": []},
            sleep_func=sleeps.append,
        )

    assert len(client.calls) == 1
    assert sleeps == []


def test_resilient_chat_completion_raises_after_max_attempts() -> None:
    client = _FakeClient([_StatusError(502), _StatusError(502), _StatusError(502)])

    with pytest.raises(LLMRetryExhaustedError) as exc_info:
        resilient_chat_completion(
            client,
            request_kwargs={"model": "gpt", "messages": []},
            max_attempts=3,
            sleep_func=lambda _seconds: None,
            random_uniform=lambda _low, _high: 0.0,
        )

    assert len(client.calls) == 3
    assert exc_info.value.telemetry.final == "failed_after_max"
    assert exc_info.value.telemetry.retry_count == 2
    assert exc_info.value.telemetry.last_status_code == 502


def test_resilient_chat_completion_stops_when_deadline_would_be_exceeded() -> None:
    client = _FakeClient([_StatusError(502), "ok"])
    sleeps: list[float] = []

    with pytest.raises(LLMRetryExhaustedError) as exc_info:
        resilient_chat_completion(
            client,
            request_kwargs={"model": "gpt", "messages": [], "timeout": 30},
            deadline=0.5,
            sleep_func=sleeps.append,
            monotonic=lambda: 0.0,
            random_uniform=lambda _low, high: high,
        )

    assert len(client.calls) == 1
    assert sleeps == []
    assert exc_info.value.telemetry.final == "failed_after_max"


def test_resilient_chat_completion_respects_retry_after_header() -> None:
    client = _FakeClient([_StatusError(429, headers={"Retry-After": "3"}), "ok"])
    sleeps: list[float] = []

    result = resilient_chat_completion(
        client,
        request_kwargs={"model": "gpt", "messages": []},
        max_attempts=2,
        sleep_func=sleeps.append,
        random_uniform=lambda _low, _high: 0.25,
    )

    assert result.response == "ok"
    assert sleeps == [3.0]
    assert result.telemetry.total_retry_wait_ms == 3000


def test_resilient_chat_completion_retry_after_does_not_shrink_backoff() -> None:
    client = _FakeClient([_StatusError(503, headers={"Retry-After": "1"}), "ok"])
    sleeps: list[float] = []

    result = resilient_chat_completion(
        client,
        request_kwargs={"model": "gpt", "messages": []},
        max_attempts=2,
        base_backoff=4.0,
        sleep_func=sleeps.append,
        random_uniform=lambda _low, _high: 0.25,
    )

    assert result.response == "ok"
    assert sleeps == [4.0]


def test_resilient_chat_completion_retries_timeout_error() -> None:
    client = _FakeClient([TimeoutError("timed out"), "ok"])
    sleeps: list[float] = []

    result = resilient_chat_completion(
        client,
        request_kwargs={"model": "gpt", "messages": []},
        sleep_func=sleeps.append,
        random_uniform=lambda _low, _high: 0.0,
    )

    assert result.response == "ok"
    assert result.telemetry.retry_count == 1


def test_resilient_responses_completion_reuses_retry_machine() -> None:
    client = _FakeResponsesClient([_StatusError(502), "ok"])
    sleeps: list[float] = []

    result = resilient_responses_completion(
        client,
        request_kwargs={"model": "gpt-5.4", "input": [], "timeout": 30},
        max_attempts=3,
        sleep_func=sleeps.append,
        random_uniform=lambda _low, _high: 0.0,
    )

    assert result.response == "ok"
    assert result.telemetry.retry_count == 1
    assert len(client.calls) == 2
    assert client.calls[0]["input"] == []
