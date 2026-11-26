"""Lazy loader for the openai SDK.

The SDK is imported on first use so that tools not needing LLM access
(the bridge adapter, file I/O helpers, analytics scripts) can run without
paying the import cost or requiring openai to be installed.
"""

from __future__ import annotations

import email.utils
import errno
import random
import ssl
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Callable, Mapping

_openai_module = None

_RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504})
_RETRYABLE_EXCEPTION_NAMES = frozenset(
    {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
    }
)
_RETRYABLE_ERRNOS = frozenset(
    {
        errno.ECONNABORTED,
        errno.ECONNRESET,
        errno.ETIMEDOUT,
        errno.EPIPE,
    }
)


@dataclass(frozen=True)
class LLMRetryTelemetry:
    attempts: int
    retry_count: int
    total_retry_wait_ms: int
    final: str
    last_error_type: str | None = None
    last_status_code: int | None = None


@dataclass(frozen=True)
class ResilientChatCompletionResult:
    response: Any
    telemetry: LLMRetryTelemetry
    request_kwargs: Mapping[str, Any]


class LLMRetryExhaustedError(RuntimeError):
    """Raised when a retryable LLM failure exhausts bounded retry policy."""

    def __init__(
        self,
        message: str,
        *,
        telemetry: LLMRetryTelemetry,
        last_exception: BaseException,
    ) -> None:
        super().__init__(message)
        self.telemetry = telemetry
        self.last_exception = last_exception


def resilient_chat_completion(
    client: Any,
    *,
    request_kwargs: Mapping[str, Any],
    deadline: float | None = None,
    max_attempts: int = 6,
    base_backoff: float = 1.0,
    max_backoff: float = 30.0,
    max_total_wait: float = 90.0,
    sleep_func: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    random_uniform: Callable[[float, float], float] = random.uniform,
) -> ResilientChatCompletionResult:
    """Call non-streaming chat completions with bounded transient retries."""

    return _resilient_completion(
        client.chat.completions.create,
        request_kwargs=request_kwargs,
        deadline=deadline,
        max_attempts=max_attempts,
        base_backoff=base_backoff,
        max_backoff=max_backoff,
        max_total_wait=max_total_wait,
        sleep_func=sleep_func,
        monotonic=monotonic,
        random_uniform=random_uniform,
    )


def resilient_responses_completion(
    client: Any,
    *,
    request_kwargs: Mapping[str, Any],
    deadline: float | None = None,
    max_attempts: int = 6,
    base_backoff: float = 1.0,
    max_backoff: float = 30.0,
    max_total_wait: float = 90.0,
    sleep_func: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
    random_uniform: Callable[[float, float], float] = random.uniform,
) -> ResilientChatCompletionResult:
    """Call non-streaming Responses completions with bounded transient retries."""

    return _resilient_completion(
        client.responses.create,
        request_kwargs=request_kwargs,
        deadline=deadline,
        max_attempts=max_attempts,
        base_backoff=base_backoff,
        max_backoff=max_backoff,
        max_total_wait=max_total_wait,
        sleep_func=sleep_func,
        monotonic=monotonic,
        random_uniform=random_uniform,
    )


def _resilient_completion(
    call_fn: Callable[..., Any],
    *,
    request_kwargs: Mapping[str, Any],
    deadline: float | None,
    max_attempts: int,
    base_backoff: float,
    max_backoff: float,
    max_total_wait: float,
    sleep_func: Callable[[float], None],
    monotonic: Callable[[], float],
    random_uniform: Callable[[float, float], float],
) -> ResilientChatCompletionResult:
    """Run one LLM transport through the shared bounded retry machine."""

    max_attempts = _positive_int(max_attempts, "max_attempts")
    base_backoff = _non_negative_float(base_backoff, "base_backoff")
    max_backoff = _non_negative_float(max_backoff, "max_backoff")
    max_total_wait = _non_negative_float(max_total_wait, "max_total_wait")

    attempts = 0
    total_wait = 0.0
    last_exception: BaseException | None = None
    last_status_code: int | None = None

    while attempts < max_attempts:
        attempts += 1
        kwargs = dict(request_kwargs)
        try:
            _apply_deadline_timeout(kwargs, deadline=deadline, monotonic=monotonic)
        except TimeoutError as exc:
            raise _retry_exhausted(
                "LLM retry deadline expired",
                attempts=attempts,
                total_wait=total_wait,
                last_exception=last_exception or exc,
                last_status_code=last_status_code,
            ) from exc
        try:
            response = call_fn(**kwargs)
        except Exception as exc:  # noqa: BLE001 - policy decides what can retry.
            status_code = _status_code_from_error(exc)
            last_exception = exc
            last_status_code = status_code
            if not is_retryable_llm_error(exc):
                raise
            if attempts >= max_attempts:
                raise _retry_exhausted(
                    "LLM retry exhausted max_attempts",
                    attempts=attempts,
                    total_wait=total_wait,
                    last_exception=exc,
                    last_status_code=status_code,
                ) from exc
            wait_s = _retry_wait_seconds(
                exc,
                attempt_index=attempts - 1,
                base_backoff=base_backoff,
                max_backoff=max_backoff,
                random_uniform=random_uniform,
            )
            if total_wait + wait_s > max_total_wait:
                raise _retry_exhausted(
                    "LLM retry would exceed max_total_wait",
                    attempts=attempts,
                    total_wait=total_wait,
                    last_exception=exc,
                    last_status_code=status_code,
                ) from exc
            if deadline is not None and monotonic() + wait_s > deadline:
                raise _retry_exhausted(
                    "LLM retry would exceed deadline",
                    attempts=attempts,
                    total_wait=total_wait,
                    last_exception=exc,
                    last_status_code=status_code,
                ) from exc
            sleep_func(wait_s)
            total_wait += wait_s
            continue

        retry_count = attempts - 1
        final = "ok_after_retry" if retry_count else "ok"
        return ResilientChatCompletionResult(
            response=response,
            telemetry=LLMRetryTelemetry(
                attempts=attempts,
                retry_count=retry_count,
                total_retry_wait_ms=round(total_wait * 1000),
                final=final,
                last_error_type=(
                    type(last_exception).__name__ if last_exception is not None else None
                ),
                last_status_code=last_status_code,
            ),
            request_kwargs=kwargs,
        )

    # Defensive only: the loop returns or raises on every reachable path.
    raise RuntimeError("_resilient_completion exhausted without terminal state")


def is_retryable_llm_error(exc: BaseException) -> bool:
    status_code = _status_code_from_error(exc)
    if status_code is not None:
        if status_code in _RETRYABLE_STATUS_CODES:
            return True
        if 400 <= status_code < 500:
            return False
        return 500 <= status_code < 600
    if type(exc).__name__ in _RETRYABLE_EXCEPTION_NAMES:
        return True
    if isinstance(exc, (TimeoutError, ConnectionError, ssl.SSLError)):
        return True
    if isinstance(exc, OSError):
        if exc.errno in _RETRYABLE_ERRNOS:
            return True
        msg = str(exc).lower()
        return any(
            needle in msg
            for needle in (
                "connection reset",
                "connection aborted",
                "timed out",
                "timeout",
                "eof occurred",
                "tls",
                "ssl",
            )
        )
    return False


def get_openai():
    """Return the imported openai module. Exits the process with a helpful
    error if openai is not installed."""
    global _openai_module
    if _openai_module is None:
        try:
            import openai as _oai
        except ImportError:
            sys.exit("ERROR: openai SDK not installed. Run: pip install openai>=1.30.0")
        _openai_module = _oai
    return _openai_module


def _retry_exhausted(
    message: str,
    *,
    attempts: int,
    total_wait: float,
    last_exception: BaseException,
    last_status_code: int | None,
) -> LLMRetryExhaustedError:
    return LLMRetryExhaustedError(
        message,
        telemetry=LLMRetryTelemetry(
            attempts=attempts,
            retry_count=max(0, attempts - 1),
            total_retry_wait_ms=round(total_wait * 1000),
            final="failed_after_max",
            last_error_type=type(last_exception).__name__,
            last_status_code=last_status_code,
        ),
        last_exception=last_exception,
    )


def _apply_deadline_timeout(
    kwargs: dict[str, Any],
    *,
    deadline: float | None,
    monotonic: Callable[[], float],
) -> None:
    if deadline is None:
        return
    remaining = deadline - monotonic()
    if remaining <= 0:
        raise TimeoutError("LLM request deadline already expired")
    existing = _timeout_seconds(kwargs.get("timeout"))
    if existing is None or remaining < existing:
        kwargs["timeout"] = max(0.1, remaining)


def _timeout_seconds(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _retry_wait_seconds(
    exc: BaseException,
    *,
    attempt_index: int,
    base_backoff: float,
    max_backoff: float,
    random_uniform: Callable[[float, float], float],
) -> float:
    backoff = min(max_backoff, base_backoff * (2**attempt_index))
    jitter = random_uniform(0.0, backoff) if backoff > 0 else 0.0
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return max(retry_after, backoff)
    return jitter


def _retry_after_seconds(exc: BaseException) -> float | None:
    headers = _headers_from_error(exc)
    value = _header_get(headers, "retry-after")
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        parsed = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return max(0.0, (parsed - datetime.now(UTC)).total_seconds())


def _headers_from_error(exc: BaseException) -> Any:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        return headers
    return getattr(exc, "headers", None)


def _header_get(headers: Any, name: str) -> Any:
    if headers is None:
        return None
    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        if value is not None:
            return value
        return getter(name.title())
    if isinstance(headers, Mapping):
        lowered = name.lower()
        for key, value in headers.items():
            if str(key).lower() == lowered:
                return value
    return None


def _status_code_from_error(exc: BaseException) -> int | None:
    for candidate in (
        getattr(exc, "status_code", None),
        getattr(getattr(exc, "response", None), "status_code", None),
    ):
        if isinstance(candidate, int) and not isinstance(candidate, bool):
            return candidate
    return None


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed


def _non_negative_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be non-negative")
    parsed = float(value)
    if parsed < 0:
        raise ValueError(f"{name} must be non-negative")
    return parsed


__all__ = [
    "LLMRetryExhaustedError",
    "LLMRetryTelemetry",
    "ResilientChatCompletionResult",
    "get_openai",
    "is_retryable_llm_error",
    "resilient_chat_completion",
    "resilient_responses_completion",
]
