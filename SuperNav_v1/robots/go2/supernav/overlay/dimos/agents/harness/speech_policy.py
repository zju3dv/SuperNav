# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Native-agent speech gating for non-repetitive Go2 voice output."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from dimos.agents.runtime_policy import AgentRuntimePolicy

_SPEECH_REQUEST_MARKERS = (
    "say aloud",
    "speak",
    "voice",
    "out loud",
    "read aloud",
    "说出来",
    "语音",
    "讲话",
    "播报",
)
SPEECH_REQUESTED_META_KEY = "dimos/speech_requested"
SPEECH_AUTHORIZED_META_KEY = "dimos/speech_authorized"


def text_requests_speech(text: str) -> bool:
    """Return whether user text explicitly asks for robot speech."""

    lowered = text.lower()
    return any(marker in lowered for marker in _SPEECH_REQUEST_MARKERS)


@dataclass
class NativeSpeechGate:
    """Track per-turn speech permission and repeated-speech suppression."""

    speak_allowed_for_turn: bool = False
    last_call_at: float = 0.0
    last_spoken_text: str | None = None
    last_spoken_at: float = 0.0

    def update_for_user_text(self, text: str) -> None:
        """Refresh the per-turn speech allowance from a human user turn."""

        self.speak_allowed_for_turn = text_requests_speech(text)

    def refusal_for_call(
        self,
        args: dict[str, Any],
        *,
        policy: AgentRuntimePolicy,
        rate_limit_s: float,
        dedupe_window_s: float,
        now: float | None = None,
    ) -> str | None:
        """Return a refusal message for a native-agent speak call, or ``None``."""

        text = str(args.get("text") or "")
        now = now if now is not None else time.monotonic()

        if policy.speak_mode == "off":
            return "Speech is disabled by runtime policy. Reply in text instead."

        if policy.speak_mode == "on_request" and not self.speak_allowed_for_turn:
            return (
                "Speech is in on_request mode. Do not call speak unless the user "
                "explicitly asks for audible speech; reply in text instead."
            )

        if text and text == self.last_spoken_text and now - self.last_spoken_at < dedupe_window_s:
            return "Skipped duplicate speech by runtime policy; reply in text instead."

        if now - self.last_call_at < rate_limit_s:
            return "Skipped speech due to runtime speech rate limit; reply in text instead."

        self.last_call_at = now
        self.last_spoken_text = text
        self.last_spoken_at = now
        if policy.speak_mode == "on_request":
            self.speak_allowed_for_turn = False
        return None


@dataclass
class ServerSpeechGate:
    """Server-side speech suppression shared by every MCP client."""

    last_call_at: float = 0.0
    last_spoken_text: str | None = None
    last_spoken_at: float = 0.0

    def refusal_for_call(
        self,
        args: dict[str, Any],
        *,
        policy: AgentRuntimePolicy,
        meta: dict[str, Any] | None,
        rate_limit_s: float,
        dedupe_window_s: float,
        now: float | None = None,
    ) -> str | None:
        """Return a dispatch refusal for a server-side ``speak`` call."""

        text = str(args.get("text") or "")
        now = now if now is not None else time.monotonic()
        meta = meta or {}

        if policy.speak_mode == "off":
            return "Runtime speech policy is off. Tool 'speak' was not executed."

        if policy.speak_mode == "on_request" and not _server_speech_request_present(meta):
            return (
                "Runtime speech policy is on_request. Tool 'speak' requires "
                f"`_meta.{SPEECH_REQUESTED_META_KEY}=true` or a native agent "
                "speech authorization."
            )

        if text and text == self.last_spoken_text and now - self.last_spoken_at < dedupe_window_s:
            return "Skipped duplicate speech by runtime policy. Tool 'speak' was not executed."

        if now - self.last_call_at < rate_limit_s:
            return "Skipped speech due to runtime speech rate limit. Tool 'speak' was not executed."

        self.last_call_at = now
        self.last_spoken_text = text
        self.last_spoken_at = now
        return None


def _server_speech_request_present(meta: dict[str, Any]) -> bool:
    return bool(meta.get(SPEECH_REQUESTED_META_KEY) or meta.get(SPEECH_AUTHORIZED_META_KEY))


@dataclass(frozen=True)
class SpeechGateLimits:
    """Resolved rate and duplicate suppression limits for speech gates."""

    rate_limit_s: float = 2.0
    dedupe_window_s: float = 30.0


def resolve_speech_gate_limits(
    runtime_config: Any | None,
    *,
    rate_limit_s: float | None = None,
    dedupe_window_s: float | None = None,
) -> SpeechGateLimits:
    """Resolve speech-gate limits from explicit overrides or GlobalConfig fields."""

    if runtime_config is None:
        from dimos.core.global_config import global_config

        runtime_config = global_config

    resolved_rate = _nonnegative_float(
        rate_limit_s
        if rate_limit_s is not None
        else getattr(runtime_config, "agent_speak_rate_limit_s", None),
        default=2.0,
    )
    resolved_dedupe = _nonnegative_float(
        dedupe_window_s
        if dedupe_window_s is not None
        else getattr(runtime_config, "agent_speak_dedupe_window_s", None),
        default=30.0,
    )
    return SpeechGateLimits(rate_limit_s=resolved_rate, dedupe_window_s=resolved_dedupe)


def _nonnegative_float(value: Any, *, default: float) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default
