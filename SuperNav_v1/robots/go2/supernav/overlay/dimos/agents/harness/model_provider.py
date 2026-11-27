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

"""Native-agent model/provider resolution for DimOS harness runs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlsplit, urlunsplit


@dataclass(frozen=True)
class NativeAgentModelConfig:
    """Model configuration used by the native LangGraph MCP client."""

    model: str
    model_provider: str | None = None
    model_base_url: str | None = None
    model_api_key: Any | None = None
    model_fixture: str | None = None


@dataclass(frozen=True)
class NativeAgentModelReadiness:
    """Display-safe model/provider readiness for native harness operators."""

    model: str
    model_provider: str | None
    context_target_provider: str
    base_url_configured: bool
    base_url_summary: str | None
    api_key_configured: bool
    fixture_configured: bool
    uses_openai_compatible_client: bool
    ready_for_native_agent: bool
    ready_for_provider_vision_roundtrip: bool
    checks: list[dict[str, Any]] = field(default_factory=list)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    def render_text(self) -> str:
        lines = ["Native agent provider readiness"]
        lines.append(f"Model: {self.model or '(empty)'}")
        lines.append(f"Provider: {self.model_provider or '(auto)'}")
        lines.append(f"Context target: {self.context_target_provider}")
        lines.append(f"Base URL configured: {'yes' if self.base_url_configured else 'no'}")
        if self.base_url_summary:
            lines.append(f"Endpoint: {self.base_url_summary}")
        lines.append(f"API key configured: {'yes' if self.api_key_configured else 'no'}")
        if self.fixture_configured:
            lines.append("Fixture: configured")
        for check in self.checks:
            marker = "PASS" if check.get("passed") else "FAIL"
            lines.append(f"[{marker}] {check.get('name')}: {check.get('message')}")
        lines.append("Native agent: " + ("READY" if self.ready_for_native_agent else "NOT READY"))
        lines.append(
            "Vision round-trip: "
            + ("READY" if self.ready_for_provider_vision_roundtrip else "NOT READY")
        )
        return "\n".join(lines)


class NativeAgentModelNotReadyError(RuntimeError):
    """Raised when native-agent model config cannot build a live provider client."""

    def __init__(self, readiness: NativeAgentModelReadiness) -> None:
        self.readiness = readiness
        super().__init__(native_agent_model_not_ready_message(readiness))


ModelFactory = Callable[[dict[str, Any]], Any]
FixtureFactory = Callable[[str], Any]


def resolve_context_target_provider(config: NativeAgentModelConfig) -> str:
    """Return the provider family used for multimodal request assembly."""

    provider = _normalized_provider(config.model_provider)
    if provider:
        return provider
    if config.model_base_url is not None or config.model_api_key is not None:
        return "openai-compatible"

    model = config.model.lower()
    if model.startswith("openai:") or model.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4")):
        return "openai"
    if ":" in model:
        return model.split(":", 1)[0]
    return "langchain"


def build_native_agent_model(
    config: NativeAgentModelConfig,
    *,
    fixture_factory: FixtureFactory | None = None,
    openai_factory: ModelFactory | None = None,
) -> str | Any:
    """Build the LangChain model handle for the native MCP client."""

    readiness = native_agent_model_readiness(config)
    if not readiness.ready_for_native_agent:
        raise NativeAgentModelNotReadyError(readiness)

    if config.model_fixture is not None:
        factory = fixture_factory or _default_fixture_factory
        return factory(config.model_fixture)

    if not _uses_openai_compatible_client(config):
        return config.model

    factory = openai_factory or _default_openai_factory
    kwargs: dict[str, Any] = {"model": config.model}
    if config.model_base_url is not None:
        kwargs["base_url"] = config.model_base_url
    if config.model_api_key is not None:
        kwargs["api_key"] = config.model_api_key
    return factory(kwargs)


def native_agent_model_readiness(config: NativeAgentModelConfig) -> NativeAgentModelReadiness:
    """Return a display-safe readiness summary without constructing a model client."""

    model = config.model.strip()
    provider = _normalized_provider(config.model_provider) or None
    context_target_provider = resolve_context_target_provider(config)
    base_url_configured = bool((config.model_base_url or "").strip())
    api_key_configured = _secret_is_configured(config.model_api_key)
    fixture_configured = bool((config.model_fixture or "").strip())
    uses_openai_client = _uses_openai_compatible_client(config)
    requires_openai_key = uses_openai_client or context_target_provider == "openai"

    checks: list[dict[str, Any]] = [
        _readiness_check(
            "model",
            bool(model),
            f"Model is configured as '{model}'." if model else "No model name is configured.",
        )
    ]

    if fixture_configured:
        client_ready = bool(model)
        checks.append(
            _readiness_check(
                "client",
                client_ready,
                "Fixture model is configured; no provider key is required."
                if client_ready
                else "Fixture model is configured, but the model name is empty.",
                {"fixture": True},
            )
        )
    elif requires_openai_key:
        client_ready = bool(model) and api_key_configured
        checks.append(
            _readiness_check(
                "client",
                client_ready,
                "OpenAI-compatible client has a model and API key."
                if client_ready
                else (
                    "OpenAI/OpenAI-compatible model needs DIMOS_AGENT_API_KEY or "
                    "OPENAI_API_KEY before the native agent can call the provider."
                ),
                {
                    "base_url_configured": base_url_configured,
                    "api_key_configured": api_key_configured,
                },
            )
        )
    else:
        client_ready = bool(model)
        checks.append(
            _readiness_check(
                "client",
                client_ready,
                "Model string will be delegated to LangChain provider resolution."
                if client_ready
                else "No model string is available for LangChain provider resolution.",
                {"delegated_to_langchain": True},
            )
        )

    roundtrip_supported = context_target_provider in {"openai", "openai-compatible"}
    roundtrip_ready = bool(model) and api_key_configured and roundtrip_supported
    checks.append(
        _readiness_check(
            "provider_vision_roundtrip",
            roundtrip_ready,
            "Configured provider can run `dimos agent-vision-roundtrip`."
            if roundtrip_ready
            else _roundtrip_not_ready_message(
                api_key_configured=api_key_configured,
                roundtrip_supported=roundtrip_supported,
            ),
            {
                "supported_provider": roundtrip_supported,
                "api_key_configured": api_key_configured,
            },
        )
    )

    return NativeAgentModelReadiness(
        model=model,
        model_provider=provider,
        context_target_provider=context_target_provider,
        base_url_configured=base_url_configured,
        base_url_summary=_safe_base_url_summary(config.model_base_url),
        api_key_configured=api_key_configured,
        fixture_configured=fixture_configured,
        uses_openai_compatible_client=uses_openai_client,
        ready_for_native_agent=client_ready,
        ready_for_provider_vision_roundtrip=roundtrip_ready,
        checks=checks,
    )


def native_agent_model_not_ready_message(readiness: NativeAgentModelReadiness) -> str:
    """Return a compact display-safe error for fail-fast startup paths."""

    failed = [check for check in readiness.checks if not check.get("passed")]
    if not failed:
        return "Native agent model provider is not ready."
    details = "; ".join(str(check.get("message") or check.get("name")) for check in failed)
    return f"Native agent model provider is not ready: {details}"


def _normalized_provider(provider: str | None) -> str:
    return (provider or "").strip().lower().replace("_", "-")


def _uses_openai_compatible_client(config: NativeAgentModelConfig) -> bool:
    provider = _normalized_provider(config.model_provider)
    return (
        provider in {"openai", "openai-compatible"}
        or config.model_base_url is not None
        or config.model_api_key is not None
    )


def _secret_is_configured(secret: Any | None) -> bool:
    if secret is None:
        return False
    get_secret_value = getattr(secret, "get_secret_value", None)
    if callable(get_secret_value):
        return bool(str(get_secret_value()).strip())
    return bool(str(secret).strip())


def _readiness_check(
    name: str,
    passed: bool,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": passed,
        "message": message,
        "details": details or {},
    }


def _roundtrip_not_ready_message(
    *,
    api_key_configured: bool,
    roundtrip_supported: bool,
) -> str:
    if not roundtrip_supported:
        return "Live visual round-trip currently supports OpenAI-compatible chat models only."
    if not api_key_configured:
        return (
            "Set DIMOS_AGENT_API_KEY or OPENAI_API_KEY before running the live "
            "provider visual round-trip."
        )
    return "Provider visual round-trip is not ready."


def _safe_base_url_summary(base_url: str | None) -> str | None:
    if not base_url:
        return None
    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.netloc:
        return "(custom endpoint)"
    netloc = parsed.netloc.rsplit("@", maxsplit=1)[-1]
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


def _default_fixture_factory(json_path: str) -> Any:
    from dimos.agents.testing import MockModel

    return MockModel(json_path=json_path)


def _default_openai_factory(kwargs: dict[str, Any]) -> Any:
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(**kwargs)
