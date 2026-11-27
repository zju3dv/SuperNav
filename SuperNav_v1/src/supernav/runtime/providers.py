"""Run-local API routing for clients; credentials remain environment references."""
from __future__ import annotations

import json
import os
from typing import Any, Mapping


def configured_provider(agent_cfg: Mapping[str, Any]) -> dict[str, Any] | None:
    value = agent_cfg.get("provider")
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("agent.provider must be an object")
    provider = dict(value)
    for key in ("id", "base_url", "env_key"):
        if not isinstance(provider.get(key), str) or not provider[key].strip():
            raise ValueError(f"agent.provider requires {key}")
    if "api_key" in provider or "oauth" in provider:
        raise ValueError("agent.provider credentials must use env_key")
    return provider


def validate_provider_environment(agent_cfg: Mapping[str, Any]) -> None:
    provider = configured_provider(agent_cfg)
    if provider and not os.environ.get(provider["env_key"], "").strip():
        raise ValueError(f"API provider requires {provider['env_key']}; refusing auth fallback")


def client_model(agent_cfg: Mapping[str, Any], name: str | None) -> str | None:
    provider = configured_provider(agent_cfg)
    if provider and name:
        return f"{provider['id']}/{name}"
    return name


def kimi_provider_config(provider: Mapping[str, Any], model: str) -> str:
    quote = json.dumps
    protocol = str(provider.get("type") or "openai_responses")
    if protocol not in {"openai", "openai_responses", "anthropic", "kimi"}:
        raise ValueError(f"unsupported Kimi provider protocol: {protocol}")
    alias = f"{provider['id']}/{model}"
    return "\n".join([
        f"default_model = {quote(alias)}", "telemetry = false", "auto_session_title = false",
        f"[providers.{quote(provider['id'])}]", f"type = {quote(protocol)}",
        f"base_url = {quote(provider['base_url'])}",
        "# API credential and temporary model are injected only into the child environment.",
        f"[models.{quote(alias)}]", f"provider = {quote(provider['id'])}", f"model = {quote(model)}",
        f"max_context_size = {int(provider.get('max_context_size', 200000))}",
        'capabilities = ["image_in", "tool_use", "thinking"]',
        'support_efforts = ["low", "medium", "high"]', 'default_effort = "high"',
        "[loop_control]", "max_attempts_per_step = 1", "",
    ])


def kimi_provider_environment(agent_cfg: Mapping[str, Any], model: str) -> dict[str, str]:
    """Kimi 0.41 supports KIMI_MODEL_* but not the newer api_key_env field."""
    provider = configured_provider(agent_cfg)
    if not provider:
        return {}
    validate_provider_environment(agent_cfg)
    return {
        "KIMI_MODEL_NAME": model,
        "KIMI_MODEL_API_KEY": os.environ[provider["env_key"]],
        "KIMI_MODEL_PROVIDER_TYPE": str(provider.get("type") or "openai_responses"),
        "KIMI_MODEL_BASE_URL": provider["base_url"],
        "KIMI_MODEL_MAX_CONTEXT_SIZE": str(provider.get("max_context_size", 200000)),
        "KIMI_MODEL_CAPABILITIES": "image_in,tool_use,thinking",
        "KIMI_MODEL_THINKING_EFFORT": "high",
    }
