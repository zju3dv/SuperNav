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

"""Opt-in provider visual round-trip validation for carried image context."""

from __future__ import annotations

import base64
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from io import BytesIO
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.messages.base import BaseMessage
from PIL import Image as PILImage, ImageDraw

from dimos.agents.harness.context_manager import ConversationContextManager
from dimos.agents.harness.image_blocks import is_image_block


class VisionRoundTripModel(Protocol):
    """Minimal model interface needed by visual round-trip validation."""

    def invoke(self, messages: list[BaseMessage]) -> Any: ...


@dataclass(frozen=True)
class VisualRoundTripImage:
    """Image payload used as the earlier-turn visual evidence."""

    block: dict[str, Any]
    source: str = "synthetic"
    artifact_id: str | None = None
    description: str | None = None

    def model_dump(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("block", None)
        payload["block_type"] = str(self.block.get("type") or "")
        return payload


@dataclass(frozen=True)
class VisualRoundTripConfig:
    """Configuration for an explicit provider vision round-trip check."""

    target_provider: str | None = None
    target_model: str | None = None
    target_base_url: str | None = None
    expected_answer: str = "red"
    square_color: str = "red"
    max_image_blocks: int = 1
    max_context_tokens: int | None = 64000
    source_image: VisualRoundTripImage | None = None
    question: str = (
        "In the visual evidence from the earlier observation, what is the color "
        "of the large square? Answer exactly one lowercase color word."
    )


@dataclass(frozen=True)
class VisualRoundTripReport:
    """Result of a provider/model visual round-trip check."""

    ok: bool
    answer: str
    expected_answer: str
    images_included: list[dict[str, Any]]
    estimated_tokens: int
    mode: str = "live"
    target_provider: str | None = None
    target_model: str | None = None
    target_base_url: str | None = None
    checks: list[dict[str, Any]] = field(default_factory=list)
    source_image: dict[str, Any] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)

    def render_text(self) -> str:
        lines = [f"Provider visual round-trip validation ({self.mode})"]
        for check in self.checks:
            marker = "PASS" if check.get("passed") else "FAIL"
            lines.append(f"[{marker}] {check.get('name')}: {check.get('message')}")
        if self.target_provider or self.target_model:
            lines.append(
                "Target: "
                f"provider={self.target_provider or '(default)'} "
                f"model={self.target_model or '(default)'}"
            )
        if self.target_base_url:
            lines.append(f"Endpoint: {self.target_base_url}")
        if self.source_image:
            lines.append(f"Source: {_source_image_summary(self.source_image)}")
        lines.append(f"Answer: {self.answer or '(empty)'}")
        lines.append(f"Overall: {'PASS' if self.ok else 'FAIL'}")
        return "\n".join(lines)


@dataclass(frozen=True)
class VisualRoundTripPreparedRequest:
    """Provider-targeted request assembled for visual round-trip validation."""

    messages: list[BaseMessage]
    images_included: list[dict[str, Any]]
    estimated_tokens: int
    checks: list[dict[str, Any]]
    source_image: VisualRoundTripImage


def prepare_visual_roundtrip_request(
    config: VisualRoundTripConfig,
) -> VisualRoundTripPreparedRequest:
    """Assemble and validate the model request without calling the provider."""

    manager = ConversationContextManager(
        max_context_tokens=config.max_context_tokens,
        max_image_blocks=config.max_image_blocks,
        target_provider=config.target_provider,
        target_model=config.target_model,
    )
    source_image = _source_image(config)
    history = _roundtrip_history(manager, config, source_image)
    assembly = manager.assemble(history)
    checks = [
        _source_image_check(source_image),
        _image_inclusion_check(assembly.images_included),
    ]
    if config.target_provider:
        checks.append(_provider_placement_check(assembly.images_included, config.target_provider))
    return VisualRoundTripPreparedRequest(
        messages=assembly.messages,
        images_included=assembly.images_included,
        estimated_tokens=assembly.estimated_tokens,
        checks=checks,
        source_image=source_image,
    )


def preflight_visual_roundtrip(config: VisualRoundTripConfig) -> VisualRoundTripReport:
    """Validate visual request assembly without issuing a model/provider call."""

    prepared = prepare_visual_roundtrip_request(config)
    checks = [
        *prepared.checks,
        _provider_call_skipped_check(all(check["passed"] for check in prepared.checks)),
    ]
    return VisualRoundTripReport(
        ok=all(check["passed"] for check in checks),
        answer="",
        expected_answer=config.expected_answer,
        images_included=prepared.images_included,
        estimated_tokens=prepared.estimated_tokens,
        mode="preflight",
        target_provider=config.target_provider,
        target_model=config.target_model,
        target_base_url=_safe_base_url_summary(config.target_base_url),
        checks=checks,
        source_image=prepared.source_image.model_dump(),
    )


def run_visual_roundtrip(
    model: VisionRoundTripModel,
    config: VisualRoundTripConfig,
) -> VisualRoundTripReport:
    """Run a cross-turn image visibility check against a model object."""

    prepared = prepare_visual_roundtrip_request(config)
    checks = list(prepared.checks)
    if not all(check["passed"] for check in checks):
        return VisualRoundTripReport(
            ok=False,
            answer="",
            expected_answer=config.expected_answer,
            images_included=prepared.images_included,
            estimated_tokens=prepared.estimated_tokens,
            target_provider=config.target_provider,
            target_model=config.target_model,
            target_base_url=_safe_base_url_summary(config.target_base_url),
            checks=checks,
            source_image=prepared.source_image.model_dump(),
        )

    response = model.invoke(prepared.messages)
    answer = _message_text(response).strip()
    checks.append(_answer_check(answer, config.expected_answer))
    return VisualRoundTripReport(
        ok=all(check["passed"] for check in checks),
        answer=answer,
        expected_answer=config.expected_answer,
        images_included=prepared.images_included,
        estimated_tokens=prepared.estimated_tokens,
        target_provider=config.target_provider,
        target_model=config.target_model,
        target_base_url=_safe_base_url_summary(config.target_base_url),
        checks=checks,
        source_image=prepared.source_image.model_dump(),
    )


def make_roundtrip_model(
    *,
    provider: str | None,
    model: str,
    api_key: str | None,
    base_url: str | None,
) -> VisionRoundTripModel:
    """Create the supported live model client for visual round-trip validation."""

    provider_name = (provider or "openai-compatible").strip().lower().replace("_", "-")
    if provider_name not in {"openai", "openai-compatible"}:
        raise ValueError(
            "Live visual round-trip currently supports OpenAI-compatible chat models only."
        )

    from langchain_openai import ChatOpenAI

    kwargs: dict[str, Any] = {"model": model}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)


def _roundtrip_history(
    manager: ConversationContextManager,
    config: VisualRoundTripConfig,
    source_image: VisualRoundTripImage,
) -> list[BaseMessage]:
    observed = manager.normalize_message(
        ToolMessage(
            content=[
                {"type": "text", "text": _observation_text(source_image)},
                deepcopy(source_image.block),
            ],
            tool_call_id=_tool_call_id(source_image),
        ),
        source="visual_roundtrip",
    )
    return [
        HumanMessage(content="Observe this image for a later question. Do not answer yet."),
        observed,
        HumanMessage(content=config.question),
    ]


def _source_image(config: VisualRoundTripConfig) -> VisualRoundTripImage:
    if config.source_image is not None:
        if not is_image_block(config.source_image.block):
            raise ValueError("Visual round-trip source_image must be an image content block.")
        return config.source_image
    return VisualRoundTripImage(
        block={
            "type": "image_url",
            "image_url": {
                "url": _solid_square_data_url(config.square_color),
                "detail": "low",
            },
        },
        source="synthetic",
        description=f"{config.square_color} square",
    )


def _observation_text(source_image: VisualRoundTripImage) -> str:
    if source_image.artifact_id:
        return (
            "Persisted visual observation for provider validation "
            f"from artifact {source_image.artifact_id}."
        )
    if source_image.description:
        return f"Visual observation for provider validation: {source_image.description}."
    return f"Visual observation for provider validation from {source_image.source}."


def _tool_call_id(source_image: VisualRoundTripImage) -> str:
    if source_image.artifact_id:
        return f"visual-roundtrip-{source_image.artifact_id}"
    return "visual-roundtrip-observe"


def _solid_square_data_url(color: str) -> str:
    image = PILImage.new("RGB", (256, 256), color="white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((32, 32, 224, 224), fill=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        return "\n".join(parts)
    return str(content)


def _source_image_check(source_image: VisualRoundTripImage) -> dict[str, Any]:
    details = source_image.model_dump()
    return {
        "name": "source_image",
        "passed": True,
        "message": (
            f"Using image source '{source_image.source}'"
            + (f" artifact {source_image.artifact_id}." if source_image.artifact_id else ".")
        ),
        "details": details,
    }


def _image_inclusion_check(images_included: list[dict[str, Any]]) -> dict[str, Any]:
    if images_included:
        return {
            "name": "model_visible_image",
            "passed": True,
            "message": f"Request assembly included {len(images_included)} model-visible image(s).",
            "details": {"images_included": images_included},
        }
    return {
        "name": "model_visible_image",
        "passed": False,
        "message": "Request assembly did not include any model-visible image.",
        "details": {},
    }


def _provider_placement_check(
    images_included: list[dict[str, Any]],
    target_provider: str,
) -> dict[str, Any]:
    undocumented = [
        {
            "artifact_id": image.get("artifact_id"),
            "placement_strategy": image.get("placement_strategy"),
            "documented_provider_input": image.get("documented_provider_input"),
            "target_provider": image.get("target_provider"),
        }
        for image in images_included
        if image.get("documented_provider_input") is not True
    ]
    if images_included and not undocumented:
        return {
            "name": "provider_image_placement",
            "passed": True,
            "message": f"Image placement is documented for provider '{target_provider}'.",
            "details": {"images_included": images_included},
        }
    return {
        "name": "provider_image_placement",
        "passed": False,
        "message": (
            f"No documented image placement was assembled for provider '{target_provider}'."
        ),
        "details": {"undocumented": undocumented, "images_included": images_included},
    }


def _provider_call_skipped_check(preflight_ok: bool) -> dict[str, Any]:
    return {
        "name": "provider_call",
        "passed": preflight_ok,
        "message": (
            "Skipped live provider call; request assembly preflight passed."
            if preflight_ok
            else "Skipped live provider call because request assembly preflight failed."
        ),
        "details": {"skipped": True},
    }


def _answer_check(answer: str, expected: str) -> dict[str, Any]:
    normalized_answer = answer.strip().lower().strip(".! ")
    normalized_expected = expected.strip().lower()
    passed = normalized_answer == normalized_expected
    return {
        "name": "provider_answer",
        "passed": passed,
        "message": (
            f"Model answered expected value '{normalized_expected}'."
            if passed
            else f"Expected '{normalized_expected}', got '{answer}'."
        ),
        "details": {"answer": answer, "expected": expected},
    }


def _source_image_summary(source_image: dict[str, Any]) -> str:
    parts = [str(source_image.get("source") or "unknown")]
    artifact_id = source_image.get("artifact_id")
    if artifact_id:
        parts.append(f"artifact={artifact_id}")
    description = source_image.get("description")
    if description:
        parts.append(str(description))
    return " ".join(parts)


def _safe_base_url_summary(base_url: str | None) -> str | None:
    if not base_url:
        return None
    parsed = urlsplit(base_url)
    if not parsed.scheme or not parsed.netloc:
        return "(custom endpoint)"
    netloc = parsed.netloc.rsplit("@", maxsplit=1)[-1]
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))
