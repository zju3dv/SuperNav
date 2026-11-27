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

"""Provider-aware placement for carried image evidence.

The context manager owns *when* an old visual artifact should be re-injected.
This module owns *where* that image sits for the configured provider/API so the
request trace can prove the model received a documented image-input shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import HumanMessage

from dimos.agents.harness.image_blocks import image_block_data_url, image_block_mime_type

OPENAI_IMAGE_INPUT_DOC = "https://platform.openai.com/docs/guides/images-vision"
ANTHROPIC_VISION_DOC = "https://docs.anthropic.com/en/docs/build-with-claude/vision"
ANTHROPIC_TOOL_USE_DOC = "https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview"


@dataclass(frozen=True)
class ImagePlacementStrategy:
    """Provider/API strategy for one model-visible image injection."""

    name: str
    provider_family: str
    provider_api: str
    placement: str
    provider_role: str
    langchain_message_type: str
    provider_image_block_type: str
    documented_provider_input: bool
    provider_docs: tuple[str, ...]
    tool_result_images_supported: bool = False
    tool_result_replay_used: bool = False
    note: str | None = None

    def metadata(self, *, target_provider: str | None, image_block_type: str) -> dict[str, Any]:
        """Return audit metadata for `request_assembled.images_included`."""

        return {
            "target_provider": target_provider,
            "target_provider_normalized": normalize_target_provider(target_provider),
            "placement_strategy": self.name,
            "provider_family": self.provider_family,
            "provider_api": self.provider_api,
            "placement": self.placement,
            "role": self.provider_role,
            "langchain_message_type": self.langchain_message_type,
            "provider_image_block_type": image_block_type,
            "documented_provider_input": self.documented_provider_input,
            "provider_docs": list(self.provider_docs),
            "tool_result_images_supported": self.tool_result_images_supported,
            "tool_result_replay_used": self.tool_result_replay_used,
            "note": self.note,
        }


@dataclass(frozen=True)
class ImagePlacement:
    """Concrete message and metadata for a re-injected image artifact."""

    message: HumanMessage
    metadata: dict[str, Any]
    content_index: int


def build_recent_image_placement(
    *,
    block: dict[str, Any],
    artifact_id: str,
    target_provider: str | None,
) -> ImagePlacement:
    """Build the provider-visible message used to re-surface an image artifact."""

    strategy = strategy_for_provider(target_provider)
    provider_block = _block_for_strategy(block, strategy)
    text_block = {
        "type": "text",
        "text": (
            f"Recent visual evidence from {artifact_id}. "
            "Use this image when answering the current task."
        ),
    }
    content = [text_block, provider_block]
    content_index = 1
    metadata = strategy.metadata(
        target_provider=target_provider,
        image_block_type=str(provider_block.get("type") or "unknown"),
    )
    metadata["content_index"] = content_index
    return ImagePlacement(
        message=HumanMessage(content=content),
        metadata=metadata,
        content_index=content_index,
    )


def strategy_for_provider(target_provider: str | None) -> ImagePlacementStrategy:
    """Return the image placement strategy for the configured provider/API."""

    provider = normalize_target_provider(target_provider)
    if provider in {"openai-responses", "openai-compatible-responses"}:
        return ImagePlacementStrategy(
            name="openai_responses_user_input_image",
            provider_family="openai",
            provider_api="responses",
            placement="human_message",
            provider_role="user",
            langchain_message_type="human",
            provider_image_block_type="input_image",
            documented_provider_input=True,
            provider_docs=(OPENAI_IMAGE_INPUT_DOC,),
        )

    if provider.startswith("openai") or provider in {"openai-compatible"}:
        return ImagePlacementStrategy(
            name="openai_chat_user_image_url",
            provider_family="openai",
            provider_api="chat_completions",
            placement="human_message",
            provider_role="user",
            langchain_message_type="human",
            provider_image_block_type="image_url",
            documented_provider_input=True,
            provider_docs=(OPENAI_IMAGE_INPUT_DOC,),
        )

    if provider in {"anthropic", "claude"} or provider.startswith(("anthropic", "claude")):
        return ImagePlacementStrategy(
            name="anthropic_messages_user_image",
            provider_family="anthropic",
            provider_api="messages",
            placement="human_message",
            provider_role="user",
            langchain_message_type="human",
            provider_image_block_type="image",
            documented_provider_input=True,
            provider_docs=(ANTHROPIC_VISION_DOC, ANTHROPIC_TOOL_USE_DOC),
            tool_result_images_supported=True,
            tool_result_replay_used=False,
            note=(
                "Re-surfaced as a user image near the current turn; not replayed "
                "as a tool_result because historical tool_result adjacency is not "
                "guaranteed during context assembly."
            ),
        )

    return ImagePlacementStrategy(
        name="langchain_user_image_fallback",
        provider_family="langchain",
        provider_api="chat_messages",
        placement="human_message",
        provider_role="user",
        langchain_message_type="human",
        provider_image_block_type=str(block_type_for_unknown_provider(target_provider)),
        documented_provider_input=False,
        provider_docs=(),
        note="Provider-specific image placement is unknown; preserving the original block.",
    )


def normalize_target_provider(target_provider: str | None) -> str:
    """Normalize provider names used by config and trace metadata."""

    provider = (target_provider or "langchain").strip().lower().replace("_", "-")
    return provider or "langchain"


def block_type_for_unknown_provider(target_provider: str | None) -> str:
    """Return the expected block type label for fallback traces."""

    provider = normalize_target_provider(target_provider)
    if provider in {"openai-responses", "openai-compatible-responses"}:
        return "input_image"
    if provider in {"anthropic", "claude"}:
        return "image"
    return "original"


def _block_for_strategy(
    block: dict[str, Any],
    strategy: ImagePlacementStrategy,
) -> dict[str, Any]:
    if strategy.provider_api == "responses":
        return _to_openai_responses_block(block)
    if strategy.provider_family == "openai":
        return _to_openai_chat_block(block)
    if strategy.provider_family == "anthropic":
        return _to_anthropic_image_block(block)
    return block


def _to_openai_chat_block(block: dict[str, Any]) -> dict[str, Any]:
    if block.get("type") == "image_url" and isinstance(block.get("image_url"), dict):
        return block
    data_url = image_block_data_url(block)
    if data_url is None:
        return block
    return {"type": "image_url", "image_url": {"url": data_url}}


def _to_openai_responses_block(block: dict[str, Any]) -> dict[str, Any]:
    if block.get("type") == "input_image":
        return block
    data_url = image_block_data_url(block)
    if data_url is not None:
        return {"type": "input_image", "image_url": data_url}
    image_url = block.get("image_url")
    if isinstance(image_url, dict) and image_url.get("url"):
        return {"type": "input_image", "image_url": str(image_url["url"])}
    return block


def _to_anthropic_image_block(block: dict[str, Any]) -> dict[str, Any]:
    if block.get("type") == "image" and isinstance(block.get("source"), dict):
        return block
    data_url = image_block_data_url(block)
    if data_url is None or ";base64," not in data_url:
        return block
    mime_type = image_block_mime_type(block) or "image/jpeg"
    encoded = data_url.split(";base64,", 1)[1]
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": mime_type,
            "data": encoded,
        },
    }
