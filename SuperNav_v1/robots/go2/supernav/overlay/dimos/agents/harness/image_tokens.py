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

"""Provider/model-aware image token estimates for context assembly."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

DEFAULT_UNKNOWN_IMAGE_TOKENS = 1500
DEFAULT_MAX_IMAGE_TOKENS = 1600
ANTHROPIC_MAX_IMAGE_PIXELS = 1_150_000
ANTHROPIC_MAX_LONG_EDGE = 1568
OPENAI_PATCH_SIZE = 32
OPENAI_PATCH_BUDGET = 1536


@dataclass(frozen=True)
class ImageTokenEstimate:
    """Traceable estimate for one model-visible image block."""

    tokens: int
    formula: str
    provider_family: str
    model_family: str
    detail: str | None
    width_px: int | None
    height_px: int | None
    metadata: dict[str, Any]

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


def estimate_image_tokens(
    width: int | None,
    height: int | None,
    *,
    target_provider: str | None = None,
    target_model: str | None = None,
    detail: str | None = None,
) -> ImageTokenEstimate:
    """Estimate image tokens from pixels and configured provider/model.

    Unknown providers fall back to the older DimOS neutral estimate so callers
    get a stable upper-ish bound instead of silently reverting to base64 length.
    """

    provider = _normalize(target_provider or "")
    model = _normalize_model(target_model)
    detail = _normalize_detail(detail)

    if width is None or height is None:
        return ImageTokenEstimate(
            tokens=DEFAULT_UNKNOWN_IMAGE_TOKENS,
            formula="unknown_dimensions_default",
            provider_family=_provider_family(provider, model),
            model_family=model or "unknown",
            detail=detail,
            width_px=None,
            height_px=None,
            metadata={},
        )

    width = max(1, int(width))
    height = max(1, int(height))
    if _is_openai_provider(provider, model):
        return _estimate_openai_tokens(width, height, model=model, detail=detail)
    if _is_anthropic_provider(provider, model):
        return _estimate_anthropic_tokens(width, height, model=model, detail=detail)
    return _estimate_neutral_tokens(width, height, provider=provider, model=model, detail=detail)


def _estimate_openai_tokens(
    width: int,
    height: int,
    *,
    model: str,
    detail: str | None,
) -> ImageTokenEstimate:
    patch_multiplier = _openai_patch_multiplier(model)
    if patch_multiplier is not None:
        patches, metadata = _openai_patch_count(width, height)
        return ImageTokenEstimate(
            tokens=math.ceil(patches * patch_multiplier),
            formula="openai_32px_patch",
            provider_family="openai",
            model_family=model or "openai_patch",
            detail=detail,
            width_px=width,
            height_px=height,
            metadata={**metadata, "patch_multiplier": patch_multiplier},
        )

    base_tokens, tile_tokens, family = _openai_tile_costs(model)
    if detail == "low":
        return ImageTokenEstimate(
            tokens=base_tokens,
            formula="openai_tile_low_detail",
            provider_family="openai",
            model_family=family,
            detail=detail,
            width_px=width,
            height_px=height,
            metadata={"base_tokens": base_tokens, "tile_tokens": tile_tokens},
        )

    tiles, metadata = _openai_tile_count(width, height)
    return ImageTokenEstimate(
        tokens=base_tokens + tile_tokens * tiles,
        formula="openai_512px_tile_high_detail",
        provider_family="openai",
        model_family=family,
        detail=detail or "high",
        width_px=width,
        height_px=height,
        metadata={**metadata, "base_tokens": base_tokens, "tile_tokens": tile_tokens},
    )


def _estimate_anthropic_tokens(
    width: int,
    height: int,
    *,
    model: str,
    detail: str | None,
) -> ImageTokenEstimate:
    resized_width, resized_height, scale = _anthropic_resized_dimensions(width, height)
    return ImageTokenEstimate(
        tokens=math.ceil((resized_width * resized_height) / 750),
        formula="anthropic_area_div_750",
        provider_family="anthropic",
        model_family=model or "claude",
        detail=detail,
        width_px=width,
        height_px=height,
        metadata={
            "resized_width_px": resized_width,
            "resized_height_px": resized_height,
            "scale": scale,
        },
    )


def _estimate_neutral_tokens(
    width: int,
    height: int,
    *,
    provider: str,
    model: str,
    detail: str | None,
) -> ImageTokenEstimate:
    tokens = max(1, min(DEFAULT_MAX_IMAGE_TOKENS, math.ceil((width * height) / 750)))
    return ImageTokenEstimate(
        tokens=tokens,
        formula="neutral_area_div_750_capped",
        provider_family=_provider_family(provider, model),
        model_family=model or "unknown",
        detail=detail,
        width_px=width,
        height_px=height,
        metadata={"max_tokens": DEFAULT_MAX_IMAGE_TOKENS},
    )


def _openai_patch_count(width: int, height: int) -> tuple[int, dict[str, Any]]:
    raw_patches_w = math.ceil(width / OPENAI_PATCH_SIZE)
    raw_patches_h = math.ceil(height / OPENAI_PATCH_SIZE)
    raw_patches = raw_patches_w * raw_patches_h
    if raw_patches <= OPENAI_PATCH_BUDGET:
        return raw_patches, {
            "raw_patches": raw_patches,
            "patches_w": raw_patches_w,
            "patches_h": raw_patches_h,
            "resized_width_px": width,
            "resized_height_px": height,
            "patch_budget": OPENAI_PATCH_BUDGET,
        }

    shrink = math.sqrt((OPENAI_PATCH_SIZE**2 * OPENAI_PATCH_BUDGET) / (width * height))
    resized_width = width * shrink
    resized_height = height * shrink
    width_patch_float = resized_width / OPENAI_PATCH_SIZE
    height_patch_float = resized_height / OPENAI_PATCH_SIZE
    whole_patch_adjust = min(
        math.floor(width_patch_float) / width_patch_float,
        math.floor(height_patch_float) / height_patch_float,
    )
    resized_width *= whole_patch_adjust
    resized_height *= whole_patch_adjust
    patches_w = max(1, math.ceil(resized_width / OPENAI_PATCH_SIZE))
    patches_h = max(1, math.ceil(resized_height / OPENAI_PATCH_SIZE))
    patches = min(OPENAI_PATCH_BUDGET, patches_w * patches_h)
    return patches, {
        "raw_patches": raw_patches,
        "patches_w": patches_w,
        "patches_h": patches_h,
        "resized_width_px": round(resized_width),
        "resized_height_px": round(resized_height),
        "patch_budget": OPENAI_PATCH_BUDGET,
        "scale": shrink * whole_patch_adjust,
    }


def _openai_tile_count(width: int, height: int) -> tuple[int, dict[str, Any]]:
    scale_to_2048 = min(1.0, 2048 / max(width, height))
    resized_width = width * scale_to_2048
    resized_height = height * scale_to_2048
    shortest = max(1.0, min(resized_width, resized_height))
    scale_to_768 = 768 / shortest
    resized_width *= scale_to_768
    resized_height *= scale_to_768
    tiles_w = max(1, math.ceil(resized_width / 512))
    tiles_h = max(1, math.ceil(resized_height / 512))
    return tiles_w * tiles_h, {
        "tiles_w": tiles_w,
        "tiles_h": tiles_h,
        "resized_width_px": round(resized_width),
        "resized_height_px": round(resized_height),
        "scale_to_2048": scale_to_2048,
        "scale_to_768": scale_to_768,
    }


def _anthropic_resized_dimensions(width: int, height: int) -> tuple[int, int, float]:
    scale = min(1.0, ANTHROPIC_MAX_LONG_EDGE / max(width, height))
    resized_width = width * scale
    resized_height = height * scale
    pixels = resized_width * resized_height
    if pixels > ANTHROPIC_MAX_IMAGE_PIXELS:
        area_scale = math.sqrt(ANTHROPIC_MAX_IMAGE_PIXELS / pixels)
        scale *= area_scale
        resized_width *= area_scale
        resized_height *= area_scale
    return max(1, round(resized_width)), max(1, round(resized_height)), scale


def _openai_patch_multiplier(model: str) -> float | None:
    if model in {"gpt-5-mini", "gpt-4.1-mini"} or (
        model.endswith("-mini") and model.startswith(("gpt-5", "gpt-4.1"))
    ):
        return 1.62
    if model in {"gpt-5-nano", "gpt-4.1-nano"} or (
        model.endswith("-nano") and model.startswith(("gpt-5", "gpt-4.1"))
    ):
        return 2.46
    if model == "o4-mini":
        return 1.72
    return None


def _openai_tile_costs(model: str) -> tuple[int, int, str]:
    if model in {"gpt-5", "gpt-5-chat-latest"}:
        return 70, 140, model
    if model in {"o1", "o1-pro", "o3"}:
        return 75, 150, model
    if model == "computer-use-preview":
        return 65, 129, model
    if model == "gpt-4o-mini":
        return 2833, 5667, model
    if model.startswith("gpt-4.1") or model.startswith("gpt-4.5"):
        return 85, 170, "gpt-4.1/4.5"
    return 85, 170, model or "gpt-4o"


def _is_openai_provider(provider: str, model: str) -> bool:
    return (
        provider.startswith("openai")
        or model.startswith(("gpt-", "chatgpt-", "o1", "o3", "o4"))
        or model == "computer-use-preview"
    )


def _is_anthropic_provider(provider: str, model: str) -> bool:
    return provider.startswith(("anthropic", "claude")) or model.startswith("claude")


def _provider_family(provider: str, model: str) -> str:
    if _is_openai_provider(provider, model):
        return "openai"
    if _is_anthropic_provider(provider, model):
        return "anthropic"
    return provider or "neutral"


def _normalize(value: str) -> str:
    return value.strip().lower().replace("_", "-")


def _normalize_model(model: str | None) -> str:
    normalized = _normalize(model or "")
    if normalized.startswith("openai:"):
        normalized = normalized.split(":", 1)[1]
    if normalized.startswith("anthropic:"):
        normalized = normalized.split(":", 1)[1]
    return normalized


def _normalize_detail(detail: str | None) -> str | None:
    normalized = _normalize(detail or "")
    return normalized if normalized in {"low", "high", "auto"} else None
