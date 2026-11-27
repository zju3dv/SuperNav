"""Shared image preprocessing for training and serving (must never diverge).

The data engine and inference service import these exact functions so image
sizes and normalization stay identical during training and serving.

Marker rendering lives here too, and deliberately AFTER the resize: a ring
drawn at capture resolution (e.g. a 12px ring on a 512px frame) shrinks to
~2px after the 96×96 policy resize and washes out; rendering at policy
resolution keeps the goal-point conditioning bold and identical between
training and inference.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

POLICY_IMAGE_SIZE = (96, 96)  # (width, height), NoMaD native input
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
POLICY_MARKER_RADIUS_PX = 5  # calibrated for 96px; scales with resolution
POLICY_MARKER_COLOR = (255, 64, 64)
CROP_FRACTION = 0.4  # crop-conditioning window side, fraction of min(H, W)


def _as_size(size) -> "tuple[int, int]":
    """int or (w, h) → (w, h); the M3 resolution knob passes a single int."""
    if isinstance(size, (int, np.integer)):
        return (int(size), int(size))
    return (int(size[0]), int(size[1]))


def _resize_rgb(rgb: np.ndarray, size: "tuple[int, int]") -> np.ndarray:
    """uint8 [H, W, 3] → uint8 [h, w, 3] at the policy resolution."""
    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("expected an [H, W, 3] RGB array")
    image = Image.fromarray(array.astype(np.uint8)).resize(size, Image.BILINEAR)
    return np.array(image)


def _normalize_chw(array_u8: np.ndarray) -> torch.Tensor:
    """uint8 [h, w, 3] → ImageNet-normalized float32 [3, h, w]."""
    tensor = torch.from_numpy(np.ascontiguousarray(array_u8)).float().div_(255.0)
    tensor = tensor.permute(2, 0, 1)
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor - mean) / std


def to_policy_tensor(
    rgb: np.ndarray, size=POLICY_IMAGE_SIZE
) -> torch.Tensor:
    """uint8 [H, W, 3] → normalized float32 [3, h, w] at the policy resolution."""
    return _normalize_chw(_resize_rgb(rgb, _as_size(size)))


def crop_around_point(
    rgb: np.ndarray, point: Sequence[float], size, fraction: float = CROP_FRACTION
) -> np.ndarray:
    """Crop-zoom conditioning: a window centered on the point (clamped inside
    the frame), resized to policy resolution. The crop itself IS the pointer —
    no marker is drawn; the goal always sits at the crop center (edge-clamp
    shifts it, which the encoder can read from context)."""
    array = np.asarray(rgb)
    height, width = array.shape[:2]
    side = max(16, int(round(fraction * min(height, width))))
    cx = int(round(float(point[0]) * (width - 1)))
    cy = int(round(float(point[1]) * (height - 1)))
    x0 = min(max(cx - side // 2, 0), width - side)
    y0 = min(max(cy - side // 2, 0), height - side)
    return _resize_rgb(array[y0 : y0 + side, x0 : x0 + side], _as_size(size))


def render_goal_marker_at_policy_resolution(
    goal_rgb: np.ndarray,
    goal_point: Sequence[float],
    size: "tuple[int, int]" = POLICY_IMAGE_SIZE,
    radius_px: int = POLICY_MARKER_RADIUS_PX,
    color: "tuple[int, int, int]" = POLICY_MARKER_COLOR,
) -> np.ndarray:
    """Resize FIRST, then draw the ring + center dot at policy resolution.

    goal_point is normalized [x, y] with (0, 0) at top-left. Returns a uint8
    [h, w, 3] array. Used by the serving backend and the data engine so the
    conditioning pixels are bit-identical between train and inference.
    """
    size = _as_size(size)
    resized = _resize_rgb(goal_rgb, size)
    width, height = size
    x = int(min(width - 1, max(0, round(float(goal_point[0]) * (width - 1)))))
    y = int(min(height - 1, max(0, round(float(goal_point[1]) * (height - 1)))))
    image = Image.fromarray(resized)
    draw = ImageDraw.Draw(image)
    # Ring size is calibrated for 96px and scales with the resolution knob.
    radius = max(3, int(round(radius_px * width / 96.0)))
    draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius], outline=color, width=2
    )
    draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=color)
    return np.array(image)


def stack_context(
    frames: Sequence[np.ndarray], context_size: int = 4, image_size=POLICY_IMAGE_SIZE
) -> torch.Tensor:
    """Latest ``context_size`` frames → [K, 3, h, w]; left-pads by repeating the oldest."""
    if not frames:
        raise ValueError("stack_context requires at least one frame")
    size = _as_size(image_size)
    recent: List[np.ndarray] = list(frames)[-context_size:]
    while len(recent) < context_size:
        recent.insert(0, recent[0])
    return torch.stack([to_policy_tensor(frame, size) for frame in recent], dim=0)


def goal_pair_tensor(
    current_rgb: np.ndarray,
    goal_rgb: np.ndarray,
    goal_point: "Sequence[float] | None" = None,
    image_size=POLICY_IMAGE_SIZE,
    conditioning: str = "marker",
) -> torch.Tensor:
    """Channel-stack (current obs, goal conditioning) → [6, h, w].

    conditioning forms (M3 ablation; identical between train and serve):
      marker — goal frame with the ring drawn at policy resolution (v1 form)
      coord  — clean goal frame; the point rides in via the coord token
      crop   — crop-zoom around the point instead of the full goal frame
    """
    size = _as_size(image_size)
    current = to_policy_tensor(current_rgb, size)
    if goal_point is None or conditioning == "coord":
        goal = to_policy_tensor(goal_rgb, size)
    elif conditioning == "marker":
        goal = _normalize_chw(
            render_goal_marker_at_policy_resolution(goal_rgb, goal_point, size)
        )
    elif conditioning == "crop":
        goal = _normalize_chw(crop_around_point(goal_rgb, goal_point, size))
    else:
        raise ValueError(f"unknown conditioning {conditioning!r}")
    return torch.cat([current, goal], dim=0)
