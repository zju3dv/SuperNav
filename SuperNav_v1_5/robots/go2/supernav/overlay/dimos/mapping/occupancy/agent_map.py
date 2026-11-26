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

"""Agent-facing 2D occupancy-map rendering."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from io import BytesIO
import math
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid

UNKNOWN_RGB = (210, 210, 210)
FREE_RGB = (255, 255, 255)
OCCUPIED_RGB = (0, 0, 0)
TRACE_RGB = (0, 210, 230)
ARROW_FILL_RGB = (230, 0, 0)
ARROW_OUTLINE_RGB = (255, 230, 0)


@dataclass(frozen=True)
class RenderedAgentOccupancyMap:
    image_block: dict[str, object]
    metadata: dict[str, object]


def render_agent_occupancy_map(
    occupancy_grid: OccupancyGrid,
    pose: PoseStamped,
    trace: list[PoseStamped] | tuple[PoseStamped, ...] = (),
    *,
    graph_nodes: list[dict[str, Any]] | tuple[dict[str, Any], ...] = (),
    output_size_px: int = 768,
    view_size_m: float = 12.0,
) -> RenderedAgentOccupancyMap:
    """Render a DimOS occupancy grid as a model-visible navigation image."""

    if (
        occupancy_grid.grid.size == 0
        or occupancy_grid.width <= 0
        or occupancy_grid.height <= 0
    ):
        raise ValueError("occupancy grid is empty")
    if occupancy_grid.resolution <= 0.0:
        raise ValueError("occupancy grid resolution must be positive")
    if output_size_px <= 0:
        raise ValueError("output_size_px must be positive")

    robot_grid = occupancy_grid.world_to_grid(pose.position)
    robot_x = float(robot_grid.x)
    robot_y = float(robot_grid.y)
    if not (0.0 <= robot_x < occupancy_grid.width and 0.0 <= robot_y < occupancy_grid.height):
        raise ValueError("pose is outside occupancy grid bounds")

    crop = _crop_bounds(
        center_x=robot_x,
        center_y=robot_y,
        width=occupancy_grid.width,
        height=occupancy_grid.height,
        cells=max(1, int(round(view_size_m / occupancy_grid.resolution))),
    )
    x0, y0, x1, y1 = crop
    crop_grid = occupancy_grid.grid[y0:y1, x0:x1]
    rgb = np.empty((crop_grid.shape[0], crop_grid.shape[1], 3), dtype=np.uint8)
    rgb[crop_grid < 0] = UNKNOWN_RGB
    known_mask = crop_grid >= 0
    if np.any(known_mask):
        costs = np.clip(crop_grid[known_mask].astype(np.int16), 0, 100)
        intensity = (255 - np.rint(costs * 2.55)).astype(np.uint8)
        rgb[known_mask] = np.stack([intensity, intensity, intensity], axis=1)

    image = PILImage.fromarray(rgb[::-1], mode="RGB")
    scale_x = output_size_px / max(1, image.width)
    scale_y = output_size_px / max(1, image.height)
    resampling = getattr(PILImage, "Resampling", PILImage)
    image = image.resize((output_size_px, output_size_px), resample=resampling.NEAREST)
    draw = ImageDraw.Draw(image)

    visible_trace = [
        _grid_to_pixel(occupancy_grid, trace_pose, crop, scale_x, scale_y)
        for trace_pose in trace
        if _pose_in_crop(occupancy_grid, trace_pose, crop)
    ]
    if len(visible_trace) >= 2:
        draw.line(
            visible_trace,
            fill=TRACE_RGB,
            width=max(3, output_size_px // 160),
            joint="curve",
        )

    robot_px = _grid_to_pixel(occupancy_grid, pose, crop, scale_x, scale_y)
    displayed_graph_nodes = _draw_graph_node_badges(
        draw,
        occupancy_grid,
        graph_nodes,
        crop,
        scale_x,
        scale_y,
        output_size_px,
    )
    _draw_facing_arrow(draw, robot_px, pose.yaw, output_size_px)

    data = _png_base64(image)
    image_block: dict[str, object] = {
        "type": "image",
        "data": data,
        "mimeType": "image/png",
        "width_px": output_size_px,
        "height_px": output_size_px,
    }
    metadata: dict[str, object] = {
        "source": "global_costmap",
        "map_frame": occupancy_grid.frame_id,
        "pose_frame": pose.frame_id,
        "grid_ts": occupancy_grid.ts,
        "resolution_m_per_cell": occupancy_grid.resolution,
        "origin_xy": [
            occupancy_grid.origin.position.x,
            occupancy_grid.origin.position.y,
        ],
        "grid_size_cells": [occupancy_grid.width, occupancy_grid.height],
        "width_px": output_size_px,
        "height_px": output_size_px,
        "crop_grid_bounds": {
            "x_min": x0,
            "y_min": y0,
            "x_max": x1,
            "y_max": y1,
        },
        "trace_points": len(visible_trace),
        "graph_nodes": displayed_graph_nodes,
        "graph_overlay": bool(displayed_graph_nodes),
    }
    return RenderedAgentOccupancyMap(image_block=image_block, metadata=metadata)


def _crop_bounds(
    *,
    center_x: float,
    center_y: float,
    width: int,
    height: int,
    cells: int,
) -> tuple[int, int, int, int]:
    crop_cells = min(max(1, cells), max(1, min(width, height)))
    half = crop_cells / 2.0
    x0 = int(math.floor(center_x - half))
    y0 = int(math.floor(center_y - half))
    x0 = min(max(0, x0), max(0, width - crop_cells))
    y0 = min(max(0, y0), max(0, height - crop_cells))
    x1 = min(width, x0 + crop_cells)
    y1 = min(height, y0 + crop_cells)
    return x0, y0, x1, y1


def _pose_in_crop(
    occupancy_grid: OccupancyGrid,
    pose: PoseStamped,
    crop: tuple[int, int, int, int],
) -> bool:
    point = occupancy_grid.world_to_grid(pose.position)
    x0, y0, x1, y1 = crop
    return x0 <= point.x < x1 and y0 <= point.y < y1


def _grid_to_pixel(
    occupancy_grid: OccupancyGrid,
    pose: PoseStamped,
    crop: tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
) -> tuple[float, float]:
    point = occupancy_grid.world_to_grid(pose.position)
    x0, y0, _x1, y1 = crop
    px = (float(point.x) - x0 + 0.5) * scale_x
    py = (y1 - float(point.y) - 0.5) * scale_y
    return px, py


def _draw_facing_arrow(
    draw: ImageDraw.ImageDraw,
    center: tuple[float, float],
    yaw_rad: float,
    output_size_px: int,
) -> None:
    length = max(28.0, output_size_px * 0.085)
    half_width = max(9.0, output_size_px * 0.018)
    tail = max(7.0, output_size_px * 0.014)
    dx = math.cos(yaw_rad)
    dy = -math.sin(yaw_rad)
    px = -dy
    py = dx
    cx, cy = center
    tip = (cx + dx * length, cy + dy * length)
    base = (cx - dx * tail, cy - dy * tail)
    polygon = [
        tip,
        (base[0] + px * half_width, base[1] + py * half_width),
        (base[0] - px * half_width, base[1] - py * half_width),
    ]
    draw.polygon(polygon, fill=ARROW_FILL_RGB, outline=ARROW_OUTLINE_RGB)
    draw.line([polygon[-1], tip, polygon[1]], fill=ARROW_OUTLINE_RGB, width=3)


def _draw_graph_node_badges(
    draw: ImageDraw.ImageDraw,
    occupancy_grid: OccupancyGrid,
    graph_nodes: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    crop: tuple[int, int, int, int],
    scale_x: float,
    scale_y: float,
    output_size_px: int,
) -> list[dict[str, Any]]:
    displayed: list[dict[str, Any]] = []
    occupied_slots: list[tuple[float, float]] = []
    radius = max(11, output_size_px // 42)
    font = None
    for raw_node in graph_nodes:
        if not isinstance(raw_node, dict) or not raw_node.get("has_pano_images"):
            continue
        slam_pose = raw_node.get("slam_pose")
        if not isinstance(slam_pose, dict):
            continue
        try:
            order_id = int(raw_node["order_id"])
            x = float(slam_pose["x"])
            y = float(slam_pose["y"])
        except (KeyError, TypeError, ValueError):
            continue
        grid_point = occupancy_grid.world_to_grid((x, y, 0.0))
        x0, y0, x1, y1 = crop
        if not (x0 <= grid_point.x < x1 and y0 <= grid_point.y < y1):
            continue
        base_px = (
            (float(grid_point.x) - x0 + 0.5) * scale_x,
            (y1 - float(grid_point.y) - 0.5) * scale_y,
        )
        badge_px = _displaced_badge_position(base_px, occupied_slots, radius, output_size_px)
        occupied_slots.append(badge_px)
        if badge_px != base_px:
            draw.line([base_px, badge_px], fill=(20, 20, 20), width=max(2, radius // 5))
        cx, cy = badge_px
        bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(bbox, fill=(0, 0, 0), outline=(255, 255, 255), width=max(2, radius // 5))
        text = str(order_id)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        draw.text((cx - tw / 2, cy - th / 2 - 1), text, fill=(255, 255, 255), font=font)
        displayed.append(
            {
                "order_id": order_id,
                "id": str(raw_node.get("id") or ""),
                "name": str(raw_node.get("name") or ""),
                "kind": str(raw_node.get("kind") or ""),
            }
        )
    return displayed


def _displaced_badge_position(
    base_px: tuple[float, float],
    occupied: list[tuple[float, float]],
    radius: int,
    output_size_px: int,
) -> tuple[float, float]:
    min_distance = radius * 2.4
    candidates = [base_px]
    for ring in range(1, 4):
        distance = ring * radius * 2.6
        for angle_deg in (0, 45, 90, 135, 180, 225, 270, 315):
            angle = math.radians(angle_deg)
            candidates.append(
                (
                    base_px[0] + math.cos(angle) * distance,
                    base_px[1] + math.sin(angle) * distance,
                )
            )
    for candidate in candidates:
        cx = min(max(radius + 2, candidate[0]), output_size_px - radius - 2)
        cy = min(max(radius + 2, candidate[1]), output_size_px - radius - 2)
        if all(math.hypot(cx - ox, cy - oy) >= min_distance for ox, oy in occupied):
            return (cx, cy)
    return (
        min(max(radius + 2, base_px[0]), output_size_px - radius - 2),
        min(max(radius + 2, base_px[1]), output_size_px - radius - 2),
    )


def _png_base64(image: PILImage.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
