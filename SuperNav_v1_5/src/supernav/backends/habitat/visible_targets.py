"""Visible navigation targets using SuperNav-owned scene-graph semantics.

Projection, occlusion and semantic filtering use the adapter's geometry helpers.
"""
from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


class SuperNavVisibleTargetsMixin:
    def _visible_nav_targets_for_view(
        self,
        *,
        session: Any,
        agent_state: Any,
        direction: str,
        width: int,
        height: int,
        depth: np.ndarray | None,
        semantic: np.ndarray | None = None,
    ) -> list[dict[str, Any]]:
        if width <= 0 or height <= 0:
            return []
        scene_graph = getattr(session, "scene_graph", None)
        if not isinstance(scene_graph, Mapping):
            return []
        from supernav.methods.navigation import oracle_local_nav as oracle_nav

        camera_position, right, up, forward = self._camera_basis_from_state(
            session=session,
            agent_state=agent_state,
        )
        hfov = float(session.settings.get("hfov", 90.0) or 90.0)
        fx = width / (2.0 * math.tan(math.radians(hfov) / 2.0))
        fy = fx
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0
        coordinate_system = oracle_nav._scene_coordinate_system(scene_graph)
        rows: list[dict[str, Any]] = []
        for index, obj in enumerate(oracle_nav._object_records(scene_graph), start=1):
            if not isinstance(obj, Mapping):
                continue
            label = str(obj.get("label") or obj.get("category") or "").strip()
            if not label:
                continue
            bbox_min = oracle_nav._as_vec3(
                obj.get("bbox_min_xyz"),
                coordinate_system=coordinate_system,
            )
            bbox_max = oracle_nav._as_vec3(
                obj.get("bbox_max_xyz"),
                coordinate_system=coordinate_system,
            )
            if bbox_min is None or bbox_max is None:
                continue
            corners = self._aabb_corners(bbox_min, bbox_max)
            projection = self._project_aabb_to_image(
                corners=corners,
                camera_position=camera_position,
                right=right,
                up=up,
                forward=forward,
                width=width,
                height=height,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
            )
            if projection is None:
                continue
            x0, y0, x1, y1, front_depth, center_depth, back_depth = projection
            if not self._line_of_sight_reaches_aabb(
                session=session,
                camera_position=camera_position,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
            ):
                continue
            area = max(1.0, float((x1 - x0 + 1) * (y1 - y0 + 1)))
            visible_fraction = min(1.0, area / float(width * height))
            basis = "projection_only"
            row_depth: float | None = None
            if depth is not None:
                basis = "projection_depth"
                visible_fraction, row_depth = self._depth_visible_fraction(
                    depth=depth,
                    bbox=(x0, y0, x1, y1),
                    image_width=width,
                    image_height=height,
                    expected_depth_range_m=(front_depth, back_depth),
                )
                if visible_fraction < 0.10:
                    continue
            semantic_overlap = self._semantic_visible_fraction(
                semantic=semantic,
                bbox=(x0, y0, x1, y1),
                image_width=width,
                image_height=height,
                object_record=obj,
            )
            if semantic_overlap is not None:
                if semantic_overlap <= 0.0:
                    continue
                basis += "_semantic"
            turn_right = self._PANORAMA_TURN_RIGHT_DEGREES.get(direction, 0)
            row: dict[str, Any] = {
                "target_ref": oracle_nav._object_target_ref(obj, index),
                "label": label,
                "direction": direction,
                "turn_right_deg": turn_right,
                "bbox_px": {
                    "x": int(x0),
                    "y": int(y0),
                    "width": int(x1 - x0 + 1),
                    "height": int(y1 - y0 + 1),
                },
                "center_px": [
                    int(round((x0 + x1) / 2.0)),
                    int(round((y0 + y1) / 2.0)),
                ],
                "visible_fraction": round(float(visible_fraction), 3),
                "visibility_basis": basis,
                "_sort_area": area,
                "_sort_depth": float(center_depth),
            }
            if row_depth is not None:
                row["depth_m"] = round(float(row_depth), 3)
            if semantic_overlap is not None:
                row["semantic_overlap"] = round(float(semantic_overlap), 3)
            rows.append(row)
        return rows
