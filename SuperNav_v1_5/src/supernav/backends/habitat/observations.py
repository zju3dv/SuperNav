"""Agent-facing observation capture using SuperNav-owned Habitat operations.

Preserves front-only and panorama evidence while rendering through the SDK.
"""
from __future__ import annotations
import math
import os
from typing import Any, Dict, Mapping, Optional
import numpy as np
from supernav.backends.habitat.simulator.types import (
    _DEFAULT_DEPTH_VIS_MAX, _DEFAULT_VISUAL_OUTPUT_DIR, HabitatAdapterError,
)

from supernav.backends.habitat.visible_targets import SuperNavVisibleTargetsMixin


class SuperNavObservationMixin(SuperNavVisibleTargetsMixin):
    def _get_panorama(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        session = self._require_session(session_id)
        front_only = self._coerce_bool(
            payload.get("front_only", False), field_name="payload.front_only"
        )
        include_depth = self._coerce_bool(
            payload.get("include_depth_analysis", False),
            field_name="payload.include_depth_analysis",
        )
        clearance = float(payload.get("clearance_threshold", 0.5))
        output_dir = payload.get("output_dir", _DEFAULT_VISUAL_OUTPUT_DIR)
        if not isinstance(output_dir, str) or not output_dir:
            raise HabitatAdapterError(
                'Field "payload.output_dir" must be a non-empty str'
            )
        depth_max = self._coerce_float(
            payload.get("depth_max", _DEFAULT_DEPTH_VIS_MAX),
            field_name="payload.depth_max",
        )
        if include_depth:
            # Verify depth sensor is available before starting panorama capture
            test_obs = self._capture_sensor_observations(session)
            if "depth_sensor" not in test_obs:
                raise HabitatAdapterError(
                    "get_panorama with include_depth_analysis=true requires "
                    "depth_sensor — enable it with hab init --depth"
                )

        agent = session.simulator.get_agent(session.agent_id)
        saved_state = agent.get_state()

        images = []
        depth_analysis = [] if include_depth else None
        visible_nav_targets: list[dict[str, Any]] = []
        visible_depth_available = False

        # Use ONE capture_seq for all 4 directions so dashboard can find them
        # at the same step number (all share pano_seq in filename).
        session.capture_counter += 1
        pano_seq = session.capture_counter

        session_dir = self._session_output_dir(output_dir, session.session_id)
        image_ref_registry: dict[str, dict[str, Any]] = {}
        directions = ("front",) if front_only else self._PANORAMA_DIRECTIONS
        for i, direction in enumerate(directions):
            obs = self._capture_sensor_observations(session)
            view_state = agent.get_state()
            heading = self._heading_degrees(session)

            visuals = self._export_visuals(
                observation=obs,
                output_dir=session_dir,
                sensors=["color_sensor"],
                depth_max=depth_max,
                session_id=session.session_id,
                step_count=session.step_count,
                capture_seq=pano_seq,
                filename_prefix=f"pano_{direction}",
                agent_image_max_size=self._coerce_int(
                    payload.get(
                        "agent_image_max_size", self._DEFAULT_AGENT_IMAGE_MAX_SIZE
                    ),
                    field_name="payload.agent_image_max_size",
                ),
            )
            color_info = visuals.get("color_sensor", {})
            image_ref = f"pano:{session.session_id}:{pano_seq}:{direction}"
            image_item = {
                "image_ref": image_ref,
                "direction": direction,
                "heading_deg": round(heading, 1),
                "path": color_info.get("path"),
            }
            for key in (
                "width",
                "height",
                "original_width",
                "original_height",
                "downsampled_for_agent",
                "agent_image_max_size",
                "original_path",
            ):
                if key in color_info:
                    image_item[key] = color_info[key]
            self._annotate_agent_panorama_view_label(
                image_item,
                label=self._PANORAMA_VIEW_LABELS.get(direction, ""),
            )
            images.append(image_item)

            depth_arr_for_visibility = self._depth_array_from_observation(obs)
            if depth_arr_for_visibility is not None:
                visible_depth_available = True
            camera_position, camera_right, camera_up, camera_forward = (
                self._camera_basis_from_state(
                    session=session,
                    agent_state=view_state,
                )
            )
            image_ref_registry[image_ref] = {
                "image_ref": image_ref,
                "capture_seq": pano_seq,
                "direction": direction,
                "width": int(image_item.get("width") or 0),
                "height": int(image_item.get("height") or 0),
                "original_width": image_item.get("original_width"),
                "original_height": image_item.get("original_height"),
                "downsampled_for_agent": image_item.get("downsampled_for_agent"),
                "agent_image_max_size": image_item.get("agent_image_max_size"),
                "original_path": image_item.get("original_path"),
                "hfov": float(session.settings.get("hfov", 90.0) or 90.0),
                "path": image_item.get("path"),
                "camera_position": camera_position.copy(),
                "camera_right": camera_right.copy(),
                "camera_up": camera_up.copy(),
                "camera_forward": camera_forward.copy(),
                "depth": (
                    depth_arr_for_visibility.copy()
                    if depth_arr_for_visibility is not None
                    else None
                ),
                "agent_state": view_state,
            }
            semantic_arr_for_visibility = self._semantic_array_from_observation(obs)
            visible_nav_targets.extend(
                self._visible_nav_targets_for_view(
                    session=session,
                    agent_state=view_state,
                    direction=direction,
                    width=int(image_item.get("width") or 0),
                    height=int(image_item.get("height") or 0),
                    depth=depth_arr_for_visibility,
                    semantic=semantic_arr_for_visibility,
                )
            )

            if include_depth and "depth_sensor" in obs:
                depth_arr = np.asarray(obs["depth_sensor"], dtype=np.float32)
                if depth_arr.ndim == 3:
                    depth_arr = depth_arr[:, :, 0]
                analysis = self._analyze_depth_array(depth_arr, clearance)
                depth_analysis.append({
                    "direction": direction,
                    "heading_deg": round(heading, 1),
                    "min_dist": analysis.get("front_center", {}).get("min_dist"),
                    "mean_dist": analysis.get("front_center", {}).get("mean_dist"),
                    "clear": analysis.get("front_center", {}).get("clear", False),
                })

            # Turn 90 degrees right for next direction (9 atomic steps × 10°)
            if i < len(directions) - 1:
                for _ in range(9):
                    session.simulator.step("turn_right")

        # Restore original state (undo all rotations)
        agent.set_state(saved_state, infer_sensor_states=True)
        session.last_sensor_obs = None  # invalidate cached observation

        public_visible_nav_targets = self._dedupe_visible_nav_targets(
            visible_nav_targets
        )
        visible_nav_targets_status = (
            "scene_graph_unavailable"
            if not isinstance(getattr(session, "scene_graph", None), Mapping)
            else ("ok" if visible_depth_available else "depth_unavailable")
        )
        overlay_max_objects = max(
            0,
            self._coerce_int(
                payload.get(
                    "visible_target_overlay_max_objects",
                    self._DEFAULT_VISIBLE_TARGET_OVERLAY_MAX_OBJECTS,
                ),
                field_name="payload.visible_target_overlay_max_objects",
            ),
        )
        overlay_aliases = self._build_visible_nav_target_overlays(
            images=images,
            visible_nav_targets=public_visible_nav_targets,
            output_dir=session_dir,
            pano_seq=pano_seq,
            max_targets_per_image=overlay_max_objects,
        )
        overlay_objlist = self._overlay_objlist_from_images(images)
        session.last_visible_nav_targets = list(public_visible_nav_targets)
        session.last_visible_nav_target_refs = {
            str(row.get("target_ref") or "")
            for row in public_visible_nav_targets
            if str(row.get("target_ref") or "")
        }
        session.last_visible_nav_targets_status = visible_nav_targets_status
        session.last_visual_overlay_aliases = overlay_aliases
        session.last_panorama_images = [dict(row) for row in images]
        session.last_visual_image_refs = image_ref_registry
        session.latest_visual_capture_seq = pano_seq

        result: Dict[str, Any] = {
            "session_id": session.session_id,
            "images": images,
            "overlay_objlist": overlay_objlist,
            "visible_nav_targets": public_visible_nav_targets,
            "visible_nav_targets_status": visible_nav_targets_status,
        }
        if depth_analysis is not None:
            result["depth_analysis"] = depth_analysis
        return result
