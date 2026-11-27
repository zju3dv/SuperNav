"""SuperNav-owned bridge navigation actions and shared scene-graph geometry."""

from __future__ import annotations

from supernav.paths import workspace_root

from typing import Any, Mapping

from habitat_contract.navigation import NavigationBackend

import numpy as np

# Helpers used directly by the handlers below.  Importing them into this module
# keeps external monkeypatches (e.g. ``oln._shortest_path``) effective because
# the handlers resolve these names against ``oracle_local_nav`` globals.
from supernav.methods.navigation.oracle_local_nav.coords import _object_coordinate_audit
from supernav.methods.navigation.oracle_local_nav.grounding_client import _call_grounding_service
from supernav.methods.navigation.oracle_local_nav.standoff import (
    _limit_path_to_horizon,
    _remaining_geodesic,
    _select_candidate,
    _shortest_path,
)
from supernav.methods.navigation.oracle_local_nav.utils import (
    _append_unique_paths,
    _movement_frame_paths,
    _positive_float,
)


def navigate_oracle_local(
    adapter: NavigationBackend,
    session_id: str | None,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Bridge action handler for ``hab_oracle_local_navigate``."""

    session = adapter.navigation_session(session_id)
    pathfinder = adapter.navigation_pathfinder(session)

    target_ref = str(payload.get("target_ref") or "").strip()
    instruction = str(payload.get("instruction") or "").strip()
    target_label = str(payload.get("target_label") or "").strip()
    if not target_ref and not instruction and not target_label:
        return {
            "ok": False,
            "backend": "oracle_local_gt",
            "status": "unresolved_local_target",
            "error": (
                "oracle local navigation requires target_ref, instruction, "
                "or target_label"
            ),
        }

    scene_graph = getattr(session, "scene_graph", None)
    if not isinstance(scene_graph, Mapping):
        return {
            "ok": False,
            "backend": "oracle_local_gt",
            "status": "unresolved_local_target",
            "error": "scene graph is not available for this session",
        }

    if target_ref:
        latest_refs = getattr(session, "last_visible_nav_target_refs", None)
        if not latest_refs:
            return {
                "ok": False,
                "backend": "oracle_local_gt",
                "status": "target_not_currently_visible",
                "target_source": "latest_visible_nav_targets",
                "query": target_ref,
                "error": (
                    "oracle local navigation requires target_ref from the "
                    "latest visible_nav_targets; no visible target set is "
                    "available yet"
                ),
            }
        if target_ref not in set(str(ref) for ref in latest_refs):
            return {
                "ok": False,
                "backend": "oracle_local_gt",
                "status": "target_not_currently_visible",
                "target_source": "latest_visible_nav_targets",
                "query": target_ref,
                "error": (
                    "target_ref is not in the latest visible_nav_targets; "
                    "capture a fresh panorama and choose one of its target_ref "
                    "values"
                ),
            }

    current_position = np.asarray(
        adapter.agent_position(session),
        dtype=np.float32,
    )
    forward = adapter.agent_forward(session)
    horizon_m = _positive_float(payload.get("horizon_m"), default=3.0)
    goal_radius = _positive_float(payload.get("goal_radius"), default=0.6)
    standoff_m = _positive_float(payload.get("standoff_m"), default=0.7)
    max_steps = max(1, int(payload.get("max_steps", 40)))

    candidate = _select_candidate(
        scene_graph=scene_graph,
        pathfinder=pathfinder,
        current_position=current_position,
        forward=forward,
        target_ref=target_ref,
        instruction=instruction,
        target_label=target_label,
        horizon_m=horizon_m,
        standoff_m=standoff_m,
    )
    if candidate is None:
        return {
            "ok": False,
            "backend": "oracle_local_gt",
            "status": "unresolved_local_target",
            "target_source": "scene_graph_object",
            "query": target_ref or target_label or instruction,
            "error": "no reachable scene-graph object matched the requested local target",
        }

    steps_remaining = max_steps
    steps_executed = 0
    blocked_steps = 0
    last_nav: dict[str, Any] = {}
    movement_frames: list[str] = []
    status = "en_route_local_target"
    while steps_remaining > 0:
        burst = min(steps_remaining, 32)
        nav = adapter.navigate_step(
            session.session_id,
            {
                "goal": candidate.local_target.tolist(),
                "goal_radius": goal_radius,
                "max_steps": burst,
                "include_metrics": True,
                "include_visuals": True,
                "include_publish_hints": True,
                "output_dir": str(
                    payload.get("output_dir") or str(workspace_root() / "data" / "runs" / "artifacts")
                ),
            },
        )
        last_nav = nav if isinstance(nav, dict) else {}
        _append_unique_paths(movement_frames, _movement_frame_paths(last_nav))
        burst_steps = int(last_nav.get("steps_executed") or 0)
        steps_executed += burst_steps
        steps_remaining -= max(1, burst_steps) if burst_steps else burst
        nav_status = str(last_nav.get("nav_status") or "")
        if bool(last_nav.get("collided")):
            blocked_steps += 1
        if nav_status == "reached":
            status = "reached_local_target"
            break
        if nav_status in {"blocked", "unreachable", "error"}:
            status = nav_status
            break
        if burst_steps <= 0:
            break

    distance = _remaining_geodesic(
        pathfinder,
        np.asarray(
            adapter.agent_position(session),
            dtype=np.float32,
        ),
        candidate.local_target,
    )
    if distance is not None and distance <= goal_radius:
        status = "reached_local_target"

    coordinate_audit = _object_coordinate_audit(scene_graph, candidate.target_ref)
    body = {
        "ok": True,
        "backend": "oracle_local_gt",
        "status": status,
        "target_source": "scene_graph_object",
        "matched_label": candidate.label,
        "steps_executed": steps_executed,
        "reachable": True,
        "blocked_steps": blocked_steps,
        "distance_to_local_target": distance,
        "nav_status": status,
        "matched_target_ref": candidate.target_ref,
        "metrics": last_nav.get("metrics"),
        "state_summary": last_nav.get("state_summary"),
        "visuals": last_nav.get("visuals", {}),
        "publish_hints": last_nav.get("publish_hints"),
        "movement_frames": movement_frames,
        "movement_frame_count": len(movement_frames),
        "_debug": {
            "object_id": candidate.object_id,
            "object_position": candidate.object_position.tolist(),
            "object_position_source": coordinate_audit.get("source_key"),
            "object_position_raw": coordinate_audit.get("raw_xyz"),
            "object_position_habitat": coordinate_audit.get("habitat_xyz"),
            "coordinate_system": coordinate_audit.get("coordinate_system"),
            "coordinate_transform_applied": coordinate_audit.get(
                "semantic_xy_floor_y_flipped"
            ),
            "coordinate_transform_audit": coordinate_audit,
            "resolved_target": candidate.target.tolist(),
            "resolved_local_target": candidate.local_target.tolist(),
            "target_geodesic_distance": candidate.target_geodesic,
            "local_geodesic_distance": candidate.local_geodesic,
            "heading_angle_deg": candidate.heading_angle_deg,
            "label_score": candidate.label_score,
            "path_points": candidate.local_path_points,
            "target_ref": target_ref,
            "instruction": instruction,
            "target_label": target_label,
        },
    }
    return {k: v for k, v in body.items() if v is not None}


# Import the visual handler after navigate_oracle_local is defined so it can
# resolve ``navigate_oracle_local`` and ``_call_grounding_service`` from this
# package's globals (preserving module-level monkeypatching).
from supernav.methods.navigation.oracle_local_nav.visual_ground_preview import navigate_visual_ground_preview  # noqa: E402
from supernav.methods.navigation.oracle_local_nav.visual_grounding import navigate_visual_local  # noqa: E402
from supernav.methods.navigation.oracle_local_nav.visual_overlay import navigate_visual_overlay  # noqa: E402
from supernav.methods.navigation.oracle_local_nav.visual_point import navigate_visual_point  # noqa: E402

# Remaining re-exports for external callers and tests.
from supernav.methods.navigation.oracle_local_nav.coords import (  # noqa: E402
    _as_vec3,
    _object_position,
    _scene_coordinate_system,
)
from supernav.methods.navigation.oracle_local_nav.grounding_client import (  # noqa: E402
    _DEFAULT_MAX_GROUNDING_CROPS,
    _ground_detections_for_panorama,
    _ground_detections_for_projection_crops,
    _grounding_image_source,
)
from supernav.methods.navigation.oracle_local_nav.local_map import (  # noqa: E402
    _local_map_object_public_row,
    _local_map_objects,
    get_oracle_local_map,
)
from supernav.methods.navigation.oracle_local_nav.match_evaluation import (  # noqa: E402
    _best_detection_projection_match,
    _visual_candidate_table,
    _write_visual_grounding_candidate_artifact,
)
from supernav.methods.navigation.oracle_local_nav.scene_graph import (  # noqa: E402
    _object_records,
    _object_target_ref,
)
from supernav.methods.navigation.oracle_local_nav.visual_grounding import (  # noqa: E402
    _visual_grounding_phrases,
)

__all__ = [
    "get_oracle_local_map",
    "navigate_oracle_local",
    "navigate_visual_ground_preview",
    "navigate_visual_local",
    "navigate_visual_overlay",
    "navigate_visual_point",
    "_as_vec3",
    "_best_detection_projection_match",
    "_call_grounding_service",
    "_DEFAULT_MAX_GROUNDING_CROPS",
    "_ground_detections_for_panorama",
    "_ground_detections_for_projection_crops",
    "_grounding_image_source",
    "_local_map_object_public_row",
    "_local_map_objects",
    "_object_coordinate_audit",
    "_object_position",
    "_object_records",
    "_object_target_ref",
    "_remaining_geodesic",
    "_scene_coordinate_system",
    "_select_candidate",
    "_shortest_path",
    "_visual_candidate_table",
    "_visual_grounding_phrases",
    "_write_visual_grounding_candidate_artifact",
]
