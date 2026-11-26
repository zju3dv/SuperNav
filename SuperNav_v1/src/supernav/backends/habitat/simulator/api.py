# Copyright (c) Meta Platforms, Inc. and its affiliates.
# Licensed under the MIT license; see THIRD_PARTY_NOTICES.md in the repository root.

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

from supernav.backends.habitat.simulator.types import _API_VERSION


class HabitatAdapterApiMixin:
    """Public adapter API description and contract metadata."""

    def _describe_api(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        del session_id, payload
        return {
            "api_version": _API_VERSION,
            "actions": {
                "describe_api": {
                    "session_required": False,
                    "payload": {},
                },
                "init_scene": {
                    "session_required": False,
                    "payload": {
                        "scene": "str (required)",
                        "scene_dataset_config_file": "str (optional)",
                        "start_position": "list[float] length 3 (optional)",
                        "start_rotation": "list[float] length 4 xyzw quaternion (optional)",
                        "default_agent_navmesh": "bool (optional; auto-disabled for GS dataset configs)",
                        "default_agent": "int (optional, default 0)",
                        "seed": "int (optional)",
                        "frustum_culling": "bool (optional)",
                        "allow_sliding": (
                            "bool (optional; per-session motion physics override, "
                            "default habitat-sim true; false = blocked step is zero displacement)"
                        ),
                        "enable_physics": (
                            "bool (optional; default from habitat_sim default_sim_settings, "
                            "typically built_with_bullet; auto false for GS dataset configs when omitted)"
                        ),
                        "sensor": {
                            "width": "int",
                            "height": "int",
                            "sensor_height": "float",
                            "hfov": "float",
                            "zfar": "float",
                            "color_sensor": "bool",
                            "depth_sensor": "bool",
                            "semantic_sensor": "bool",
                            "third_person_color_sensor": (
                                "bool (optional, default false; enables an "
                                "over-the-shoulder RGB camera at ~1.5m behind "
                                "and 1.2m above the agent)"
                            ),
                            "third_person_sensor_uuid": (
                                "str (optional, default 'third_rgb_sensor'; "
                                "name of the third-person sensor in visuals "
                                "and observation payloads)"
                            ),
                        },
                    },
                },
                "get_scene_info": {
                    "session_required": True,
                    "payload": {
                        "refresh_observation": "bool (optional, default false)",
                    },
                },
                "set_agent_state": {
                    "session_required": True,
                    "payload": {
                        "position": "list[float] length 3 (required)",
                        "rotation": "list[float] length 4 (optional)",
                        "snap_to_navmesh": "bool (optional, default false)",
                        "infer_sensor_states": "bool (optional, default true)",
                        "include_observation_data": "bool (optional, default false)",
                        "max_observation_elements": "int (optional, default 2048)",
                    },
                },
                "reset_agent_pose": {
                    "session_required": True,
                    "payload": {
                        "x": "float (required) — world-frame X in metres",
                        "y": "float (required) — world-frame Y in metres",
                        "z": "float (required) — world-frame Z in metres",
                        "yaw": "float (required) — heading in radians about Y",
                    },
                },
                "sample_navigable_point": {
                    "session_required": True,
                    "payload": {
                        "near": "list[float] length 3 (optional)",
                        "distance": "float (optional, default 3.0)",
                        "max_tries": "int (optional, default 100)",
                        "seed": "int (optional)",
                    },
                },
                "find_shortest_path": {
                    "session_required": True,
                    "payload": {
                        "start": "list[float] length 3 (optional, default current pose)",
                        "end": "list[float] length 3 (required)",
                        "snap_start": "bool (optional, default true)",
                        "snap_end": "bool (optional, default true)",
                    },
                },
                "get_topdown_map": {
                    "session_required": True,
                    "payload": {
                        "output_dir": "str (optional, default data/runs/visuals)",
                        "height": "float (optional, default current agent y)",
                        "meters_per_pixel": "float (optional, default 0.05)",
                        "goal": "list[float] length 3 (optional)",
                        "path_points": "list[list[float]] (optional)",
                    },
                },
                "navigate_step": {
                    "session_required": True,
                    "payload": {
                        "goal": "list[float] length 3 (required)",
                        "goal_radius": "float (optional)",
                        "max_steps": "int (optional, default 1) — greedy actions to execute in one call",
                        "until_reached": "bool (optional, default false) — keep stepping until reached, blocked, or error",
                        "dt": "float (optional, default 1/60)",
                        "include_observation_data": "bool (optional, default false)",
                        "max_observation_elements": "int (optional, default 2048)",
                        "include_metrics": "bool (optional, default true)",
                        "include_visuals": "bool (optional, default false) — include exported visuals in response; trace frames are still recorded for replay",
                        "include_publish_hints": "bool (optional, default false)",
                        "output_dir": "str (optional, default data/runs/visuals)",
                        "sensors": "list[str] (optional)",
                        "depth_max": "float (optional, default 10.0)",
                    },
                },
                "step_action": {
                    "session_required": True,
                    "payload": {
                        "action": "str|int (required) — move_forward | move_backward | turn_left | turn_right | look_up | look_down",
                        "degrees": "float (optional) — for turn_left/turn_right/look_up/look_down: rotate this many degrees, auto-decomposed into 10°/step atomic turns (e.g. degrees=30 → 3 steps)",
                        "distance": "float (optional) — for move_forward/move_backward: move this many metres, auto-decomposed into 0.25m/step atomic steps (e.g. distance=1.0 → 4 steps); stops early on collision",
                        "dt": "float (optional, default 1/60)",
                        "include_observation_data": "bool (optional, default false)",
                        "max_observation_elements": "int (optional, default 2048)",
                    },
                },
                "step_and_capture": {
                    "session_required": True,
                    "payload": {
                        "action": "str|int (required) — move_forward | move_backward | turn_left | turn_right | look_up | look_down",
                        "degrees": "float (optional) — for turn_left/turn_right/look_up/look_down: rotate this many degrees, auto-decomposed into 10°/step atomic turns",
                        "distance": "float (optional) — for move_forward/move_backward: move this many metres, auto-decomposed into 0.25m/step atomic steps; stops early on collision",
                        "dt": "float (optional, default 1/60)",
                        "include_observation_data": "bool (optional, default false)",
                        "max_observation_elements": "int (optional, default 2048)",
                        "output_dir": "str (optional, default data/runs/visuals)",
                        "sensors": "list[str] (optional)",
                        "depth_max": "float (optional, default 10.0)",
                        "include_metrics": "bool (optional, default true)",
                        "include_publish_hints": "bool (optional, default true)",
                    },
                },
                "get_observation": {
                    "session_required": True,
                    "payload": {
                        "refresh": "bool (optional, default false)",
                        "include_observation_data": "bool (optional, default false)",
                        "max_observation_elements": "int (optional, default 2048)",
                    },
                },
                "get_visuals": {
                    "session_required": True,
                    "payload": {
                        "refresh": "bool (optional, default false)",
                        "output_dir": "str (optional, default data/runs/visuals)",
                        "sensors": "list[str] (optional)",
                        "depth_max": "float (optional, default 10.0)",
                        "include_metrics": "bool (optional, default false)",
                    },
                },
                "export_video_trace": {
                    "session_required": True,
                    "payload": {
                        "output_dir": "str (optional, default data/runs/visuals)",
                        "sensor": "str (optional, default color_sensor)",
                        "fps": "float (optional, default 6.0)",
                        "step_start": "int (optional)",
                        "step_end": "int (optional)",
                        "include_metrics": "bool (optional, default true)",
                        "include_publish_hints": "bool (optional, default true)",
                    },
                },
                "get_metrics": {
                    "session_required": True,
                    "payload": {},
                },
                "get_runtime_status": {
                    "session_required": False,
                    "payload": {},
                },
                "get_collision_profile": {
                    "session_required": True,
                    "payload": {},
                },
                "get_scene_graph": {
                    "session_required": True,
                    "payload": {
                        "query_type": "str (required) — 'all', 'room', or 'object'",
                        "room_type": "str (optional) — filter rooms by type label (e.g. 'kitchen', 'bedroom')",
                        "object_label": "str (optional) — filter objects by label (e.g. 'chair', 'table')",
                        "max_results": "int (optional, default 10) — max nodes returned",
                    },
                },
                "navigate_oracle_local": {
                    "session_required": True,
                    "payload": {
                        "target_ref": "str (required) — internal/debug target ref from the latest projected visible target set",
                        "max_steps": "int (optional, default 40)",
                        "horizon_m": "float (optional, default 3.0) — local-hop geodesic cap",
                        "goal_radius": "float (optional, default 0.6)",
                        "standoff_m": "float (optional, default 0.7)",
                    },
                },
                "navigate_visual_local": {
                    "session_required": True,
                    "payload": {
                        "instruction": "str (required) — visual target phrase sent to Grounding DINO",
                        "max_steps": "int (optional, default 40)",
                        "horizon_m": "float (optional, default 3.0) — local-hop geodesic cap",
                        "goal_radius": "float (optional, default 0.6)",
                        "standoff_m": "float (optional, default 0.7)",
                        "output_dir": "str (optional, default data/runs/visuals)",
                    },
                },
                "navigate_visual_ground_preview": {
                    "session_required": True,
                    "payload": {
                        "image_ref": "str (required) — image_ref from the latest panorama_images row",
                        "phrase": "str (required for preview) — natural-language target description",
                        "confirm_token": "str (optional) — token from preview_ready; only multiple candidates require confirm",
                        "candidate_id": "int (optional) — numbered preview candidate to confirm when preview_ready",
                        "mode": "str (optional, default box) — box or point",
                        "max_candidates": "int (optional, default 4)",
                        "score_threshold": "float (optional, default 0.30)",
                        "horizon_m": "float (optional, default 40.0) — max geodesic distance for a reachable candidate",
                        "goal_radius": "float (optional, default 0.3)",
                        "max_steps": "int (optional) — direct or confirmed move cap; omit to navigate until reached, blocked, or error",
                        "standoff_m": "float (optional, default 0.7)",
                        "output_dir": "str (optional, default data/runs/visuals)",
                    },
                },
                "navigate_visual_overlay": {
                    "session_required": True,
                    "payload": {
                        "image_ref": "str (required) — image_ref from the latest overlay panorama_images row",
                        "target_alias": "str (required) — scene-graph obj_<number> alias visible on that overlay image",
                        "max_steps": "int (optional, default 40)",
                        "horizon_m": "float (optional, default 40.0) — overlay approach geodesic cap",
                        "goal_radius": "float (optional, default 0.6)",
                        "standoff_m": "float (optional, default 0.7)",
                        "output_dir": "str (optional, default data/runs/visuals)",
                    },
                },
                "navigate_visual_point": {
                    "session_required": True,
                    "payload": {
                        "image_ref": "str (required) — image_ref from the latest panorama_images row",
                        "point": "array[number, number] (optional) — normalized [x, y] visual anchor",
                        "bbox": "object|array (optional) — normalized {x,y,width,height} or [x,y,width,height]",
                        "anchor": "str (optional, default auto) — auto, center, bottom_center, or lower_band",
                        "intent": "str (optional, default approach) — approach, pass_through, or inspect",
                        "search_radius": "float (optional, default 0.10) — normalized local search radius",
                        "max_steps": "int (optional, default 20)",
                        "horizon_m": "float (optional, default 40.0) — local-hop geodesic cap",
                        "goal_radius": "float (optional, default 0.3)",
                        "standoff_m": "float (optional, default 0.7)",
                        "output_dir": "str (optional, default data/runs/visuals)",
                    },
                },
                "get_oracle_local_map": {
                    "session_required": True,
                    "payload": {
                        "radius_m": (
                            "float (optional, default 5.0) — local map radius "
                            "around the agent"
                        ),
                        "max_objects": "int (optional, default 30)",
                        "label_filter": "str (optional) — filter object labels",
                        "meters_per_pixel": "float (optional, default 0.05)",
                        "include_image": "bool (optional, default true)",
                        "output_dir": "str (optional, default data/runs/visuals)",
                    },
                },
                "close_session": {
                    "session_required": True,
                    "payload": {},
                },
                "get_panorama": {
                    "session_required": True,
                    "payload": {
                        "include_depth_analysis": "bool (optional, default false) — include per-direction depth analysis",
                        "clearance_threshold": "float (optional, default 0.5) — clearance in meters for depth analysis",
                    },
                },
                "analyze_depth": {
                    "session_required": True,
                    "payload": {
                        "clearance_threshold": "float (optional, default 0.5) — min distance in meters to consider clear",
                    },
                },
                "query_depth": {
                    "session_required": True,
                    "payload": {
                        "points": "list of [u, v] pixel coordinates (optional)",
                        "bbox": "[x1, y1, x2, y2] pixel region (optional)",
                        "coordinate_space": "'rgb' (default, latest visible RGB image) or 'depth'",
                        "rgb_width": "int (optional) — visible RGB image width for coordinate mapping",
                        "rgb_height": "int (optional) — visible RGB image height for coordinate mapping",
                    },
                },
                "depth_grid": {
                    "session_required": True,
                    "payload": {
                        "rows": "int (optional, default 5) — grid row count",
                        "cols": "int (optional, default 5) — grid column count",
                        "clearance_threshold": "float (optional, default 0.5) — min distance in meters to consider clear",
                    },
                },
                "side_depth_grid": {
                    "session_required": True,
                    "payload": {
                        "rows": "int (optional, default 5) — grid row count for each side",
                        "cols": "int (optional, default 5) — grid column count for each side",
                        "clearance_threshold": "float (optional, default 0.5) — min distance in meters to consider clear",
                    },
                },
            },
            "supported_actions": list(self.SUPPORTED_ACTIONS),
        }
