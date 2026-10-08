# Copyright (c) Meta Platforms, Inc. and its affiliates.
# Licensed under the MIT license; see THIRD_PARTY_NOTICES.md in the repository root.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_DEFAULT_MAX_OBSERVATION_ELEMENTS = 2048
_DEFAULT_DEPTH_VIS_MAX = 10.0
_DEFAULT_VISUAL_OUTPUT_DIR = "data/runs/visuals"
_DEFAULT_TOPDOWN_METERS_PER_PIXEL = 0.05
_DEFAULT_VIDEO_FPS = 6.0
_API_VERSION = "habitat-gs/v1"
from habitat_contract.navigation_state import HabitatAdapterError

SUPPORTED_ACTIONS = (
    "describe_api",
    "init_scene",
    "get_scene_info",
    "set_agent_state",
    "reset_agent_pose",
    "sample_navigable_point",
    "find_shortest_path",
    "get_topdown_map",
    "navigate_oracle_local",
    "navigate_visual_ground_preview",
    "navigate_visual_overlay",
    "navigate_visual_point",
    "get_oracle_local_map",
    "navigate_step",
    "step_action",
    "step_and_capture",
    "get_observation",
    "get_visuals",
    "export_video_trace",
    "get_metrics",
    "get_runtime_status",
    "get_collision_profile",
    "close_session",
    "get_panorama",
    "analyze_depth",
    "query_depth",
    "depth_grid",
    "side_depth_grid",
    "get_scene_graph",
)


@dataclass
class _Session:
    session_id: str
    simulator: Any
    scene: str
    settings: Dict[str, Any]
    agent_id: int = 0
    step_count: int = 0
    last_action: Optional[str] = None
    last_sensor_obs: Optional[Dict[str, Any]] = None
    created_at_s: float = 0.0
    last_activity_s: float = 0.0
    trajectory: List[List[float]] = field(default_factory=list)
    # Full evaluator/audit path. Public trajectory is capped for response size.
    audit_trajectory: List[List[float]] = field(default_factory=list)
    collision_points: List[Dict[str, Any]] = field(default_factory=list)
    last_goal: Optional[List[float]] = None
    last_collision: Optional[Dict[str, Any]] = None
    mapless: bool = False
    is_gaussian: bool = False  # True if scene uses GS rendering
    capture_counter: int = 0  # unique per-capture counter for image filenames
    camera_pitch_deg: float = 0.0  # Positive values mean the agent is looking down.
    # SPL metrics: accumulate real path length independently of trajectory cap
    cumulative_path_length: float = 0.0
    # Start-of-episode geodesic distance to goal (l_opt in SPL formula).
    initial_geodesic_distance: Optional[float] = None
    # Evaluation ground-truth goal — invisible to the agent.
    # Used only by _build_debug_snapshot and SPL computation.
    # For pointnav this equals last_goal; for other tasks this is independent
    # so the agent never sees eval coordinates or polar signals derived from them.
    eval_goal: Optional[List[float]] = None
    # Precomputed scene graph loaded from room_object_scene_graph.json at init_scene.
    # None if the file was not found for the current scene.
    scene_graph: Optional[Dict[str, Any]] = None
    # Internal visible navigation targets from the latest panorama capture.
    # These are not MCP-agent-visible by default; visual local navigation uses
    # them to bind detector boxes to scene-graph object refs.
    last_panorama_images: List[Dict[str, Any]] = field(default_factory=list)
    last_visible_nav_target_refs: set[str] = field(default_factory=set)
    last_visible_nav_targets: List[Dict[str, Any]] = field(default_factory=list)
    last_visible_nav_targets_status: Optional[str] = None
    # Session-private overlay alias registry for the latest panorama capture.
    # Shape: image_ref -> target_ref -> {target_ref, direction, capture_seq}.
    last_visual_overlay_aliases: Dict[str, Dict[str, Dict[str, Any]]] = field(
        default_factory=dict
    )
    # Session-local image metadata for visual point navigation. Entries are
    # keyed by public image_ref and may include private camera basis/depth data.
    last_visual_image_refs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    latest_visual_capture_seq: Optional[int] = None
    pending_visual_point_preview: Optional[Dict[str, Any]] = None
    pending_visual_ground_preview: Optional[Dict[str, Any]] = None
    # Visual robot (third-person camera proxy)
    visual_robot_enabled: bool = False
    visual_robot_kind: Optional[str] = None  # "articulated" | "rigid"
    visual_robot_obj: Any = None
    visual_robot_template_handle: Optional[str] = None
