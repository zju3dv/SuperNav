"""Keep engineering defaults distinct from a confirmed comparison protocol."""

from dataclasses import asdict, dataclass
import math


BUILD_COMMIT = "f0825767cd50d69f666c7f282e54abfe58f1e917"
CHECKPOINT_SHA256 = "efd3f50def5f65d0aa57da1d2bbe1b159cf7b14e870d82495d4b4167c0aa62dd"


def vertical_fov(horizontal: float, width: int, height: int) -> float:
    return math.degrees(2 * math.atan(math.tan(math.radians(horizontal) / 2) * height / width))


def horizontal_fov(vertical: float, width: int, height: int) -> float:
    return math.degrees(2 * math.atan(math.tan(math.radians(vertical) / 2) * width / height))


@dataclass(frozen=True)
class NativeConfig:
    profile: str = "packaged_geometry_probe"
    width: int = 640
    height: int = 480
    vertical_fov_deg: float = 120.0
    grid_size_m: float = 0.1
    move_magnitude_m: float = 0.1
    turn_degrees: float = 10.0
    snap_to_grid: bool = False
    visibility_distance_m: float = 20.0
    agent_mode: str = "default"
    max_actions: int = 500
    localnav_stop_probability: float = 0.1
    localnav_execute_actions: int = 3
    localnav_confidence_threshold: float = 0.2
    camera_height_m: float | None = None
    success_scoring: str = "withheld"

    def __post_init__(self):
        if self.profile not in ("packaged_geometry_probe", "neednav_h125_unscored", "neednav_fourview_unscored") or self.success_scoring != "withheld":
            raise ValueError("Formal evaluation requires a confirmed Lixing baseline protocol and separate scorer")
        if self.profile == "neednav_h125_unscored" and self.camera_height_m != 1.25:
            raise ValueError("The height-controlled cohort requires a 1.25 m camera")
        if self.profile == "packaged_geometry_probe" and self.camera_height_m is not None:
            raise ValueError("Camera changes require the separate height-controlled cohort")
        if self.profile == "neednav_fourview_unscored" and (
            self.camera_height_m != 1.25 or (self.width, self.height) != (640, 480)
            or abs(self.vertical_fov_deg - vertical_fov(90, 640, 480)) > .001
            or self.max_actions != 500 or self.turn_degrees != 10 or self.move_magnitude_m != .1
        ):
            raise ValueError("Four-view protocol requires 640x480, HFOV 90, height 1.25m, 500 units, 0.1m/10deg")
        if self.width <= 0 or self.height <= 0 or not 0 < self.vertical_fov_deg < 180:
            raise ValueError("Invalid camera settings")
        for value in (self.grid_size_m, self.move_magnitude_m, self.turn_degrees):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid action quantum")
        if self.max_actions <= 0 or self.localnav_execute_actions <= 0:
            raise ValueError("Invalid engineering safety limit")
        if not 0 <= self.localnav_stop_probability <= 1:
            raise ValueError("Invalid stop probability")

    def controller_kwargs(self):
        return dict(width=self.width, height=self.height, fieldOfView=self.vertical_fov_deg,
                    gridSize=self.grid_size_m, snapToGrid=self.snap_to_grid,
                    rotateStepDegrees=self.turn_degrees, visibilityDistance=self.visibility_distance_m,
                    agentMode=self.agent_mode, renderDepthImage=False, renderInstanceSegmentation=False,
                    makeAgentsVisible=False)

    def receipt(self):
        result = dict(**asdict(self), simulator="ai2thor-5.0.0", build_commit=BUILD_COMMIT,
                    horizontal_fov_deg=horizontal_fov(self.vertical_fov_deg, self.width, self.height),
                    baseline_alignment_confirmed=False,
                    frozen_from_package=["vertical_fov_deg", "grid_size_m", "move_magnitude_m", "snap_to_grid", "visibility_distance_m"],
                    provisional=["width", "height", "turn_degrees", "agent_mode", "max_actions", "observation_permissions"],
                    body_height_m_source_default=1.8, body_radius_m_source_default=0.2,
                    camera_implementation="native_main" if self.camera_height_m is None else "synchronized_egocentric_render_camera",
                    camera_height_reference="house floor plane; all frozen houses must have one floor elevation",
                    controller_variant="native_quantized_nomad_collision_return_to_brain",
                    observation_permissions="front RGB only; no semantic metadata, GT, depth or global map",
                    budget_scope="engineering primitive-action safety cap, not a final comparison budget",
                    collision_policy="native AI2-THOR physics; not claimed equivalent to Habitat allow_sliding")
        if self.profile == "neednav_fourview_unscored":
            result.update(observation_views=["front", "right", "back", "left"], wall_budget_s=3600,
                          observation_permissions="four directional RGB; no semantic metadata, GT, depth or global map",
                          camera_implementation="four_synchronized_egocentric_render_cameras",
                          observation_budget_cost=0, side_goal_alignment="physical turns charged per 10 degrees")
            result["frozen_from_package"].remove("vertical_fov_deg")
            result["provisional"].extend(["vertical_fov_deg", "camera_height_m", "wall_budget_s"])
        return result
