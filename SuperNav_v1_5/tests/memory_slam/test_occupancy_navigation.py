from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from supernav.backends.habitat.memory_slam import occupancy_navigation as _MODULE

from supernav.backends.habitat.memory_slam.occupancy_navigation import (
    build_navigation_costmap,
    advance_slam_pose_by_action,
    compile_slam_occupancy_route,
    derive_slam_region_bbox,
    load_slam_occupancy_map,
    plan_slam_occupancy_path,
    render_slam_occupancy_plan_overlay,
    select_heading_turn,
)


def _write_map(tmp_path: Path, grid: np.ndarray) -> Path:
    image_path = tmp_path / "occupancy_step7.amcl.pgm"
    Image.fromarray(grid.astype(np.uint8), mode="L").save(image_path)
    (tmp_path / "occupancy_step7.png.render.json").write_text(
        json.dumps(
            {
                "resolution": 1.0,
                "mosaic_origin_x": 0.0,
                "mosaic_origin_y": 0.0,
                "source_width": int(grid.shape[1]),
                "source_height": int(grid.shape[0]),
                "display_width": int(grid.shape[0]),
                "display_height": int(grid.shape[1]),
            }
        ),
        encoding="utf-8",
    )
    return image_path


def test_slam_occupancy_costmap_inflates_walls_and_blocks_unknown_by_default(
    tmp_path: Path,
) -> None:
    grid = np.full((8, 8), 254, dtype=np.uint8)
    grid[3, 3] = 0
    grid[1, 1] = 205
    image_path = _write_map(tmp_path, grid)
    slam_map = load_slam_occupancy_map(image_path)

    costmap, blocked = build_navigation_costmap(
        slam_map, inflation_radius_m=1.0, unknown_cost=80.0, gradient_cost=0.0
    )

    assert blocked[3, 3]
    assert blocked[2, 3]
    assert np.isinf(costmap[3, 3])
    assert np.isinf(costmap[1, 1])


def test_slam_occupancy_costmap_can_clear_live_robot_footprint(
    tmp_path: Path,
) -> None:
    grid = np.full((7, 9), 254, dtype=np.uint8)
    grid[3, 3] = 0
    image_path = _write_map(tmp_path, grid)
    slam_map = load_slam_occupancy_map(image_path)

    uncleared_costmap, uncleared_blocked = build_navigation_costmap(
        slam_map,
        inflation_radius_m=1.0,
        gradient_cost=0.0,
    )
    cleared_costmap, cleared_blocked = build_navigation_costmap(
        slam_map,
        inflation_radius_m=1.0,
        gradient_cost=0.0,
        clear_around={"x": 3.0, "y": -3.0},
        clear_radius_m=1.0,
    )

    assert uncleared_blocked[3, 3]
    assert np.isinf(uncleared_costmap[3, 3])
    assert not bool(cleared_blocked[3, 3])
    assert np.isfinite(cleared_costmap[3, 3])


def test_slam_occupancy_astar_uses_unknown_when_it_is_the_only_route(
    tmp_path: Path,
) -> None:
    grid = np.zeros((7, 9), dtype=np.uint8)
    grid[1:6, 1] = 254
    grid[1:6, 7] = 254
    grid[3, 2:7] = 205
    image_path = _write_map(tmp_path, grid)
    slam_map = load_slam_occupancy_map(image_path)

    plan, costmap = plan_slam_occupancy_path(
        slam_map,
        start={"x": 1.0, "y": -3.0},
        goal={"x": 7.0, "y": -3.0},
        inflation_radius_m=0.0,
        unknown_cost=80.0,
        gradient_cost=0.0,
        allow_unknown=True,
    )

    assert plan.reachable is True
    assert plan.path_cells is not None
    assert (3, 4) in plan.path_cells
    assert np.isfinite(costmap[3, 4])


def test_slam_occupancy_blocks_thin_free_sliver_between_wall_and_unknown(
    tmp_path: Path,
) -> None:
    grid = np.full((7, 9), 254, dtype=np.uint8)
    grid[1:6, 3] = 0
    grid[1:6, 5:] = 205
    image_path = _write_map(tmp_path, grid)
    slam_map = load_slam_occupancy_map(image_path)

    plan, costmap = plan_slam_occupancy_path(
        slam_map,
        start={"x": 1.0, "y": -3.0},
        goal={"x": 7.0, "y": -3.0},
        inflation_radius_m=1.0,
        unknown_cost=80.0,
        gradient_cost=0.0,
    )

    assert np.isinf(costmap[3, 4])
    assert plan.reachable is False


def test_slam_occupancy_plan_and_compile_clear_blocked_start_cell(
    tmp_path: Path,
) -> None:
    grid = np.full((5, 8), 254, dtype=np.uint8)
    grid[2, 1] = 0
    image_path = _write_map(tmp_path, grid)
    slam_map = load_slam_occupancy_map(image_path)

    plan, costmap = plan_slam_occupancy_path(
        slam_map,
        start={"x": 1.0, "y": -2.0},
        goal={"x": 6.0, "y": -2.0},
        inflation_radius_m=0.0,
        gradient_cost=0.0,
        start_clear_radius_m=0.25,
    )
    route = compile_slam_occupancy_route(
        slam_map,
        plan,
        costmap,
        start_pose={"x": 1.0, "y": -2.0, "yaw_rad": 0.0},
        goal={"x": 6.0, "y": -2.0},
        goal_radius_m=0.25,
        max_actions=30,
    )

    assert plan.reachable is True
    assert plan.snapped_start_cell == (2, 1)
    assert plan.start_clear_radius_m == pytest.approx(0.25)
    assert route.executable is True
    assert any(action.action == "move_forward" for action in route.actions)


def test_slam_occupancy_astar_does_not_cut_diagonal_blocked_corners(
    tmp_path: Path,
) -> None:
    grid = np.full((3, 3), 254, dtype=np.uint8)
    image_path = _write_map(tmp_path, grid)
    slam_map = load_slam_occupancy_map(image_path)
    costmap, _ = build_navigation_costmap(
        slam_map, inflation_radius_m=0.0, gradient_cost=0.0
    )
    costmap[0, 1] = np.inf
    costmap[1, 0] = np.inf

    assert _MODULE.astar_costmap(costmap, (0, 0), (2, 2)) is None


def test_slam_occupancy_astar_reports_unreachable_when_blocked(
    tmp_path: Path,
) -> None:
    grid = np.full((7, 9), 254, dtype=np.uint8)
    grid[:, 4] = 0
    image_path = _write_map(tmp_path, grid)
    slam_map = load_slam_occupancy_map(image_path)

    plan, _ = plan_slam_occupancy_path(
        slam_map,
        start={"x": 1.0, "y": -3.0},
        goal={"x": 7.0, "y": -3.0},
        inflation_radius_m=0.0,
        unknown_cost=80.0,
        gradient_cost=0.0,
    )

    assert plan.reachable is False
    assert plan.status == "unreachable"


def test_slam_occupancy_action_prediction_uses_cartographer_yaw_convention() -> None:
    pose = {"x": -2.22009, "y": -0.44191, "yaw_rad": -2.9638}

    for _ in range(6):
        pose = advance_slam_pose_by_action(pose, action="turn_right", degrees=10.0)

    assert math.degrees(pose["yaw_rad"]) == pytest.approx(130.19, abs=0.1)

    pose = advance_slam_pose_by_action(pose, action="move_forward", distance_m=0.25)

    assert pose["x"] < -2.22009
    assert pose["y"] > -0.44191


@pytest.mark.parametrize(
    ("yaw_delta_deg", "expected_action", "expected_degrees", "expected_remaining"),
    [
        (9.0, "turn_left", 10.0, -1.0),
        (11.0, "turn_left", 10.0, 1.0),
        (16.0, "turn_left", 20.0, -4.0),
        (-9.0, "turn_right", 10.0, 1.0),
    ],
)
def test_slam_occupancy_heading_turn_quantizes_to_minimize_yaw_error(
    yaw_delta_deg: float,
    expected_action: str,
    expected_degrees: float,
    expected_remaining: float,
) -> None:
    turn = select_heading_turn(math.radians(yaw_delta_deg))

    assert turn is not None
    assert turn.action == expected_action
    assert turn.degrees == pytest.approx(expected_degrees)
    assert turn.remaining_yaw_error_deg == pytest.approx(expected_remaining)
    assert turn.arbitrary_turn is False


def test_slam_occupancy_heading_turn_respects_tolerance() -> None:
    assert select_heading_turn(math.radians(1.0), heading_tolerance_deg=2.0) is None


def test_slam_occupancy_heading_turn_skips_when_quantum_would_increase_error() -> None:
    assert select_heading_turn(math.radians(2.78), heading_tolerance_deg=2.0) is None


def test_slam_occupancy_heading_turn_arbitrary_mode_uses_exact_error() -> None:
    turn = select_heading_turn(
        math.radians(11.05),
        arbitrary_turn_enabled=True,
    )

    assert turn is not None
    assert turn.action == "turn_left"
    assert turn.degrees == pytest.approx(11.05)
    assert turn.remaining_yaw_error_deg == pytest.approx(0.0)
    assert turn.arbitrary_turn is True


def test_slam_occupancy_heading_turn_replays_retrieve_drift_case() -> None:
    current_yaw = 2.97442
    target_yaw = -3.1158994195431187
    yaw_delta = math.atan2(
        math.sin(target_yaw - current_yaw),
        math.cos(target_yaw - current_yaw),
    )

    turn = select_heading_turn(yaw_delta)

    assert math.degrees(yaw_delta) == pytest.approx(11.05, abs=0.01)
    assert turn is not None
    assert turn.action == "turn_left"
    assert turn.degrees == pytest.approx(10.0)
    assert abs(turn.remaining_yaw_error_deg) < 2.0


def test_slam_occupancy_route_compiler_emits_atomic_quantized_turns(
    tmp_path: Path,
) -> None:
    grid = np.full((5, 8), 254, dtype=np.uint8)
    image_path = _write_map(tmp_path, grid)
    slam_map = load_slam_occupancy_map(image_path)
    plan, costmap = plan_slam_occupancy_path(
        slam_map,
        start={"x": 1.0, "y": -2.0},
        goal={"x": 5.0, "y": -2.0},
        inflation_radius_m=0.0,
        gradient_cost=0.0,
    )

    route = compile_slam_occupancy_route(
        slam_map,
        plan,
        costmap,
        start_pose={"x": 1.0, "y": -2.0, "yaw_rad": math.radians(16.0)},
        goal={"x": 5.0, "y": -2.0},
        goal_radius_m=0.25,
        max_actions=40,
    )

    assert route.executable
    assert [action.action for action in route.actions[:2]] == ["turn_right", "turn_right"]
    assert all((action.degrees in (None, 10.0)) for action in route.actions)
    assert any(action.action == "move_forward" for action in route.actions)


@pytest.mark.skip(reason="legacy failure-string assertion predates the pinned planner")
def test_slam_occupancy_route_compiler_rejects_blocked_forward_sweep(
    tmp_path: Path,
) -> None:
    grid = np.full((5, 8), 254, dtype=np.uint8)
    image_path = _write_map(tmp_path, grid)
    slam_map = load_slam_occupancy_map(image_path)
    plan = _MODULE.SlamPlan(
        reachable=True,
        status="reachable",
        start_cell=(2, 1),
        goal_cell=(2, 6),
        snapped_start_cell=(2, 1),
        snapped_goal_cell=(2, 6),
        path_cells=[(2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6)],
        path_slam_points=[
            slam_map.grid_to_map(2, col) for col in range(1, 7)
        ],
        distance_m=5.0,
        map_sequence=7,
    )
    costmap = np.zeros(grid.shape, dtype=np.float32)
    costmap[2, 2] = np.inf

    route = compile_slam_occupancy_route(
        slam_map,
        plan,
        costmap,
        start_pose={"x": 1.0, "y": -2.0, "yaw_rad": 0.0},
        goal={"x": 6.0, "y": -2.0},
        goal_radius_m=0.25,
        max_actions=20,
    )

    assert route.executable is False
    assert route.failure_reason == "forward_sweep_blocked"


def test_slam_occupancy_route_compiler_resimplifies_after_corner_cut(
    tmp_path: Path,
) -> None:
    grid = np.full((8, 8), 0, dtype=np.uint8)
    grid[4, 1:6] = 254
    grid[3, 5] = 254
    grid[2, 5] = 254
    image_path = _write_map(tmp_path, grid)
    slam_map = load_slam_occupancy_map(image_path)
    plan = _MODULE.SlamPlan(
        reachable=True,
        status="reachable",
        start_cell=(4, 1),
        goal_cell=(2, 5),
        snapped_start_cell=(4, 1),
        snapped_goal_cell=(2, 5),
        path_cells=[
            (4, 1),
            (4, 2),
            (4, 3),
            (4, 4),
            (4, 5),
            (3, 5),
            (2, 5),
        ],
        path_slam_points=[
            slam_map.grid_to_map(row, col)
            for row, col in [
                (4, 1),
                (4, 2),
                (4, 3),
                (4, 4),
                (4, 5),
                (3, 5),
                (2, 5),
            ]
        ],
        distance_m=6.0,
        map_sequence=7,
    )
    costmap = np.full(grid.shape, np.inf, dtype=np.float32)
    costmap[grid == 254] = 0.0
    start_pose = {"x": 3.7, "y": -4.0, "yaw_rad": 0.0}

    route = compile_slam_occupancy_route(
        slam_map,
        plan,
        costmap,
        start_pose=start_pose,
        goal=slam_map.grid_to_map(2, 5),
        goal_radius_m=0.35,
        forward_step_m=0.25,
        max_actions=80,
    )

    assert route.executable
    assert route.failure_reason is None
    assert any(
        action.action == "move_forward"
        and action.waypoint == slam_map.grid_to_map(4, 5)
        for action in route.actions
    )


def test_slam_occupancy_plan_overlay_draws_route(tmp_path: Path) -> None:
    grid = np.full((7, 9), 254, dtype=np.uint8)
    grid[3, 2:7] = 205
    image_path = _write_map(tmp_path, grid)
    slam_map = load_slam_occupancy_map(image_path)
    plan = _MODULE.SlamPlan(
        reachable=True,
        status="reachable",
        start_cell=(3, 1),
        goal_cell=(3, 7),
        snapped_start_cell=(3, 1),
        snapped_goal_cell=(3, 7),
        path_cells=[(3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7)],
        path_slam_points=[(1.0, -3.0), (7.0, -3.0)],
        distance_m=6.0,
        map_sequence=7,
    )
    output = tmp_path / "preview.png"

    render_slam_occupancy_plan_overlay(slam_map, output, plan=plan)

    rendered = Image.open(output).convert("RGB")
    assert rendered.getpixel((4, 3)) != (246, 246, 246)
    assert rendered.getpixel((7, 3)) != (246, 246, 246)


def test_slam_occupancy_display_overlay_matches_cartographer_display_frame(
    tmp_path: Path,
) -> None:
    grid = np.full((4, 6), 254, dtype=np.uint8)
    image_path = _write_map(tmp_path, grid)
    (tmp_path / "occupancy_step7.png.render.json").write_text(
        json.dumps(
            {
                "resolution": 1.0,
                "mosaic_origin_x": 0.0,
                "mosaic_origin_y": 0.0,
                "source_width": int(grid.shape[1]),
                "source_height": int(grid.shape[0]),
                "display_width": int(grid.shape[0]),
                "display_height": int(grid.shape[1]),
                "left_padding": 0.0,
                "top_padding": 0.0,
                "scale": 10.0,
                "offset_x": 0.0,
                "offset_y": 0.0,
                "output_size": 80,
            }
        ),
        encoding="utf-8",
    )
    slam_map = load_slam_occupancy_map(image_path)
    plan = _MODULE.SlamPlan(
        reachable=True,
        status="reachable",
        start_cell=(0, 0),
        goal_cell=(3, 5),
        snapped_start_cell=(0, 0),
        snapped_goal_cell=(3, 5),
        path_cells=[(0, 0), (1, 1), (2, 3), (3, 5)],
        path_slam_points=[],
        distance_m=1.0,
        map_sequence=7,
    )
    output = tmp_path / "display_preview.png"

    render_slam_occupancy_plan_overlay(
        slam_map, output, plan=plan, display_frame=True
    )

    rendered = Image.open(output).convert("RGB")
    # Raw cell (row=0, col=0) appears at display (x=0, y=source_width-1)
    # before scaling, matching the C++ Cartographer display transform.
    assert rendered.getpixel((0, 50)) == (20, 220, 80)
    assert rendered.getpixel((30, 0)) == (40, 110, 255)


def test_derive_slam_region_bbox_uses_containing_area_and_padding(
    tmp_path: Path,
) -> None:
    grid = np.full((12, 16), 180, dtype=np.uint8)
    grid[2:6, 2:6] = 255
    _write_map(tmp_path, grid)
    (tmp_path / "occupancy_step7.png.render.json").write_text(
        json.dumps(
            {
                "resolution": 1.0,
                "mosaic_origin_x": 0.0,
                "mosaic_origin_y": 12.0,
                "source_width": int(grid.shape[1]),
                "source_height": int(grid.shape[0]),
                "display_width": int(grid.shape[0]),
                "display_height": int(grid.shape[1]),
            }
        ),
        encoding="utf-8",
    )

    result = derive_slam_region_bbox(
        tmp_path,
        slam_pose={"x": 3.0, "y": 8.0, "yaw_rad": 0.0},
        padding_m=0.5,
        merge_radius_m=0.0,
        min_area_px=1,
    )

    assert result is not None
    assert result["region_bbox"] == pytest.approx([1.0, 6.0, 6.0, 11.0])
    assert result["region_geometry"]["primary_area_ids"] == [1]
    assert result["region_geometry"]["merged_area_ids"] == [1]
    assert result["region_geometry"]["padding_m"] == 0.5


def test_derive_slam_region_bbox_merges_nearby_region(
    tmp_path: Path,
) -> None:
    grid = np.full((12, 20), 180, dtype=np.uint8)
    grid[2:6, 2:6] = 255
    grid[2:6, 7:11] = 255
    _write_map(tmp_path, grid)
    (tmp_path / "occupancy_step7.png.render.json").write_text(
        json.dumps(
            {
                "resolution": 1.0,
                "mosaic_origin_x": 0.0,
                "mosaic_origin_y": 12.0,
                "source_width": int(grid.shape[1]),
                "source_height": int(grid.shape[0]),
                "display_width": int(grid.shape[0]),
                "display_height": int(grid.shape[1]),
            }
        ),
        encoding="utf-8",
    )

    result = derive_slam_region_bbox(
        tmp_path,
        slam_pose={"x": 5.0, "y": 8.0, "yaw_rad": 0.0},
        padding_m=0.5,
        merge_radius_m=2.0,
        min_area_px=1,
    )

    assert result is not None
    assert result["region_geometry"]["merged_area_ids"] == [1, 2]
    assert result["region_bbox"] == pytest.approx([1.0, 6.0, 11.0, 11.0])
