"""Offline differential fixtures; no robot, external model, or network is used.

Run through audit_snapshot.py so the same fixtures execute in separate Python
processes against the approved source and the assembled release. Transport,
planner completion, clocks, and artifact destinations are controlled; map
rendering, annotation, tool filtering, target resolution, caption selection,
memory persistence, and A* execute their real implementations.
"""

import argparse
import base64
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
from io import BytesIO
from pathlib import Path
import queue
import sys
import tempfile
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
import uuid


def compact(value):
    """Compare image bytes and structured payloads, not JSON whitespace."""
    if isinstance(value, dict):
        result = {key: compact(item) for key, item in value.items()}
        if value.get("type") == "image" and "data" in value:
            result["data"] = (
                "sha256:" + hashlib.sha256(base64.b64decode(value["data"])).hexdigest()
            )
        if value.get("type") == "text":
            try:
                result["text"] = compact(json.loads(value["text"]))
            except (ValueError, TypeError):
                pass
        return result
    if isinstance(value, (list, tuple)):
        return [compact(item) for item in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tree", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.tree))
    import dimos

    assert Path(dimos.__file__).resolve().is_relative_to(args.tree.resolve())
    # Load only the SDK constants. Importing its package initializer opens the
    # host audio stack, which is unrelated to these fixtures and needs hardware.
    spec = importlib.util.find_spec("unitree_webrtc_connect")
    sdk = ModuleType("unitree_webrtc_connect")
    sdk.__path__ = list(spec.submodule_search_locations)
    sys.modules[sdk.__name__] = sdk
    # Both processes use the freshly built binary, whose C++ source is checked
    # byte-for-byte by audit_snapshot before this probe is allowed to run.
    import dimos.navigation.replanning_a_star as astar_package

    astar_package.__path__.append(
        str(args.runtime / "dimos/navigation/replanning_a_star")
    )
    import numpy as np
    from dimos.agents.skill_result import SkillResult
    from dimos.agents.mcp.external_harness_authority_proxy import (
        ExternalHarnessAuthorityProxy,
        MovementAuthorityEnvelope,
        _overlay_graph_badges_on_image_block,
        _sanitize_agent_navigation_result,
    )
    from dimos.agents.mcp.test_external_harness_authority_proxy import (
        FakeMcpAdapter,
        PixelGoalMcpAdapter,
    )
    from dimos.mapping.occupancy.room_segmentation import render_room_segmentation
    from dimos.mapping.occupancy.habitat_area_segmentation import build_slam_area_nodes
    from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
    from dimos.msgs.geometry_msgs.Pose import Pose
    from dimos.msgs.geometry_msgs.Vector3 import Vector3
    from dimos.msgs.nav_msgs.OccupancyGrid import OccupancyGrid
    from dimos.navigation.base import NavigationState
    from dimos.navigation.replanning_a_star import min_cost_astar as astar
    from dimos.robot.unitree.unitree_skill_container import (
        UnitreeSkillContainer,
        _MOVEMENT_IMAGE_ATTACHMENT_TOOLS,
        _SLAM_MAP_ATTACHMENT_TOOLS,
    )

    assert astar._USE_CPP, (
        "Build the A* extension first; fallback must not hide a broken build"
    )
    package = (
        "pixelgoal"
        if (args.tree / "dimos/robot/unitree/pixelgoal").exists()
        else "nomad"
    )
    caption_type = importlib.import_module(
        f"dimos.robot.unitree.{package}.area_captioning"
    ).Go2AreaCaptionWorker
    pixel_module = importlib.import_module(
        f"dimos.robot.unitree.{package}.pixel_goal_map_module"
    )
    result = {}
    grid = np.full((80, 120), -1, dtype=np.int8)
    grid[8:72, 10:110] = 0
    grid[8:72, 10] = grid[8:72, 109] = 100
    grid[8, 10:110] = grid[71, 10:110] = 100
    grid[8:72, 60] = 100
    grid[35:46, 60] = 0
    grid[18:24, 25:32] = 100
    occupancy = OccupancyGrid(
        grid=grid,
        resolution=0.1,
        origin=Pose(position=Vector3(-3, -2, 0)),
        frame_id="world",
        ts=123,
    )
    pose = PoseStamped(position=Vector3(-0.5, 0, 0.3), frame_id="world", ts=123)

    with tempfile.TemporaryDirectory(prefix="go2-snapshot-probe-") as tmp:
        root = Path(tmp)
        with (
            patch("time.time", return_value=1800000000.0),
            patch("time.monotonic", return_value=1000.0),
            patch("uuid.uuid4", return_value=uuid.UUID(int=1)),
        ):
            robot = object.__new__(UnitreeSkillContainer)
            robot._latest_global_costmap = occupancy
            robot._latest_global_costmap_received_at_s = 999.0
            robot._slam_pose_trace = []
            robot._floor_plan_size_px = robot._2d_slam_map_size_px = 240
            robot._map_frame_pose = lambda frame: pose if frame == "world" else None
            robot._current_base_pose = robot._current_direct_motion_pose = lambda: pose
            robot._tf = SimpleNamespace(
                get=lambda *_: SimpleNamespace(to_pose=lambda: pose)
            )
            for name in [
                "_record_2d_slam_map_artifact",
                "_record_floor_plan_artifact",
                "_record_area_segmentation_artifact",
                "_attach_latest_camera_image",
                "_export_task_rerun_recording",
                "_record_latest_motion_result",
                "_update_active_navigation_task",
                "_finish_memory_motion_recorder",
            ]:
                setattr(robot, name, lambda *a, **k: None)
            robot._start_memory_motion_recorder = lambda *_: None
            result["attachment_tool_sets"] = [
                sorted(_MOVEMENT_IMAGE_ATTACHMENT_TOOLS),
                sorted(_SLAM_MAP_ATTACHMENT_TOOLS),
            ]
            attachments = {}
            for tool in sorted(
                _SLAM_MAP_ATTACHMENT_TOOLS
                | {"observe", "observe_surround", "go2_situation", "pixelgoal_status"}
            ):
                robot._slam_pose_trace = []
                outcome = SkillResult.ok()
                robot._trace_movement_skill_result(tool, outcome)
                expected_images = (
                    1
                    if tool
                    in {
                        "go2_navigate_to_memory_node",
                        "go2_report_relocalization_prior",
                        "go2_slam_map_snapshot",
                    }
                    else 3
                    if tool in _MOVEMENT_IMAGE_ATTACHMENT_TOOLS
                    else 0
                )
                assert len(outcome.content_blocks) == expected_images, (
                    tool,
                    outcome.metadata,
                )
                attachments[tool] = compact(outcome.agent_encode())
            result["tool_map_attachments"] = attachments
            availability = {}
            for label, age, include_pose, available_pose in [
                ("stale", 10, True, pose),
                ("no_pose", 1, True, None),
                ("static", 1, False, None),
                ("current", 1, True, pose),
            ]:
                robot._latest_global_costmap_received_at_s = 1000 - age
                robot._map_frame_pose = lambda _, p=available_pose: p
                outcome = SkillResult.ok()
                robot._attach_2d_slam_map(
                    "go2_slam_map_snapshot", outcome, include_pose=include_pose
                )
                availability[label] = compact(outcome.agent_encode())
            result["map_availability"] = availability
            robot._latest_global_costmap_received_at_s = 999
            robot._map_frame_pose = lambda _: pose

            maps = {}
            for label, native_grid in [
                ("two_rooms", grid),
                ("empty", np.full_like(grid, -1)),
                (
                    "boundary",
                    np.pad(np.zeros((20, 35), dtype=np.int8), 1, constant_values=100),
                ),
            ]:
                og = OccupancyGrid(
                    grid=native_grid,
                    resolution=0.1,
                    origin=Pose(position=Vector3(-3, -2, 0)),
                    frame_id="world",
                    ts=123,
                )
                rendered = render_room_segmentation(og, pose=pose)
                nodes = build_slam_area_nodes(
                    rendered.occupancy_grid, rendered.region_labels
                )
                image = copy.deepcopy(rendered.image_block)
                metadata = {**rendered.metadata, "slam_area_nodes": nodes}
                displayed = _overlay_graph_badges_on_image_block(
                    image,
                    metadata,
                    [
                        {
                            "id": "start",
                            "order_id": 1,
                            "kind": "area",
                            "name": "start",
                            "has_pano_images": True,
                            "slam_pose": {"x": -0.5, "y": 0},
                        },
                        {
                            "id": "nearby",
                            "order_id": 2,
                            "kind": "area",
                            "name": "nearby",
                            "has_pano_images": True,
                            "slam_pose": {"x": -0.49, "y": 0.01},
                        },
                    ],
                )
                maps[label] = {
                    "raw": compact(rendered.image_block),
                    "annotated": compact(image),
                    "metadata": metadata,
                    "nodes": nodes,
                    "displayed": displayed,
                    "labels_sha256": hashlib.sha256(
                        rendered.region_labels.tobytes()
                    ).hexdigest(),
                }
            result["segmentation_and_annotation"] = maps

            navigation = {}
            raw_navigation = {}
            for mode in [
                "arrived",
                "rejected",
                "dispatch_timeout",
                "terminal_timeout",
                "no_path",
                "invalid",
                "stale",
                "frame_mismatch",
                "no_odom",
                "not_configured",
            ]:
                calls = []
                robot._latest_global_costmap_received_at_s = (
                    990 if mode == "stale" else 999
                )
                robot._navigation_not_configured = lambda: mode == "not_configured"
                robot._tf = SimpleNamespace(
                    get=lambda *_: (
                        None
                        if mode == "no_odom"
                        else SimpleNamespace(to_pose=lambda: pose)
                    )
                )

                def dispatch(name, goal, **kwargs):
                    calls.append(
                        [
                            name,
                            [
                                goal.position.x,
                                goal.position.y,
                                goal.orientation.to_euler().z,
                            ],
                            kwargs,
                        ]
                    )
                    if mode == "dispatch_timeout":
                        raise TimeoutError()
                    return mode != "rejected"

                robot._navigation_call = dispatch
                robot._navigation_cancel_goal = lambda **kw: (
                    calls.append(["cancel_goal", kw]) or True
                )
                robot._navigation_failure_or = lambda fallback: fallback
                robot._wait_for_memory_navigation_terminal = lambda **kw: (
                    "TIMEOUT"
                    if mode == "terminal_timeout"
                    else "NO_PATH"
                    if mode == "no_path"
                    else "ARRIVED",
                    NavigationState.IDLE,
                    True,
                )
                robot._slam_pose_trace = []
                outcome = inspect.unwrap(
                    UnitreeSkillContainer.go2_navigate_to_memory_node
                )(
                    robot,
                    "start",
                    target_slam_pose=None
                    if mode == "invalid"
                    else {"x": -0.5, "y": 0, "yaw_rad": 0.5, "frame_id": "world"},
                    map_frame="other" if mode == "frame_mismatch" else "world",
                    map_sequence=7,
                    timeout_s=8,
                )
                assert (
                    outcome.error_code
                    == {
                        "arrived": None,
                        "rejected": "NO_PATH",
                        "dispatch_timeout": "TIMEOUT",
                        "terminal_timeout": "TIMEOUT",
                        "no_path": "NO_PATH",
                        "invalid": "INVALID_TARGET",
                        "stale": "STALE_MAP",
                        "frame_mismatch": "FRAME_MISMATCH",
                        "no_odom": "NO_ODOM",
                        "not_configured": "NOT_CONFIGURED",
                    }[mode]
                )
                navigation[mode] = {
                    "calls": calls,
                    "result": compact(outcome.agent_encode()),
                }
                raw_navigation[mode] = {"content": outcome.agent_encode()}
            result["memory_navigation"] = navigation

            # Real terminal map path, including its pose-unavailable behavior.
            terminal = object.__new__(pixel_module.PixelGoalMapNavModule)
            terminal._state_lock = threading.RLock()
            terminal._latest_costmap = occupancy
            terminal._latest_costmap_received_at_s = 999
            terminal._latest_odom = pose
            terminal._caption_worker = None
            terminal._sink = SimpleNamespace(event=lambda *a, **k: None)
            terminal_results = []
            raw_terminal_image = None
            for odom in [pose, None]:
                terminal._latest_odom = odom
                payload = {}
                block = terminal._terminal_2d_slam_map_content(payload)
                if odom is not None:
                    raw_terminal_image = block
                terminal_results.append({"payload": payload, "image": compact(block)})
            result["pixelgoal_terminal_maps"] = terminal_results

            # Same fake MCP transport as the snapshot tests, but compare full
            # output schemas, maps, stored state, and forwarding, not assertions.
            class FixtureAdapter(PixelGoalMcpAdapter):
                def call_tool(self, name, arguments=None, *, meta=None):
                    response = super().call_tool(name, arguments, meta=meta)
                    if name == "go2_navigate_to_memory_node":
                        return copy.deepcopy(raw_navigation["arrived"])
                    if name == "go2_slam_map_snapshot":
                        return {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(
                                        {
                                            "success": True,
                                            "metadata": terminal_results[0]["payload"],
                                        }
                                    ),
                                },
                                copy.deepcopy(raw_terminal_image),
                            ]
                        }
                    if name in {"pixelgoal_status", "go2_report_relocalization_prior"}:
                        payload = json.loads(response["content"][0]["text"])
                        location = (
                            payload
                            if name == "pixelgoal_status"
                            else payload.setdefault("metadata", {})
                        )
                        location["2d_slam_map"] = terminal_results[0]["payload"][
                            "2d_slam_map"
                        ]
                        response["content"][0]["text"] = json.dumps(payload)
                        response["content"].append(copy.deepcopy(raw_terminal_image))
                    return response

            upstream = FixtureAdapter()
            upstream.tools.extend(FakeMcpAdapter(include_required_tools=True).tools)
            upstream.tools.append(
                {
                    "name": "go2_navigate_to_memory_node",
                    "inputSchema": {"type": "object"},
                    "_meta": {"dimos/uses": ["movement"]},
                }
            )
            upstream.tools.append(
                {
                    "name": "go2_report_relocalization_prior",
                    "inputSchema": {"type": "object"},
                }
            )
            envelope = MovementAuthorityEnvelope.create(
                task_goal="probe",
                scope="offline",
                area="fixture",
                duration_s=60,
                allowed_movement_tools=(
                    "relative_move",
                    "pixelgoal_navigate",
                    "go2_navigate_to_memory_node",
                ),
            )
            proxy = ExternalHarnessAuthorityProxy(
                upstream, envelope=envelope, graph_memory_path=root / "graph.json"
            )
            result["visible_tools_authorized"] = proxy.list_tools()
            result["visible_tools_without_authority"] = ExternalHarnessAuthorityProxy(
                upstream
            ).list_tools()
            proxy._cache_navigation_context_from_result(
                upstream.call_tool("go2_slam_map_snapshot")
            )
            actions = [
                (
                    "go2_apply_graph_mutations",
                    {
                        "mutations": [
                            {
                                "op": "create",
                                "node": {
                                    "id": "hall",
                                    "kind": "area",
                                    "name": "hall",
                                    "description": "Hall with cabinet.",
                                    "records": [],
                                    "pano_images": {"front": "artifact:front"},
                                    "waypoints": {},
                                },
                            }
                        ]
                    },
                ),
                ("go2_query_route_context", {"return_images": False}),
                ("go2_inspect_graph_node", {"id": "hall", "return_images": False}),
                ("go2_report_graph_position", {"id": "hall"}),
                ("go2_navigate_to_memory_node", {"node_id": "hall", "timeout_s": 8}),
                ("pixelgoal_navigate", {"point": [0.5, 0.6], "direction": "front"}),
                (
                    "pixelgoal_navigate",
                    {"point": [0.5, 0.6], "direction": "front", "frame_ts": 1},
                ),
                ("stop_movement", {}),
                ("pixelgoal_stop", {}),
                ("reset_pixel_goal_map", {}),
                ("go2_slam_map_snapshot", {}),
                ("go2_report_relocalization_prior", {}),
            ]
            outputs = {}
            for i, (name, arguments) in enumerate(actions):
                outputs[f"{i}:{name}"] = compact(proxy.call_tool(name, arguments))
            result["proxy_tool_results"] = outputs
            result["proxy_forwarded_calls"] = compact(upstream.calls)
            result["persisted_memory"] = json.loads((root / "graph.json").read_text())
            # Resolve persistent and ephemeral targets only from current,
            # frame-consistent map metadata, and reject stale/legacy anchors.
            metadata = copy.deepcopy(maps["two_rooms"]["metadata"])
            metadata.update(
                status="current", map_frame="world", pose={"frame_id": "world"}
            )
            proxy._latest_slam_map_metadata = metadata
            targets = {}
            for target in ["hall", "slam_area_1", "slam_area_missing", "missing", ""]:
                targets[target] = proxy._resolve_memory_navigation_target(target)
            assert targets["hall"][1] is None and targets["slam_area_1"][1] is None
            metadata["status"] = "static"
            targets["static"] = proxy._resolve_memory_navigation_target("hall")
            metadata["status"] = "current"
            metadata["pose"] = {"frame_id": "other"}
            targets["wrong_robot_frame"] = proxy._resolve_memory_navigation_target(
                "hall"
            )
            result["memory_target_resolution"] = targets
            sanitized = _sanitize_agent_navigation_result(
                copy.deepcopy(raw_navigation["arrived"])
            )
            result["agent_geometry_visibility"] = compact(sanitized)

            # Caption selection uses explicit metric grid geometry; the old
            # source tests accidentally assumed a 1m default grid resolution.
            worker = object.__new__(caption_type)
            worker.root = root
            keyframes = [
                {
                    "keyframe_id": "inside",
                    "step_count": 1,
                    "pose": {"x": 3.1, "y": 2.1},
                },
                {
                    "keyframe_id": "ray",
                    "step_count": 2,
                    "pose": {"x": 1, "y": 6, "yaw_rad": 0},
                },
            ]
            worker.keyframes_path.write_text(
                "".join(json.dumps(row) + "\n" for row in keyframes)
            )
            worker.results_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "job_id": f"{key}:{view}",
                            "caption": f"{key}:{view}",
                            "status": "ready",
                        }
                    )
                    + "\n"
                    for key in ["inside", "ray"]
                    for view in ["front", "right", "back", "left"]
                )
            )
            areas = np.zeros((10, 10), dtype=np.int32)
            areas[2, 3] = 1
            areas[6, 5] = 2
            caption_grid = OccupancyGrid(
                grid=np.zeros((10, 10), dtype=np.int8), resolution=1
            )
            caption_nodes = [
                {"area_id": 1, "centroid_grid": [2, 3]},
                {"area_id": 2, "centroid_grid": [6, 5]},
                {"area_id": 3, "centroid_grid": [9, 9]},
            ]
            result["caption_selection"] = worker.enrich_nodes(
                caption_nodes, caption_grid, areas
            )
            assert [node["caption_status"] for node in caption_nodes] == [
                "ready",
                "ready",
                "unavailable",
            ]
            # Run the actual encoder/queue/HTTP request/result-persistence path
            # synchronously. Only urlopen is replaced; no caption server call.
            from dimos.msgs.sensor_msgs.Image import Image

            caption_exchange = {}
            for mode in ["ready", "missing_caption", "http_error"]:
                worker.root = root / mode
                worker.backend_url = "http://caption.invalid/caption"
                worker.timeout_s = 60
                worker._queue = queue.Queue(maxsize=16)
                worker._processing_lock = threading.Lock()
                worker._processing = worker._closed = False
                captured = []

                def respond(request, timeout):
                    document = json.loads(request.data)
                    captured.append(
                        {
                            "url": request.full_url,
                            "payload": document,
                            "timeout": timeout,
                        }
                    )
                    if mode == "http_error":
                        raise OSError("offline fixture")
                    return BytesIO(
                        json.dumps(
                            {
                                "results": [
                                    {
                                        "job_id": item["job_id"],
                                        **(
                                            {"caption": "  corridor  "}
                                            if mode == "ready"
                                            else {}
                                        ),
                                    }
                                    for item in document["items"]
                                ]
                            }
                        ).encode()
                    )

                image = Image.from_numpy(np.full((8, 10, 3), 120, dtype=np.uint8))
                worker.submit(
                    {
                        direction: image
                        for direction in ["front", "right", "back", "left"]
                    },
                    pose,
                    3,
                )
                worker._queue.put(None)
                with patch("urllib.request.urlopen", side_effect=respond):
                    worker._run()
                rows = [
                    json.loads(line)
                    for line in worker.results_path.read_text().splitlines()
                ]
                assert len(rows) == 4 and all(
                    row["status"] == ("ready" if mode == "ready" else "error")
                    for row in rows
                )
                caption_exchange[mode] = {
                    "requests": captured,
                    "results": rows,
                    "jobs": worker._read_jsonl(worker.jobs_path),
                    "keyframes": worker._read_jsonl(worker.keyframes_path),
                }
            result["caption_transport_and_persistence"] = caption_exchange

            paths = {}
            for label, cost in [
                ("free", 0),
                ("unknown", -1),
                ("49", 49),
                ("50", 50),
                ("99", 99),
                ("blocked", 100),
            ]:
                costgrid = np.zeros((24, 30), dtype=np.int8)
                costgrid[:, 14] = cost
                for gap in [False, True]:
                    costgrid[18:20, 14] = 0 if gap else cost
                    og = OccupancyGrid(
                        grid=costgrid,
                        resolution=0.1,
                        origin=Pose(position=Vector3(-3, -2, 0)),
                        frame_id="world",
                        ts=123,
                    )
                    for penalty in [0.5, 1.0]:
                        for cpp in [False, True]:
                            path = astar.min_cost_astar(
                                og,
                                og.grid_to_world((26, 5)),
                                og.grid_to_world((4, 5)),
                                unknown_penalty=penalty,
                                use_cpp=cpp,
                            )
                            paths[f"{label}:gap={gap}:unknown={penalty}:cpp={cpp}"] = (
                                None
                                if path is None
                                else [[p.position.x, p.position.y] for p in path.poses]
                            )
            result["astar_paths"] = paths

            # Actual forced-prior trigger both before/after the first map.
            from dimos.mapping.relocalization.test_module import (
                _module_with_attempt_capture,
                _SmallMap,
            )
            from dimos_lcm.std_msgs import Bool

            reloc = []
            for before in [False, True]:
                module, attempts = _module_with_attempt_capture()
                cloud = _SmallMap(1)
                if before:
                    module._remember_global_map(cloud)
                module._on_relocalization_request(Bool(data=True))
                pending_before = module._pending_forced_relocalization
                if not before:
                    module._remember_global_map(cloud)
                reloc.append(
                    {
                        "normal_threshold_passed": module._has_enough_points(cloud),
                        "pending_before_map": pending_before,
                        "attempts": [x.point_count for x in attempts],
                        "pending_after_map": module._pending_forced_relocalization,
                    }
                )
            result["relocalization_forced_prior"] = reloc

            import open3d as o3d
            from dimos.mapping.relocalization.relocalize import (
                _crop_global_map_to_prior,
            )

            cloud = o3d.geometry.PointCloud()
            cloud.points = o3d.utility.Vector3dVector(
                np.random.default_rng(42).uniform(-4, 4, (10000, 3))
            )
            crops = []
            for prior in [
                None,
                {"slam_pose": {"x": 1, "y": -1}, "region_radius_m": 3},
                {"slam_pose": {"x": 200, "y": 200}, "region_radius_m": 1},
            ]:
                cropped, tag = _crop_global_map_to_prior(cloud, prior)
                crops.append(
                    {
                        "tag": tag,
                        "points_sha256": hashlib.sha256(
                            np.asarray(cropped.points).tobytes()
                        ).hexdigest(),
                        "count": len(cropped.points),
                    }
                )
            result["relocalization_prior_crop"] = crops

            from dimos.agents.mcp.external_harness_codex import (
                prepare_codex_harness_run,
            )

            source_home = root / "empty-codex-home"
            source_home.mkdir()
            run = prepare_codex_harness_run(
                run_root=root / "run",
                task="find the cabinet",
                mcp_url="http://127.0.0.1:9990/mcp",
                source_codex_home=source_home,
                copy_auth=False,
                mode="authority-proxy",
                prompt_mode="long_instruction",
                authority_allowed_movement_tools=(
                    "pixelgoal_navigate",
                    "go2_navigate_to_memory_node",
                ),
            )
            result["generated_prompt_and_skills"] = {
                "prompt": run.prompt_path.read_text(),
                "skills": {
                    str(path.relative_to(run.workspace)): path.read_text()
                    for path in sorted((run.workspace / ".agents/skills").rglob("*.md"))
                },
            }
            assert (
                len(
                    [
                        name
                        for name in result["generated_prompt_and_skills"]["skills"]
                        if name.endswith("SKILL.md")
                    ]
                )
                == 5
            )

            from dimos.core.coordination.module_coordinator import ModuleCoordinator
            from dimos.core.global_config import GlobalConfig

            events = []

            class PreparedModule:
                rpcs = {"prepare_stop": object(), "stop": object()}

            coordinator = ModuleCoordinator(g=GlobalConfig(n_workers=0, viewer="none"))
            coordinator._managers = {}
            coordinator._deployed_modules[PreparedModule] = SimpleNamespace(
                prepare_stop=lambda: events.append("prepare_stop"),
                stop=lambda: events.append("stop"),
            )
            coordinator.stop()
            result["durable_shutdown_order"] = events

        # Paths are deployment-specific, not robot behavior. Do not normalize
        # timestamps, coordinates, navigation outcomes, or image bytes.
        serialized = (
            json.dumps(result, sort_keys=True)
            .replace(str(root), "<fixture>")
            .replace(root.name, "<fixture>")
        )
        print("SNAPSHOT_RESULT=" + serialized)


if __name__ == "__main__":
    main()
