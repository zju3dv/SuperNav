from __future__ import annotations

import base64
import json
import shlex
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import supernav.methods.navigation.tools  # noqa: F401
from supernav.backends.habitat.memory_slam.area_captioning import (
    SlamAreaCaptionJob,
    SlamAreaCaptionWorker,
)
from supernav.backends.habitat.memory_slam.area_segmentation import (
    segment_slam_occupancy,
)
from supernav.backends.habitat.memory_slam.cartographer import (
    CartographerScan,
    CartographerStream,
    _pbstream_available,
    _require_relocalizer_dependencies,
    _require_worker_capabilities,
    _stream_config_from_environment,
    _write_memory_nodes,
    _write_reported_memory_node_prior,
    odometry_from_habitat_state,
    scan_from_depth,
)
from supernav.backends.habitat.memory_slam.occupancy_navigation import (
    load_slam_occupancy_map,
    plan_slam_occupancy_path,
)
from supernav.backends.habitat.memory_slam.qwen_caption_server import (
    DEFAULT_MODEL,
    QwenCaptionServer,
)
from supernav.backends.habitat.memory_slam.runtime import HabitatMemorySlamMixin
from supernav.experiments.config import load_experiment_config
from supernav.methods.navigation.graph_memory import GraphMemoryStore
from supernav.methods.navigation.prompts import build_prompt, global_task_agent_policy
from supernav.methods.navigation.tools.base import ToolContext, ToolRegistry
from supernav.methods.navigation.tools.graph_memory import ReportMemoryPosTool
from supernav.methods.navigation.tools.graph_memory import _store as memory_node_store
from supernav.methods.navigation.tools.session import InitSceneTool
from supernav.runtime.config import merged_arm


def _node(node_id: str = "kitchen-01") -> dict[str, object]:
    return {
        "id": node_id,
        "kind": "area",
        "name": "kitchen",
        "description": "A kitchen with a doorway.",
        "records": ["Observed directly."],
        "pano_images": {"front": "front.png"},
    }


def test_graph_memory_v7_drops_private_navigation_goal(tmp_path: Path) -> None:
    path = tmp_path / "memory.json"
    path.write_text(json.dumps({
        "version": 6,
        "next_order_id": 2,
        "nodes": {
            "kitchen-01": {
                **_node(),
                "order_id": 1,
                "slam_pose": {"sequence": 4, "x": 1.0, "y": 2.0, "yaw_rad": 0.25},
                "navigation_goal": [10.0, 0.0, 20.0],
            }
        },
    }))

    store = GraphMemoryStore(path)
    assert "navigation_goal" not in store.inspect_node("kitchen-01", return_images=True)
    store.update_node("kitchen-01", {"description": "updated"})
    document = json.loads(path.read_text())
    assert document["version"] == 7
    assert "navigation_goal" not in document["nodes"]["kitchen-01"]


def test_graph_memory_lifecycle_keeps_slam_pose(tmp_path: Path) -> None:
    store = GraphMemoryStore(tmp_path / "memory.json")
    pose = {"sequence": 4, "x": 1.0, "y": 2.0, "yaw_rad": 0.25}
    store.add_node(_node(), slam_pose=pose)
    assert store.map_nodes()[0]["slam_pose"] == pose
    assert store.grep_node("doorway")[0]["id"] == "kitchen-01"


def test_default_memory_node_store_is_scoped_to_the_active_session(
    tmp_path: Path,
) -> None:
    bridge = SimpleNamespace(session_id="session-a")
    context = ToolContext(
        bridge=bridge,
        session_id="session-a",
        output_dir=str(tmp_path),
        nav_mode="navmesh",
        task_type="chat",
    )

    first = memory_node_store(context)
    assert first.path == tmp_path / "session-a" / "memory_nodes.json"

    bridge.session_id = "session-b"
    second = memory_node_store(context)
    assert second.path == tmp_path / "session-b" / "memory_nodes.json"


def test_graph_memory_registry_exports_only_occupancy_navigation() -> None:
    names = {tool.metadata.name for tool in ToolRegistry.list_all()}
    expected = {
        "list_nodes", "grep_node", "inspect_node", "query_node_context",
        "add_node", "upsert_node", "update_node", "apply_memory_node_mutations",
        "delete_node", "report_memory_pos", "navigate_to_memory_node_occ",
    }
    assert expected <= names
    assert "navigate_to_memory_node" not in names


def test_spatial_memory_recipe_uses_release_terms_and_explicit_gpu() -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_experiment_config(
        root / "configs/experiments/habitat-spatial-memory.json"
    )
    environment = config["mcp"]["environment"]
    arm = config["arms"]["default"]

    assert config["skill_runtime"]["skill_set_id"] == (
        "supernav-1.5-structured-spatial-memory"
    )
    assert environment["HAB_GPU_DEVICE_ID"] == "0"
    assert environment["HAB_STRUCTURED_SPATIAL_MEMORY_ENABLED"] == "1"
    assert environment["HAB_MCP_SPATIAL_MEMORY"] == "0"
    assert "hab_apply_memory_node_mutations" in arm["tool_whitelist"]
    assert "hab_apply_graph_mutations" not in arm["tool_whitelist"]
    expected_skills = ["memory-nodes", "2d-slam", "spatial-memory-navigation"]
    assert arm["movement"] == "spatial_memory"
    assert arm["skill_name"] == "memory-nodes"
    assert arm["skill_names"] == expected_skills
    assert arm["required_skill_names"] == expected_skills
    assert config["prompts_dir"] == "configs/benchmarks/spatial_memory/prompts"


@pytest.mark.parametrize(
    "experiment",
    ["habitat-spatial-memory", "habitat-spatial-memory-slam-only"],
)
def test_spatial_memory_prompt_uses_the_research_policy_surface(
    experiment: str,
) -> None:
    root = Path(__file__).resolve().parents[2]
    config = load_experiment_config(
        root / "configs/experiments" / f"{experiment}.json"
    )
    arm = merged_arm(config, "default")
    arm["_agent_backend"] = "codex"
    prompt = build_prompt(
        arm_name="default",
        arm_cfg=arm,
        instruction="Find the requested object.",
        scene="test-scene",
        scene_dataset_config_file="/tmp/scenes.json",
        spawn={"start_position": [0.0, 0.2, 0.0]},
        workspace_root=root,
        prompts_dir=root / config["prompts_dir"],
    )
    policy = global_task_agent_policy(config, arm_cfg=arm)

    assert "hab_navigate_to_memory_node_occ" in prompt
    assert "one-call visual point navigation" in prompt
    assert "There is no preview/confirmation round trip" in prompt
    assert "replans from the current position after collision" in prompt
    assert "slim reference lines at normalized x/y coordinates" in prompt
    assert "potentially duplicated or ambiguous objects" in prompt
    assert "MUST spend one `hab_visual_point_navigate` hop" in prompt
    assert "target's base is cut by the frame's bottom edge" in prompt
    assert "memory nodes" in prompt
    for skill_name in arm["required_skill_names"]:
        assert f"`/{skill_name}`" in prompt
    assert "hab_register_spatial_junction" not in prompt
    assert "LocateAnything" not in prompt
    assert "two-step visual point navigation" not in prompt
    assert "graph-memory" not in prompt
    assert "agent-visible Cartographer `2d_slam_map`" in policy
    assert "Remain mapless" not in policy
    assert "do not query maps" not in policy


def test_spatial_memory_caption_ablation_changes_only_caption_service() -> None:
    root = Path(__file__).resolve().parents[2]
    full = load_experiment_config(
        root / "configs/experiments/habitat-spatial-memory.json"
    )
    no_caption = load_experiment_config(
        root / "configs/experiments/habitat-spatial-memory-slam-only.json"
    )

    assert merged_arm(full, "default") == merged_arm(no_caption, "default")
    assert full["prompts_dir"] == no_caption["prompts_dir"]
    assert full["mcp"]["environment"]["HAB_SLAM_AREA_CAPTION_ENABLED"] == "1"
    assert (
        no_caption["mcp"]["environment"]["HAB_SLAM_AREA_CAPTION_ENABLED"]
        == "0"
    )


def test_structured_spatial_memory_overrides_legacy_ledger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from supernav.methods.navigation import mcp_server

    monkeypatch.setenv("HAB_STRUCTURED_SPATIAL_MEMORY_ENABLED", "1")
    monkeypatch.setenv("HAB_MCP_SPATIAL_MEMORY", "1")

    assert mcp_server._structured_spatial_memory_enabled() is True
    assert mcp_server._spatial_memory_enabled() is False


def test_cartographer_publishes_memory_node_sidecar_name(tmp_path: Path) -> None:
    _write_memory_nodes([tmp_path], [{"id": "kitchen-01"}])

    assert json.loads((tmp_path / "memory_nodes.json").read_text()) == {
        "nodes": [{"id": "kitchen-01"}]
    }
    assert not (tmp_path / "graph_memory_nodes.json").exists()


def test_map_labels_skip_slam_poses_outside_rendered_crop(tmp_path: Path) -> None:
    source = tmp_path / "occupancy_step1.png"
    metadata = {
        "resolution": 1.0,
        "scale": 1.0,
        "mosaic_origin_x": 0,
        "mosaic_origin_y": 10,
        "source_width": 20,
        "display_width": 10,
        "display_height": 20,
        "left_padding": 0,
        "top_padding": 0,
        "offset_x": 10,
        "offset_y": 0,
    }
    (tmp_path / f"{source.name}.render.json").write_text(json.dumps(metadata))
    image = Image.new("RGB", (100, 100), (255, 255, 255))

    # This pose projects onto the output canvas only because of offset_x, but
    # lies left of the actual occupancy-map crop and must not receive a label.
    HabitatMemorySlamMixin._draw_map_labels(
        image,
        tmp_path,
        source,
        [{"order_id": 8, "name": "stale node", "x": 0.0, "y": 11.0}],
        "memory",
    )
    assert np.all(np.asarray(image) == 255)

    HabitatMemorySlamMixin._draw_map_labels(
        image,
        tmp_path,
        source,
        [{"order_id": 1, "name": "visible node", "x": 0.0, "y": 5.0}],
        "memory",
    )
    assert np.any(np.asarray(image) != 255)

    visible = HabitatMemorySlamMixin._map_nodes_in_rendered_crop(
        tmp_path,
        source,
        [
            {"order_id": 8, "name": "stale node", "x": 0.0, "y": 11.0},
            {"order_id": 1, "name": "visible node", "x": 0.0, "y": 5.0},
        ],
        "memory",
    )
    assert [node["order_id"] for node in visible] == [1]


def test_memory_slam_status_uses_scan_matched_pose_not_odometry(
    tmp_path: Path,
) -> None:
    (tmp_path / "occupancy.png.poses.jsonl").write_text(
        json.dumps(
            {
                "sequence": 17,
                "x": 12.5,
                "y": -3.25,
                "yaw_rad": 1.2,
                "odometry_x": 99.0,
                "odometry_y": 88.0,
                "odometry_yaw_rad": -0.5,
            }
        )
        + "\n"
    )

    class Stream:
        artifact_dir = tmp_path

        @staticmethod
        def status() -> dict[str, object]:
            return {
                "artifact_dir": str(tmp_path),
                "latest_odometry": [99.0, 88.0, -0.5],
                "latest_odometry_sequence": 17,
            }

    session = SimpleNamespace(cartographer_stream=Stream())
    status = HabitatMemorySlamMixin()._memory_slam_status(session)

    assert status["slam_pose"] == {
        "sequence": 17,
        "x": 12.5,
        "y": -3.25,
        "yaw_rad": 1.2,
    }


def test_slam_area_nodes_come_from_visual_map_and_skip_marked_area(
    tmp_path: Path,
) -> None:
    visual = tmp_path / "visual"
    planning = tmp_path / "planning"
    visual.mkdir()
    planning.mkdir()
    metadata = {
        "resolution": 1.0,
        "mosaic_origin_x": 0,
        "mosaic_origin_y": 10,
        "source_width": 10,
        "source_height": 10,
        "display_width": 10,
    }
    for root in (visual, planning):
        Image.fromarray(np.full((10, 10), 254, dtype=np.uint8)).save(
            root / "occupancy_step9.amcl.pgm"
        )
        (root / "occupancy_step9.png.render.json").write_text(
            json.dumps(metadata)
        )

    source = visual / "occupancy_step9.png"
    segmentation_source = visual / "occupancy_step9.amcl.pgm"
    visual_segmentation = {
        "areas": [
            {
                "id": 1,
                "centroid": [2.0, 2.0],
                "bbox": [1, 1, 4, 4],
                "pixel_count": 16,
            },
            {
                "id": 2,
                "centroid": [6.0, 6.0],
                "bbox": [5, 5, 8, 8],
                "pixel_count": 16,
            },
        ]
    }
    nodes, payload = HabitatMemorySlamMixin._publish_slam_area_nodes(
        artifact_dir=visual,
        source=source,
        segmentation_source=segmentation_source,
        segmentation_result=visual_segmentation,
        memory_nodes=[{"x": 2.0, "y": 8.0}],
        planning_artifact_dir=planning,
    )

    assert [node["label"] for node in nodes] == ["A2"]
    assert nodes[0]["centroid_px"] == [6, 6]
    assert nodes[0]["center_slam_pose"] == {
        "sequence": 9,
        "x": 6.0,
        "y": 4.0,
        "yaw_rad": 0.0,
    }
    assert payload["source_map"] == source.name
    assert payload["resample_source"] == "planning_occupancy"
    assert payload["decisions"][0]["reason"] == "contains_memory_node"


def test_cartographer_defaults_to_authoritative_simulator_odometry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NAV_ARTIFACTS_DIR", str(tmp_path))
    result = _stream_config_from_environment("session", {"width": 16, "hfov": 90.0})
    assert result[2] is True
    assert result[3] == "simulator_evaluator"
    assert result[4] == "authoritative"


def test_init_forwards_both_frozen_dual_stream_states(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []

    class Bridge:
        session_id = None

        def call(self, action: str, payload: dict) -> dict:
            calls.append((action, payload))
            return {"session_id": "session", "is_gaussian": True}

    monkeypatch.setenv("HAB_CARTOGRAPHER_ENABLED", "1")
    monkeypatch.setenv("HAB_CARTOGRAPHER_LOAD_STATE", "/maps/visual.pbstream")
    monkeypatch.setenv(
        "HAB_CARTOGRAPHER_PLANNING_LOAD_STATE", "/maps/planning.pbstream"
    )
    monkeypatch.delenv("SUPERNAV_MEMORY_NODES_PATH", raising=False)
    context = ToolContext(
        bridge=Bridge(),
        session_id="",
        output_dir=str(tmp_path),
        nav_mode="navmesh",
        task_type="chat",
    )

    result = InitSceneTool().execute({"scene": "test"}, context)

    assert result.ok is True
    localization = calls[0][1]["memory_slam"]["localization"]
    assert localization["load_state_path"] == "/maps/visual.pbstream"
    assert localization["planning_load_state_path"] == "/maps/planning.pbstream"
    assert "memory_nodes_path" not in calls[0][1]["memory_slam"]


def test_init_forwards_an_explicit_prior_session_memory_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []

    class Bridge:
        session_id = None

        def call(self, action: str, payload: dict) -> dict:
            calls.append((action, payload))
            return {"session_id": "session-2", "is_gaussian": True}

    seed = tmp_path / "session-1" / "memory_nodes.json"
    seed.parent.mkdir()
    seed.write_text(
        '{"version":7,"next_order_id":1,"nodes":{}}\n', encoding="utf-8"
    )
    monkeypatch.setenv("HAB_CARTOGRAPHER_ENABLED", "1")
    monkeypatch.setenv("SUPERNAV_MEMORY_NODES_PATH", str(seed))
    context = ToolContext(
        bridge=Bridge(),
        session_id="",
        output_dir=str(tmp_path),
        nav_mode="navmesh",
        task_type="chat",
    )

    result = InitSceneTool().execute({"scene": "test"}, context)

    assert result.ok is True
    assert calls[0][1]["memory_slam"]["memory_nodes_path"] == str(seed)
    assert memory_node_store(context).path == seed


def test_init_forwards_private_paired_caption_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []

    class Bridge:
        session_id = None

        def call(self, action: str, payload: dict) -> dict:
            calls.append((action, payload))
            return {"session_id": "session-2", "is_gaussian": True}

    evidence = tmp_path / "snapshot" / "captions"
    evidence.mkdir(parents=True)
    monkeypatch.setenv("HAB_CARTOGRAPHER_ENABLED", "1")
    context = ToolContext(
        bridge=Bridge(),
        session_id="",
        output_dir=str(tmp_path),
        nav_mode="navmesh",
        task_type="chat",
    )

    result = InitSceneTool().execute(
        {
            "scene": "test",
            "_slam_area_caption_sources": [
                {
                    "artifact_dir": str(evidence),
                    "evidence_origin": "exploration",
                    "source_session_id": "session-1",
                }
            ],
        },
        context,
    )

    assert result.ok is True
    sources = calls[0][1]["memory_slam"]["caption_sources"]
    assert sources == [
        {
            "artifact_dir": str(evidence),
            "evidence_origin": "exploration",
            "source_session_id": "session-1",
        }
    ]
    session = SimpleNamespace(settings={})
    HabitatMemorySlamMixin()._configure_memory_slam_session(
        session, {"memory_slam": {"caption_sources": sources}}
    )
    assert session.settings["slam_area_caption_sources"] == sources


def test_init_selects_habitat_python_for_relocalizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []

    class Bridge:
        session_id = None

        def call(self, action: str, payload: dict) -> dict:
            calls.append((action, payload))
            return {"session_id": "session", "is_gaussian": True}

    habitat_python = tmp_path / "habitat env" / "bin" / "python"
    monkeypatch.setenv("HAB_CARTOGRAPHER_ENABLED", "1")
    monkeypatch.setenv("SUPERNAV_HABITAT_PYTHON", str(habitat_python))
    monkeypatch.delenv("HAB_CARTOGRAPHER_RELOCALIZER_COMMAND", raising=False)
    context = ToolContext(
        bridge=Bridge(),
        session_id="",
        output_dir=str(tmp_path),
        nav_mode="navmesh",
        task_type="chat",
    )

    result = InitSceneTool().execute({"scene": "test"}, context)

    assert result.ok is True
    command = shlex.split(calls[0][1]["memory_slam"]["relocalizer_command"])
    assert command[0] == str(habitat_python)
    assert command[1].endswith("/memory_slam/relocalize.py")


def test_report_memory_pos_forwards_mcp_tool_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, dict]] = []

    class Bridge:
        def call(self, action: str, payload: dict) -> dict:
            calls.append((action, payload))
            if action == "get_memory_slam_status":
                return {"enabled": True}
            return {"accepted": True}

    nodes_path = tmp_path / "memory_nodes.json"
    nodes_path.write_text(
        '{"version":7,"next_order_id":2,"nodes":{"kitchen":{'
        '"id":"kitchen","kind":"area","name":"Kitchen",'
        '"description":"Kitchen area","records":[],"pano_images":{},'
        '"order_id":1,"slam_pose":null}}}'
        "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SUPERNAV_MEMORY_NODES_PATH", str(nodes_path))
    context = ToolContext(
        bridge=Bridge(),
        session_id="session",
        output_dir=str(tmp_path),
        nav_mode="navmesh",
        task_type="chat",
        tool_seq=17,
    )

    result = ReportMemoryPosTool().execute({"id": "kitchen"}, context)

    assert result.ok is True
    assert (
        "check_relocalization_prior",
        {
            "node_id": "kitchen",
            "tool_seq": 17,
            "node": result.body["current_node"],
        },
    ) in calls


def test_reported_memory_node_prior_matches_worker_snapshot_contract(
    tmp_path: Path,
) -> None:
    payload = _write_reported_memory_node_prior(
        tmp_path,
        session_id="session",
        node={
            "id": "kitchen",
            "kind": "area",
            "name": "Kitchen",
            "slam_pose": {"sequence": 4, "x": 1.0, "y": 2.0, "yaw_rad": 0.25},
        },
        tool_seq=17,
    )

    assert payload["tool_seq"] == 17
    assert payload["node_id"] == "kitchen"
    assert json.loads((tmp_path / "reported_memory_node.json").read_text()) == payload


def test_loaded_pbstream_is_a_ready_frozen_final_artifact(tmp_path: Path) -> None:
    loaded = tmp_path / "frozen.pbstream"
    loaded.write_bytes(b"frozen-state")

    assert _pbstream_available(tmp_path / "missing.pbstream", str(loaded)) is True
    assert _pbstream_available(tmp_path / "missing.pbstream", None) is False


def test_cartographer_rejects_non_public_odometry_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NAV_ARTIFACTS_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="simulator_evaluator"):
        _stream_config_from_environment(
            "session", {"cartographer_use_odometry": True, "cartographer_odometry_source": "superodom"}
        )


def test_authoritative_worker_capability_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    class Completed:
        stdout = '{"authoritative_odometry": true}'
    monkeypatch.setattr(
        "supernav.backends.habitat.memory_slam.cartographer.subprocess.run",
        lambda *args, **kwargs: Completed(),
    )
    _require_worker_capabilities("/tmp/worker {input_fifo} {occupancy_path}", authoritative=True)


def test_authoritative_worker_capability_probe_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Completed:
        stdout = '{"authoritative_odometry": false}'
    monkeypatch.setattr(
        "supernav.backends.habitat.memory_slam.cartographer.subprocess.run",
        lambda *args, **kwargs: Completed(),
    )
    with pytest.raises(RuntimeError, match="lacks authoritative_odometry"):
        _require_worker_capabilities("worker {input_fifo} {occupancy_path}", authoritative=True)


def test_relocalizer_dependency_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    class Completed:
        returncode = 0

    calls: list[tuple[list[str], dict]] = []

    def run(command: list[str], **kwargs: object) -> Completed:
        calls.append((command, kwargs))
        return Completed()

    monkeypatch.setattr(
        "supernav.backends.habitat.memory_slam.cartographer.subprocess.run", run
    )

    _require_relocalizer_dependencies("/runtime/python /package/relocalize.py")

    assert calls[0][0] == [
        "/runtime/python",
        "/package/relocalize.py",
        "--check-open3d",
    ]


def test_depth_scan_and_habitat_odometry_conversion() -> None:
    depth = np.full((7, 8), 2.0, dtype=np.float32)
    scan = scan_from_depth(
        depth, sequence=3, timestamp_s=0.3, beam_count=3,
        range_min_m=0.15, range_max_m=8.0,
    )
    assert scan.ranges_m == [2.828, 2.0, 2.828]
    assert odometry_from_habitat_state([0.0, 0.0, -2.0], [1.0, 0.0, 0.0, 0.0]) == (2.0, -0.0, 0.0)


def test_cartographer_queue_drains_on_close(tmp_path: Path) -> None:
    stream = CartographerStream(
        artifact_dir=tmp_path,
        width=8,
        hfov_deg=90.0,
        sensor_height_m=1.5,
        beam_count=3,
        range_min_m=0.15,
        range_max_m=8.0,
        queue_size=4,
    )
    stream.submit(CartographerScan(1, 0.1, [1.0, 1.5, 2.0]))
    summary = stream.close(2.0)
    assert summary["processed"] == 1
    assert json.loads((tmp_path / "summary.json").read_text())["processed"] == 1


def test_area_segmentation_and_occupancy_planning(tmp_path: Path) -> None:
    display = np.full((8, 8, 3), 255, dtype=np.uint8)
    display[0, :] = 0
    display[-1, :] = 0
    display[:, 0] = 0
    display[:, -1] = 0
    display_path = tmp_path / "display.png"
    Image.fromarray(display).save(display_path)
    segmented = segment_slam_occupancy(display_path, min_area_px=1, min_frontier_px=1)
    assert segmented["areas"]

    grid = np.full((8, 8), 254, dtype=np.uint8)
    grid[0, :] = grid[-1, :] = 0
    grid[:, 0] = grid[:, -1] = 0
    map_path = tmp_path / "occupancy.png.amcl.pgm"
    Image.fromarray(grid).save(map_path)
    (tmp_path / "occupancy.png.render.json").write_text(json.dumps({
        "resolution": 1.0, "mosaic_origin_x": 0.0, "mosaic_origin_y": 7.0,
        "source_width": 8, "source_height": 8, "display_width": 8,
    }))
    slam_map = load_slam_occupancy_map(map_path)
    plan, _ = plan_slam_occupancy_path(slam_map, start={"x": 2, "y": 2}, goal={"x": 5, "y": 5})
    assert plan.reachable


def test_caption_defaults_external_model_and_inline_transport(tmp_path: Path) -> None:
    assert DEFAULT_MODEL == "Qwen/Qwen3.5-2B"
    image_path = tmp_path / "view.png"
    Image.new("RGB", (4, 4), color=(20, 30, 40)).save(image_path)
    worker = SlamAreaCaptionWorker(
        session_id="s", artifact_dir=tmp_path, backend_url="http://caption.invalid",
        check_health=False,
    )
    try:
        payload = worker._caption_payload(SlamAreaCaptionJob(
            job_id="j", keyframe_id="k", session_id="s", view="front",
            image_path=str(image_path), heading_deg=0.0, step_count=1,
            cartographer_sequence=1,
        ))
    finally:
        worker.close()
    assert "image_path" not in payload
    assert base64.b64decode(payload["image_data"]).startswith(b"\x89PNG")


def test_qwen_server_loads_through_injected_backend() -> None:
    class Processor:
        pass
    class Model:
        def eval(self) -> None:
            self.ready = True
        def parameters(self):
            return iter(())
    server = QwenCaptionServer(model_loader=lambda _server: (Processor(), Model()), warmup=False)
    server.load()
    assert server.health()["ok"] is True
    assert server.health()["protocol"] == "supernav-caption/v1"
