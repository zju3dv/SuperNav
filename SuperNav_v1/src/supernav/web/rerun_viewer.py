#!/usr/bin/env python3
"""Live Rerun dashboard for habitat-gs navigation sessions.

Headless-friendly defaults:
    supernav viewer --session-id <session_id>
    supernav viewer --session-id <session_id> --save /tmp/habitat_nav.rrd
    supernav viewer --session-id <session_id> --connect 127.0.0.1:9876
    supernav viewer --serve-web 9090
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import sys
import tempfile
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

from supernav.paths import asset_root, workspace_root

ROOT = workspace_root()

from supernav.backends.habitat.bridge_client import BridgeClient
from supernav.evaluation.session_trace_adapters import (
    ObservabilityRecord,
    ObservabilityTokens,
    ObservabilityToolCall,
)

rr: Any = None
DEFAULT_CONFIG_PATH = asset_root() / "configs/viewers/rerun.yaml"


def _safe_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _load_yaml_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.is_file():
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyYAML is required to read configs/viewers/rerun.yaml. "
            "Install it in the SuperNav environment or pass ports via CLI."
        ) from exc

    with config_path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"Config file must contain a YAML mapping: {config_path}")
    return data


def _default_output_dir() -> str:
    for env_name in (
        "NAV_VISUAL_OUTPUT_DIR",
        "HAB_VISUAL_OUTPUT_DIR",
        "NAV_ARTIFACTS_DIR",
    ):
        raw = os.environ.get(env_name, "").strip()
        if raw:
            return raw
    user = os.environ.get("USER", "").strip() or getpass.getuser().strip() or "user"
    safe_user = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in user)
    return str(Path(tempfile.gettempdir()) / f"habitat_gs_visuals_{safe_user}")


def _read_image(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    if path.startswith("zip://"):
        return _read_zip_image(path)
    file_path = Path(path)
    if not file_path.is_file():
        return None
    return _image_file_to_array(file_path)


def _read_zip_image(uri: str) -> Optional[np.ndarray]:
    archive_text, sep, member = uri.removeprefix("zip://").partition("!/")
    if not sep or not archive_text or not member:
        return None
    archive = Path(archive_text)
    if not archive.is_file():
        return None
    try:
        with zipfile.ZipFile(archive) as zf, zf.open(member) as fh, Image.open(
            fh
        ) as image:
            return _image_to_array(image)
    except (KeyError, OSError, ValueError, zipfile.BadZipFile):
        return None


def _image_file_to_array(path: Path) -> Optional[np.ndarray]:
    try:
        with Image.open(path) as image:
            return _image_to_array(image)
    except (OSError, ValueError):
        return None


def _image_to_array(image: Image.Image) -> np.ndarray:
    if image.mode not in ("RGB", "RGBA", "L"):
        image = image.convert("RGB")
    array = np.asarray(image)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[2] == 4:
        return array[:, :, :3]
    return array


def _strip_colored_topdown_overlays(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] < 3:
        return image
    rgb = image[:, :, :3]
    channel_max = rgb.max(axis=2)
    channel_min = rgb.min(axis=2)
    colored_overlay = (channel_max - channel_min > 20) & (channel_max > 80)
    if not colored_overlay.any():
        return image
    cleaned = image.copy()
    cleaned[colored_overlay, :3] = 255
    return cleaned


MappingLike = Dict[str, Any]


def _first_path(item: MappingLike) -> Optional[str]:
    for key in ("path", "mapped_path"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return None


# Matches the per-step frame filenames written by `_export_visuals` in the
# bridge. Capture_seq is 6 zero-padded digits; sensor is a slash-free token.
# Panorama frames (prefix `pano_<dir>_`) and files prefixed with the
# session_id are intentionally skipped — the tail feed is meant for the
# "one RGB/depth frame per simulator step" stream that `_navigate_step`
# (and `step_and_capture`) emit, not for panorama bursts or video mp4s.
_TAIL_SENSOR_PATTERN = re.compile(
    r"^step(?P<seq>\d{6})_(?P<sensor>color_sensor|depth_sensor|third_rgb_sensor|semantic_sensor)\.png$"
)


def _scan_tail_dir(tail_dir: str, after_seq: int) -> List[Tuple[int, Dict[str, str]]]:
    """Return new per-step frame groups under ``tail_dir`` (sorted by seq).

    Each element is ``(capture_seq, {sensor_name: png_path})``. Frames with
    ``seq <= after_seq`` are filtered out so a caller can use this to implement
    an "emit only what is new since last tick" cursor.
    """
    buckets: Dict[int, Dict[str, str]] = {}
    try:
        entries = os.listdir(tail_dir)
    except (FileNotFoundError, NotADirectoryError, PermissionError):
        return []
    for name in entries:
        match = _TAIL_SENSOR_PATTERN.match(name)
        if match is None:
            continue
        seq = int(match.group("seq"))
        if seq <= after_seq:
            continue
        sensor = match.group("sensor")
        buckets.setdefault(seq, {})[sensor] = os.path.join(tail_dir, name)
    return sorted(buckets.items(), key=lambda item: item[0])


def _entity_component(name: str) -> str:
    cleaned = "".join(
        ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in name
    )
    cleaned = cleaned.strip("_")
    return cleaned or "sensor"


def _vector3(value: Any) -> Optional[np.ndarray]:
    """Normalize a 3D vector from either list/tuple or dict ``{x,y,z}`` form.

    The bridge's SG payload uses dict form for ``position_xyz`` /
    ``bbox_*_xyz`` fields, while topdown trajectory points use list form.
    This helper accepts both so callers don't need to special-case.
    """
    # Dict form: {"x": .., "y": .., "z": ..}
    if isinstance(value, dict):
        try:
            return np.asarray(
                [float(value["x"]), float(value["y"]), float(value["z"])],
                dtype=np.float32,
            )
        except (KeyError, TypeError, ValueError):
            return None
    # List/tuple form: [x, y, z]
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        try:
            return np.asarray(
                [float(value[0]), float(value[1]), float(value[2])],
                dtype=np.float32,
            )
        except (TypeError, ValueError):
            return None
    return None


def _pose_position(pose: Any) -> Optional[list[float]]:
    if not isinstance(pose, dict):
        return None
    for key in ("position", "current_position"):
        vec = _vector3(pose.get(key))
        if vec is not None:
            return [float(vec[0]), float(vec[1]), float(vec[2])]
    for key in ("ego_pose", "pose", "state_summary"):
        nested = pose.get(key)
        if isinstance(nested, dict):
            found = _pose_position(nested)
            if found is not None:
                return found
        else:
            vec = _vector3(nested)
            if vec is not None:
                return [float(vec[0]), float(vec[1]), float(vec[2])]
    vec = _vector3(pose)
    if vec is not None:
        return [float(vec[0]), float(vec[1]), float(vec[2])]
    return None


def _ground_truth_goal_positions(ground_truth: MappingLike) -> list[Any]:
    for key in (
        "goal_position",
        "target_position",
        "position",
        "final_position",
        "eval_goal_position",
        "goal",
        "gt_goal",
        "target_position_world",
    ):
        value = ground_truth.get(key)
        if _vector3(value) is not None:
            return [value]
    target_pose = ground_truth.get("target_pose")
    if isinstance(target_pose, dict):
        position = target_pose.get("position")
        if _vector3(position) is not None:
            return [position]
    object_instances = ground_truth.get("object_instances")
    if isinstance(object_instances, list):
        positions: list[Any] = []
        for item in object_instances:
            if not isinstance(item, dict):
                continue
            position = item.get("position")
            if _vector3(position) is not None:
                positions.append(position)
        if positions:
            return positions
    debug = ground_truth.get("_debug")
    if isinstance(debug, dict):
        return _ground_truth_goal_positions(debug)
    return []


def _vector3_items(value: Any) -> list[Any]:
    if _vector3(value) is not None:
        return [value]
    if not isinstance(value, list):
        return []
    return [item for item in value if _vector3(item) is not None]


def _ground_truth_reference_path(ground_truth: MappingLike) -> list[Any]:
    return _vector3_items(ground_truth.get("reference_path"))


def _ground_truth_waypoints(ground_truth: MappingLike) -> list[Any]:
    return _vector3_items(ground_truth.get("waypoints"))


def _topdown_goal_candidates(topdown: MappingLike) -> list[Any]:
    goals = topdown.get("goals")
    if _vector3(goals) is not None:
        return [goals]
    if isinstance(goals, list):
        candidates = [item for item in goals if _vector3(item) is not None]
        if candidates:
            return candidates
    goal = topdown.get("goal")
    return [goal] if _vector3(goal) is not None else []


def _topdown_reference_path_candidates(topdown: MappingLike) -> list[Any]:
    return _vector3_items(topdown.get("reference_path"))


def _topdown_waypoint_candidates(topdown: MappingLike) -> list[Any]:
    return _vector3_items(topdown.get("waypoints"))


def _vector2_to_3(value: Any, z: float = 0.0) -> Optional[np.ndarray]:
    """Normalize a 2D ``{x, y}`` dict into a 3D vector with the given z.

    Room nodes in the SG payload carry ``centroid_xy`` rather than a full
    3D position. The 2D point lives in the navmesh's XY plane; we lift it
    to 3D by pinning z (floor height) to a configurable constant.
    """
    if isinstance(value, dict):
        try:
            return np.asarray(
                [float(value["x"]), float(value["y"]), float(z)],
                dtype=np.float32,
            )
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            return np.asarray(
                [float(value[0]), float(value[1]), float(z)],
                dtype=np.float32,
            )
        except (TypeError, ValueError):
            return None
    return None


def _project_topdown(
    points: Iterable[Any],
    *,
    map_bounds: Any,
    meters_per_pixel: float,
) -> list[list[float]]:
    if (
        not isinstance(map_bounds, list)
        or len(map_bounds) != 2
        or meters_per_pixel <= 0
    ):
        return []
    min_corner = _vector3(map_bounds[0])
    if min_corner is None:
        return []
    projected: list[list[float]] = []
    for point in points:
        vec = _vector3(point)
        if vec is None:
            continue
        px = float((vec[0] - min_corner[0]) / meters_per_pixel)
        py = float((vec[2] - min_corner[2]) / meters_per_pixel)
        projected.append([px, py])
    return projected


def _try_log_text(entity: str, text: str) -> None:
    if hasattr(rr, "TextDocument"):
        try:
            rr.log(entity, rr.TextDocument(text, media_type="markdown"))
            return
        except Exception:
            pass
    if hasattr(rr, "TextLog"):
        try:
            rr.log(entity, rr.TextLog(text))
        except Exception:
            pass


def _log_raw_llm_call_payload(call_id: str, raw: MappingLike) -> None:
    input_payload: Dict[str, Any] = {"request": raw.get("request")}
    if "wire_request" in raw:
        input_payload["wire_request"] = raw.get("wire_request")
    output_payload: Dict[str, Any] = {"response": raw.get("response")}
    _try_log_text(
        "agent/llm_raw_input",
        f"### {call_id}\n\n```json\n{_safe_json(input_payload)}\n```",
    )
    _try_log_text(
        "agent/llm_raw_output",
        f"### {call_id}\n\n```json\n{_safe_json(output_payload)}\n```",
    )


def _set_step_timeline(step_count: Optional[int], wall_time_s: float) -> None:
    if isinstance(step_count, int) and hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("step", step_count)
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("wall_time", wall_time_s)


def _try_construct(
    factory: Any, kwargs_candidates: Iterable[Dict[str, Any]], *arg_candidates: tuple
) -> Any:
    for kwargs in kwargs_candidates:
        try:
            return factory(**kwargs)
        except TypeError:
            continue
    for args in arg_candidates:
        try:
            return factory(*args)
        except TypeError:
            continue
    return None


def _make_named_view(rrb: Any, class_name: str, *, name: str, origin: str) -> Any:
    ctor = getattr(rrb, class_name, None)
    if ctor is None:
        return None
    return _try_construct(
        ctor,
        (
            {"name": name, "origin": origin},
            {"origin": origin, "name": name},
            {"name": name, "contents": [origin]},
            {"contents": [origin], "name": name},
            {"origin": origin},
            {"contents": [origin]},
        ),
        (origin,),
    )


def _make_container(
    rrb: Any, class_name: str, children: Iterable[Any], *, name: Optional[str] = None
) -> Any:
    ctor = getattr(rrb, class_name, None)
    child_list = [child for child in children if child is not None]
    if ctor is None or not child_list:
        return None

    kwargs_candidates = [{"name": name}] if name else []
    kwargs_candidates.append({})
    for kwargs in kwargs_candidates:
        try:
            return ctor(*child_list, **kwargs)
        except TypeError:
            continue
    return None


def _configure_default_blueprint() -> None:
    if rr is None or not hasattr(rr, "send_blueprint"):
        return
    try:
        import rerun.blueprint as rrb  # type: ignore[import-not-found]
    except ImportError:
        return

    rgb_view = _make_named_view(
        rrb, "Spatial2DView", name="RGB", origin="world/rgb/first_person"
    )
    depth_view = _make_named_view(
        rrb, "Spatial2DView", name="Depth", origin="world/depth/first_person"
    )
    third_person_view = _make_named_view(
        rrb, "Spatial2DView", name="Third Person RGB", origin="world/third_person/rgb"
    )
    bev_view = _make_named_view(rrb, "Spatial2DView", name="BEV", origin="world/bev")

    left_col = _make_container(rrb, "Vertical", (rgb_view, depth_view))
    right_col = _make_container(rrb, "Vertical", (third_person_view, bev_view))
    visual_page = _make_container(
        rrb, "Horizontal", (left_col, right_col), name="Visuals"
    )
    if visual_page is None:
        visual_page = _make_container(
            rrb,
            "Vertical",
            (rgb_view, depth_view, third_person_view, bev_view),
            name="Visuals",
        )
    if visual_page is None:
        return

    text_views = [
        _make_named_view(
            rrb, "TextDocumentView", name="Metrics", origin="debug/metrics"
        ),
        _make_named_view(
            rrb, "TextDocumentView", name="Lifecycle", origin="debug/lifecycle"
        ),
        _make_named_view(rrb, "TextDocumentView", name="Errors", origin="debug/errors"),
        _make_named_view(
            rrb, "TextDocumentView", name="BEV Status", origin="world/bev/status"
        ),
        _make_named_view(
            rrb,
            "TextDocumentView",
            name="Depth Analysis",
            origin="world/depth_analysis",
        ),
        _make_named_view(rrb, "TextLogView", name="Errors Log", origin="debug/errors"),
        _make_named_view(
            rrb,
            "TextDocumentView",
            name="SG Summary",
            origin="world/scene_graph/summary",
        ),
        _make_named_view(
            rrb, "TextDocumentView", name="SG Status", origin="debug/sg_status"
        ),
    ]
    text_page = _make_container(rrb, "Vertical", text_views, name="Text")

    # Agent tab: 3D scene graph + tool call TextLog side by side.
    # NOTE: the SG 3D view MUST use an explicit `contents` filter, not
    # just `origin`. Without the filter rerun auto-includes every
    # entity the view can find, including the 2D entities logged under
    # /world/{rgb,depth,bev,path,agent}/*, and each of those triggers a
    # "2D visualizers require a pinhole ancestor" warning inside the
    # 3D view.
    sg_3d_view: Any = None
    sg_ctor = getattr(rrb, "Spatial3DView", None)
    if sg_ctor is not None:
        sg_3d_view = _try_construct(
            sg_ctor,
            (
                {
                    "name": "Scene Graph 3D",
                    "origin": "/world/scene_graph",
                    "contents": ["+/world/scene_graph/**"],
                },
                {
                    "name": "Scene Graph 3D",
                    "origin": "world/scene_graph",
                    "contents": ["+world/scene_graph/**"],
                },
                {
                    "origin": "/world/scene_graph",
                    "contents": ["+/world/scene_graph/**"],
                },
            ),
        )
        if sg_3d_view is None:
            # Last-resort fallback: no contents filter (will show warnings)
            sg_3d_view = _make_named_view(
                rrb,
                "Spatial3DView",
                name="Scene Graph 3D",
                origin="world/scene_graph",
            )

    tool_call_view = _make_named_view(
        rrb, "TextLogView", name="Tool Calls", origin="agent/tool_calls"
    )
    llm_reasoning_view = _make_named_view(
        rrb, "TextDocumentView", name="LLM Reasoning", origin="agent/llm_reasoning"
    )
    llm_raw_input_view = _make_named_view(
        rrb, "TextDocumentView", name="LLM Raw Input", origin="agent/llm_raw_input"
    )
    llm_raw_output_view = _make_named_view(
        rrb, "TextDocumentView", name="LLM Raw Output", origin="agent/llm_raw_output"
    )
    tool_decision_view = _make_named_view(
        rrb,
        "TextDocumentView",
        name="Tool Decisions",
        origin="agent/tool_call_decisions",
    )
    agent_text_col = _make_container(
        rrb,
        "Vertical",
        (
            llm_reasoning_view,
            llm_raw_input_view,
            llm_raw_output_view,
            tool_decision_view,
            tool_call_view,
        ),
    )
    agent_page = _make_container(
        rrb, "Horizontal", (sg_3d_view, agent_text_col), name="Agent"
    )
    if agent_page is None:
        agent_page = (
            sg_3d_view or agent_text_col or tool_call_view
        )  # fallback: single view

    all_pages = [p for p in (visual_page, agent_page, text_page) if p is not None]
    tabs = (
        _make_container(rrb, "Tabs", all_pages, name=None)
        if len(all_pages) > 1
        else None
    )
    root = tabs if tabs is not None else visual_page

    blueprint_ctor = getattr(rrb, "Blueprint", None)
    if blueprint_ctor is None:
        return
    try:
        blueprint = blueprint_ctor(root, auto_layout=False, auto_views=False)
        rr.send_blueprint(blueprint)
    except TypeError:
        try:
            blueprint = blueprint_ctor(root)
            rr.send_blueprint(blueprint)
        except TypeError:
            pass


def _configure_rerun(args: argparse.Namespace) -> None:
    global rr
    try:
        import rerun as rr_module
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime guard
        raise SystemExit(
            "rerun-sdk is required for supernav viewer. "
            "Install dependencies from requirements.txt first."
        ) from exc

    rr = rr_module
    rr.init(args.app_id)
    # NOTE: _configure_default_blueprint() must be called AFTER the sink
    # (serve_grpc / save) is set up so send_blueprint() reaches the viewer.
    if args.save:
        if not hasattr(rr, "save"):
            raise SystemExit("This rerun-sdk build does not support rr.save().")
        rr.save(args.save)
        _configure_default_blueprint()
    elif args.serve_web is not None:
        if hasattr(rr, "serve_grpc") and hasattr(rr, "serve_web_viewer"):
            server_uri = rr.serve_grpc(grpc_port=args.grpc_port)
            rr.serve_web_viewer(
                web_port=args.serve_web,
                open_browser=False,
                connect_to=server_uri,
            )
            _configure_default_blueprint()
            # Build a direct URL that includes the gRPC connection parameter so
            # the browser auto-connects even if the viewer's embed doesn't fire.
            import urllib.parse

            encoded = urllib.parse.quote(server_uri, safe="")
            direct_url = f"http://localhost:{args.serve_web}/?url={encoded}"
            print(
                f"[rerun-nav-viewer] web viewer: {direct_url}",
                file=sys.stderr,
            )
        elif hasattr(rr, "serve_web"):  # noqa: SIM106
            try:
                rr.serve_web(port=args.serve_web, open_browser=False)
            except TypeError:
                try:
                    rr.serve_web(args.serve_web)
                except TypeError as exc:
                    raise SystemExit(
                        "Failed to call rr.serve_web() with this rerun-sdk build. "
                        "Check the installed rerun-sdk version."
                    ) from exc
            _configure_default_blueprint()
            print(
                f"[rerun-nav-viewer] web viewer: http://localhost:{args.serve_web}",
                file=sys.stderr,
            )
        else:
            raise SystemExit(
                "This rerun-sdk build supports neither serve_grpc+serve_web_viewer nor serve_web()."
            )
    elif args.spawn:
        if not hasattr(rr, "spawn"):
            raise SystemExit("This rerun-sdk build does not support rr.spawn().")
        rr.spawn()
    elif args.connect:
        host, sep, port = args.connect.partition(":")
        if not sep:
            raise SystemExit('--connect must use "HOST:PORT"')
        if hasattr(rr, "connect_grpc"):
            rr.connect_grpc(f"rerun+http://{host}:{int(port)}/proxy")
        elif hasattr(rr, "connect_tcp"):  # noqa: SIM106
            rr.connect_tcp(host, int(port))
        else:
            raise SystemExit(
                "This rerun-sdk build supports neither rr.connect_grpc() nor rr.connect_tcp()."
            )
    else:
        default_path = f"/tmp/{args.app_id}.rrd"
        if not hasattr(rr, "save"):
            raise SystemExit(
                "No Rerun sink selected and this rerun-sdk build does not support rr.save(). "
                "Use --connect or --spawn explicitly."
            )
        rr.save(default_path)
        print(
            f"[rerun-nav-viewer] headless mode: writing recording to {default_path}",
            file=sys.stderr,
        )


@dataclass
class FrameSnapshot:
    wall_time_s: float
    session_id: str
    step_count: Optional[int]
    metrics: Optional[Dict[str, Any]]
    visuals: Dict[str, Any]
    topdown: Optional[Dict[str, Any]]
    depth_analysis: Optional[Dict[str, Any]]
    # Scene graph (static per session, fetched once)
    scene_graph: Optional[Dict[str, Any]] = None
    # Slice of action ring from runtime status
    action_ring: Optional[list] = None


class OfflineNavReplay:
    """Replay common observability records without bridge access."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.bundle = Path(args.offline_bundle).expanduser().resolve(strict=False)
        self._trajectory_points: list[Any] = []
        self._frame_record_count = 0
        self._frame_record_index = 0
        self._raw_llm_calls_by_id: dict[str, Dict[str, Any]] = {}

    def run(self) -> None:
        self._raw_llm_calls_by_id = self._load_raw_llm_calls()
        records = self._load_records()
        self._frame_record_count = sum(1 for record in records if record.kind == "frame")
        self._frame_record_index = 0
        ground_truth = self._load_json("ground_truth.json") or {}
        topdown = self._load_topdown()
        self._log_video_asset()

        for record in records:
            if hasattr(rr, "set_time_sequence"):
                rr.set_time_sequence("seq", int(record.seq))
            if hasattr(rr, "set_time_seconds"):
                rr.set_time_seconds(
                    "wall_time",
                    float(record.seq) / max(float(self.args.offline_fps), 1.0),
                )
            if record.kind == "llm_call":
                self._log_llm_call(record)
            elif record.kind == "frame":
                self._frame_record_index += 1
                self._log_frame(record, topdown=topdown, ground_truth=ground_truth)
        self._log_raw_llm_calls_without_records(
            {
                record.call_id
                for record in records
                if record.kind == "llm_call" and record.call_id
            }
        )

    def _load_records(self) -> tuple[ObservabilityRecord, ...]:
        common_names = (
            "observability.jsonl",
            "common_observability.jsonl",
            "taskh_observability.jsonl",
        )
        if self.bundle.is_file():
            return self._read_common_records(self.bundle)
        common = self._find_file(common_names)
        if common is not None:
            return self._read_common_records(common)
        if self._raw_llm_calls_by_id:
            return ()
        raise FileNotFoundError(f"No observability JSONL found under: {self.bundle}")

    def _read_common_records(self, path: Path) -> tuple[ObservabilityRecord, ...]:
        records: list[ObservabilityRecord] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            record = self._record_from_row(row, default_seq=len(records))
            if record is not None:
                records.append(record)
        return tuple(records)

    def _record_from_row(
        self, row: Dict[str, Any], *, default_seq: int
    ) -> Optional[ObservabilityRecord]:
        kind = row.get("kind")
        call_id = row.get("call_id")
        if not isinstance(kind, str) or not isinstance(call_id, str):
            return None
        seq = int(row.get("seq", default_seq))
        if kind == "llm_call":
            tool_calls = []
            raw_tool_calls = row.get("tool_calls")
            if isinstance(raw_tool_calls, list):
                for item in raw_tool_calls:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name")
                    args = item.get("args")
                    if isinstance(name, str):
                        tool_calls.append(
                            ObservabilityToolCall(
                                name=name,
                                args=args if isinstance(args, dict) else {},
                            )
                        )
            tokens = (
                ObservabilityTokens.from_mapping(row.get("tokens"))
                if isinstance(row.get("tokens"), dict)
                else None
            )
            return ObservabilityRecord(
                kind="llm_call",
                seq=seq,
                call_id=call_id,
                reasoning=str(row.get("reasoning") or ""),
                tool_calls=tuple(tool_calls),
                tokens=tokens,
            )
        if kind == "frame" and isinstance(row.get("image_path"), str):
            return ObservabilityRecord(
                kind="frame",
                seq=seq,
                call_id=call_id,
                image_path=str(row["image_path"]),
                pose=row.get("pose") if isinstance(row.get("pose"), dict) else {},
                action=str(row.get("action") or "unknown"),
            )
        return None

    def _load_json(self, name: str) -> Optional[Dict[str, Any]]:
        path = self._find_file((name,))
        if path is None:
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return data if isinstance(data, dict) else None

    def _load_topdown(self) -> Optional[Dict[str, Any]]:
        return (
            self._load_json("topdown_response.json")
            or self._load_json("topdown.json")
            or self._load_json("topdown_map.json")
        )

    def _find_file(self, names: Iterable[str]) -> Optional[Path]:
        name_tuple = tuple(names)
        if self.bundle.is_file():
            if self.bundle.name in set(name_tuple):
                return self.bundle
            roots = (self.bundle.parent,)
        else:
            roots = (self.bundle,)
        for root in roots:
            for name in name_tuple:
                candidate = root / name
                if candidate.is_file():
                    return candidate
            for name in name_tuple:
                matches = sorted(
                    path for path in root.glob(f"**/{name}") if path.is_file()
                )
                if matches:
                    return matches[0]
        return None

    def _load_raw_llm_calls(self) -> dict[str, Dict[str, Any]]:
        raw_path = self._find_file(("raw_llm_calls.jsonl",))
        if raw_path is None:
            return {}
        by_id: dict[str, Dict[str, Any]] = {}
        for line in raw_path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            call_id = row.get("call_id")
            if isinstance(call_id, str) and call_id:
                by_id[call_id] = row
        return by_id

    def _resolve_path(self, path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        if path.startswith(("http://", "https://", "zip://")):
            return path
        candidate = Path(path)
        if candidate.is_absolute():
            return str(candidate)
        root = self.bundle.parent if self.bundle.is_file() else self.bundle
        return str(root / candidate)

    def _log_llm_call(self, record: ObservabilityRecord) -> None:
        reasoning = record.reasoning.strip()
        _try_log_text(
            "agent/llm_reasoning",
            reasoning or "_No non-tool assistant text recorded._",
        )
        tool_payload = [tool.to_json() for tool in record.tool_calls]
        _try_log_text(
            "agent/tool_call_decisions",
            f"```json\n{_safe_json(tool_payload)}\n```",
        )
        if record.tokens:
            _try_log_text(
                "agent/llm_tokens",
                f"```json\n{_safe_json(record.tokens.to_json())}\n```",
            )
        self._log_raw_llm_call(record.call_id)

    def _log_raw_llm_call(self, call_id: str) -> None:
        raw = self._raw_llm_calls_by_id.get(call_id)
        if not raw:
            _try_log_text(
                "agent/llm_raw_input",
                f"_No raw API input recorded for `{call_id}`._",
            )
            _try_log_text(
                "agent/llm_raw_output",
                f"_No raw API output recorded for `{call_id}`._",
            )
            return
        _log_raw_llm_call_payload(call_id, raw)

    def _log_raw_llm_calls_without_records(self, seen_call_ids: set[str]) -> None:
        pending = [
            raw
            for call_id, raw in self._raw_llm_calls_by_id.items()
            if call_id not in seen_call_ids
        ]
        for raw in sorted(pending, key=lambda row: int(row.get("seq") or 0)):
            call_id = raw.get("call_id")
            if not isinstance(call_id, str) or not call_id:
                continue
            seq = raw.get("seq")
            if hasattr(rr, "set_time_sequence"):
                rr.set_time_sequence("seq", int(seq) if isinstance(seq, int) else 0)
            if hasattr(rr, "set_time_seconds"):
                seconds = (
                    float(seq) / max(float(self.args.offline_fps), 1.0)
                    if isinstance(seq, int)
                    else 0.0
                )
                rr.set_time_seconds("wall_time", seconds)
            _log_raw_llm_call_payload(call_id, raw)

    def _log_frame(
        self,
        record: ObservabilityRecord,
        *,
        topdown: Optional[Dict[str, Any]],
        ground_truth: Dict[str, Any],
    ) -> None:
        image = _read_image(self._resolve_path(record.image_path))
        if image is not None:
            rr.log("world/rgb/first_person", rr.Image(image))
        _try_log_text(
            "agent/current_frame",
            f"- call_id: `{record.call_id}`\n"
            f"- action: `{record.action}`\n"
            f"- image_path: `{record.image_path}`",
        )
        current_position = _pose_position(record.pose)
        if current_position is not None:
            self._trajectory_points.append(current_position)
        self._log_offline_topdown(
            topdown,
            current_position=current_position,
            ground_truth=ground_truth,
        )

    def _log_offline_topdown(
        self,
        topdown: Optional[Dict[str, Any]],
        *,
        current_position: Optional[list[float]],
        ground_truth: Dict[str, Any],
    ) -> None:
        if not isinstance(topdown, dict):
            return
        payload = dict(topdown)
        saved_trajectory = (
            topdown.get("trajectory_points")
            if isinstance(topdown.get("trajectory_points"), list)
            else []
        )
        trajectory_prefix = self._offline_trajectory_prefix(saved_trajectory)
        state_summary = dict(payload.get("state_summary") or {})
        if trajectory_prefix:
            state_summary["position"] = trajectory_prefix[-1]
            payload["trajectory_points"] = trajectory_prefix
        elif current_position is not None:
            state_summary["position"] = current_position
        if state_summary:
            payload["state_summary"] = state_summary
        if self._trajectory_points and not payload.get("trajectory_points"):
            payload["trajectory_points"] = list(self._trajectory_points)
        goals = _ground_truth_goal_positions(ground_truth)
        if goals:
            payload["goals"] = goals
            if payload.get("goal") is None:
                payload["goal"] = goals[0]
        elif _vector3(payload.get("goal")) is not None:
            payload["goals"] = [payload["goal"]]
        reference_path = _ground_truth_reference_path(ground_truth)
        if reference_path:
            payload["reference_path"] = reference_path
        waypoints = _ground_truth_waypoints(ground_truth)
        if waypoints:
            payload["waypoints"] = waypoints
        self._log_topdown_payload(payload)

    def _offline_trajectory_prefix(self, trajectory: list[Any]) -> list[Any]:
        if not trajectory:
            return []
        if self._frame_record_count <= 1:
            return list(trajectory)
        frame_index = max(1, min(self._frame_record_index, self._frame_record_count))
        prefix_len = 1 + int(
            (frame_index - 1) * (len(trajectory) - 1) / (self._frame_record_count - 1)
        )
        return list(trajectory[: max(1, min(len(trajectory), prefix_len))])

    def _log_topdown_payload(self, topdown: Dict[str, Any]) -> None:
        topdown_item = topdown.get("topdown_map")
        if isinstance(topdown_item, dict):
            image = _read_image(self._resolve_path(_first_path(topdown_item)))
            if image is not None:
                image = _strip_colored_topdown_overlays(image)
                rr.log("world/bev", rr.Image(image))

        map_bounds = topdown.get("map_bounds")
        meters_per_pixel = float(topdown.get("meters_per_pixel", 0.0) or 0.0)
        trajectory = _project_topdown(
            topdown.get("trajectory_points", []),
            map_bounds=map_bounds,
            meters_per_pixel=meters_per_pixel,
        )
        goal_points = _project_topdown(
            _topdown_goal_candidates(topdown),
            map_bounds=map_bounds,
            meters_per_pixel=meters_per_pixel,
        )
        reference_path = _project_topdown(
            _topdown_reference_path_candidates(topdown),
            map_bounds=map_bounds,
            meters_per_pixel=meters_per_pixel,
        )
        waypoint_points = _project_topdown(
            _topdown_waypoint_candidates(topdown),
            map_bounds=map_bounds,
            meters_per_pixel=meters_per_pixel,
        )
        state_summary = topdown.get("state_summary")
        current_position = (
            state_summary.get("position") if isinstance(state_summary, dict) else None
        )
        agent_point = _project_topdown(
            [current_position],
            map_bounds=map_bounds,
            meters_per_pixel=meters_per_pixel,
        )
        if trajectory and hasattr(rr, "LineStrips2D"):
            rr.log(
                "world/bev/path/trajectory",
                rr.LineStrips2D([trajectory], colors=[[80, 220, 120]]),
            )
        if len(reference_path) >= 2 and hasattr(rr, "LineStrips2D"):
            rr.log(
                "world/bev/gt/reference_path",
                rr.LineStrips2D([reference_path], colors=[[80, 160, 255]]),
            )
        if waypoint_points and hasattr(rr, "Points2D"):
            rr.log(
                "world/bev/gt/waypoints",
                rr.Points2D(
                    waypoint_points,
                    colors=[[255, 215, 0]],
                    radii=[3.5] * len(waypoint_points),
                ),
            )
        if goal_points and hasattr(rr, "Points2D"):
            rr.log(
                "world/bev/goal",
                rr.Points2D(
                    goal_points,
                    colors=[[255, 0, 0]],
                    radii=[4.0] * len(goal_points),
                ),
            )
        if agent_point and hasattr(rr, "Points2D"):
            rr.log(
                "world/bev/agent/position",
                rr.Points2D(agent_point, colors=[[0, 255, 0]], radii=[4.0]),
            )

    def _log_video_asset(self) -> None:
        videos = [] if self.bundle.is_file() else sorted(self.bundle.glob("**/*.mp4"))
        if not videos:
            return
        video_path = videos[0]
        _try_log_text("world/video/status", f"MP4: `{video_path}`")
        asset_video = getattr(rr, "AssetVideo", None)
        if asset_video is None:
            return
        try:
            rr.log("world/video", asset_video(path=str(video_path)))
        except Exception:
            pass


class NavViewer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.bridge = BridgeClient(host=args.bridge_host, port=args.bridge_port)
        self.session_id = args.session_id
        self._last_error: Optional[str] = None
        # Scene graph: fetched once (static per session), then cached
        self._sg_cache: Optional[Dict[str, Any]] = None
        self._sg_logged: bool = False
        # Action ring: track last seen timestamp to emit only new entries
        self._action_ring_last_t: float = 0.0
        # Explicit per-step frame directory; start at its current maximum
        # sequence so future frames are shown without replaying its history.
        self._resolved_tail_dir: Optional[str] = args.tail_frames_dir
        self._last_tailed_seq: int = -1
        self._tail_baseline_set: bool = False

    def _resolve_tail_dir(self) -> Optional[str]:
        if not self.args.tail_frames:
            return None
        return self._resolved_tail_dir

    def _drain_tail_frames(self) -> int:
        """Emit any new per-step frames from the tail directory to rerun.

        Returns the number of seq groups drained this tick.

        Two things to be careful about:

        1. **Catch-up cap**: a long ``navigate()`` burst can produce
           dozens of new PNGs between ticks. We cap per-tick emission
           by ``--tail-drain-limit`` and keep the *tail* of the queue
           (newest frames) rather than the head, so the viewer never
           falls further behind. The dropped frames are only missed
           from the live feed — they remain on disk for
           ``export_video_trace`` to read later.

        2. **Per-frame timestamps + pacing**: every drained frame is
           stamped with a fresh ``time.time()`` value AND the drain
           sleeps ~``1 / --tail-playback-fps`` between frames. Without
           this, rerun's follow-latest mode would collapse the whole
           burst to the last frame (they would all share one wall_time
           stamp). Pacing turns the burst into a smooth
           ``fps``-frame-per-second replay.
        """
        if not self.args.tail_frames:
            return 0
        tail_dir = self._resolve_tail_dir()
        if tail_dir is None:
            return 0

        # On first resolve, set the cursor to the current max so we only
        # emit frames that land after the viewer attaches. Without this
        # the viewer would replay every historical frame in the session.
        if not self._tail_baseline_set:
            existing = _scan_tail_dir(tail_dir, after_seq=-1)
            if existing:
                self._last_tailed_seq = existing[-1][0]
            self._tail_baseline_set = True
            return 0

        frames = _scan_tail_dir(tail_dir, after_seq=self._last_tailed_seq)
        if not frames:
            return 0

        drain_limit = max(1, int(self.args.tail_drain_limit))
        if len(frames) > drain_limit:
            # Skip dropped-in-middle frames: we prefer to stay fresh
            # (keep the newest) over showing a few extra stale frames.
            # Advance the cursor past the dropped ones so we do not try
            # to re-emit them next tick.
            dropped = frames[:-drain_limit]
            self._last_tailed_seq = dropped[-1][0]
            frames = frames[-drain_limit:]

        playback_fps = max(1.0, float(self.args.tail_playback_fps))
        frame_interval_s = 1.0 / playback_fps
        total = len(frames)
        for index, (seq, sensor_paths) in enumerate(frames):
            visuals: Dict[str, Any] = {}
            for sensor, path in sensor_paths.items():
                mode = "L" if "depth" in sensor else "RGB"
                visuals[sensor] = {"path": path, "mode": mode}
            # Stamp each frame with a unique (step_count, wall_time_s)
            # pair so rerun's timeline sees a monotonic progression
            # instead of one flash-collision. `seq` is the capture
            # counter from the bridge — monotonic per session.
            _set_step_timeline(seq, time.time())
            self._log_visual_sensors(visuals)
            self._last_tailed_seq = seq
            if index + 1 < total:
                time.sleep(frame_interval_s)
        return total

    def _resolve_session_id(self) -> Optional[str]:
        if not self.session_id:
            runtime = self.bridge.call(
                "get_runtime_status", {}, timeout=self.args.bridge_timeout,
            )
            sessions = runtime.get("sessions", [])
            if isinstance(sessions, list) and sessions:
                sid = sessions[-1].get("session_id")
                if isinstance(sid, str) and sid:
                    self.session_id = sid
        return self.session_id

    def _capture(self) -> FrameSnapshot:
        wall_time_s = time.time()
        session_id = self._resolve_session_id()
        if not isinstance(session_id, str) or not session_id:
            raise RuntimeError(
                "Unable to resolve session_id. Provide --session-id."
            )

        self.bridge.session_id = session_id

        # Image data comes from disk (via _drain_tail_frames) whenever
        # tail mode is active AND the tail directory has been resolved
        # successfully. In that case we skip the get_visuals RPC
        # entirely so a long navigate() burst cannot block the metadata
        # path either. metrics is populated from the fallback sources
        # (topdown) below.
        use_tail = self.args.tail_frames and self._resolved_tail_dir is not None
        if use_tail:
            visuals = {"visuals": {}}
            metrics: Optional[Dict[str, Any]] = None
        else:
            visuals = self.bridge.call(
                "get_visuals",
                {
                    "refresh": True,
                    "include_metrics": True,
                    "output_dir": self.args.output_dir,
                    "depth_max": self.args.depth_max,
                },
                timeout=self.args.bridge_timeout,
            )
            metrics = visuals.get("metrics")

        depth_analysis = None
        try:
            depth_analysis = self.bridge.call(
                "analyze_depth",
                {"clearance_threshold": self.args.clearance_threshold},
                timeout=self.args.bridge_timeout,
            )
        except Exception:
            depth_analysis = None

        topdown = None
        try:
            topdown = self.bridge.call(
                "get_topdown_map",
                {
                    "output_dir": self.args.output_dir,
                    "meters_per_pixel": self.args.meters_per_pixel,
                    "show_pose": True,
                    "show_traj": self.args.show_traj,
                    "show_collisions": self.args.show_collisions,
                    "show_path": self.args.show_path,
                    "collision_limit": self.args.collision_limit,
                },
                timeout=self.args.bridge_timeout,
            )
        except Exception:
            topdown = None

        step_count = None
        for candidate in (
            visuals.get("step_count"),
            topdown.get("step_count") if isinstance(topdown, dict) else None,
            metrics.get("step_count") if isinstance(metrics, dict) else None,
        ):
            if isinstance(candidate, int):
                step_count = candidate
                break

        # --- Scene graph (static per session; fetch once) ---
        scene_graph: Optional[Dict[str, Any]] = None
        if not self._sg_cache:
            try:
                sg_resp = self.bridge.call(
                    "get_scene_graph",
                    {"query_type": "all", "max_results": 500},
                    timeout=self.args.bridge_timeout,
                )
                if sg_resp.get("scene_graph_available") and sg_resp.get("nodes"):
                    self._sg_cache = sg_resp
                    scene_graph = sg_resp
                elif not sg_resp.get("scene_graph_available"):
                    _try_log_text(
                        "debug/sg_status",
                        "SG not available for this session (mapless mode or no SG file).",
                    )
                else:
                    _try_log_text(
                        "debug/sg_status",
                        f"SG response had no nodes: {_safe_json(sg_resp)[:300]}",
                    )
            except Exception as exc:
                _try_log_text("debug/sg_status", f"SG fetch error: {exc}")
        elif not self._sg_logged:
            scene_graph = self._sg_cache  # trigger logging on first frame after fetch

        # --- Action ring (recent agent tool calls from bridge) ---
        action_ring: Optional[list] = None
        try:
            runtime = self.bridge.call(
                "get_runtime_status",
                {},
                timeout=self.args.bridge_timeout,
            )
            action_ring = runtime.get("action_ring")
        except Exception:
            pass

        return FrameSnapshot(
            wall_time_s=wall_time_s,
            session_id=session_id,
            step_count=step_count,
            metrics=metrics if isinstance(metrics, dict) else None,
            visuals=(
                visuals.get("visuals", {})
                if isinstance(visuals.get("visuals"), dict)
                else {}
            ),
            topdown=topdown if isinstance(topdown, dict) else None,
            depth_analysis=depth_analysis if isinstance(depth_analysis, dict) else None,
            scene_graph=scene_graph,
            action_ring=action_ring,
        )

    def _log_frame(self, frame: FrameSnapshot) -> None:
        _set_step_timeline(frame.step_count, frame.wall_time_s)
        metrics = frame.metrics or {}
        summary = [
            f"session_id: `{frame.session_id}`",
            f"step_count: `{frame.step_count}`",
            f"last_action: `{metrics.get('last_action')}`",
            f"collisions: `{metrics.get('collision_count', 0)}`",
        ]
        _try_log_text("debug/session", "\n".join(f"- {line}" for line in summary))
        _try_log_text("debug/metrics", f"```json\n{_safe_json(metrics)}\n```")

        if isinstance(frame.step_count, int) and hasattr(rr, "Scalar"):
            rr.log("debug/step_count", rr.Scalar(frame.step_count))
        if hasattr(rr, "Scalar"):
            for entity, value in (
                (
                    "debug/collisions",
                    metrics.get("collision_count"),
                ),
                ("debug/trajectory_length", metrics.get("trajectory_length")),
                ("debug/step_time_s", metrics.get("step_time_s")),
            ):
                if isinstance(value, (int, float)):
                    rr.log(entity, rr.Scalar(value))

        self._log_visual_sensors(frame.visuals)

        if isinstance(frame.depth_analysis, dict):
            _try_log_text(
                "world/depth_analysis",
                f"```json\n{_safe_json(frame.depth_analysis)}\n```",
            )
            center = frame.depth_analysis.get("front_center")
            if isinstance(center, dict) and hasattr(rr, "Scalar"):
                for entity, value in (
                    ("world/depth/front_center_min_m", center.get("min_dist")),
                    ("world/depth/front_center_mean_m", center.get("mean_dist")),
                ):
                    if isinstance(value, (int, float)):
                        rr.log(entity, rr.Scalar(value))

        self._log_topdown(frame.topdown)

        if frame.scene_graph is not None:
            self._log_scene_graph(frame.scene_graph)

        if isinstance(frame.action_ring, list):
            self._log_action_ring(frame.action_ring)

    def _log_visual_sensors(self, visuals: Dict[str, Any]) -> None:
        if not isinstance(visuals, dict):
            return

        for sensor_name, sensor_item in visuals.items():
            if not isinstance(sensor_name, str) or not isinstance(sensor_item, dict):
                continue

            image = _read_image(_first_path(sensor_item))
            if image is None:
                continue

            sensor_lower = sensor_name.lower()
            mode = str(sensor_item.get("mode", "")).upper()
            is_semantic = "semantic" in sensor_lower
            is_depth = bool(
                (image.ndim == 2 or mode == "L" or "depth" in sensor_lower)
                and not is_semantic
            )
            entity_suffix = _entity_component(sensor_name)

            if sensor_name == "color_sensor":
                # Match blueprint origin: world/rgb/first_person
                rr.log("world/rgb/first_person", rr.Image(image))
                continue
            if sensor_name == "depth_sensor":
                # Match blueprint origin: world/depth/first_person
                rr.log("world/depth/first_person", rr.Image(image))
                continue

            if is_semantic:
                rr.log(f"world/semantic/{entity_suffix}", rr.Image(image))
            elif is_depth:
                rr.log(f"world/depth/{entity_suffix}", rr.Image(image))
            else:
                if "third" in sensor_lower:
                    # Match blueprint origin: world/third_person/rgb
                    rr.log("world/third_person/rgb", rr.Image(image))
                else:
                    rr.log(f"world/rgb/{entity_suffix}", rr.Image(image))

    def _log_topdown(self, topdown: Optional[Dict[str, Any]]) -> None:
        if not isinstance(topdown, dict):
            _try_log_text("world/bev/status", "Topdown unavailable for this session.")
            return

        topdown_item = topdown.get("topdown_map")
        if not isinstance(topdown_item, dict):
            _try_log_text("world/bev/status", "Topdown response missing topdown_map.")
            return

        image = _read_image(_first_path(topdown_item))
        if image is not None:
            rr.log("world/bev", rr.Image(image))

        map_bounds = topdown.get("map_bounds")
        meters_per_pixel = float(topdown.get("meters_per_pixel", 0.0) or 0.0)
        trajectory = _project_topdown(
            topdown.get("trajectory_points", []),
            map_bounds=map_bounds,
            meters_per_pixel=meters_per_pixel,
        )
        path_points = _project_topdown(
            topdown.get("path_points", []),
            map_bounds=map_bounds,
            meters_per_pixel=meters_per_pixel,
        )
        collisions_world = [
            item.get("position")
            for item in topdown.get("collision_points", [])
            if isinstance(item, dict)
        ]
        collision_points = _project_topdown(
            collisions_world,
            map_bounds=map_bounds,
            meters_per_pixel=meters_per_pixel,
        )
        goal_points = _project_topdown(
            _topdown_goal_candidates(topdown),
            map_bounds=map_bounds,
            meters_per_pixel=meters_per_pixel,
        )
        reference_path = _project_topdown(
            _topdown_reference_path_candidates(topdown),
            map_bounds=map_bounds,
            meters_per_pixel=meters_per_pixel,
        )
        waypoint_points = _project_topdown(
            _topdown_waypoint_candidates(topdown),
            map_bounds=map_bounds,
            meters_per_pixel=meters_per_pixel,
        )
        current_position = None
        state_summary = topdown.get("state_summary")
        if isinstance(state_summary, dict):
            current_position = state_summary.get("position")
        agent_point = _project_topdown(
            [current_position],
            map_bounds=map_bounds,
            meters_per_pixel=meters_per_pixel,
        )

        if trajectory and hasattr(rr, "LineStrips2D"):
            rr.log(
                "world/bev/path/trajectory",
                rr.LineStrips2D([trajectory], colors=[[80, 220, 120]]),
            )
        if path_points and hasattr(rr, "LineStrips2D"):
            rr.log(
                "world/bev/path/planned",
                rr.LineStrips2D([path_points], colors=[[255, 80, 80]]),
            )
        if len(reference_path) >= 2 and hasattr(rr, "LineStrips2D"):
            rr.log(
                "world/bev/gt/reference_path",
                rr.LineStrips2D([reference_path], colors=[[80, 160, 255]]),
            )
        if waypoint_points and hasattr(rr, "Points2D"):
            rr.log(
                "world/bev/gt/waypoints",
                rr.Points2D(
                    waypoint_points,
                    colors=[[255, 215, 0]],
                    radii=[3.5] * len(waypoint_points),
                ),
            )
        if collision_points and hasattr(rr, "Points2D"):
            rr.log(
                "world/bev/collisions",
                rr.Points2D(collision_points, colors=[[255, 215, 0]], radii=[3.0]),
            )
        if goal_points and hasattr(rr, "Points2D"):
            rr.log(
                "world/bev/goal",
                rr.Points2D(
                    goal_points,
                    colors=[[255, 0, 0]],
                    radii=[4.0] * len(goal_points),
                ),
            )
        if agent_point and hasattr(rr, "Points2D"):
            rr.log(
                "world/bev/agent/position",
                rr.Points2D(agent_point, colors=[[0, 255, 0]], radii=[4.0]),
            )

        if (
            agent_point
            and isinstance(state_summary, dict)
            and hasattr(rr, "LineStrips2D")
        ):
            heading_deg = state_summary.get("heading_deg")
            if isinstance(heading_deg, (int, float)):
                heading_rad = np.deg2rad(float(heading_deg))
                start = np.asarray(agent_point[0], dtype=np.float32)
                tip = (
                    start
                    + np.asarray(
                        [np.cos(heading_rad), np.sin(heading_rad)], dtype=np.float32
                    )
                    * 10.0
                )
                rr.log(
                    "world/bev/agent/heading",
                    rr.LineStrips2D(
                        [[start.tolist(), tip.tolist()]], colors=[[0, 255, 255]]
                    ),
                )

        bev_summary = {
            "meters_per_pixel": meters_per_pixel,
            "image_size": topdown.get("image_size"),
            "map_bounds": map_bounds,
            "show_path": bool(path_points),
            "show_traj": bool(trajectory),
            "show_gt_reference_path": bool(reference_path),
            "gt_waypoint_count": len(waypoint_points),
            "show_collisions": bool(collision_points),
        }
        _try_log_text("world/bev/status", f"```json\n{_safe_json(bev_summary)}\n```")

    # ------------------------------------------------------------------
    # Scene-graph visualization (logged once; SG is static per session)
    # ------------------------------------------------------------------

    def _log_scene_graph(self, sg: Dict[str, Any]) -> None:
        """Render SG nodes and object bounding boxes in 3D space."""
        nodes = sg.get("nodes", [])
        if not nodes:
            return

        room_nodes = [n for n in nodes if n.get("type") == "room"]
        obj_nodes = [n for n in nodes if n.get("type") == "object"]

        # --- Room centroids (cyan spheres) ---
        # Room nodes carry 2D `centroid_xy` (navmesh plane), lift to 3D.
        room_pos: list = []
        room_labels: list = []
        for r in room_nodes:
            pos = _vector2_to_3(r.get("centroid_xy"))
            if pos is None:
                # Fallback: some datasets might use position_xyz for rooms
                pos = _vector3(r.get("position_xyz"))
            if pos is not None:
                room_pos.append(pos.tolist())
                room_labels.append(str(r.get("id", "room")))

        if room_pos and hasattr(rr, "Points3D"):
            rr.log(
                "world/scene_graph/rooms",
                rr.Points3D(
                    room_pos,
                    colors=[[0, 210, 255]] * len(room_pos),  # cyan
                    radii=[0.18] * len(room_pos),
                    labels=room_labels,
                ),
            )

        # --- Object centroids (amber spheres) + bounding boxes ---
        obj_pos: list = []
        obj_labels: list = []
        bbox_centers: list = []
        bbox_half_sizes: list = []

        for o in obj_nodes:
            pos = _vector3(o.get("position_xyz"))
            if pos is None:
                continue
            obj_pos.append(pos.tolist())
            obj_labels.append(str(o.get("label", "object")))

            bmin = _vector3(o.get("bbox_min_xyz"))
            bmax = _vector3(o.get("bbox_max_xyz"))
            if bmin is not None and bmax is not None:
                center = ((bmin + bmax) / 2.0).tolist()
                half = ((bmax - bmin) / 2.0).tolist()
                # Skip degenerate boxes (any dimension < 1 mm)
                if all(h > 0.0005 for h in half):
                    bbox_centers.append(center)
                    bbox_half_sizes.append(half)

        if obj_pos and hasattr(rr, "Points3D"):
            rr.log(
                "world/scene_graph/objects",
                rr.Points3D(
                    obj_pos,
                    colors=[[255, 165, 0]] * len(obj_pos),  # amber
                    radii=[0.06] * len(obj_pos),
                    labels=obj_labels,
                ),
            )

        if bbox_centers and hasattr(rr, "Boxes3D"):
            rr.log(
                "world/scene_graph/bboxes",
                rr.Boxes3D(
                    centers=bbox_centers,
                    half_sizes=bbox_half_sizes,
                    colors=[[255, 165, 0, 60]]
                    * len(bbox_centers),  # semi-transparent amber
                ),
            )

        # Summary text
        _try_log_text(
            "world/scene_graph/summary",
            f"**Scene Graph**\n"
            f"- Rooms: {len(room_nodes)}\n"
            f"- Objects: {len(obj_nodes)}\n"
            f"- Bboxes logged: {len(bbox_centers)}\n"
            f"- Scene: `{sg.get('scene_id', '-')}`",
        )

        self._sg_logged = True

    # ------------------------------------------------------------------
    # Tool-call log (emit new entries from the action ring each poll)
    # ------------------------------------------------------------------

    def _log_action_ring(self, action_ring: list) -> None:
        """Forward new bridge action records to the rerun TextLog panel."""
        if not action_ring or not hasattr(rr, "TextLog"):
            return

        # Only emit entries we haven't seen yet (compare by timestamp)
        new_entries = [
            entry
            for entry in action_ring
            if isinstance(entry, dict)
            and float(entry.get("t", 0)) > self._action_ring_last_t
        ]
        if not new_entries:
            return

        level_ok = getattr(getattr(rr, "TextLogLevel", None), "INFO", "INFO")
        level_err = getattr(getattr(rr, "TextLogLevel", None), "ERROR", "ERROR")

        for entry in new_entries:
            action = entry.get("action", "?")
            ok = bool(entry.get("ok", True))
            sid = str(entry.get("session_id") or "")[:8]
            err_msg = entry.get("error", "")

            if ok:
                text = f"[{action}]  session={sid}  ✓"
                level = level_ok
            else:
                text = f"[{action}]  session={sid}  ✗  {err_msg}"
                level = level_err

            try:
                rr.log("agent/tool_calls", rr.TextLog(text, level=level))
            except TypeError:
                try:
                    rr.log("agent/tool_calls", rr.TextLog(text))
                except Exception:
                    pass

        self._action_ring_last_t = max(float(e.get("t", 0)) for e in new_entries)

    def run(self) -> None:
        while True:
            try:
                # Drain per-step frames from disk first. This path never
                # blocks on the bridge, so it stays responsive during
                # long navigate() RPCs that would otherwise freeze the
                # single-threaded HTTP server for the duration of the
                # whole burst.
                self._drain_tail_frames()
                frame = self._capture()
                self._log_frame(frame)
                self._last_error = None
            except KeyboardInterrupt:
                return
            except Exception as exc:
                message = str(exc)
                if message != self._last_error:
                    _try_log_text("debug/errors", message)
                    self._last_error = message
            time.sleep(self.args.poll_interval_ms / 1000.0)


def parse_args() -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help=argparse.SUPPRESS,
    )
    pre_args, remaining = pre_parser.parse_known_args()
    config_path = Path(pre_args.config).expanduser()
    config = _load_yaml_config(config_path)

    parser = argparse.ArgumentParser(
        description="Live Rerun viewer for habitat-gs nav sessions."
    )
    parser.add_argument(
        "--config",
        default=str(config_path),
        help="Path to YAML config file. CLI flags override config values.",
    )
    parser.add_argument(
        "--session-id", default=None, help="Existing habitat-gs session id."
    )
    parser.add_argument(
        "--offline-bundle",
        default=None,
        help=(
            "Replay a saved session directory or common observability JSONL "
            "instead of connecting to a live bridge."
        ),
    )
    parser.add_argument(
        "--offline-fps",
        type=float,
        default=30.0,
        help="Synthetic wall-time FPS for --offline-bundle frame replay.",
    )
    parser.add_argument("--bridge-host", default="127.0.0.1", help="Bridge host.")
    parser.add_argument("--bridge-port", type=int, default=18911, help="Bridge port.")
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=None,
        help="Explicit Rerun gRPC/proxy port for --serve-web. Use this if the default port is occupied.",
    )
    parser.add_argument(
        "--bridge-timeout",
        type=float,
        default=15.0,
        help="Bridge call timeout in seconds.",
    )
    parser.add_argument(
        "--poll-interval-ms",
        type=int,
        default=400,
        help="Polling interval in milliseconds.",
    )
    parser.add_argument(
        "--tail-frames",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When enabled (default), read per-step RGB/depth/third_person frames "
            "directly from --tail-frames-dir instead of calling "
            "`get_visuals` via the bridge. The bridge is single-threaded, so "
            "long `navigate()` RPCs block HTTP polls; disk tailing lets the "
            "viewer see every intermediate simulator step without waiting."
        ),
    )
    parser.add_argument(
        "--tail-frames-dir",
        default=None,
        help=(
            "Directory to tail for per-step frames."
        ),
    )
    parser.add_argument(
        "--tail-drain-limit",
        type=int,
        default=64,
        help=(
            "Max number of per-step frames to drain from disk per tick. "
            "Keeps the logging loop responsive when catching up after a long "
            "navigate burst; the rest will be picked up next tick."
        ),
    )
    parser.add_argument(
        "--tail-playback-fps",
        type=float,
        default=30.0,
        help=(
            "Pacing rate for the tailed per-step frames (frames per second). "
            "Each drained frame gets a fresh `time.time()` stamp and the drain "
            "loop sleeps ~1/fps between frames, so rerun replays the burst at "
            "a human-watchable speed instead of flashing the whole navigate() "
            "burst in one tick. Combined with --tail-drain-limit to cap how "
            "long a single tick can spend pacing. Set to a large value "
            "(e.g. 1000) to effectively disable pacing."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=_default_output_dir(),
        help=(
            "Where bridge writes captured frames. "
            "Defaults to NAV_VISUAL_OUTPUT_DIR/HAB_VISUAL_OUTPUT_DIR/NAV_ARTIFACTS_DIR "
            "or a per-user temp directory."
        ),
    )
    parser.add_argument(
        "--depth-max",
        type=float,
        default=10.0,
        help="Depth visualization max for get_visuals.",
    )
    parser.add_argument(
        "--clearance-threshold",
        type=float,
        default=0.5,
        help="Depth analysis clearance threshold.",
    )
    parser.add_argument(
        "--meters-per-pixel", type=float, default=0.05, help="Topdown map resolution."
    )
    parser.add_argument(
        "--collision-limit",
        type=int,
        default=20,
        help="Number of collisions to request in BEV.",
    )
    parser.add_argument(
        "--show-traj",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request trajectory overlay in topdown.",
    )
    parser.add_argument(
        "--show-collisions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request collision overlay in topdown.",
    )
    parser.add_argument(
        "--show-path",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Request planned path overlay in topdown.",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Write a .rrd recording file. Defaults to /tmp/<app-id>.rrd in headless mode.",
    )
    parser.add_argument(
        "--serve-web",
        type=int,
        default=None,
        metavar="PORT",
        help="Serve a Rerun web viewer on the given port.",
    )
    parser.add_argument(
        "--spawn", action="store_true", help="Spawn a local Rerun viewer window."
    )
    parser.add_argument(
        "--connect",
        default=None,
        help='Connect to an existing Rerun viewer as "HOST:PORT".',
    )
    parser.add_argument(
        "--app-id", default="habitat-gs-nav", help="Rerun application id."
    )
    parser.set_defaults(
        bridge_host=config.get("bridge_host", "127.0.0.1"),
        bridge_port=config.get("bridge_port", 18911),
        serve_web=config.get("web_port"),
        grpc_port=config.get("grpc_port"),
    )
    args = parser.parse_args(remaining)
    sink_count = (
        int(bool(args.save))
        + int(bool(args.spawn))
        + int(bool(args.connect))
        + int(args.serve_web is not None)
    )
    if sink_count > 1:
        parser.error(
            "--save, --serve-web, --spawn, and --connect are mutually exclusive."
        )
    return args


def main() -> None:
    args = parse_args()
    _configure_rerun(args)
    if args.offline_bundle:
        OfflineNavReplay(args).run()
    else:
        NavViewer(args).run()


if __name__ == "__main__":
    main()
