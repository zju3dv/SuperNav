from __future__ import annotations

import asyncio
import base64
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


def _tiny_png(path: Path, *, marker: bytes) -> None:
    # Valid 1x1 PNG with a deterministic trailing marker so tests can
    # distinguish color and depth payloads after MCP base64 encoding.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMB/"
        "ax8fkAAAAAASUVORK5CYII="
    )
    path.write_bytes(png + marker)


@pytest.mark.parametrize("enabled,overrides,max_steps,horizon_m", [
    (False, {}, 20, 40.0),
    (True, {}, 200, 10000.0),
    (True, {"max_steps": 75, "horizon_m": 80.0}, 75, 80.0),
])
def test_geo_based_executor_setting_reaches_bridge_payload(
    monkeypatch, tmp_path, enabled, overrides, max_steps, horizon_m,
):
    from supernav.methods.navigation.tools.navigation_oracle import VisualPointNavigateTool
    from supernav.methods.navigation.oracle_local_nav.visual_point import _geo_based_executor_mode_enabled

    monkeypatch.setenv("HAB_VISUAL_POINT_GEO_BASED_EXECUTOR", "1" if enabled else "0")
    calls = []

    def call(action, payload):
        calls.append((action, payload))
        return {"ok": True, "status": "synthetic-navigation-result"}

    result = VisualPointNavigateTool().execute(
        {"image_ref": "synthetic:front", "point": [0.5, 0.5], **overrides},
        SimpleNamespace(bridge=SimpleNamespace(call=call), output_dir=str(tmp_path)),
    )
    assert result.ok and len(calls) == 1
    action, payload = calls[0]
    assert action == "navigate_visual_point"
    assert payload["point"] == [0.5, 0.5]
    assert payload["max_steps"] == max_steps and payload["horizon_m"] == horizon_m
    assert payload.get("geo_based_executor", False) is enabled
    # The bridge process can receive the mode through the payload alone.
    monkeypatch.delenv("HAB_VISUAL_POINT_GEO_BASED_EXECUTOR")
    assert _geo_based_executor_mode_enabled(payload) is enabled


@pytest.fixture
def mcp_module(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("NAV_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setenv("HAB_ENABLE_ORACLE_LOCAL_NAV", "1")
    monkeypatch.setenv("HAB_MCP_GATE_ENABLED", "0")
    from supernav.methods.navigation.tools import navigation_oracle
    from supernav.methods.navigation.tools.base import ToolRegistry

    ToolRegistry.unregister("oracle_local_map")
    ToolRegistry.unregister("oracle_local_navigate")
    importlib.reload(navigation_oracle)
    sys.modules.pop("supernav.methods.navigation.mcp_server", None)
    mod = importlib.import_module("supernav.methods.navigation.mcp_server")
    mod._bridge.session_id = "s1"
    calls: list[tuple[str, Any]] = []

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        calls.append((action, payload))
        return {"loop_id": "fake_loop", "pid": 1, "status": "started"}

    monkeypatch.setattr(mod._bridge, "call", fake_call)
    mod._calls = calls  # type: ignore[attr-defined]
    return mod


def _converted_tool_content(
    mcp_module: Any, name: str, args: dict[str, Any] | None = None
):
    result = asyncio.run(
        mcp_module.mcp._tool_manager.call_tool(
            name,
            args or {},
            convert_result=True,
        )
    )
    # FastMCP returns (unstructured_content, structured_content) when
    # the wrapper has a structured-output schema. The MCP wire content
    # blocks are the first element.
    if isinstance(result, tuple):
        return result[0]
    return result


def _text_payload(content: list[Any]) -> dict[str, Any]:
    texts = [
        getattr(item, "text", None)
        for item in content
        if getattr(item, "type", None) == "text"
    ]
    assert texts, content
    return json.loads(texts[0])


def _image_blocks(content: list[Any]) -> list[Any]:
    return [item for item in content if getattr(item, "type", None) == "image"]


def _make_panorama_images(tmp_path: Path) -> dict[str, Path]:
    colors: dict[str, Path] = {}
    for direction in ("front", "right", "back", "left"):
        path = tmp_path / f"pano_{direction}_color_sensor.png"
        _tiny_png(path, marker=direction.encode("ascii"))
        colors[direction] = path
    return colors


def _expected_panorama_rows(colors: dict[str, Path]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(colors["front"]),
            "direction": "front",
            "view": "front",
            "turn_right_deg": 0,
        },
        {
            "path": str(colors["right"]),
            "direction": "right",
            "view": "right",
            "turn_right_deg": 90,
        },
        {
            "path": str(colors["back"]),
            "direction": "back",
            "view": "back",
            "turn_right_deg": 180,
        },
        {
            "path": str(colors["left"]),
            "direction": "left",
            "view": "left",
            "turn_right_deg": 270,
        },
    ]


def _visible_targets() -> list[dict[str, Any]]:
    return [
        {
            "target_ref": "obj-chair-1",
            "label": "chair",
            "direction": "front",
            "turn_right_deg": 0,
            "bbox_px": {"x": 10, "y": 12, "width": 20, "height": 18},
            "center_px": [20, 21],
            "depth_m": 2.4,
            "visible_fraction": 0.32,
            "visibility_basis": "projection_depth",
            "object_position": [1.0, 0.2, 3.0],
            "_debug": {"bbox_min_xyz": [0.0, 0.0, 0.0]},
        }
    ]


def _caption_texts(content: list[Any]) -> list[str]:
    texts = [
        getattr(item, "text", None)
        for item in content[1:]
        if getattr(item, "type", None) == "text"
    ]
    return [text for text in texts if isinstance(text, str)]


def _assert_captioned_images(
    content: list[Any],
    expected_captions: list[str],
) -> list[Any]:
    assert _caption_texts(content) == expected_captions
    images = _image_blocks(content)
    assert len(images) == len(expected_captions)
    image_index = 0
    for idx in range(1, len(content), 2):
        assert getattr(content[idx], "type", None) == "text"
        assert getattr(content[idx], "text", None) == expected_captions[image_index]
        assert getattr(content[idx + 1], "type", None) == "image"
        image_index += 1
    return images


def test_mcp_turn_returns_inline_rgb_image_block_without_depth_or_gt(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    color = tmp_path / "look_color_sensor.png"
    depth = tmp_path / "look_depth_sensor.png"
    _tiny_png(color, marker=b"COLOR")
    _tiny_png(depth, marker=b"DEPTH")

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        return {
            "visuals": {
                "color_sensor": {"path": str(color)},
                "depth_sensor": {"path": str(depth)},
            },
            "agent_state": {"heading": 1.5},
            "ground_truth": {"target_position": [9, 9, 9]},
            "eval_goal_position": [8, 8, 8],
            "reference_path": "/secret/gt.json",
            "score": {"strict_task_success": True},
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(mcp_module, "hab_turn", {"direction": "left"})
    payload = _text_payload(content)
    images = _image_blocks(content)

    assert payload["images"] == [str(color), str(depth)]
    assert payload["agent_state"]["heading"] == 1.5
    assert "visuals" not in payload
    payload_text = json.dumps(payload, ensure_ascii=False)
    for forbidden in (
        "ground_truth",
        "target_position",
        "eval_goal_position",
        "reference_path",
        "strict_task_success",
    ):
        assert forbidden not in payload_text

    assert _caption_texts(content) == ["Current egocentric view"]
    assert len(images) == 1
    assert images[0].mimeType == "image/png"
    assert images[0].data == base64.b64encode(color.read_bytes()).decode("ascii")
    assert images[0].data != base64.b64encode(depth.read_bytes()).decode("ascii")


def test_mcp_oracle_local_map_is_not_registered(
    mcp_module: Any,
) -> None:
    with pytest.raises(Exception):
        _converted_tool_content(mcp_module, "hab_oracle_local_map")


def test_mcp_oracle_local_navigate_keeps_movement_frames_for_replay(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    movement = []
    for index in (1, 2):
        path = tmp_path / f"step{index:06d}_color_sensor.png"
        _tiny_png(path, marker=f"MOVE{index}".encode("ascii"))
        movement.append(path)
    colors = _make_panorama_images(tmp_path)
    calls: list[tuple[str, Any]] = []

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        calls.append((action, payload))
        if action == "navigate_oracle_local":
            assert payload["output_dir"] == str(tmp_path)
            return {
                "ok": True,
                "backend": "oracle_local_gt",
                "status": "reached_local_target",
                "movement_frames": [str(path) for path in movement],
                "movement_frame_count": len(movement),
                "_debug": {"resolved_local_target": [1.0, 0.2, 2.0]},
            }
        assert action == "get_panorama"
        return {
            "images": [
                {"path": str(colors[direction]), "direction": direction}
                for direction in ("front", "right", "back", "left")
            ],
            "visible_nav_targets": _visible_targets(),
            "visible_nav_targets_status": "ok",
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(
        mcp_module,
        "hab_oracle_local_navigate",
        {"target_ref": "obj-chair-1"},
    )
    payload = _text_payload(content)
    images = _image_blocks(content)

    assert payload["movement_frames"] == [str(path) for path in movement]
    assert payload["movement_frame_count"] == 2
    assert "_debug" not in json.dumps(payload)
    assert payload["images"] == [
        str(colors[direction]) for direction in ("front", "right", "back", "left")
    ]
    assert "visible_nav_targets_status" not in payload
    assert "visible_nav_targets" not in payload
    payload_text = json.dumps(payload, ensure_ascii=False)
    assert "_debug" not in payload_text
    assert "object_position" not in payload_text
    assert _caption_texts(content) == [
        "Front view",
        "Right view",
        "Back view",
        "Left view",
    ]
    assert len(images) == 4
    assert [action for action, _ in calls] == [
        "navigate_oracle_local",
        "get_panorama",
    ]
    assert calls[0][1]["target_ref"] == "obj-chair-1"
    assert "target_label" not in calls[0][1]
    assert "instruction" not in calls[0][1]


def test_mcp_visual_point_preview_does_not_capture_panorama(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    overlay = tmp_path / "visual_point_overlay.png"
    _tiny_png(overlay, marker=b"OVERLAY")
    calls: list[tuple[str, Any]] = []

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        calls.append((action, payload))
        assert action == "navigate_visual_point"
        return {
            "ok": True,
            "backend": "visual_point_navmesh",
            "status": "preview_ready",
            "nav_status": "preview_ready",
            "confirm_token": "confirm-1",
            "selected_anchor": {
                "image_ref": "pano:s1:7:right",
                "direction": "right",
                "point": [0.5, 0.8],
            },
            "overlay_image": str(overlay),
            "planned_path_m": 1.25,
            "pre_move_self_check": {"required": True},
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(
        mcp_module,
        "hab_visual_point_navigate",
        {"image_ref": "pano:s1:7:right", "point": [0.5, 0.8]},
    )
    payload = _text_payload(content)
    images = _image_blocks(content)

    assert [action for action, _ in calls] == ["navigate_visual_point"]
    assert payload["status"] == "preview_ready"
    assert payload["confirm_token"] == "confirm-1"
    assert payload["pre_move_self_check"]["required"] is True
    assert payload["overlay_image"] == str(overlay)
    assert "panorama_images" not in payload
    assert len(images) == 1
    assert images[0].data == base64.b64encode(overlay.read_bytes()).decode("ascii")


def test_mcp_visual_ground_preview_resolves_view_shortcut(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    front = tmp_path / "pano_front_color_sensor.png"
    _tiny_png(front, marker=b"FRONT")
    mcp_module._shape_strip_visuals(
        {
            "images": [
                {
                    "image_ref": "pano:s1:9:front",
                    "path": str(front),
                    "direction": "front",
                }
            ]
        }
    )
    calls: list[tuple[str, Any]] = []

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        calls.append((action, payload))
        assert action == "navigate_visual_ground_preview"
        assert payload["image_ref"] == "pano:s1:9:front"
        assert "view" not in payload
        return {
            "ok": True,
            "backend": "locate_anything_preview",
            "status": "preview_ready",
            "confirm_token": "ground-confirm-1",
            "candidate_count": 2,
            "candidates": [
                {
                    "candidate_id": 0,
                    "reachable": True,
                    "snapped_point": [1.0, 0.2, 3.0],
                }
            ],
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(
        mcp_module,
        "hab_visual_ground_preview",
        {
            "view": "front",
            "phrase": "cabinet on the right side",
            "include_images": False,
        },
    )
    payload = _text_payload(content)

    assert [action for action, _ in calls] == ["navigate_visual_ground_preview"]
    assert payload["status"] == "preview_ready"
    assert payload["confirm_token"] == "ground-confirm-1"
    assert "snapped_point" not in json.dumps(payload)


def test_mcp_visual_point_navigate_resolves_view_shortcut(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    right = tmp_path / "pano_right_color_sensor.png"
    _tiny_png(right, marker=b"RIGHT")
    mcp_module._shape_strip_visuals(
        {
            "images": [
                {
                    "image_ref": "pano:s1:9:right",
                    "path": str(right),
                    "direction": "right",
                }
            ]
        }
    )
    calls: list[tuple[str, Any]] = []

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        calls.append((action, payload))
        assert action == "navigate_visual_point"
        assert payload["image_ref"] == "pano:s1:9:right"
        assert "view" not in payload
        return {
            "ok": True,
            "backend": "visual_point_navmesh",
            "status": "preview_ready",
            "confirm_token": "point-confirm-1",
            "overlay_image": None,
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(
        mcp_module,
        "hab_visual_point_navigate",
        {"view": "right", "point": [0.5, 0.8], "include_images": False},
    )
    payload = _text_payload(content)

    assert [action for action, _ in calls] == ["navigate_visual_point"]
    assert payload["status"] == "preview_ready"
    assert payload["confirm_token"] == "point-confirm-1"


def test_mcp_visual_view_shortcut_requires_cached_panorama(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, Any]] = []

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        calls.append((action, payload))
        raise AssertionError("view resolution failure should not dispatch")

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(
        mcp_module,
        "hab_visual_ground_preview",
        {"view": "front", "phrase": "cabinet", "include_images": False},
    )
    payload = _text_payload(content)

    assert calls == []
    assert payload["ok"] is False
    assert payload["status"] == "invalid_view"
    assert "No latest panorama image_ref is cached" in payload["error"]


def test_mcp_visual_point_confirm_hides_old_direction_and_inlines_latest_panorama(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    movement = tmp_path / "step000001_color_sensor.png"
    _tiny_png(movement, marker=b"MOVE")
    colors = _make_panorama_images(tmp_path)
    calls: list[tuple[str, Any]] = []

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        calls.append((action, payload))
        if action == "navigate_visual_point":
            assert payload["confirm_token"] == "confirm-1"
            return {
                "ok": True,
                "backend": "visual_point_navmesh",
                "status": "reached_visual_point",
                "nav_status": "reached_visual_point",
                "image_ref": "pano:s1:7:right",
                "direction": "right",
                "selected_anchor": {
                    "image_ref": "pano:s1:7:right",
                    "direction": "right",
                    "point": [0.5, 0.8],
                },
                "steps_executed": 3,
                "planned_path_m": 1.25,
                "movement_frames": [str(movement)],
                "movement_frame_count": 1,
            }
        assert action == "get_panorama"
        return {
            "images": [
                {"path": str(colors[direction]), "direction": direction}
                for direction in ("front", "right", "back", "left")
            ],
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(
        mcp_module,
        "hab_visual_point_navigate",
        {
            "image_ref": "pano:s1:7:right",
            "point": [0.5, 0.8],
            "confirm_token": "confirm-1",
        },
    )
    payload = _text_payload(content)
    images = _image_blocks(content)

    assert [action for action, _ in calls] == ["navigate_visual_point", "get_panorama"]
    assert "image_ref" not in payload
    assert "direction" not in payload
    assert payload["selected_anchor"]["image_ref"] == "pano:s1:7:right"
    assert payload["selected_anchor"]["direction"] == "right"
    assert payload["movement_frames"] == [str(movement)]
    assert payload["panorama_images"] == _expected_panorama_rows(colors)
    assert payload["post_move_observation_reset"]["old_direction_expired"] is True
    assert _caption_texts(content) == [
        "Front view",
        "Right view",
        "Back view",
        "Left view",
    ]
    assert len(images) == 4


def test_mcp_visual_ground_preview_confirm_inlines_latest_panorama(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    colors = _make_panorama_images(tmp_path)
    calls: list[tuple[str, Any]] = []

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        calls.append((action, payload))
        if action == "navigate_visual_ground_preview":
            assert payload["confirm_token"] == "ground-confirm-1"
            assert payload["candidate_id"] == 0
            return {
                "ok": True,
                "backend": "locate_anything_preview",
                "status": "reached_visual_ground_candidate",
                "nav_status": "reached_visual_ground_candidate",
                "image_ref": "pano:s1:7:right",
                "direction": "right",
                "selected_candidate_id": 0,
                "steps_executed": 3,
                "planned_path_m": 1.25,
            }
        assert action == "get_panorama"
        return {
            "images": [
                {"path": str(colors[direction]), "direction": direction}
                for direction in ("front", "right", "back", "left")
            ],
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(
        mcp_module,
        "hab_visual_ground_preview",
        {
            "image_ref": "pano:s1:7:right",
            "confirm_token": "ground-confirm-1",
            "candidate_id": 0,
        },
    )
    payload = _text_payload(content)
    images = _image_blocks(content)

    assert [action for action, _ in calls] == [
        "navigate_visual_ground_preview",
        "get_panorama",
    ]
    assert "image_ref" not in payload
    assert "direction" not in payload
    assert payload["selected_candidate_id"] == 0
    assert payload["panorama_images"] == _expected_panorama_rows(colors)
    assert payload["post_move_observation_reset"]["old_direction_expired"] is True
    assert _caption_texts(content) == [
        "Front view",
        "Right view",
        "Back view",
        "Left view",
    ]
    assert len(images) == 4


def test_visual_ground_response_hides_cached_portal_geometry(
    mcp_module: Any,
) -> None:
    body = {
        "ok": True,
        "status": "preview_ready",
        "candidate_count": 1,
        "candidates": [
            {
                "candidate_id": 0,
                "anchor_strategy": "portal_normal",
                "portal_width_m": 1.1,
                "normal_standoff_m": 0.7,
                "arrival_alignment_deg": 4.0,
                "snapped_point": [1.0, 0.2, -2.0],
                "_private": {
                    "portal_midpoint": [1.0, 0.8, -2.5],
                    "portal_normal": [0.0, 0.0, 1.0],
                    "left_jamb": [0.4, 0.8, -2.5],
                    "right_jamb": [1.6, 0.8, -2.5],
                },
            }
        ],
    }

    shaped, _ = mcp_module._shape_visual_ground_preview_response(body, {})
    encoded = json.dumps(shaped)

    assert "snapped_point" not in encoded
    assert "_private" not in encoded
    assert "portal_midpoint" not in encoded
    assert shaped["candidates"][0]["portal_width_m"] == 1.1
    assert shaped["candidates"][0]["normal_standoff_m"] == 0.7
    assert shaped["candidates"][0]["arrival_alignment_deg"] == 4.0


def test_mcp_visual_overlay_hides_old_alias_top_level(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    colors = _make_panorama_images(tmp_path)

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        if action == "navigate_visual_overlay":
            return {
                "ok": True,
                "backend": "visual_overlay_navmesh",
                "status": "reached_local_target",
                "image_ref": "pano:s1:7:left",
                "target_alias": "obj_189",
                "direction": "left",
                "selected_overlay_alias": {
                    "image_ref": "pano:s1:7:left",
                    "target_alias": "obj_189",
                    "direction": "left",
                },
                "steps_executed": 4,
            }
        assert action == "get_panorama"
        return {
            "images": [
                {"path": str(colors[direction]), "direction": direction}
                for direction in ("front", "right", "back", "left")
            ],
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(
        mcp_module,
        "hab_visual_overlay_navigate",
        {"image_ref": "pano:s1:7:left", "target_alias": "obj_189"},
    )
    payload = _text_payload(content)

    assert "image_ref" not in payload
    assert "target_alias" not in payload
    assert "direction" not in payload
    assert payload["selected_overlay_alias"]["target_alias"] == "obj_189"
    assert payload["selected_overlay_alias"]["direction"] == "left"
    assert payload["panorama_images"] == _expected_panorama_rows(colors)


def test_mcp_visual_local_navigate_sends_phrase_only_and_hides_refs(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    colors = _make_panorama_images(tmp_path)
    calls: list[tuple[str, Any]] = []

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        calls.append((action, payload))
        if action == "navigate_visual_local":
            assert payload["instruction"] == "the red sofa"
            assert "target_ref" not in payload
            assert "target_label" not in payload
            return {
                "ok": True,
                "backend": "visual_local_grounded",
                "status": "reached_local_target",
                "visual_grounding": {
                    "direction": "front",
                    "detection_confidence": 0.91,
                    "bbox_px": {"x": 10, "y": 12, "width": 20, "height": 18},
                },
                "matched_target_ref": "internal-chair-1",
                "_debug": {
                    "matched_target_ref": "internal-chair-1",
                    "visual_grounding_candidates_path": "/tmp/candidates.json",
                    "candidate_table": [
                        {
                            "scene_target_ref": "internal-chair-1",
                            "scene_label": "partition",
                            "reject_reason": "semantic_incompatible",
                        }
                    ],
                },
            }
        assert action == "get_panorama"
        return {
            "images": [
                {"path": str(colors[direction]), "direction": direction}
                for direction in ("front", "right", "back", "left")
            ],
            "visible_nav_targets": _visible_targets(),
            "visible_nav_targets_status": "ok",
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(
        mcp_module,
        "hab_visual_local_navigate",
        {"instruction": "the red sofa"},
    )
    payload = _text_payload(content)

    assert payload["visual_grounding"]["direction"] == "front"
    payload_text = json.dumps(payload, ensure_ascii=False)
    assert "internal-chair-1" not in payload_text
    assert "candidate_table" not in payload_text
    assert "partition" not in payload_text
    assert "visible_nav_targets" not in payload_text
    assert [action for action, _ in calls] == [
        "navigate_visual_local",
        "get_panorama",
    ]


def test_mcp_movement_shaper_hides_visible_nav_targets_and_keeps_depth_stats(
    mcp_module: Any,
    tmp_path: Path,
) -> None:
    colors = []
    for direction in ("front", "right", "back", "left", "extra"):
        path = tmp_path / f"pano_{direction}_color_sensor.png"
        _tiny_png(path, marker=direction.encode("ascii"))
        colors.append(path)
    depth = tmp_path / "pano_front_depth_sensor.png"
    _tiny_png(depth, marker=b"DEPTH")

    raw_body = {
        "images": [
            *(
                {
                    "path": str(path),
                    "direction": direction,
                    "overlay_aliases": ["obj-chair-1"],
                    "overlay_path": str(
                        path.with_name(f"{path.stem}_visible_overlay.png")
                    ),
                }
                for direction, path in zip(
                    ("front", "right", "back", "left", "extra"),
                    colors,
                )
            ),
            {"path": str(depth), "direction": "front_depth"},
        ],
        "depth_analysis": {"front": {"min": 1.0}},
        "visible_nav_targets": [
            {
                **_visible_targets()[0],
                "depth_m": None,
                "visibility_basis": "projection_only",
            }
        ],
        "overlay_objlist": {"front": ["obj-chair-1"]},
        "visible_nav_targets_status": "depth_unavailable",
        "answer": "secret-answer",
    }

    payload, inline_paths = mcp_module._shape_strip_visuals(raw_body)

    assert payload["panorama_images"] == [
        {
            "path": str(colors[0]),
            "direction": "front",
            "view": "front",
            "turn_right_deg": 0,
        },
        {
            "path": str(colors[1]),
            "direction": "right",
            "view": "right",
            "turn_right_deg": 90,
        },
        {
            "path": str(colors[2]),
            "direction": "back",
            "view": "back",
            "turn_right_deg": 180,
        },
        {
            "path": str(colors[3]),
            "direction": "left",
            "view": "left",
            "turn_right_deg": 270,
        },
    ]
    assert payload["images"] == [str(path) for path in colors[:4]]
    assert payload["depth_analysis"] == {"front": {"min": 1.0}}
    assert "visible_nav_targets_status" not in payload
    assert "visible_nav_targets" not in payload
    assert "overlay_objlist" not in payload
    assert "overlay_aliases" not in json.dumps(payload, ensure_ascii=False)
    assert "object_position" not in json.dumps(payload, ensure_ascii=False)
    assert "answer" not in json.dumps(payload, ensure_ascii=False)
    assert inline_paths == [str(path) for path in colors[:4]]
    assert str(depth) not in inline_paths


def test_mcp_movement_shaper_uses_overlay_paths_when_enabled(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    clean = tmp_path / "pano_front_color_sensor.png"
    overlay = tmp_path / "pano_front_visible_overlay.png"
    _tiny_png(clean, marker=b"CLEAN")
    _tiny_png(overlay, marker=b"OVLY")
    monkeypatch.setenv("HAB_MCP_VISIBLE_TARGET_OVERLAYS", "1")

    payload, inline_paths = mcp_module._shape_strip_visuals(
        {
            "images": [
                {
                    "image_ref": "pano:s1:1:front",
                    "path": str(clean),
                    "overlay_path": str(overlay),
                    "overlay_aliases": ["obj-chair-1", "obj-table-1"],
                    "direction": "front",
                }
            ],
            "overlay_objlist": {"front": ["obj-chair-1", "obj-table-1"]},
            "visible_nav_targets": _visible_targets(),
            "visible_nav_targets_status": "ok",
        }
    )

    assert payload["panorama_images"] == [
        {
            "image_ref": "pano:s1:1:front",
            "path": str(overlay),
            "overlay_path": str(overlay),
            "overlay_aliases": ["obj-chair-1", "obj-table-1"],
            "direction": "front",
            "view": "front",
            "turn_right_deg": 0,
        }
    ]
    assert payload["overlay_objlist"] == {"front": ["obj-chair-1", "obj-table-1"]}
    assert inline_paths == [str(overlay)]
    assert "visible_nav_targets" not in json.dumps(payload)
    assert str(clean) not in inline_paths


def test_mcp_visible_target_overlay_max_objects_env(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert mcp_module._visible_target_overlay_max_objects() == 4
    monkeypatch.setenv("HAB_MCP_VISIBLE_TARGET_OVERLAY_MAX_OBJECTS", "5")
    assert mcp_module._visible_target_overlay_max_objects() == 5
    monkeypatch.setenv("HAB_MCP_VISIBLE_TARGET_OVERLAY_MAX_OBJECTS", "-2")
    assert mcp_module._visible_target_overlay_max_objects() == 0
    monkeypatch.setenv("HAB_MCP_VISIBLE_TARGET_OVERLAY_MAX_OBJECTS", "invalid")
    assert mcp_module._visible_target_overlay_max_objects() == 4


def test_mcp_turn_prefers_egocentric_color_sensor_over_other_rgb_sensors(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    semantic = tmp_path / "step000001_semantic_sensor.png"
    third_person = tmp_path / "step000001_third_person_color_sensor.png"
    color = tmp_path / "step000001_color_sensor.png"
    _tiny_png(semantic, marker=b"SEMANTIC")
    _tiny_png(third_person, marker=b"THIRD")
    _tiny_png(color, marker=b"EGO")

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        return {
            "visuals": {
                "color_sensor": {"path": str(color)},
                "third_person_color_sensor": {"path": str(third_person)},
                "semantic_sensor": {"path": str(semantic)},
            }
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(mcp_module, "hab_turn", {"direction": "left"})
    images = _image_blocks(content)

    assert _caption_texts(content) == ["Current egocentric view"]
    assert len(images) == 1
    assert images[0].data == base64.b64encode(color.read_bytes()).decode("ascii")


def test_mcp_inline_image_blocks_refuse_paths_outside_artifacts_root(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "step000001_color_sensor.png"
    outside = tmp_path.parent / "outside_color_sensor.png"
    _tiny_png(allowed, marker=b"ALLOWED")
    _tiny_png(outside, marker=b"OUTSIDE")

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        return {
            "panorama_images": [
                {"direction": "front", "path": str(allowed)},
                {"direction": "right", "path": str(outside)},
            ],
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(mcp_module, "hab_turn", {"direction": "left"})
    images = _image_blocks(content)

    assert _caption_texts(content) == ["Front view"]
    assert len(images) == 1
    assert images[0].data == base64.b64encode(allowed.read_bytes()).decode("ascii")
    assert images[0].data != base64.b64encode(outside.read_bytes()).decode("ascii")


def test_mcp_depth_only_observation_keeps_path_text_without_inline_depth_png(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    depth = tmp_path / "depth_only_depth_sensor.png"
    _tiny_png(depth, marker=b"DEPTH")

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        return {
            "visuals": {"depth_sensor": {"path": str(depth)}},
            "depth_analysis": {"front": {"min": 1.0}},
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(mcp_module, "hab_turn", {"direction": "left"})
    payload = _text_payload(content)

    assert payload["images"] == [str(depth)]
    assert payload["depth_analysis"] == {"front": {"min": 1.0}}
    assert _image_blocks(content) == []


def test_mcp_rgb_under_depth_named_parent_still_inlines_color_image(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parent = tmp_path / "depth-debug-run"
    parent.mkdir()
    color = parent / "step000001_color_sensor.png"
    depth = parent / "step000001_depth_sensor.png"
    _tiny_png(color, marker=b"COLOR")
    _tiny_png(depth, marker=b"DEPTH")

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        return {
            "visuals": {
                "color_sensor": {"path": str(color)},
                "depth_sensor": {"path": str(depth)},
            },
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(mcp_module, "hab_turn", {"direction": "left"})
    images = _image_blocks(content)

    assert _caption_texts(content) == ["Current egocentric view"]
    assert len(images) == 1
    assert images[0].data == base64.b64encode(color.read_bytes()).decode("ascii")


def test_mcp_movement_returns_surround_paths_and_inlines_by_default(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    colors: dict[str, Path] = {}
    for direction in ("front", "right", "back", "left"):
        path = tmp_path / f"pano_{direction}_color_sensor.png"
        _tiny_png(path, marker=direction.encode("ascii"))
        colors[direction] = path

    bridge_calls: list[tuple[str, Any]] = []

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        bridge_calls.append((action, payload))
        if action == "get_panorama":
            return {
                "images": [
                    {"path": str(colors[direction]), "direction": direction}
                    for direction in ("front", "right", "back", "left")
                ],
                "visible_nav_targets": _visible_targets(),
                "visible_nav_targets_status": "ok",
            }
        return {
            "visuals": {"color_sensor": {"path": str(tmp_path / "single_color.png")}},
            "metrics": {"step_count": 2},
            "gt_distance_to_goal_m": 0.1,
            "trace_frame_paths": [str(tmp_path / "internal_step.png")],
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    for tool_name, args in (
        ("hab_forward", {"distance_m": 0.5}),
        ("hab_navigate", {"x": 1.0, "y": 0.2, "z": 1.0}),
    ):
        content = _converted_tool_content(mcp_module, tool_name, args)
        payload = _text_payload(content)
        images = _image_blocks(content)
        assert payload["images"] == [
            str(colors["front"]),
            str(colors["right"]),
            str(colors["back"]),
            str(colors["left"]),
        ]
        assert "trace_frame_paths" not in payload
        assert payload["panorama_images"] == [
            {
                "path": str(colors["front"]),
                "direction": "front",
                "view": "front",
                "turn_right_deg": 0,
            },
            {
                "path": str(colors["right"]),
                "direction": "right",
                "view": "right",
                "turn_right_deg": 90,
            },
            {
                "path": str(colors["back"]),
                "direction": "back",
                "view": "back",
                "turn_right_deg": 180,
            },
            {
                "path": str(colors["left"]),
                "direction": "left",
                "view": "left",
                "turn_right_deg": 270,
            },
        ]
        assert "visible_nav_targets_status" not in payload
        assert "visible_nav_targets" not in payload
        assert "object_position" not in json.dumps(payload, ensure_ascii=False)
        assert payload["metrics"] == {"step_count": 2}
        assert "gt_distance_to_goal_m" not in json.dumps(payload)
        assert _caption_texts(content) == [
            "Front view",
            "Right view",
            "Back view",
            "Left view",
        ]
        assert len(images) == 4
        assert [img.data for img in images] == [
            base64.b64encode(colors[direction].read_bytes()).decode("ascii")
            for direction in ("front", "right", "back", "left")
        ]

    navigate_payload = next(
        payload for action, payload in bridge_calls if action == "navigate_step"
    )
    assert navigate_payload["include_visuals"] is True

    content = _converted_tool_content(
        mcp_module,
        "hab_forward",
        {"distance_m": 0.5, "include_images": False},
    )
    images = _image_blocks(content)
    payload = _text_payload(content)
    assert payload["panorama_image_paths"] == [
        str(colors[direction]) for direction in ("front", "right", "back", "left")
    ]
    assert images == []


def test_mcp_init_scene_returns_initial_surround_paths_and_inline_images(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    colors = _make_panorama_images(tmp_path)
    calls: list[tuple[str, Any]] = []

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        calls.append((action, payload))
        if action == "init_scene":
            return {
                "session_id": "s-init",
                "is_gaussian": True,
                "scene": payload["scene"],
            }
        assert action == "get_panorama"
        return {
            "images": [
                {"path": str(colors[direction]), "direction": direction}
                for direction in ("front", "right", "back", "left")
            ],
            "visible_nav_targets": _visible_targets(),
            "visible_nav_targets_status": "ok",
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(
        mcp_module, "hab_init_scene", {"scene": "test_scene"}
    )
    payload = _text_payload(content)
    images = _image_blocks(content)

    assert payload["session_id"] == "s-init"
    assert payload["is_gaussian"] is True
    assert payload["scene_info"]["scene"] == "test_scene"
    assert payload["panorama_images"] == _expected_panorama_rows(colors)
    assert "visible_nav_targets_status" not in payload
    assert "visible_nav_targets" not in payload
    assert "object_position" not in json.dumps(payload, ensure_ascii=False)
    assert payload["panorama_image_paths"] == [
        str(colors[direction]) for direction in ("front", "right", "back", "left")
    ]
    assert payload["images"] == [
        str(colors[direction]) for direction in ("front", "right", "back", "left")
    ]
    assert [action for action, _ in calls] == ["init_scene", "get_panorama"]
    assert len(images) == 4
    assert [img.data for img in images] == [
        base64.b64encode(colors[direction].read_bytes()).decode("ascii")
        for direction in ("front", "right", "back", "left")
    ]


def test_mcp_init_scene_surround_failure_does_not_fail_init(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        if action == "init_scene":
            return {"session_id": "s-init", "is_gaussian": False}
        assert action == "get_panorama"
        raise RuntimeError("camera unavailable")

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(
        mcp_module, "hab_init_scene", {"scene": "test_scene"}
    )
    payload = _text_payload(content)

    assert payload["session_id"] == "s-init"
    assert payload["is_gaussian"] is False
    assert "RuntimeError: camera unavailable" in payload["panorama_error"]
    assert "panorama_images" not in payload
    assert _image_blocks(content) == []


def test_mcp_set_pose_returns_surround_paths_and_inline_images(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    colors = _make_panorama_images(tmp_path)
    calls: list[tuple[str, Any]] = []

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        calls.append((action, payload))
        if action == "set_agent_state":
            return {"agent_state": {"position": payload["position"]}}
        assert action == "get_panorama"
        return {
            "images": [
                {"path": str(colors[direction]), "direction": direction}
                for direction in ("front", "right", "back", "left")
            ]
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(
        mcp_module,
        "hab_set_pose",
        {"session_id": "s1", "x": 1.25, "z": -2.5, "yaw": 0.75},
    )
    payload = _text_payload(content)
    images = _image_blocks(content)

    assert payload["ok"] is True
    assert payload["x"] == 1.25
    assert payload["z"] == -2.5
    assert payload["yaw"] == 0.75
    assert payload["position"] == [1.25, 1.2, -2.5]
    assert payload["panorama_images"] == _expected_panorama_rows(colors)
    assert payload["panorama_image_paths"] == [
        str(colors[direction]) for direction in ("front", "right", "back", "left")
    ]
    assert payload["images"] == [
        str(colors[direction]) for direction in ("front", "right", "back", "left")
    ]
    assert [action for action, _ in calls] == ["set_agent_state", "get_panorama"]
    assert calls[0][1]["snap_to_navmesh"] is True
    assert len(images) == 4
    assert [img.data for img in images] == [
        base64.b64encode(colors[direction].read_bytes()).decode("ascii")
        for direction in ("front", "right", "back", "left")
    ]


def test_mcp_set_pose_surround_failure_does_not_fail_pose(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        if action == "set_agent_state":
            return {"agent_state": {"position": payload["position"]}}
        assert action == "get_panorama"
        raise RuntimeError("camera unavailable")

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(
        mcp_module,
        "hab_set_pose",
        {"session_id": "s1", "x": 1.25, "z": -2.5, "yaw": 0.75},
    )
    payload = _text_payload(content)

    assert payload["ok"] is True
    assert payload["position"] == [1.25, 1.2, -2.5]
    assert "RuntimeError: camera unavailable" in payload["panorama_error"]
    assert "panorama_images" not in payload
    assert _image_blocks(content) == []


def test_mcp_single_view_capture_is_hidden_by_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("NAV_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.delenv("HAB_ENABLE_LEGACY_SEE", raising=False)
    monkeypatch.delenv("HAB_ENABLE_LEGACY_LOOK", raising=False)
    sys.modules.pop("supernav.methods.navigation.mcp_server", None)
    mod = importlib.import_module("supernav.methods.navigation.mcp_server")

    with pytest.raises(Exception):
        _converted_tool_content(mod, "hab_see")
    with pytest.raises(Exception):
        _converted_tool_content(mod, "hab_look")


def _assert_no_eval_only_payload_fields(payload: dict[str, Any]) -> None:
    forbidden_keys = {
        "_debug",
        "gt_position",
        "gt_heading_deg",
        "gt_goal",
        "gt_geodesic_distance",
        "ground_truth",
        "eval_goal_position",
        "target_position",
        "target_positions",
        "reference_path",
        "oracle_pass",
        "oracle_score",
        "answer_key",
        "score",
        "score_table",
        "strict_task_success",
    }
    seen: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in forbidden_keys or str(key).startswith("gt_"):
                    seen.append(str(key))
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(payload)
    assert seen == []


def test_mcp_unshaped_scene_graph_redacts_eval_only_fields_at_dispatch_boundary(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        assert action == "get_scene_graph"
        return {
            "nodes": [
                {
                    "label": "wine bottle",
                    "position": [1.0, 0.2, 3.0],
                    "_debug": {"gt_position": [9.0, 0.2, 9.0]},
                    "ground_truth": {"target_position": [9.0, 0.2, 9.0]},
                    "oracle_score": 1.0,
                }
            ],
            "answer_key": "secret",
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(
        mcp_module,
        "hab_scene_graph",
        {"query_type": "object", "object_label": "wine"},
    )
    payload = _text_payload(content)

    assert payload["nodes"][0]["label"] == "wine bottle"
    assert payload["nodes"][0]["position"] == [1.0, 0.2, 3.0]
    _assert_no_eval_only_payload_fields(payload)


def test_mcp_topdown_redacts_eval_only_fields_after_response_shaping(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    topdown = tmp_path / "topdown_color_sensor.png"
    _tiny_png(topdown, marker=b"TOPDOWN")

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        assert action == "get_topdown_map"
        return {
            "topdown_map": {"path": str(topdown)},
            "metrics": {
                "_debug": {"gt_position": [1.0, 0.2, 3.0]},
                "target_positions": [[1.0, 0.2, 3.0]],
                "score": {"strict_task_success": True},
            },
            "reference_path": "/hidden/ref.json",
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    content = _converted_tool_content(mcp_module, "hab_topdown", {"show_path": True})
    payload = _text_payload(content)

    assert payload["images"] == [str(topdown)]
    assert "topdown_map" not in payload
    _assert_no_eval_only_payload_fields(payload)


def test_direct_mcp_wrapper_returns_json_with_real_image_path(
    mcp_module: Any,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    color = tmp_path / "direct_color_sensor.png"
    _tiny_png(color, marker=b"DIRECT")

    def fake_call(action: str, payload: Any = None) -> dict[str, Any]:
        return {
            "visuals": {"color_sensor": {"path": str(color)}},
            "agent_state": {"heading": 2.0},
        }

    monkeypatch.setattr(mcp_module._bridge, "call", fake_call)

    raw = mcp_module.hab_turn(direction="left")

    assert isinstance(raw, str)
    payload = json.loads(raw)
    assert payload["images"] == [str(color)]
    assert payload["agent_state"]["heading"] == 2.0


def test_front_only_motion_returns_one_image_and_invalidates_old_refs(
    mcp_module, monkeypatch, tmp_path
):
    monkeypatch.setenv("HAB_MCP_OBSERVATION_MODE", "front")
    colors = _make_panorama_images(tmp_path)
    calls = []

    def capture(action, payload):
        calls.append((action, payload))
        return {
            "images": [
                {
                    "direction": "front",
                    "path": str(colors["front"]),
                    "image_ref": "fresh",
                }
            ]
        }

    monkeypatch.setattr(mcp_module._bridge, "call", capture)
    mcp_module._cache_mcp_panorama_images([{"direction": "right", "image_ref": "old"}])
    body = mcp_module._attach_mcp_surround_view(
        "turn", {"images": [str(colors["back"])], "overlay_image": "old.png"}
    )
    assert calls[0][1]["front_only"] is True
    assert len(body["panorama_images"]) == 1
    assert "overlay_image" not in body
    assert mcp_module._resolve_latest_view_image_ref("right") == ""
    assert (
        mcp_module._resolve_visual_view_args(
            "visual_point_navigate", {"point": [0.5, 0.5]}
        )["image_ref"]
        == "fresh"
    )
    for args in ({"view": "right"}, {"image_ref": "old"}):
        assert "_view_resolution_error" in mcp_module._resolve_visual_view_args(
            "visual_point_navigate", args
        )
    shaped, paths = mcp_module._shape_strip_visuals(body)
    assert paths == [str(colors["front"])]


def test_front_capture_failure_does_not_reuse_previous_image(mcp_module, monkeypatch):
    monkeypatch.setenv("HAB_MCP_OBSERVATION_MODE", "front")
    mcp_module._cache_mcp_panorama_images([{"direction": "front", "image_ref": "old"}])
    monkeypatch.setattr(
        mcp_module,
        "_capture_mcp_surround_view",
        lambda: {"panorama_error": "capture failed"},
    )
    body = mcp_module._attach_mcp_surround_view(
        "visual_point_navigate", {"images": ["old.png"]}
    )
    assert body["panorama_error"] == "capture failed"
    assert mcp_module._resolve_latest_view_image_ref("front") == ""
    assert mcp_module._shape_strip_visuals(body)[1] == []


def test_front_tools_hide_multi_view_arguments_and_panorama(mcp_module, monkeypatch):
    from supernav.methods.navigation.tools.base import ToolRegistry

    monkeypatch.setenv("HAB_MCP_OBSERVATION_MODE", "front")
    monkeypatch.setenv("HAB_MCP_MINIMAL_TASK", "1")
    tools = {tool.metadata.name: tool for tool in ToolRegistry.list_all()}
    params = {
        p.name for p in mcp_module._extract_params(tools["visual_point_navigate"])
    }
    assert "point" in params
    assert not params.intersection({"view", "image_ref"})
    assert mcp_module._extract_params(tools["init_scene"]) == []
    assert {p.name for p in mcp_module._extract_params(tools["close_session"])} == {
        "outcome",
        "reason",
    }
    assert not mcp_module._mcp_tool_allowed_by_whitelist("hab_panorama")
    assert not mcp_module._mcp_tool_allowed_by_whitelist("hab_look_around")


@pytest.mark.parametrize("mode,count", [("front", 1), ("surround", 4)])
def test_turn_wire_response_respects_observation_mode(
    mcp_module, monkeypatch, tmp_path, mode, count
):
    monkeypatch.setenv("HAB_MCP_OBSERVATION_MODE", mode)
    colors = _make_panorama_images(tmp_path)

    def call(action, payload=None):
        if action == "get_panorama":
            directions = ["front"] if payload["front_only"] else list(colors)
            return {
                "images": [
                    {"direction": d, "path": str(colors[d]), "image_ref": f"fresh:{d}"}
                    for d in directions
                ]
            }
        return {
            "metrics": {"step_count": 9},
            "visuals": {"color_sensor": {"path": str(colors["back"])}},
        }

    monkeypatch.setattr(mcp_module._bridge, "call", call)
    content = _converted_tool_content(
        mcp_module, "hab_turn", {"direction": "right", "degrees": 90}
    )
    assert len(_image_blocks(content)) == count
    assert len(_text_payload(content)["panorama_images"]) == count
    assert _image_blocks(content)[0].data == base64.b64encode(
        colors["front"].read_bytes()
    ).decode("ascii")


@pytest.mark.parametrize("mode", ["front", "surround"])
def test_minimal_process_exposes_exact_whitelist_without_artifact_resources(
    tmp_path, mode
):
    import os
    import subprocess

    script = """
import asyncio, json
from supernav.methods.navigation.mcp_server import mcp
async def main():
    tools = await mcp.list_tools()
    resources = await mcp.list_resources()
    templates = await mcp.list_resource_templates()
    print(json.dumps({'tools': [t.name for t in tools], 'resources': len(resources), 'templates': len(templates)}))
asyncio.run(main())
"""
    names = [
        "hab_init_scene",
        "hab_turn",
        "hab_visual_point_navigate",
        "hab_close_session",
    ]
    env = dict(
        os.environ,
        HAB_MCP_MINIMAL_TASK="1",
        HAB_MCP_OBSERVATION_MODE=mode,
        HAB_MCP_TOOL_WHITELIST=",".join(names),
        HAB_ENABLE_LOOK_AROUND="1",
        HAB_ENABLE_LEGACY_SEE="1",
        HAB_ENABLE_ORACLE_LOCAL_NAV="1",
        HAB_VISUAL_POINT_GEO_BASED_EXECUTOR="1",
        HAB_VISUAL_POINT_POINT_ONLY="1",
    )
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[3] / "tools")
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    inventory = json.loads(result.stdout)
    assert set(inventory["tools"]) == set(names)
    assert inventory["resources"] == inventory["templates"] == 0
