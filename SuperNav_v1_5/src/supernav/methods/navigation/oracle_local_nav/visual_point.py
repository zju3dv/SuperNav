"""Image-point grounded local navigation handler."""

from __future__ import annotations

from supernav.paths import workspace_root

import math
import os
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from habitat_contract.navigation import NavigationBackend, NavigationPathfinder, NavigationSession, RaycastScene

import numpy as np

from supernav.methods.navigation.oracle_local_nav.utils import _append_unique_paths, _movement_frame_paths, _positive_float

_MIN_MEANINGFUL_DISPLACEMENT_M = 0.15
_MIN_MEANINGFUL_PLANNED_PATH_M = 0.30
_PREVIEW_TOKEN_TTL_S = 180.0
_PENDING_PREVIEW_ATTR = "pending_visual_point_preview"
_DEFAULT_VISUAL_NAV_HORIZON_M = 40.0
# Geo-based executor mode: single-call execution without the preview/confirm
# round-trip, no 32-step follower cap, and a practically unbounded horizon.
# Enabled via process env or, per request, via the MCP-forwarded payload flag
# (arm-level environment reaches the MCP process, not this bridge process).
_GEO_BASED_EXECUTOR_ENV = "HAB_VISUAL_POINT_GEO_BASED_EXECUTOR"
_GEO_BASED_EXECUTOR_DEFAULT_MAX_STEPS = 200
_GEO_BASED_EXECUTOR_VISUAL_NAV_HORIZON_M = 10000.0


def navigate_visual_point(
    adapter: NavigationBackend,
    session_id: str | None,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Bridge action handler for point/bbox-grounded local navigation."""

    session = adapter.navigation_session(session_id)
    pathfinder = adapter.navigation_pathfinder(session)
    confirm_token = str(payload.get("confirm_token") or "").strip()
    geo_based_executor = _geo_based_executor_mode_enabled(payload)

    image_ref = str(payload.get("image_ref") or "").strip()
    if not image_ref:
        return _failure(
            "invalid_image_ref", "visual point navigation requires image_ref"
        )
    if _point_only_mode_enabled() and payload.get("bbox") is not None:
        return _failure(
            "bbox_not_allowed",
            "visual_point point-only mode rejects bbox; use point=[x,y]",
            image_ref=image_ref,
        )
    if _point_only_mode_enabled() and payload.get("point") is None:
        return _failure(
            "point_required",
            "visual_point point-only mode requires point=[x,y]",
            image_ref=image_ref,
        )

    registry = getattr(session, "last_visual_image_refs", {}) or {}
    image_meta = registry.get(image_ref) if isinstance(registry, Mapping) else None
    if not isinstance(image_meta, Mapping):
        return _failure("invalid_image_ref", "image_ref is not valid for this session")
    latest_seq = getattr(session, "latest_visual_capture_seq", None)
    if latest_seq is not None and image_meta.get("capture_seq") != latest_seq:
        if confirm_token:
            _clear_pending_preview(session)
            return _failure(
                "stale_confirm_token",
                "a newer visual capture invalidated the preview",
                image_ref=image_ref,
            )
        return _failure(
            "stale_image_ref", "image_ref is not from the latest visual capture"
        )

    if confirm_token and not geo_based_executor:
        invalid = _validate_pending_preview(
            adapter=adapter,
            session=session,
            payload=payload,
            image_ref=image_ref,
            image_meta=image_meta,
            confirm_token=confirm_token,
        )
        if invalid is not None:
            _clear_pending_preview(session)
            return invalid
        _clear_pending_preview(session)

    parsed = _parse_visual_anchor(payload, image_meta)
    if parsed.get("error"):
        return _failure(
            str(parsed["status"]), str(parsed["error"]), image_ref=image_ref
        )

    current_position = np.asarray(
        adapter.agent_position(session),
        dtype=np.float32,
    )
    goal_radius = _positive_float(payload.get("goal_radius"), default=0.3)
    original_annotation = _original_annotation(
        payload=payload,
        image_ref=image_ref,
        image_meta=image_meta,
        anchor=parsed,
    )
    candidates = _visual_nav_candidates(
        image_meta=image_meta,
        anchor=parsed,
        current_position=current_position,
        pathfinder=pathfinder,
        horizon_m=_positive_float(
            payload.get("horizon_m"),
            default=(
                _GEO_BASED_EXECUTOR_VISUAL_NAV_HORIZON_M
                if geo_based_executor
                else _DEFAULT_VISUAL_NAV_HORIZON_M
            ),
        ),
        standoff_m=_positive_float(payload.get("standoff_m"), default=0.7),
        goal_radius=goal_radius,
    )
    overlay = _write_visual_point_overlay(
        output_dir=str(payload.get("output_dir") or str(workspace_root() / "data" / "runs" / "artifacts")),
        image_meta=image_meta,
        anchor=parsed,
        candidates=candidates,
    )
    live_overlay = getattr(adapter, "publish_live_overlay", None)
    if live_overlay is not None:
        live_overlay(session, overlay, tool="hab_visual_point_navigate", image_ref=image_ref,
                     capture_seq=image_meta.get("capture_seq"), direction=image_meta.get("direction"),
                     point=payload.get("point"), anchor_px=parsed.get("anchor_px"))
    sample_count = len(
        _anchor_sample_pixels(
            parsed,
            width=int(image_meta.get("width") or 0),
            height=int(image_meta.get("height") or 0),
        )
    )
    viable = [item for item in candidates if item.get("reachable")]
    if not viable:
        statuses = {str(item.get("status") or "") for item in candidates}
        status = "no_reachable_point_near_anchor"
        if "no_depth_near_anchor" in statuses and len(statuses) == 1:
            status = "no_depth_near_anchor"
        elif "path_exceeds_horizon" in statuses:
            status = "path_exceeds_horizon"
        elif "snap_too_far" in statuses:
            status = "snap_too_far"
        return _failure(
            status,
            "no reachable navmesh point was found near the visual anchor",
            image_ref=image_ref,
            overlay_image=overlay,
            original_annotation=original_annotation,
        )

    best = min(
        viable,
        key=lambda item: (
            float(item.get("score") or 0.0),
            float(item.get("geodesic_distance") or 0.0),
        ),
    )
    planned_path_m = _optional_float(best.get("geodesic_distance"))
    refined_annotation = _refined_annotation(
        image_ref=image_ref,
        image_meta=image_meta,
        anchor=parsed,
        best=best,
        overlay=overlay,
        planned_path_m=planned_path_m,
        displacement_m=None,
        sample_count=sample_count,
    )
    selected_anchor = _selected_anchor(
        payload=payload,
        image_ref=image_ref,
        image_meta=image_meta,
        parsed=parsed,
    )
    if not confirm_token and not geo_based_executor:
        token = secrets.token_urlsafe(18)
        setattr(
            session,
            _PENDING_PREVIEW_ATTR,
            {
                "token": token,
                "created_at": time.monotonic(),
                "image_ref": image_ref,
                "capture_seq": image_meta.get("capture_seq"),
                "point": _anchor_arg_signature(payload.get("point")),
                "bbox": _anchor_arg_signature(payload.get("bbox")),
                "agent_position": current_position.tolist(),
                "agent_forward": _agent_forward_signature(adapter, session),
            },
        )
        result = {
            "ok": True,
            "backend": "visual_point_navmesh",
            "status": "preview_ready",
            "nav_status": "preview_ready",
            "confirm_token": token,
            "selected_anchor": selected_anchor,
            "selected_anchor_px": parsed.get("anchor_px"),
            "overlay_image": overlay,
            "reachable": True,
            "distance_to_visual_target": best.get("target_distance_m"),
            "planned_path_m": planned_path_m,
            "pre_move_self_check": {
                "required": True,
                "instruction": (
                    "Inspect overlay_image before confirming. If the marker is "
                    "not on the intended reachable target/floor, discard this "
                    "preview and call hab_visual_point_navigate again with a "
                    "better point. To move, repeat the same image_ref and point "
                    "with confirm_token."
                ),
            },
            "original_annotation": original_annotation,
            "refined_annotation": refined_annotation,
            "refine_diagnostics": _refine_diagnostics(
                candidates=candidates,
                best=best,
                sample_count=sample_count,
                planned_path_m=planned_path_m,
                displacement_m=None,
            ),
            "_debug": {
                "selected_goal": best.get("snapped"),
                "selected_hint": best.get("hint"),
                "candidate_count": len(candidates),
                "selected_candidate": best,
            },
        }
        return {key: value for key, value in result.items() if value is not None}

    max_steps = max(
        1,
        int(
            payload.get(
                "max_steps", _GEO_BASED_EXECUTOR_DEFAULT_MAX_STEPS if geo_based_executor else 20
            )
        ),
    )
    nav = adapter.navigate_step(
        session.session_id,
        {
            "goal": np.asarray(best["snapped"], dtype=np.float32).tolist(),
            "goal_radius": goal_radius,
            "max_steps": max_steps if geo_based_executor else min(max_steps, 32),
            "include_metrics": True,
            "include_visuals": True,
            "include_publish_hints": True,
            "output_dir": str(payload.get("output_dir") or str(workspace_root() / "data" / "runs" / "artifacts")),
        },
    )
    body = dict(nav) if isinstance(nav, Mapping) else {}
    steps_executed = int(body.get("steps_executed") or 0)
    planned_path_m = _optional_float(best.get("geodesic_distance"))
    final_position = _extract_agent_position(body)
    if final_position is None:
        try:
            final_position = np.asarray(
                adapter.agent_position(session),
                dtype=np.float32,
            )
        except Exception:
            final_position = None
    displacement_m = (
        round(_horizontal_distance(current_position, final_position), 4)
        if final_position is not None
        else None
    )
    refined_annotation["displacement_m"] = displacement_m
    status = str(body.get("nav_status") or "en_route_visual_point")
    if status == "reached":
        status = "reached_visual_point"
    elif status in {"blocked", "unreachable", "error"}:
        status = "navigation_blocked"
    reason: str | None = None
    if steps_executed <= 0:
        status = "self_check_required"
        reason = "near_zero_motion"
    elif displacement_m is not None and displacement_m < _MIN_MEANINGFUL_DISPLACEMENT_M:
        status = "self_check_required"
        reason = "near_zero_motion"
    elif planned_path_m is not None and planned_path_m < _MIN_MEANINGFUL_PLANNED_PATH_M:
        status = "self_check_required"
        reason = "target_too_close"
    message = _self_check_message(reason) if reason else None
    movement_frames: list[str] = []
    _append_unique_paths(movement_frames, _movement_frame_paths(body))
    result = {
        "ok": True,
        "backend": "visual_point_navmesh",
        "status": status,
        "nav_status": status,
        "image_ref": image_ref,
        "direction": image_meta.get("direction"),
        "selected_anchor": selected_anchor,
        "selected_anchor_px": parsed.get("anchor_px"),
        "overlay_image": overlay,
        "steps_executed": steps_executed,
        "reason": reason,
        "message": message,
        "reachable": True,
        "distance_to_visual_target": best.get("target_distance_m"),
        "planned_path_m": planned_path_m,
        "displacement_m": displacement_m,
        "original_annotation": original_annotation,
        "refined_annotation": refined_annotation,
        "refine_diagnostics": _refine_diagnostics(
            candidates=candidates,
            best=best,
            sample_count=sample_count,
            planned_path_m=planned_path_m,
            displacement_m=displacement_m,
        ),
        "metrics": body.get("metrics"),
        "state_summary": body.get("state_summary"),
        "visuals": body.get("visuals", {}),
        "publish_hints": body.get("publish_hints"),
        "movement_frames": movement_frames,
        "movement_frame_count": len(movement_frames),
        "_debug": {
            "selected_goal": best.get("snapped"),
            "selected_hint": best.get("hint"),
            "candidate_count": len(candidates),
            "selected_candidate": best,
        },
    }
    return {key: value for key, value in result.items() if value is not None}


def _failure(
    status: str,
    error: str,
    *,
    image_ref: str | None = None,
    overlay_image: str | None = None,
    original_annotation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ok": False,
        "backend": "visual_point_navmesh",
        "status": status,
        "error": error,
    }
    if image_ref:
        body["image_ref"] = image_ref
    if overlay_image:
        body["overlay_image"] = overlay_image
    if original_annotation:
        body["original_annotation"] = original_annotation
    return body


def _clear_pending_preview(session: NavigationSession) -> None:
    try:
        setattr(session, _PENDING_PREVIEW_ATTR, None)
    except Exception:
        pass


def _validate_pending_preview(
    *,
    adapter: NavigationBackend,
    session: NavigationSession,
    payload: Mapping[str, Any],
    image_ref: str,
    image_meta: Mapping[str, Any],
    confirm_token: str,
) -> dict[str, Any] | None:
    pending = getattr(session, _PENDING_PREVIEW_ATTR, None)
    if not isinstance(pending, Mapping):
        return _failure(
            "invalid_confirm_token",
            "confirm_token does not match an active visual point preview",
            image_ref=image_ref,
        )
    if str(pending.get("token") or "") != confirm_token:
        return _failure(
            "invalid_confirm_token",
            "confirm_token does not match the active visual point preview",
            image_ref=image_ref,
        )
    created_at = pending.get("created_at")
    if not isinstance(created_at, (int, float)):
        return _failure(
            "invalid_confirm_token",
            "visual point preview token is malformed",
            image_ref=image_ref,
        )
    if time.monotonic() - float(created_at) > _PREVIEW_TOKEN_TTL_S:
        return _failure(
            "expired_confirm_token",
            "visual point preview expired; request a fresh preview",
            image_ref=image_ref,
        )
    if str(pending.get("image_ref") or "") != image_ref:
        return _failure(
            "confirm_mismatch",
            "confirm_token must be used with the same image_ref as the preview",
            image_ref=image_ref,
        )
    if pending.get("capture_seq") != image_meta.get("capture_seq"):
        return _failure(
            "stale_confirm_token",
            "visual capture changed after preview; request a fresh preview",
            image_ref=image_ref,
        )
    latest_seq = getattr(session, "latest_visual_capture_seq", None)
    if latest_seq is not None and pending.get("capture_seq") != latest_seq:
        return _failure(
            "stale_confirm_token",
            "a newer visual capture invalidated the preview",
            image_ref=image_ref,
        )
    if pending.get("point") != _anchor_arg_signature(payload.get("point")):
        return _failure(
            "confirm_mismatch",
            "confirm_token must be used with the same point as the preview",
            image_ref=image_ref,
        )
    if pending.get("bbox") != _anchor_arg_signature(payload.get("bbox")):
        return _failure(
            "confirm_mismatch",
            "confirm_token must be used with the same bbox as the preview",
            image_ref=image_ref,
        )
    current_position = np.asarray(
        adapter.agent_position(session),
        dtype=np.float32,
    )
    old_position = pending.get("agent_position")
    if _vector_changed(old_position, current_position.tolist(), tolerance=1e-3):
        return _failure(
            "stale_confirm_token",
            "agent movement invalidated the visual point preview",
            image_ref=image_ref,
        )
    old_forward = pending.get("agent_forward")
    new_forward = _agent_forward_signature(adapter, session)
    if old_forward is not None and _vector_changed(
        old_forward, new_forward, tolerance=1e-3
    ):
        return _failure(
            "stale_confirm_token",
            "agent heading changed after preview; request a fresh preview",
            image_ref=image_ref,
        )
    return None


def _selected_anchor(
    *,
    payload: Mapping[str, Any],
    image_ref: str,
    image_meta: Mapping[str, Any],
    parsed: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "image_ref": image_ref,
        "direction": image_meta.get("direction"),
        "point": payload.get("point"),
        "bbox": payload.get("bbox"),
        "anchor_px": parsed.get("anchor_px"),
        "point_px": parsed.get("point_px"),
        "bbox_px": parsed.get("bbox_px"),
        "capture_seq": image_meta.get("capture_seq"),
    }
    return {key: value for key, value in body.items() if value is not None}


def _anchor_arg_signature(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {
            str(key): _anchor_arg_signature(value[key])
            for key in sorted(value.keys(), key=str)
        }
    if isinstance(value, (list, tuple)):
        return [_anchor_arg_signature(item) for item in value]
    try:
        return round(float(value), 6)
    except (TypeError, ValueError):
        return value


def _agent_forward_signature(adapter: NavigationBackend, session: NavigationSession) -> list[float] | None:
    forward_fn = getattr(adapter, "agent_forward", None)
    if not callable(forward_fn):
        return None
    try:
        forward = np.asarray(forward_fn(session), dtype=np.float32).tolist()
    except Exception:
        return None
    return [round(float(value), 6) for value in forward[:3]]


def _vector_changed(
    old: Any,
    new: Any,
    *,
    tolerance: float,
) -> bool:
    if old is None or new is None:
        return False
    try:
        left = np.asarray(old, dtype=np.float32)
        right = np.asarray(new, dtype=np.float32)
    except Exception:
        return True
    if left.shape != right.shape:
        return True
    return bool(np.linalg.norm(left - right) > tolerance)


def _point_only_mode_enabled() -> bool:
    return str(os.environ.get("HAB_VISUAL_POINT_POINT_ONLY", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _truthy_flag(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _geo_based_executor_mode_enabled(payload: Mapping[str, Any] | None = None) -> bool:
    if _truthy_flag(os.environ.get(_GEO_BASED_EXECUTOR_ENV, "")):
        return True
    return payload is not None and _truthy_flag(payload.get("geo_based_executor"))


def _original_annotation(
    *,
    payload: Mapping[str, Any],
    image_ref: str,
    image_meta: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "image_ref": image_ref,
        "direction": image_meta.get("direction"),
        "anchor_px": anchor.get("anchor_px"),
        "anchor": anchor.get("anchor"),
        "intent": anchor.get("intent"),
    }
    point = payload.get("point")
    if point is not None:
        body["point"] = point
    bbox = payload.get("bbox")
    if bbox is not None:
        body["bbox"] = bbox
    if anchor.get("point_px") is not None:
        body["point_px"] = anchor.get("point_px")
    if anchor.get("bbox_px") is not None:
        body["bbox_px"] = anchor.get("bbox_px")
    return {key: value for key, value in body.items() if value is not None}


def _refined_annotation(
    *,
    image_ref: str,
    image_meta: Mapping[str, Any],
    anchor: Mapping[str, Any],
    best: Mapping[str, Any],
    overlay: str | None,
    planned_path_m: float | None,
    displacement_m: float | None,
    sample_count: int,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "image_ref": image_ref,
        "direction": image_meta.get("direction"),
        "selected_anchor_px": anchor.get("anchor_px"),
        "candidate_status": best.get("status"),
        "overlay_image": overlay,
        "depth_sample_count": sample_count,
        "selected_sample_px": best.get("sample_px"),
        "selected_depth_m": best.get("depth_m"),
        "snap_distance_m": best.get("snap_distance"),
        "target_distance_m": best.get("target_distance_m"),
        "planned_path_m": planned_path_m,
        "displacement_m": displacement_m,
    }
    return {key: value for key, value in body.items() if value is not None}


def _refine_diagnostics(
    *,
    candidates: list[dict[str, Any]],
    best: Mapping[str, Any],
    sample_count: int,
    planned_path_m: float | None,
    displacement_m: float | None,
) -> dict[str, Any]:
    rejected: dict[str, int] = {}
    for candidate in candidates:
        status = str(candidate.get("status") or "unknown")
        if candidate.get("reachable"):
            continue
        rejected[status] = rejected.get(status, 0) + 1
    body: dict[str, Any] = {
        "method": "depth_first",
        "candidate_count": len(candidates),
        "reachable_candidate_count": sum(
            1 for item in candidates if item.get("reachable")
        ),
        "depth_sample_count": sample_count,
        "selected_sample_px": best.get("sample_px"),
        "selected_depth_m": best.get("depth_m"),
        "selected_candidate_status": best.get("status"),
        "snap_distance_m": best.get("snap_distance"),
        "target_distance_m": best.get("target_distance_m"),
        "planned_path_m": planned_path_m,
        "displacement_m": displacement_m,
        "rejected_status_counts": rejected,
    }
    return {key: value for key, value in body.items() if value is not None}


def _self_check_message(reason: str | None) -> str | None:
    if reason == "target_too_close":
        return (
            "The selected visual target snapped to a reachable point that is already "
            "too close. Inspect the overlay and choose a farther point on reachable "
            "floor or on the object's lower/front area."
        )
    if reason == "near_zero_motion":
        return (
            "The selected visual target was valid visually, but the navigation hop "
            "produced near-zero movement. Inspect the overlay and choose a farther "
            "point on reachable floor or on the object's lower/front area."
        )
    return None


def _parse_visual_anchor(
    payload: Mapping[str, Any],
    image_meta: Mapping[str, Any],
) -> dict[str, Any]:
    width = int(image_meta.get("width") or 0)
    height = int(image_meta.get("height") or 0)
    if width <= 0 or height <= 0:
        return {"status": "invalid_image_ref", "error": "image_ref has no image size"}
    try:
        point = _normalized_point(payload.get("point"), width=width, height=height)
        bbox = _normalized_bbox(payload.get("bbox"), width=width, height=height)
    except (TypeError, ValueError):
        error = (
            "point must use numeric normalized coordinates"
            if _point_only_mode_enabled()
            else "point/bbox must use numeric normalized coordinates"
        )
        return {
            "status": "missing_visual_anchor",
            "error": error,
        }
    if point is None and bbox is None:
        error = (
            "visual point navigation requires point"
            if _point_only_mode_enabled()
            else "visual point navigation requires point and/or bbox"
        )
        return {
            "status": "missing_visual_anchor",
            "error": error,
        }
    anchor = str(payload.get("anchor") or "auto").strip().lower()
    if anchor not in {"auto", "center", "bottom_center", "lower_band"}:
        anchor = "auto"
    intent = str(payload.get("intent") or "approach").strip().lower()
    if intent not in {"approach", "pass_through", "inspect"}:
        intent = "approach"
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        if anchor == "center":
            ax, ay = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        else:
            ax, ay = (x0 + x1) / 2.0, y0 + 0.88 * (y1 - y0)
    else:
        ax, ay = point or (width / 2.0, height / 2.0)
    search_radius = _positive_float(payload.get("search_radius"), default=0.10)
    return {
        "point_px": list(point) if point is not None else None,
        "bbox_px": list(bbox) if bbox is not None else None,
        "anchor": anchor,
        "intent": intent,
        "search_radius": max(0.01, min(0.30, search_radius)),
        "anchor_px": [int(round(ax)), int(round(ay))],
    }


def _optional_float(value: Any) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _extract_agent_position(body: Mapping[str, Any]) -> np.ndarray | None:
    candidates: list[Mapping[str, Any]] = []
    metrics = body.get("metrics")
    if isinstance(metrics, Mapping):
        agent_state = metrics.get("agent_state")
        if isinstance(agent_state, Mapping):
            candidates.append(agent_state)
        state_summary = metrics.get("state_summary")
        if isinstance(state_summary, Mapping):
            candidates.append(state_summary)
    agent_state = body.get("agent_state")
    if isinstance(agent_state, Mapping):
        candidates.append(agent_state)
    state_summary = body.get("state_summary")
    if isinstance(state_summary, Mapping):
        candidates.append(state_summary)
    for candidate in candidates:
        position = candidate.get("position")
        if not isinstance(position, (list, tuple)) or len(position) < 3:
            continue
        try:
            return np.asarray([float(v) for v in position[:3]], dtype=np.float32)
        except (TypeError, ValueError):
            continue
    return None


def _normalized_point(
    value: Any,
    *,
    width: int,
    height: int,
) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    x = max(0.0, min(1.0, float(value[0])))
    y = max(0.0, min(1.0, float(value[1])))
    return x * (width - 1), y * (height - 1)


def _normalized_bbox(
    value: Any,
    *,
    width: int,
    height: int,
) -> tuple[float, float, float, float] | None:
    if isinstance(value, Mapping):
        raw = (value.get("x"), value.get("y"), value.get("width"), value.get("height"))
    elif isinstance(value, (list, tuple)) and len(value) >= 4:
        raw = (value[0], value[1], value[2], value[3])
    else:
        return None
    x = max(0.0, min(1.0, float(raw[0])))
    y = max(0.0, min(1.0, float(raw[1])))
    w = max(0.0, min(1.0, float(raw[2])))
    h = max(0.0, min(1.0, float(raw[3])))
    x0 = x * (width - 1)
    y0 = y * (height - 1)
    x1 = min(width - 1.0, (x + w) * (width - 1))
    y1 = min(height - 1.0, (y + h) * (height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def _visual_nav_candidates(
    *,
    image_meta: Mapping[str, Any],
    anchor: Mapping[str, Any],
    current_position: np.ndarray,
    pathfinder: NavigationPathfinder,
    horizon_m: float,
    standoff_m: float,
    goal_radius: float = 0.3,
    simulator: RaycastScene | None = None,
    enforce_approach_side: bool = False,
    max_snap_distance_m: float = 0.75,
    require_target_visibility: bool = False,
) -> list[dict[str, Any]]:
    depth = image_meta.get("depth")
    if not isinstance(depth, np.ndarray) or depth.ndim != 2:
        return [{"status": "no_depth_near_anchor", "reachable": False}]
    hint_records = _world_hint_records_from_anchor(image_meta=image_meta, anchor=anchor)
    if enforce_approach_side:
        hint_records = _foreground_hint_records(hint_records)
    if not hint_records:
        return [{"status": "no_depth_near_anchor", "reachable": False}]
    candidates: list[dict[str, Any]] = []
    for record in hint_records:
        hint = np.asarray(record["hint"], dtype=np.float32)
        for point, point_hint in _candidate_points_for_hint(
            hint=hint,
            current_position=current_position,
            standoff_m=standoff_m,
            approach_only=enforce_approach_side,
        ):
            candidate = _validate_candidate(
                point=point,
                hint=point_hint,
                current_position=current_position,
                pathfinder=pathfinder,
                horizon_m=horizon_m,
                standoff_m=standoff_m,
                goal_radius=goal_radius,
                simulator=simulator,
                enforce_approach_side=enforce_approach_side,
                max_snap_distance_m=max_snap_distance_m,
                require_target_visibility=require_target_visibility,
                visibility_eye_height_m=max(
                    0.2,
                    float(
                        np.asarray(
                            image_meta.get("camera_position", current_position),
                            dtype=np.float32,
                        )[1]
                        - current_position[1]
                    ),
                ),
            )
            candidate["sample_px"] = record.get("sample_px")
            candidate["depth_m"] = record.get("depth_m")
            candidates.append(candidate)
    return candidates


def _world_hint_records_from_anchor(
    *,
    image_meta: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> list[dict[str, Any]]:
    width = int(image_meta.get("width") or 0)
    height = int(image_meta.get("height") or 0)
    depth = image_meta.get("depth")
    if not isinstance(depth, np.ndarray):
        return []
    samples = _anchor_sample_pixels(anchor, width=width, height=height)
    records: list[dict[str, Any]] = []
    for px, py in samples:
        depth_m = _median_depth_near(
            depth,
            px,
            py,
            image_width=width,
            image_height=height,
        )
        if depth_m is None:
            continue
        records.append(
            {
                "hint": _unproject_pixel(image_meta, px, py, depth_m),
                "sample_px": [int(round(px)), int(round(py))],
                "depth_m": round(float(depth_m), 4),
            }
        )
    return records


def _world_hints_from_anchor(
    *,
    image_meta: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> list[np.ndarray]:
    return [
        np.asarray(record["hint"], dtype=np.float32)
        for record in _world_hint_records_from_anchor(
            image_meta=image_meta,
            anchor=anchor,
        )
    ]


def _foreground_hint_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the nearest coherent depth cluster instead of bbox background."""
    depths = [
        float(row["depth_m"]) for row in records if row.get("depth_m") is not None
    ]
    if not depths:
        return records
    nearest = min(depths)
    tolerance = max(0.25, 0.20 * nearest)
    return [
        row
        for row in records
        if row.get("depth_m") is not None
        and float(row["depth_m"]) <= nearest + tolerance
    ]


def _anchor_sample_pixels(
    anchor: Mapping[str, Any],
    *,
    width: int,
    height: int,
) -> list[tuple[float, float]]:
    bbox = anchor.get("bbox_px")
    if isinstance(bbox, list) and len(bbox) == 4:
        x0, y0, x1, y1 = [float(v) for v in bbox]
        lower_y = y0 + 0.88 * (y1 - y0)
        xs = [x0 + frac * (x1 - x0) for frac in (0.25, 0.5, 0.75)]
        return [
            (
                max(0.0, min(width - 1.0, x)),
                max(0.0, min(height - 1.0, lower_y)),
            )
            for x in xs
        ]
    ax, ay = anchor.get("anchor_px") or [width / 2.0, height / 2.0]
    radius = float(anchor.get("search_radius") or 0.10) * min(width, height)
    offsets = [(0.0, 0.0), (-0.5, 0.0), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5)]
    return [
        (
            max(0.0, min(width - 1.0, float(ax) + ox * radius)),
            max(0.0, min(height - 1.0, float(ay) + oy * radius)),
        )
        for ox, oy in offsets
    ]


def _median_depth_near(
    depth: np.ndarray,
    px: float,
    py: float,
    *,
    image_width: int,
    image_height: int,
) -> float | None:
    dh, dw = depth.shape[:2]
    dx = int(round(float(px) * (dw - 1) / max(1, image_width - 1)))
    dy = int(round(float(py) * (dh - 1) / max(1, image_height - 1)))
    radius = max(1, int(round(min(dw, dh) * 0.02)))
    x0, x1 = max(0, dx - radius), min(dw, dx + radius + 1)
    y0, y1 = max(0, dy - radius), min(dh, dy + radius + 1)
    region = np.asarray(depth[y0:y1, x0:x1], dtype=np.float32)
    valid = region[np.isfinite(region) & (region > 0.05)]
    if valid.size == 0:
        return None
    median = float(np.median(valid))
    trimmed = valid[np.abs(valid - median) <= max(0.25, 0.20 * median)]
    if trimmed.size == 0:
        return median
    return float(np.median(trimmed))


def _unproject_pixel(
    image_meta: Mapping[str, Any],
    px: float,
    py: float,
    depth_m: float,
) -> np.ndarray:
    width = int(image_meta.get("width") or 1)
    height = int(image_meta.get("height") or 1)
    hfov = float(image_meta.get("hfov") or 90.0)
    fx = width / (2.0 * math.tan(math.radians(hfov) / 2.0))
    fy = fx
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    x_right = (float(px) - cx) / fx * depth_m
    y_up = -(float(py) - cy) / fy * depth_m
    camera_position = np.asarray(image_meta["camera_position"], dtype=np.float32)
    right = np.asarray(image_meta["camera_right"], dtype=np.float32)
    up = np.asarray(image_meta["camera_up"], dtype=np.float32)
    forward = np.asarray(image_meta["camera_forward"], dtype=np.float32)
    return (
        camera_position
        + right * float(x_right)
        + up * float(y_up)
        + forward * float(depth_m)
    ).astype(np.float32)


def _candidate_points_for_hint(
    *,
    hint: np.ndarray,
    current_position: np.ndarray,
    standoff_m: float,
    approach_only: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    points: list[tuple[np.ndarray, np.ndarray]] = []
    floor_hint = np.asarray([hint[0], current_position[1], hint[2]], dtype=np.float32)
    if not approach_only:
        points.append((floor_hint, hint))
    direction = hint - current_position
    direction[1] = 0.0
    norm = float(np.linalg.norm(direction))
    if norm > 1e-6:
        unit = direction / norm
        points.append((floor_hint - unit * standoff_m, hint))
        if approach_only:
            lateral = np.asarray([-unit[2], 0.0, unit[0]], dtype=np.float32)
            for radius in (max(standoff_m, 0.9), max(standoff_m, 1.2)):
                base = floor_hint - unit * radius
                for offset in (-0.6, -0.3, 0.3, 0.6):
                    points.append((base + lateral * offset, hint))
            return points
    for radius in (0.3, 0.6, 0.9, 1.2):
        for idx in range(12):
            theta = 2.0 * math.pi * (idx / 12.0)
            raw = floor_hint + np.asarray(
                [math.cos(theta) * radius, 0.0, math.sin(theta) * radius],
                dtype=np.float32,
            )
            points.append((raw.astype(np.float32), hint))
    return points


def _validate_candidate(
    *,
    point: np.ndarray,
    hint: np.ndarray,
    current_position: np.ndarray,
    pathfinder: NavigationPathfinder,
    horizon_m: float,
    standoff_m: float,
    goal_radius: float = 0.3,
    simulator: RaycastScene | None = None,
    enforce_approach_side: bool = False,
    max_snap_distance_m: float = 0.75,
    require_target_visibility: bool = False,
    visibility_eye_height_m: float = 1.5,
    approach_plane_origin: np.ndarray | None = None,
    approach_plane_normal: np.ndarray | None = None,
    lateral_offset_m: float = 0.0,
) -> dict[str, Any]:
    try:
        snapped = np.asarray(pathfinder.snap_point(point), dtype=np.float32)
    except Exception:
        return {"status": "snap_failed", "reachable": False}
    if np.any(np.isnan(snapped)):
        return {"status": "snap_failed", "reachable": False}
    snap_dist = _horizontal_distance(point, snapped)
    if snap_dist > max_snap_distance_m:
        return {
            "status": "snap_too_far",
            "reachable": False,
            "snap_distance": snap_dist,
        }
    if abs(float(snapped[1] - current_position[1])) > 1.25:
        return {
            "status": "snap_too_far",
            "reachable": False,
            "snap_distance": snap_dist,
        }
    side_margin = _approach_plane_margin(
        point=snapped,
        hint=hint,
        current_position=current_position,
        plane_origin=approach_plane_origin,
        plane_normal=approach_plane_normal,
    )
    if enforce_approach_side and (side_margin is None or side_margin < 0.05):
        return {
            "status": "candidate_beyond_target",
            "reachable": False,
            "snap_distance": round(float(snap_dist), 4),
            "target_side_margin_m": (
                round(float(side_margin), 4) if side_margin is not None else None
            ),
        }
    try:
        path = pathfinder.shortest_path(
            np.asarray(current_position, dtype=np.float32), snapped
        )
    except Exception:
        path = None
    if path is None:
        return {"status": "no_reachable_point_near_anchor", "reachable": False}
    geodesic = float(path.geodesic_distance)
    if geodesic > horizon_m:
        return {
            "status": "path_exceeds_horizon",
            "reachable": False,
            "geodesic_distance": geodesic,
        }
    if enforce_approach_side:
        for path_point in list(getattr(path, "points", []) or []):
            path_margin = _approach_plane_margin(
                point=np.asarray(path_point, dtype=np.float32),
                hint=hint,
                current_position=current_position,
                plane_origin=approach_plane_origin,
                plane_normal=approach_plane_normal,
            )
            if path_margin is None or path_margin < -0.05:
                return {
                    "status": "path_crosses_target_plane",
                    "reachable": False,
                    "snap_distance": round(float(snap_dist), 4),
                    "geodesic_distance": round(float(geodesic), 4),
                    "target_side_margin_m": round(float(path_margin or 0.0), 4),
                }
    if require_target_visibility and not _candidate_can_see_hint(
        simulator=simulator,
        snapped=snapped,
        hint=hint,
        eye_height_m=visibility_eye_height_m,
    ):
        return {
            "status": "target_not_visible_from_candidate",
            "reachable": False,
            "snap_distance": round(float(snap_dist), 4),
            "geodesic_distance": round(float(geodesic), 4),
            "target_side_margin_m": round(float(side_margin or 0.0), 4),
        }
    target_dist = (
        float(side_margin)
        if approach_plane_origin is not None
        and approach_plane_normal is not None
        and side_margin is not None
        else _horizontal_distance(snapped, hint)
    )
    standoff_error = abs(target_dist - standoff_m)
    near_zero_penalty = 1.0 if geodesic < goal_radius + 0.15 else 0.0
    arrival_alignment_deg = _terminal_path_alignment_degrees(
        path_points=list(getattr(path, "points", []) or []),
        desired_forward=(
            -np.asarray(approach_plane_normal, dtype=np.float32)
            if approach_plane_normal is not None
            else None
        ),
    )
    alignment_penalty = (
        float(arrival_alignment_deg) / 90.0 * 0.5
        if arrival_alignment_deg is not None
        else 0.0
    )
    score = (
        2.0 * standoff_error
        + snap_dist
        + 0.35 * abs(float(lateral_offset_m))
        + alignment_penalty
        + near_zero_penalty
    )
    return {
        "status": "ok",
        "reachable": True,
        "snapped": snapped.tolist(),
        "hint": np.asarray(hint, dtype=np.float32).tolist(),
        "score": round(float(score), 4),
        "snap_distance": round(float(snap_dist), 4),
        "geodesic_distance": round(float(geodesic), 4),
        "target_distance_m": round(float(target_dist), 4),
        "standoff_error_m": round(float(standoff_error), 4),
        "normal_standoff_m": (
            round(float(side_margin), 4)
            if approach_plane_origin is not None and side_margin is not None
            else None
        ),
        "lateral_offset_m": round(float(lateral_offset_m), 4),
        "arrival_alignment_deg": arrival_alignment_deg,
        "target_side_margin_m": (
            round(float(side_margin), 4) if side_margin is not None else None
        ),
    }


def _approach_plane_margin(
    *,
    point: np.ndarray,
    hint: np.ndarray,
    current_position: np.ndarray,
    plane_origin: np.ndarray | None,
    plane_normal: np.ndarray | None,
) -> float | None:
    if plane_origin is None or plane_normal is None:
        return _approach_side_margin(
            point=point,
            hint=hint,
            current_position=current_position,
        )
    normal = np.asarray(plane_normal, dtype=np.float32).copy()
    normal[1] = 0.0
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-6:
        return None
    delta = np.asarray(point, dtype=np.float32) - np.asarray(
        plane_origin, dtype=np.float32
    )
    delta[1] = 0.0
    return float(np.dot(delta, normal / norm))


def _terminal_path_alignment_degrees(
    *,
    path_points: list[Any],
    desired_forward: np.ndarray | None,
) -> float | None:
    if desired_forward is None or len(path_points) < 2:
        return None
    terminal = np.asarray(path_points[-1], dtype=np.float32) - np.asarray(
        path_points[-2], dtype=np.float32
    )
    terminal[1] = 0.0
    desired = np.asarray(desired_forward, dtype=np.float32).copy()
    desired[1] = 0.0
    terminal_norm = float(np.linalg.norm(terminal))
    desired_norm = float(np.linalg.norm(desired))
    if terminal_norm <= 1e-6 or desired_norm <= 1e-6:
        return None
    cosine = float(
        np.clip(np.dot(terminal / terminal_norm, desired / desired_norm), -1.0, 1.0)
    )
    return round(float(math.degrees(math.acos(cosine))), 2)


def _approach_side_margin(
    *,
    point: np.ndarray,
    hint: np.ndarray,
    current_position: np.ndarray,
) -> float | None:
    """Positive distance means point is on the agent-facing side of the target."""
    target_direction = np.asarray(hint, dtype=np.float32) - np.asarray(
        current_position, dtype=np.float32
    )
    target_direction[1] = 0.0
    norm = float(np.linalg.norm(target_direction))
    if norm <= 1e-6:
        return None
    unit = target_direction / norm
    target_to_point = np.asarray(hint, dtype=np.float32) - np.asarray(
        point, dtype=np.float32
    )
    target_to_point[1] = 0.0
    return float(np.dot(target_to_point, unit))


def _candidate_can_see_hint(
    *,
    simulator: RaycastScene | None,
    snapped: np.ndarray,
    hint: np.ndarray,
    eye_height_m: float,
) -> bool:
    """Reject a candidate when scene geometry blocks the grounded surface."""
    if simulator is None or not hasattr(simulator, "cast_ray"):
        return True
    origin = np.asarray(snapped, dtype=np.float32).copy()
    origin[1] += max(0.2, float(eye_height_m))
    ray_vector = np.asarray(hint, dtype=np.float32) - origin
    distance = float(np.linalg.norm(ray_vector))
    if distance <= 1e-4:
        return True
    try:
        hits = simulator.cast_ray(
            origin, (ray_vector / distance).astype(np.float32),
            max_distance=distance + 0.25,
        )
    except Exception:
        return True
    if not hits:
        return True
    first_distance = float(hits[0])
    return bool(first_distance + 0.25 >= distance)


def _horizontal_distance(left: np.ndarray, right: np.ndarray) -> float:
    delta = np.asarray(right, dtype=np.float32) - np.asarray(left, dtype=np.float32)
    delta[1] = 0.0
    return float(np.linalg.norm(delta))


def _write_visual_point_overlay(
    *,
    output_dir: str,
    image_meta: Mapping[str, Any],
    anchor: Mapping[str, Any],
    candidates: list[Mapping[str, Any]],
) -> str | None:
    path = image_meta.get("path")
    if not isinstance(path, str) or not path:
        return None
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        return None
    draw = ImageDraw.Draw(image)
    bbox = anchor.get("bbox_px")
    if isinstance(bbox, list) and len(bbox) == 4:
        draw.rectangle(
            tuple(float(v) for v in bbox),
            outline=(255, 230, 0),
            width=3,
        )
    ax, ay = anchor.get("anchor_px") or [image.width / 2, image.height / 2]
    r = 5
    draw.ellipse((ax - r, ay - r, ax + r, ay + r), outline=(0, 255, 255), width=3)
    label = (
        "visual point ok"
        if any(item.get("reachable") for item in candidates)
        else "no reachable point"
    )
    color = (
        (0, 210, 40)
        if any(item.get("reachable") for item in candidates)
        else (255, 80, 80)
    )
    draw.text((4, 3), label, fill=color)
    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            prefix="visual_point_",
            dir=str(out_dir),
            delete=False,
        ) as handle:
            out_path = handle.name
        image.save(out_path)
        return out_path
    except Exception:
        return None
