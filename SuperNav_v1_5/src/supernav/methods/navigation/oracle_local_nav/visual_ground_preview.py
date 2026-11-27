"""LocateAnything-based visual grounding preview.

This module implements a bridge action that detects a target described by a
natural-language phrase and converts each detection into navmesh-validated
candidate goals. A single reachable candidate is executed directly; multiple
candidates return an overlay image with numbered markers so the agent can
confirm one cached snapped navmesh point by id.
"""

from __future__ import annotations

from supernav.paths import workspace_root

import math
import os
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from habitat_contract.navigation import NavigationBackend, NavigationPathfinder, NavigationSession, RaycastScene

import numpy as np

from supernav.methods.navigation.oracle_local_nav.locate_anything_client import _locate_anything_detections
from supernav.methods.navigation.oracle_local_nav.utils import _append_unique_paths, _movement_frame_paths, _positive_float
from supernav.methods.navigation.oracle_local_nav.visual_point import (
    _agent_forward_signature,
    _extract_agent_position,
    _horizontal_distance,
    _median_depth_near,
    _unproject_pixel,
    _validate_candidate,
    _visual_nav_candidates,
    _vector_changed,
)

_MIN_MEANINGFUL_DISPLACEMENT_M = 0.15
_MIN_MEANINGFUL_PLANNED_PATH_M = 0.30
_PREVIEW_TOKEN_TTL_S = 180.0
_PENDING_PREVIEW_ATTR = "pending_visual_ground_preview"
_DEFAULT_LOCATE_ANYTHING_TIMEOUT_S = 30.0
_DEFAULT_LOCATE_ANYTHING_SCORE_THRESHOLD = 0.30
_DEFAULT_LOCATE_ANYTHING_MAX_CANDIDATES = 4
_DEFAULT_LOCATE_ANYTHING_MODE = "box"
_DEFAULT_GOAL_RADIUS_M = 0.3
_DEFAULT_VISUAL_NAV_HORIZON_M = 40.0
_MAX_APPROACH_SNAP_DISTANCE_M = 0.35
_PORTAL_JAMB_MAX_MAD_M = 0.20
_PORTAL_MIN_WIDTH_M = 0.55
_PORTAL_MAX_WIDTH_M = 2.40
_PORTAL_MAX_TANGENT_FORWARD_COS = 0.90
_DEFAULT_VISUAL_OUTPUT_DIR = os.environ.get(
    "NAV_ARTIFACTS_DIR", str(workspace_root() / "data" / "runs" / "artifacts")
)


def _candidate_rank_value(item: Mapping[str, Any]) -> float:
    for key in ("rank_score", "nav_score", "score"):
        value = _optional_float(item.get(key))
        if value is not None:
            return value
    return 1e9


def _has_portal_semantics(phrase: str, detection: Mapping[str, Any]) -> bool:
    label = str(detection.get("label") or "")
    text = f"{phrase} {label}".lower()
    return bool(
        re.search(
            r"\b(door|doorway|opening|entrance|passage|archway|threshold)\b",
            text,
        )
    )


def navigate_visual_ground_preview(
    adapter: NavigationBackend,
    session_id: str | None,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Bridge action handler for visual grounding preview.

    Inputs:
        image_ref: image_ref from the latest panorama_images row.
        phrase: natural-language target description for preview calls.
        confirm_token: token returned by preview_ready; present only to move.
        candidate_id: numbered candidate to execute with confirm_token.
        mode: "box" (default) or "point".
        max_candidates: maximum number of candidates to return (default 4).
        score_threshold: minimum model score (default 0.30).
        horizon_m: geodesic cap for navmesh validation (default 40.0).
        standoff_m: preferred stop distance from object centers (default 0.7).
        output_dir: where to write overlay images.

    Outputs:
        preview: overlay image plus numbered candidates and confirm_token.
        confirm: one bounded navigation hop to the cached candidate snapped point.
    """
    session = adapter.navigation_session(session_id)
    pathfinder = adapter.navigation_pathfinder(session)

    image_ref = str(payload.get("image_ref") or "").strip()
    if not image_ref:
        return _failure("invalid_image_ref", "visual_ground_preview requires image_ref")

    registry = getattr(session, "last_visual_image_refs", {}) or {}
    image_meta = registry.get(image_ref) if isinstance(registry, Mapping) else None
    if not isinstance(image_meta, Mapping):
        return _failure("invalid_image_ref", "image_ref is not valid for this session")

    latest_seq = getattr(session, "latest_visual_capture_seq", None)
    if latest_seq is not None and image_meta.get("capture_seq") != latest_seq:
        if str(payload.get("confirm_token") or "").strip():
            _clear_pending_preview(session)
            return _failure(
                "stale_confirm_token",
                "a newer visual capture invalidated the preview",
                image_ref=image_ref,
            )
        return _failure(
            "stale_image_ref",
            "image_ref is not from the latest visual capture",
        )

    confirm_token = str(payload.get("confirm_token") or "").strip()
    if confirm_token:
        return _confirm_candidate(
            adapter=adapter,
            session=session,
            payload=payload,
            image_ref=image_ref,
            image_meta=image_meta,
            confirm_token=confirm_token,
        )

    phrase = str(payload.get("phrase") or "").strip()
    if not phrase:
        return _failure("missing_phrase", "visual_ground_preview requires a phrase")
    _clear_pending_preview(session)

    mode = str(payload.get("mode") or _DEFAULT_LOCATE_ANYTHING_MODE).strip().lower()
    if mode not in {"box", "point"}:
        mode = _DEFAULT_LOCATE_ANYTHING_MODE

    score_threshold = _positive_float(
        payload.get("score_threshold"), default=_DEFAULT_LOCATE_ANYTHING_SCORE_THRESHOLD
    )
    max_candidates = max(
        1,
        int(
            payload.get("max_candidates")
            or os.environ.get("LOCATE_ANYTHING_MAX_CANDIDATES")
            or _DEFAULT_LOCATE_ANYTHING_MAX_CANDIDATES
        ),
    )
    horizon_m = _positive_float(
        payload.get("horizon_m"), default=_DEFAULT_VISUAL_NAV_HORIZON_M
    )
    standoff_m = _positive_float(payload.get("standoff_m"), default=0.7)
    goal_radius = _positive_float(
        payload.get("goal_radius"),
        default=_DEFAULT_GOAL_RADIUS_M,
    )
    output_dir = str(payload.get("output_dir") or _DEFAULT_VISUAL_OUTPUT_DIR)
    session_output_dir = _resolve_session_output_dir(
        adapter,
        output_dir,
        session.session_id,
    )

    grounding_source = _grounding_image_source(image_meta)
    if grounding_source is None:
        return _failure("invalid_image_ref", "image_ref has no image path")

    try:
        detections = _locate_anything_detections(
            image_path=str(grounding_source["path"]),
            phrase=phrase,
            mode=mode,
            timeout_s=_positive_float(
                payload.get("timeout_s"), default=_DEFAULT_LOCATE_ANYTHING_TIMEOUT_S
            ),
        )
    except Exception as exc:
        return _failure(
            "grounding_service_error",
            f"LocateAnything service call failed: {exc}",
        )

    detections = _scale_detections_to_target(detections, grounding_source)
    detections = [
        d for d in detections if float(d.get("score") or 0.0) >= score_threshold
    ]
    if not detections:
        return _failure(
            "no_visual_grounding",
            "LocateAnything returned no detections above the score threshold",
            image_ref=image_ref,
        )

    current_position = np.asarray(
        adapter.agent_position(session),
        dtype=np.float32,
    )

    candidates: list[dict[str, Any]] = []
    for det in detections:
        candidates.extend(
            _build_candidates(
                image_meta=image_meta,
                detection=det,
                phrase=phrase,
                mode=mode,
                current_position=current_position,
                pathfinder=pathfinder,
                simulator=adapter.navigation_scene(session),
                horizon_m=horizon_m,
                standoff_m=standoff_m,
                goal_radius=goal_radius,
            )
        )

    if not candidates:
        return _failure(
            "no_reachable_candidate",
            "Detections were found but none produced a reachable navmesh point",
            image_ref=image_ref,
            detections=_public_detections(detections),
        )

    candidates.sort(
        key=lambda item: (
            not bool(item.get("reachable")),
            _candidate_rank_value(item),
            -float(item.get("score") or 0.0),
        )
    )
    detected_candidate_count = len(candidates)
    candidates = candidates[:max_candidates]
    for idx, candidate in enumerate(candidates):
        candidate["candidate_id"] = idx

    safe_phrase = re.sub(r"[^a-zA-Z0-9_]+", "_", phrase)[:40].strip("_") or "preview"
    overlay = _write_preview_overlay(
        output_dir=session_output_dir,
        image_meta=image_meta,
        candidates=candidates,
        prefix=f"locate_anything_preview_{safe_phrase}_",
    )
    live_overlay = getattr(adapter, "publish_live_overlay", None)
    if live_overlay is not None:
        live_overlay(session, overlay, tool="hab_visual_ground_preview", image_ref=image_ref,
                     capture_seq=image_meta.get("capture_seq"), direction=image_meta.get("direction"),
                     phrase=phrase)
    public_detections = _public_detections(detections)
    preview_record = {
        "created_at": time.monotonic(),
        "image_ref": image_ref,
        "capture_seq": image_meta.get("capture_seq"),
        "agent_position": current_position.tolist(),
        "agent_forward": _agent_forward_signature(adapter, session),
        "candidates": candidates,
        "overlay_image": overlay,
        "phrase": phrase,
        "mode": mode,
        "detections": public_detections,
        "output_dir": output_dir,
    }

    reachable_candidates = [c for c in candidates if c.get("reachable")]
    if not reachable_candidates:
        return _failure(
            "no_reachable_candidate",
            "LocateAnything detections did not produce a reachable navmesh point",
            image_ref=image_ref,
            detections=public_detections,
        )

    if (
        detected_candidate_count == 1
        and len(candidates) == 1
        and not bool(candidates[0].get("requires_confirmation"))
    ):
        return _execute_candidate_navigation(
            adapter=adapter,
            session=session,
            payload=payload,
            image_ref=image_ref,
            image_meta=image_meta,
            pending=preview_record,
            candidate_id=0,
            candidate=candidates[0],
            auto_confirmed=True,
            candidate_count=len(candidates),
        )

    token = secrets.token_urlsafe(18)
    preview_record["token"] = token
    setattr(
        session,
        _PENDING_PREVIEW_ATTR,
        preview_record,
    )

    return {
        "ok": True,
        "backend": "locate_anything_preview",
        "status": "preview_ready",
        "image_ref": image_ref,
        "direction": image_meta.get("direction"),
        "phrase": phrase,
        "mode": mode,
        "confirm_token": token,
        "overlay_image": overlay,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "detections": public_detections,
        "pre_move_self_check": {
            "required": True,
            "instruction": (
                "Inspect overlay_image, choose a candidate_id, then call "
                "hab_visual_ground_preview again with the same image_ref, "
                "confirm_token, and candidate_id to move. Do not call "
                "hab_visual_point_navigate for this preview."
            ),
        },
    }


def _confirm_candidate(
    *,
    adapter: NavigationBackend,
    session: NavigationSession,
    payload: Mapping[str, Any],
    image_ref: str,
    image_meta: Mapping[str, Any],
    confirm_token: str,
) -> dict[str, Any]:
    pending = getattr(session, _PENDING_PREVIEW_ATTR, None)
    invalid = _validate_pending_preview(
        adapter=adapter,
        session=session,
        pending=pending,
        image_ref=image_ref,
        image_meta=image_meta,
        confirm_token=confirm_token,
    )
    if invalid is not None:
        _clear_pending_preview(session)
        return invalid

    candidates = (
        list(pending.get("candidates") or []) if isinstance(pending, Mapping) else []
    )
    candidate_id = _candidate_id(payload.get("candidate_id"))
    if candidate_id is None or candidate_id < 0 or candidate_id >= len(candidates):
        _clear_pending_preview(session)
        return _failure(
            "invalid_candidate_id",
            "candidate_id must identify a candidate from the active preview",
            image_ref=image_ref,
        )
    candidate = candidates[candidate_id]
    if not isinstance(candidate, Mapping) or not candidate.get("reachable"):
        _clear_pending_preview(session)
        return _failure(
            "candidate_not_reachable",
            "selected candidate is not reachable",
            image_ref=image_ref,
        )
    snapped = candidate.get("snapped_point")
    if not isinstance(snapped, (list, tuple)) or len(snapped) < 3:
        _clear_pending_preview(session)
        return _failure(
            "candidate_missing_goal",
            "selected candidate has no cached snapped navmesh point",
            image_ref=image_ref,
        )

    _clear_pending_preview(session)
    return _execute_candidate_navigation(
        adapter=adapter,
        session=session,
        payload=payload,
        image_ref=image_ref,
        image_meta=image_meta,
        pending=pending,
        candidate_id=candidate_id,
        candidate=candidate,
        auto_confirmed=False,
        candidate_count=len(candidates),
    )


def _execute_candidate_navigation(
    *,
    adapter: NavigationBackend,
    session: NavigationSession,
    payload: Mapping[str, Any],
    image_ref: str,
    image_meta: Mapping[str, Any],
    pending: Mapping[str, Any],
    candidate_id: int,
    candidate: Mapping[str, Any],
    auto_confirmed: bool,
    candidate_count: int,
) -> dict[str, Any]:
    snapped = candidate.get("snapped_point")
    if not isinstance(snapped, (list, tuple)) or len(snapped) < 3:
        return _failure(
            "candidate_missing_goal",
            "selected candidate has no cached snapped navmesh point",
            image_ref=image_ref,
        )

    current_position = np.asarray(
        adapter.agent_position(session),
        dtype=np.float32,
    )
    goal_radius = _positive_float(
        payload.get("goal_radius"),
        default=_DEFAULT_GOAL_RADIUS_M,
    )
    output_dir = str(
        payload.get("output_dir")
        or pending.get("output_dir")
        or _DEFAULT_VISUAL_OUTPUT_DIR
    )
    max_steps = payload.get("max_steps")
    nav = adapter.navigate_step(
        session.session_id,
        {
            "goal": [float(v) for v in snapped[:3]],
            "goal_radius": goal_radius,
            "until_reached": max_steps is None,
            **({"max_steps": max(1, int(max_steps))} if max_steps is not None else {}),
            "include_metrics": True,
            "include_visuals": True,
            "include_publish_hints": True,
            "output_dir": output_dir,
        },
    )
    body = dict(nav) if isinstance(nav, Mapping) else {}
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
    planned_path_m = _optional_float(candidate.get("planned_path_m"))
    translation_steps = int(body.get("steps_executed") or 0)
    raw_nav_status = str(body.get("nav_status") or "en_route_visual_ground_candidate")
    navigation_reached = raw_nav_status == "reached"
    status = raw_nav_status
    if navigation_reached:
        status = "reached_visual_ground_candidate"
    elif status in {"blocked", "unreachable", "error"}:
        status = "navigation_blocked"

    reason: str | None = None
    if translation_steps <= 0 or (
        displacement_m is not None and displacement_m < _MIN_MEANINGFUL_DISPLACEMENT_M
    ):
        status = "self_check_required"
        reason = "near_zero_motion"
    elif planned_path_m is not None and planned_path_m < _MIN_MEANINGFUL_PLANNED_PATH_M:
        status = "self_check_required"
        reason = "target_too_close"

    movement_frames: list[str] = []
    _append_unique_paths(movement_frames, _movement_frame_paths(body))
    try:
        orientation = _orient_to_portal(
            adapter=adapter,
            session=session,
            candidate=candidate,
            output_dir=output_dir,
            navigation_reached=navigation_reached,
        )
    except Exception as exc:
        orientation = {
            "steps_executed": 0,
            "turn_bodies": [],
            "last_body": None,
            "adjustment": {"status": "failed", "reason": str(exc)},
        }
    for turn_body in orientation["turn_bodies"]:
        _append_unique_paths(movement_frames, _movement_frame_paths(turn_body))
    orientation_steps = int(orientation["steps_executed"])
    final_body = orientation.get("last_body") or body
    result = {
        "ok": True,
        "backend": "locate_anything_preview",
        "status": status,
        "nav_status": status,
        "image_ref": image_ref,
        "direction": image_meta.get("direction"),
        "phrase": pending.get("phrase"),
        "mode": pending.get("mode"),
        "auto_confirmed": auto_confirmed,
        "candidate_count": candidate_count,
        "candidates": pending.get("candidates"),
        "detections": pending.get("detections"),
        "selected_candidate_id": candidate_id,
        "selected_candidate": dict(candidate),
        "overlay_image": pending.get("overlay_image"),
        "steps_executed": translation_steps + orientation_steps,
        "translation_steps_executed": translation_steps,
        "orientation_steps_executed": orientation_steps,
        "orientation_adjustment": orientation["adjustment"],
        "reason": reason,
        "message": _self_check_message(reason),
        "reachable": True,
        "distance_to_visual_target": candidate.get("distance_to_visual_target")
        or candidate.get("target_distance_m"),
        "planned_path_m": planned_path_m,
        "displacement_m": displacement_m,
        "goal_radius": goal_radius,
        "metrics": final_body.get("metrics") or body.get("metrics"),
        "state_summary": final_body.get("state_summary"),
        "visuals": final_body.get("visuals", {}),
        "publish_hints": final_body.get("publish_hints"),
        "movement_frames": movement_frames,
        "movement_frame_count": len(movement_frames),
    }
    return {key: value for key, value in result.items() if value is not None}


def _orient_to_portal(
    *,
    adapter: NavigationBackend,
    session: NavigationSession,
    candidate: Mapping[str, Any],
    output_dir: str,
    navigation_reached: bool,
) -> dict[str, Any]:
    private = candidate.get("_private")
    midpoint = private.get("portal_midpoint") if isinstance(private, Mapping) else None
    if (
        not navigation_reached
        or candidate.get("portal_confidence") != "high"
        or not isinstance(midpoint, (list, tuple))
        or len(midpoint) < 3
    ):
        return {
            "steps_executed": 0,
            "turn_bodies": [],
            "last_body": None,
            "adjustment": {"status": "not_applicable"},
        }
    initial_error = _portal_heading_error_degrees(adapter, session, midpoint)
    if initial_error is None:
        return {
            "steps_executed": 0,
            "turn_bodies": [],
            "last_body": None,
            "adjustment": {"status": "unavailable", "reason": "heading_unavailable"},
        }
    if abs(initial_error) <= 5.0:
        return {
            "steps_executed": 0,
            "turn_bodies": [],
            "last_body": None,
            "adjustment": {
                "status": "aligned",
                "requested_degrees": 0.0,
                "executed_degrees": 0.0,
                "residual_degrees": round(abs(initial_error), 2),
            },
        }
    step_count = min(18, int(round(abs(initial_error) / 10.0)))
    action = "turn_right" if initial_error > 0.0 else "turn_left"
    turn_bodies: list[dict[str, Any]] = []
    failure_reason: str | None = None
    for _ in range(step_count):
        try:
            raw = adapter.step_and_capture(
                session.session_id,
                {
                    "action": action,
                    "degrees": 10,
                    "include_metrics": True,
                    "include_publish_hints": True,
                    "output_dir": output_dir,
                },
            )
        except Exception as exc:
            failure_reason = str(exc)
            break
        turn_body = dict(raw) if isinstance(raw, Mapping) else {}
        turn_bodies.append(turn_body)
        if turn_body.get("observation_error"):
            failure_reason = str(turn_body["observation_error"])
            break
    residual = _portal_heading_error_degrees(adapter, session, midpoint)
    executed_steps = sum(int(row.get("steps_taken") or 0) for row in turn_bodies)
    aligned = residual is not None and abs(residual) <= 5.0
    if failure_reason:
        adjustment_status = "partial" if executed_steps else "failed"
    elif aligned:
        adjustment_status = "aligned"
    else:
        adjustment_status = "incomplete"
    adjustment = {
        "status": adjustment_status,
        "action": action,
        "requested_degrees": round(abs(initial_error), 2),
        "executed_degrees": executed_steps * 10,
        "residual_degrees": round(abs(residual), 2) if residual is not None else None,
        "reason": failure_reason,
    }
    return {
        "steps_executed": executed_steps,
        "turn_bodies": turn_bodies,
        "last_body": turn_bodies[-1] if turn_bodies else None,
        "adjustment": {
            key: value for key, value in adjustment.items() if value is not None
        },
    }


def _portal_heading_error_degrees(
    adapter: NavigationBackend,
    session: NavigationSession,
    midpoint: list[float] | tuple[float, ...],
) -> float | None:
    try:
        position = np.asarray(
            adapter.agent_position(session),
            dtype=np.float32,
        )
        forward = np.asarray(adapter.agent_forward(session), dtype=np.float32)
        desired = np.asarray(midpoint[:3], dtype=np.float32) - position
    except Exception:
        return None
    forward[1] = 0.0
    desired[1] = 0.0
    if not np.all(np.isfinite(position)) or not np.all(np.isfinite(forward)):
        return None
    if not np.all(np.isfinite(desired)):
        return None
    if float(np.linalg.norm(forward)) <= 1e-6 or float(np.linalg.norm(desired)) <= 1e-6:
        return None
    current_heading = math.degrees(math.atan2(float(forward[2]), float(forward[0])))
    desired_heading = math.degrees(math.atan2(float(desired[2]), float(desired[0])))
    error = (desired_heading - current_heading + 180.0) % 360.0 - 180.0
    return error if math.isfinite(error) else None


def _validate_pending_preview(
    *,
    adapter: NavigationBackend,
    session: NavigationSession,
    pending: Any,
    image_ref: str,
    image_meta: Mapping[str, Any],
    confirm_token: str,
) -> dict[str, Any] | None:
    if not isinstance(pending, Mapping):
        return _failure(
            "invalid_confirm_token",
            "confirm_token does not match an active visual ground preview",
            image_ref=image_ref,
        )
    if str(pending.get("token") or "") != confirm_token:
        return _failure(
            "invalid_confirm_token",
            "confirm_token does not match the active visual ground preview",
            image_ref=image_ref,
        )
    created_at = pending.get("created_at")
    if not isinstance(created_at, (int, float)):
        return _failure(
            "invalid_confirm_token",
            "visual ground preview token is malformed",
            image_ref=image_ref,
        )
    if time.monotonic() - float(created_at) > _PREVIEW_TOKEN_TTL_S:
        return _failure(
            "expired_confirm_token",
            "visual ground preview expired; request a fresh preview",
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
    current_position = np.asarray(
        adapter.agent_position(session),
        dtype=np.float32,
    )
    if _vector_changed(
        pending.get("agent_position"),
        current_position.tolist(),
        tolerance=1e-3,
    ):
        return _failure(
            "stale_confirm_token",
            "agent movement invalidated the visual ground preview",
            image_ref=image_ref,
        )
    old_forward = pending.get("agent_forward")
    new_forward = _agent_forward_signature(adapter, session)
    if old_forward is not None and _vector_changed(
        old_forward,
        new_forward,
        tolerance=1e-3,
    ):
        return _failure(
            "stale_confirm_token",
            "agent heading changed after preview; request a fresh preview",
            image_ref=image_ref,
        )
    return None


def _clear_pending_preview(session: NavigationSession) -> None:
    try:
        setattr(session, _PENDING_PREVIEW_ATTR, None)
    except Exception:
        pass


def _candidate_id(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


def _grounding_image_source(image_meta: Mapping[str, Any]) -> dict[str, Any] | None:
    agent_path = str(image_meta.get("path") or "")
    original_path = str(image_meta.get("original_path") or "")
    use_original = bool(original_path) and (
        os.path.exists(original_path) or not agent_path
    )
    path = original_path if use_original else agent_path
    if not path:
        return None
    try:
        target_width = float(image_meta.get("width") or 0)
        target_height = float(image_meta.get("height") or 0)
        source_width_value = image_meta.get(
            "original_width" if use_original else "width"
        )
        source_height_value = image_meta.get(
            "original_height" if use_original else "height"
        )
        source_width = float(source_width_value or 0)
        source_height = float(source_height_value or 0)
    except (TypeError, ValueError):
        return None
    if (
        target_width <= 0
        or target_height <= 0
        or source_width <= 0
        or source_height <= 0
    ):
        target_width = target_height = source_width = source_height = 1.0
    return {
        "path": path,
        "grounding_source": "original" if use_original else "agent",
        "target_width": target_width,
        "target_height": target_height,
        "source_width": source_width,
        "source_height": source_height,
    }


def _scale_detections_to_target(
    detections: list[dict[str, Any]],
    source: Mapping[str, Any],
) -> list[dict[str, Any]]:
    try:
        source_width = float(source.get("source_width") or 0)
        source_height = float(source.get("source_height") or 0)
        target_width = float(source.get("target_width") or 0)
        target_height = float(source.get("target_height") or 0)
    except (TypeError, ValueError):
        return detections
    if (
        source_width <= 0
        or source_height <= 0
        or target_width <= 0
        or target_height <= 0
    ):
        return detections
    scale_x = target_width / source_width
    scale_y = target_height / source_height
    scaled: list[dict[str, Any]] = []
    for det in detections:
        row = dict(det)
        box = row.get("box_xyxy")
        if isinstance(box, (list, tuple)) and len(box) >= 4:
            try:
                x0, y0, x1, y1 = [float(v) for v in box[:4]]
            except (TypeError, ValueError):
                pass
            else:
                row["box_xyxy"] = (
                    x0 * scale_x,
                    y0 * scale_y,
                    x1 * scale_x,
                    y1 * scale_y,
                )
        point = row.get("point_xy")
        if isinstance(point, (list, tuple)) and len(point) >= 2:
            try:
                px, py = float(point[0]), float(point[1])
            except (TypeError, ValueError):
                pass
            else:
                row["point_xy"] = (px * scale_x, py * scale_y)
        row["grounding_source"] = source.get("grounding_source")
        row["grounding_source_size"] = [
            int(round(source_width)),
            int(round(source_height)),
        ]
        row["grounding_target_size"] = [
            int(round(target_width)),
            int(round(target_height)),
        ]
        scaled.append(row)
    return scaled


def _self_check_message(reason: str | None) -> str | None:
    if reason == "target_too_close":
        return (
            "The cached visual candidate is already too close. Inspect the "
            "latest panorama before choosing another candidate or closing."
        )
    if reason == "near_zero_motion":
        return (
            "The cached visual candidate was valid, but the navigation hop "
            "produced near-zero movement. Inspect the latest panorama before "
            "choosing another candidate or closing."
        )
    return None


def _resolve_session_output_dir(adapter: NavigationBackend, output_dir: str, session_id: str) -> str:
    helper = getattr(adapter, "session_output_dir", None)
    if callable(helper):
        return str(helper(output_dir, session_id))
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _portal_normal_enabled() -> bool:
    value = str(os.environ.get("HAB_VISUAL_GROUND_PORTAL_NORMAL", "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _estimate_portal_geometry(
    *,
    image_meta: Mapping[str, Any],
    bbox_px: list[int],
    current_position: np.ndarray,
    standoff_m: float,
) -> dict[str, Any] | None:
    depth = image_meta.get("depth")
    width = int(image_meta.get("width") or 0)
    height = int(image_meta.get("height") or 0)
    if (
        not isinstance(depth, np.ndarray)
        or depth.ndim != 2
        or width <= 1
        or height <= 1
    ):
        return None
    x0, y0, x1, y1 = [float(value) for value in bbox_px]
    box_width = max(1.0, x1 - x0)
    box_height = max(1.0, y1 - y0)
    jambs: list[np.ndarray] = []
    support_counts: list[int] = []
    residuals: list[float] = []
    edge_samples = (
        (x0, (0.0, 0.02, 0.04)),
        (x1, (0.0, -0.02, -0.04)),
    )
    for edge_x, x_fractions in edge_samples:
        samples: list[dict[str, Any]] = []
        for x_fraction in x_fractions:
            px = max(0.0, min(width - 1.0, edge_x + x_fraction * box_width))
            for y_fraction in (0.45, 0.60, 0.75):
                py = max(0.0, min(height - 1.0, y0 + y_fraction * box_height))
                depth_m = _median_depth_near(
                    depth,
                    px,
                    py,
                    image_width=width,
                    image_height=height,
                )
                if depth_m is not None:
                    samples.append({"px": px, "py": py, "depth_m": depth_m})
        if len(samples) < 3:
            return None
        nearest = min(float(row["depth_m"]) for row in samples)
        tolerance = max(0.25, 0.20 * nearest)
        coherent = [
            row for row in samples if float(row["depth_m"]) <= nearest + tolerance
        ]
        if len(coherent) < 3:
            return None
        try:
            points = np.asarray(
                [
                    _unproject_pixel(
                        image_meta,
                        float(row["px"]),
                        float(row["py"]),
                        float(row["depth_m"]),
                    )
                    for row in coherent
                ],
                dtype=np.float32,
            )
        except (KeyError, TypeError, ValueError):
            return None
        median = np.median(points, axis=0).astype(np.float32)
        xz_residuals = np.linalg.norm(points[:, [0, 2]] - median[[0, 2]], axis=1)
        residual = float(np.median(xz_residuals))
        if residual > _PORTAL_JAMB_MAX_MAD_M:
            return None
        jambs.append(median)
        support_counts.append(len(coherent))
        residuals.append(residual)

    left, right = jambs
    tangent = right - left
    tangent[1] = 0.0
    portal_width = float(np.linalg.norm(tangent))
    if not _PORTAL_MIN_WIDTH_M <= portal_width <= _PORTAL_MAX_WIDTH_M:
        return None
    tangent /= portal_width
    camera_forward = image_meta.get("camera_forward")
    if camera_forward is not None:
        forward = np.asarray(camera_forward, dtype=np.float32).copy()
        if forward.shape == (3,):
            forward[1] = 0.0
            forward_norm = float(np.linalg.norm(forward))
            if forward_norm > 1e-6 and abs(
                float(np.dot(tangent, forward / forward_norm))
            ) > _PORTAL_MAX_TANGENT_FORWARD_COS:
                return None
    midpoint = ((left + right) / 2.0).astype(np.float32)
    if _horizontal_distance(current_position, midpoint) <= standoff_m + 0.15:
        return None
    normal = np.asarray([-tangent[2], 0.0, tangent[0]], dtype=np.float32)
    toward_agent = np.asarray(current_position, dtype=np.float32) - midpoint
    toward_agent[1] = 0.0
    if float(np.dot(normal, toward_agent)) < 0.0:
        normal *= -1.0
    return {
        "left_jamb": left,
        "right_jamb": right,
        "midpoint": midpoint,
        "tangent": tangent,
        "normal": normal,
        "width_m": portal_width,
        "support_count": min(support_counts),
        "residual_m": max(residuals),
    }


def _portal_candidate_points(
    *,
    portal: Mapping[str, Any],
    standoff_m: float,
) -> list[dict[str, Any]]:
    midpoint = np.asarray(portal["midpoint"], dtype=np.float32)
    tangent = np.asarray(portal["tangent"], dtype=np.float32)
    normal = np.asarray(portal["normal"], dtype=np.float32)
    lateral = min(0.30, 0.20 * float(portal["width_m"]))
    rows: list[dict[str, Any]] = []
    for distance in (standoff_m, max(standoff_m, 0.9), max(standoff_m, 1.2)):
        for offset in (0.0, -lateral, lateral):
            rows.append(
                {
                    "point": (midpoint + normal * distance + tangent * offset).astype(
                        np.float32
                    ),
                    "normal_standoff_m": float(distance),
                    "lateral_offset_m": float(offset),
                }
            )
    return rows


def _portal_candidate_for_detection(
    *,
    image_meta: Mapping[str, Any],
    detection: Mapping[str, Any],
    bbox_px: list[int],
    current_position: np.ndarray,
    pathfinder: NavigationPathfinder,
    simulator: RaycastScene,
    horizon_m: float,
    standoff_m: float,
    goal_radius: float,
) -> dict[str, Any] | None:
    portal = _estimate_portal_geometry(
        image_meta=image_meta,
        bbox_px=bbox_px,
        current_position=current_position,
        standoff_m=standoff_m,
    )
    if portal is None:
        return None
    viable: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for spec in _portal_candidate_points(portal=portal, standoff_m=standoff_m):
        raw_point = np.asarray(spec["point"], dtype=np.float32).copy()
        raw_point[1] = float(current_position[1])
        validated = _validate_candidate(
            point=raw_point,
            hint=np.asarray(portal["midpoint"], dtype=np.float32),
            current_position=current_position,
            pathfinder=pathfinder,
            horizon_m=horizon_m,
            standoff_m=standoff_m,
            goal_radius=goal_radius,
            simulator=simulator,
            enforce_approach_side=True,
            max_snap_distance_m=_MAX_APPROACH_SNAP_DISTANCE_M,
            require_target_visibility=True,
            approach_plane_origin=np.asarray(portal["midpoint"], dtype=np.float32),
            approach_plane_normal=np.asarray(portal["normal"], dtype=np.float32),
            lateral_offset_m=float(spec["lateral_offset_m"]),
        )
        if validated.get("reachable"):
            viable.append((spec, validated))
    if not viable:
        return None
    _, best = min(viable, key=lambda row: _candidate_rank_value(row[1]))
    width = int(image_meta.get("width") or 0)
    height = int(image_meta.get("height") or 0)
    point_px = _bbox_anchor_px(bbox_px)
    row = {
        "candidate_id": None,
        "point": _normalize_pixel(
            float(point_px[0]), float(point_px[1]), width, height
        ),
        "point_px": point_px,
        "bbox_px": bbox_px,
        "score": round(float(detection.get("score") or 0.0), 4),
        "nav_score": best.get("score"),
        "rank_score": best.get("score"),
        "anchor_strategy": "portal_normal",
        "portal_confidence": "high",
        "portal_width_m": round(float(portal["width_m"]), 4),
        "portal_support_count": int(portal["support_count"]),
        "portal_residual_m": round(float(portal["residual_m"]), 4),
        "normal_standoff_m": best.get("normal_standoff_m"),
        "lateral_offset_m": best.get("lateral_offset_m"),
        "arrival_alignment_deg": best.get("arrival_alignment_deg"),
        "requires_confirmation": True,
        "reachable": True,
        "status": "ok",
        "planned_path_m": best.get("geodesic_distance"),
        "distance_to_visual_target": best.get("target_distance_m"),
        "target_distance_m": best.get("target_distance_m"),
        "snap_distance_m": best.get("snap_distance"),
        "standoff_error_m": best.get("standoff_error_m"),
        "target_side_margin_m": best.get("target_side_margin_m"),
        "snapped_point": best.get("snapped"),
        "_private": {
            "portal_midpoint": np.asarray(portal["midpoint"]).tolist(),
            "portal_normal": np.asarray(portal["normal"]).tolist(),
            "left_jamb": np.asarray(portal["left_jamb"]).tolist(),
            "right_jamb": np.asarray(portal["right_jamb"]).tolist(),
        },
    }
    return {key: value for key, value in row.items() if value is not None}


def _build_candidates(
    *,
    image_meta: Mapping[str, Any],
    detection: Mapping[str, Any],
    phrase: str,
    mode: str,
    current_position: np.ndarray,
    pathfinder: NavigationPathfinder,
    simulator: RaycastScene,
    horizon_m: float,
    standoff_m: float,
    goal_radius: float,
) -> list[dict[str, Any]]:
    width = int(image_meta.get("width") or 0)
    height = int(image_meta.get("height") or 0)
    if width <= 0 or height <= 0:
        return []

    if mode == "point":
        point_xy = detection.get("point_xy")
        if not isinstance(point_xy, (list, tuple)) or len(point_xy) < 2:
            return []
        px, py = float(point_xy[0]), float(point_xy[1])
        anchor: dict[str, Any] = {
            "anchor_px": [int(round(px)), int(round(py))],
            "search_radius": 0.10,
        }
        bbox_px = None
        anchor_specs = [
            {
                "anchor_strategy": "point",
                "anchor": anchor,
                "point_px": anchor["anchor_px"],
            }
        ]
    else:
        box_xyxy = detection.get("box_xyxy")
        if not isinstance(box_xyxy, (list, tuple)) or len(box_xyxy) < 4:
            return []
        x0, y0, x1, y1 = [float(v) for v in box_xyxy[:4]]
        bbox_px = [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]
        anchor_specs = _anchor_specs_for_detection(
            bbox_px=bbox_px,
            width=width,
            height=height,
            phrase=phrase,
            detection=detection,
        )

    candidates: list[dict[str, Any]] = []
    unreachable: list[dict[str, Any]] = []
    door_like = _is_door_like_detection(
        bbox_px=bbox_px,
        width=width,
        height=height,
        phrase=phrase,
        detection=detection,
    )
    if (
        door_like
        and bbox_px is not None
        and _portal_normal_enabled()
        and _has_portal_semantics(phrase, detection)
    ):
        portal_candidate = _portal_candidate_for_detection(
            image_meta=image_meta,
            detection=detection,
            bbox_px=bbox_px,
            current_position=current_position,
            pathfinder=pathfinder,
            simulator=simulator,
            horizon_m=horizon_m,
            standoff_m=standoff_m,
            goal_radius=goal_radius,
        )
        if portal_candidate is not None:
            return [portal_candidate]
    for spec in anchor_specs:
        anchor = spec["anchor"]
        raw_candidates = _visual_nav_candidates(
            image_meta=image_meta,
            anchor=anchor,
            current_position=current_position,
            pathfinder=pathfinder,
            horizon_m=horizon_m,
            standoff_m=standoff_m,
            goal_radius=goal_radius,
            simulator=simulator,
            enforce_approach_side=True,
            max_snap_distance_m=_MAX_APPROACH_SNAP_DISTANCE_M,
            require_target_visibility=True,
        )
        viable = [c for c in raw_candidates if c.get("reachable")]
        if not viable:
            best_unreachable = min(
                raw_candidates,
                key=_candidate_rank_value,
                default=None,
            )
            if best_unreachable is not None:
                unreachable.append(
                    _candidate_row(
                        image_meta=image_meta,
                        detection=detection,
                        bbox_px=bbox_px,
                        best=best_unreachable,
                        spec=spec,
                        reachable=False,
                        door_like=door_like,
                    )
                )
            continue
        best = min(viable, key=_candidate_rank_value)
        candidates.append(
            _candidate_row(
                image_meta=image_meta,
                detection=detection,
                bbox_px=bbox_px,
                best=best,
                spec=spec,
                reachable=True,
                door_like=door_like,
            )
        )

    if candidates:
        return _dedupe_candidates(candidates)
    return unreachable[:1]


def _anchor_specs_for_detection(
    *,
    bbox_px: list[int],
    width: int,
    height: int,
    phrase: str,
    detection: Mapping[str, Any],
) -> list[dict[str, Any]]:
    x0, y0, x1, y1 = [float(v) for v in bbox_px]
    box_h = max(1.0, y1 - y0)
    lower_y = y0 + 0.88 * box_h
    center_x = (x0 + x1) / 2.0
    center_y = (y0 + y1) / 2.0
    specs: list[dict[str, Any]] = [
        {
            "anchor_strategy": "lower_band",
            "anchor": {"bbox_px": bbox_px, "anchor": "lower_band"},
            "point_px": [int(round(center_x)), int(round(lower_y))],
        }
    ]
    if not _is_door_like_detection(
        bbox_px=bbox_px,
        width=width,
        height=height,
        phrase=phrase,
        detection=detection,
    ):
        return specs

    below_y = min(height - 1.0, y1 + max(4.0, 0.12 * box_h))
    probe_radius = max(0.02, min(0.12, 0.05 * box_h / max(1.0, min(width, height))))
    extra_points = [
        ("bbox_center", center_x, center_y, 0.06),
        ("below_bbox_floor", center_x, below_y, probe_radius),
    ]
    for strategy, px, py, radius in extra_points:
        point_px = [
            int(round(max(0.0, min(width - 1.0, px)))),
            int(round(max(0.0, min(height - 1.0, py)))),
        ]
        specs.append(
            {
                "anchor_strategy": strategy,
                "anchor": {
                    "anchor_px": point_px,
                    "search_radius": radius,
                },
                "point_px": point_px,
            }
        )
    return specs


def _candidate_row(
    *,
    image_meta: Mapping[str, Any],
    detection: Mapping[str, Any],
    bbox_px: list[int] | None,
    best: Mapping[str, Any],
    spec: Mapping[str, Any],
    reachable: bool,
    door_like: bool,
) -> dict[str, Any]:
    width = int(image_meta.get("width") or 0)
    height = int(image_meta.get("height") or 0)
    sample_px = best.get("sample_px")
    fallback_px = spec.get("point_px") or _bbox_anchor_px(bbox_px)
    anchor_px = sample_px if sample_px else fallback_px
    if not isinstance(anchor_px, (list, tuple)) or len(anchor_px) < 2:
        anchor_px = [0, 0]
    norm_point = _normalize_pixel(
        float(anchor_px[0]), float(anchor_px[1]), width, height
    )
    depth_diagnostics = _depth_diagnostics(
        image_meta=image_meta,
        point_px=[float(anchor_px[0]), float(anchor_px[1])],
        bbox_px=bbox_px,
    )
    penalties = (
        _ranking_penalties(
            best=best,
            strategy=str(spec.get("anchor_strategy") or "unknown"),
            door_like=door_like,
            depth_diagnostics=depth_diagnostics,
        )
        if reachable
        else {}
    )
    nav_score = _optional_float(best.get("score"))
    penalty_total = sum(float(value) for value in penalties.values())
    rank_score = (
        round(float(nav_score or 0.0) + penalty_total, 4) if reachable else nav_score
    )
    row = {
        "candidate_id": None,
        "point": norm_point,
        "point_px": [int(round(float(anchor_px[0]))), int(round(float(anchor_px[1])))],
        "bbox_px": bbox_px,
        "score": round(float(detection.get("score") or 0.0), 4),
        "nav_score": nav_score,
        "rank_score": rank_score,
        "ranking_penalties": penalties,
        "depth_diagnostics": depth_diagnostics,
        "anchor_strategy": str(spec.get("anchor_strategy") or "unknown"),
        "reachable": bool(reachable),
        "status": (
            "ok" if reachable else str(best.get("status") or "no_reachable_point")
        ),
        "planned_path_m": best.get("geodesic_distance") if reachable else None,
        "distance_to_visual_target": (
            best.get("target_distance_m") if reachable else None
        ),
        "target_distance_m": best.get("target_distance_m") if reachable else None,
        "snap_distance_m": best.get("snap_distance"),
        "standoff_error_m": best.get("standoff_error_m") if reachable else None,
        "target_side_margin_m": best.get("target_side_margin_m"),
        "sample_px": sample_px,
        "snapped_point": best.get("snapped") if reachable else None,
    }
    return {key: value for key, value in row.items() if value is not None}


def _is_door_like_detection(
    *,
    bbox_px: list[int] | None,
    width: int,
    height: int,
    phrase: str,
    detection: Mapping[str, Any],
) -> bool:
    label = str(detection.get("label") or "")
    text = f"{phrase} {label}".lower()
    if re.search(r"\b(door|doorway|opening|glass|partition|entrance)\b", text):
        return True
    if bbox_px is None:
        return False
    x0, y0, x1, y1 = [float(v) for v in bbox_px]
    box_w = max(1.0, x1 - x0)
    box_h = max(1.0, y1 - y0)
    return bool(
        box_h >= 1.8 * box_w
        and box_h >= 0.16 * max(1.0, float(height))
        and box_w <= 0.18 * max(1.0, float(width))
    )


def _ranking_penalties(
    *,
    best: Mapping[str, Any],
    strategy: str,
    door_like: bool,
    depth_diagnostics: Mapping[str, Any],
) -> dict[str, float]:
    penalties: dict[str, float] = {}
    planned_path = _optional_float(best.get("geodesic_distance"))
    if door_like and strategy == "lower_band":
        penalties["thin_opening_lower_band"] = 0.75
        if planned_path is not None and planned_path < 2.25:
            penalties["suspicious_short_opening_path"] = 1.75
    discontinuity = _optional_float(depth_diagnostics.get("local_depth_range_m"))
    anchor_depth = _optional_float(depth_diagnostics.get("anchor_depth_m"))
    if discontinuity is not None and anchor_depth is not None:
        threshold = max(0.45, 0.30 * max(0.1, anchor_depth))
        if discontinuity > threshold:
            penalties["depth_discontinuity"] = min(
                1.5, round(discontinuity / threshold, 4)
            )
    center_depth = _optional_float(depth_diagnostics.get("bbox_center_depth_m"))
    if (
        door_like
        and anchor_depth is not None
        and center_depth is not None
        and anchor_depth + max(0.35, 0.22 * center_depth) < center_depth
    ):
        penalties["foreground_depth_contamination"] = 1.5
    return {key: round(float(value), 4) for key, value in penalties.items()}


def _depth_diagnostics(
    *,
    image_meta: Mapping[str, Any],
    point_px: list[float],
    bbox_px: list[int] | None,
) -> dict[str, Any]:
    depth = image_meta.get("depth")
    width = int(image_meta.get("width") or 0)
    height = int(image_meta.get("height") or 0)
    if (
        not isinstance(depth, np.ndarray)
        or depth.ndim != 2
        or width <= 0
        or height <= 0
    ):
        return {}
    anchor_stats = _depth_window_stats(
        depth=depth,
        px=float(point_px[0]),
        py=float(point_px[1]),
        image_width=width,
        image_height=height,
    )
    body: dict[str, Any] = {}
    if anchor_stats:
        body["anchor_depth_m"] = anchor_stats.get("median")
        body["local_depth_range_m"] = anchor_stats.get("range")
        body["valid_depth_count"] = anchor_stats.get("count")
    if bbox_px is not None:
        x0, y0, x1, y1 = [float(v) for v in bbox_px]
        center_stats = _depth_window_stats(
            depth=depth,
            px=(x0 + x1) / 2.0,
            py=(y0 + y1) / 2.0,
            image_width=width,
            image_height=height,
        )
        if center_stats:
            body["bbox_center_depth_m"] = center_stats.get("median")
    return {key: value for key, value in body.items() if value is not None}


def _depth_window_stats(
    *,
    depth: np.ndarray,
    px: float,
    py: float,
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    dh, dw = depth.shape[:2]
    dx = int(round(float(px) * (dw - 1) / max(1, image_width - 1)))
    dy = int(round(float(py) * (dh - 1) / max(1, image_height - 1)))
    radius = max(1, int(round(min(dw, dh) * 0.02)))
    x0, x1 = max(0, dx - radius), min(dw, dx + radius + 1)
    y0, y1 = max(0, dy - radius), min(dh, dy + radius + 1)
    region = np.asarray(depth[y0:y1, x0:x1], dtype=np.float32)
    valid = region[np.isfinite(region) & (region > 0.05)]
    if valid.size == 0:
        return {}
    p10 = float(np.percentile(valid, 10))
    p90 = float(np.percentile(valid, 90))
    return {
        "median": round(float(np.median(valid)), 4),
        "range": round(max(0.0, p90 - p10), 4),
        "count": int(valid.size),
    }


def _dedupe_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for candidate in sorted(
        candidates,
        key=_candidate_rank_value,
    ):
        snapped = candidate.get("snapped_point")
        if not isinstance(snapped, (list, tuple)) or len(snapped) < 3:
            kept.append(candidate)
            continue
        duplicate = False
        for existing in kept:
            other = existing.get("snapped_point")
            if not isinstance(other, (list, tuple)) or len(other) < 3:
                continue
            if (
                _horizontal_distance(
                    np.asarray(snapped[:3], dtype=np.float32),
                    np.asarray(other[:3], dtype=np.float32),
                )
                < 0.15
            ):
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _bbox_anchor_px(bbox_px: list[int] | None) -> list[int]:
    if bbox_px is None:
        return [0, 0]
    x0, y0, x1, y1 = bbox_px
    return [int(round((x0 + x1) / 2.0)), int(round(y0 + 0.88 * (y1 - y0)))]


def _normalize_pixel(px: float, py: float, width: int, height: int) -> list[float]:
    if width <= 1 or height <= 1:
        return [0.0, 0.0]
    return [
        round(max(0.0, min(1.0, float(px) / (width - 1))), 6),
        round(max(0.0, min(1.0, float(py) / (height - 1))), 6),
    ]


def _public_detections(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for det in detections:
        row: dict[str, Any] = {"score": round(float(det.get("score") or 0.0), 4)}
        if "box_xyxy" in det:
            row["box_xyxy"] = det["box_xyxy"]
        if "point_xy" in det:
            row["point_xy"] = det["point_xy"]
        if "label" in det:
            row["label"] = det["label"]
        if "grounding_source" in det:
            row["grounding_source"] = det["grounding_source"]
        if "grounding_source_size" in det:
            row["grounding_source_size"] = det["grounding_source_size"]
        if "grounding_target_size" in det:
            row["grounding_target_size"] = det["grounding_target_size"]
        rows.append(row)
    return rows


def _write_preview_overlay(
    *,
    output_dir: str,
    image_meta: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    prefix: str = "locate_anything_preview_",
) -> str | None:
    path = image_meta.get("path")
    if not isinstance(path, str) or not path:
        return None
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None
    try:
        image = Image.open(path).convert("RGB")
    except Exception:
        return None

    draw = ImageDraw.Draw(image)
    font = _overlay_font(ImageFont)

    for idx, candidate in enumerate(candidates):
        point_px = candidate.get("point_px")
        bbox_px = candidate.get("bbox_px")
        reachable = bool(candidate.get("reachable"))
        color = (0, 255, 0) if reachable else (255, 0, 0)

        if isinstance(bbox_px, (list, tuple)) and len(bbox_px) == 4:
            draw.rectangle(tuple(float(v) for v in bbox_px), outline=color, width=2)

        if isinstance(point_px, (list, tuple)) and len(point_px) == 2:
            px, py = int(point_px[0]), int(point_px[1])
            r = 6
            draw.ellipse((px - r, py - r, px + r, py + r), outline=color, width=3)
            label = f"{idx}"
            text_w, text_h = _text_size(draw, label, font)
            tx = max(0, min(image.width - text_w - 1, px + 8))
            ty = max(0, min(image.height - text_h - 1, py - text_h - 8))
            _draw_outlined_text(draw, (tx, ty), label, fill=color, font=font)

    try:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            prefix=prefix,
            dir=str(out_dir),
            delete=False,
        ) as handle:
            out_path = handle.name
        image.save(out_path)
        return out_path
    except Exception:
        return None


def _overlay_font(image_font_module: Any) -> Any:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ):
        if os.path.exists(path):
            try:
                return image_font_module.truetype(path, size=12)
            except Exception:
                pass
    return image_font_module.load_default()


def _text_size(draw: Any, text: str, font: Any) -> tuple[int, int]:
    try:
        box = draw.textbbox((0, 0), text, font=font)
        return int(box[2] - box[0]), int(box[3] - box[1])
    except Exception:
        return len(text) * 7, 12


def _draw_outlined_text(
    draw: Any,
    xy: tuple[int, int],
    text: str,
    *,
    fill: tuple[int, int, int],
    font: Any,
) -> None:
    x, y = xy
    outline = (255, 255, 255)
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        draw.text((x + dx, y + dy), text, fill=outline, font=font)
    draw.text((x, y), text, fill=fill, font=font)


def _failure(
    status: str,
    error: str,
    *,
    image_ref: str | None = None,
    detections: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ok": False,
        "backend": "locate_anything_preview",
        "status": status,
        "error": error,
    }
    if image_ref:
        body["image_ref"] = image_ref
    if detections is not None:
        body["detections"] = detections
    return body
