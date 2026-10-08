"""Overlay-alias grounded local navigation handler."""

from __future__ import annotations

from supernav.paths import workspace_root

import re
from typing import Any, Mapping

from habitat_contract.navigation import NavigationBackend, NavigationSession

from supernav.methods.navigation.oracle_local_nav import navigate_oracle_local
from supernav.methods.navigation.oracle_local_nav.utils import _positive_float
from supernav.methods.navigation.oracle_local_nav.visual_grounding import _turn_toward_projected_direction


_IMAGE_REF_RE = re.compile(r"^pano:[^:]+:(\d+):(?:front|right|back|left)$")
_DEFAULT_OVERLAY_HORIZON_M = 40.0


def navigate_visual_overlay(
    adapter: NavigationBackend,
    session_id: str | None,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Bridge action handler for scene-graph overlay alias navigation."""

    session = adapter.navigation_session(session_id)
    adapter.navigation_pathfinder(session)

    image_ref = str(payload.get("image_ref") or "").strip()
    target_alias = str(payload.get("target_alias") or "").strip()
    if not image_ref:
        return _failure("invalid_image_ref", "visual overlay navigation requires image_ref")
    if not target_alias:
        return _failure(
            "unknown_target_alias",
            "visual overlay navigation requires target_alias such as obj_189",
            image_ref=image_ref,
        )

    image_meta = _latest_image_meta(session, image_ref)
    if image_meta is None:
        latest_seq = getattr(session, "latest_visual_capture_seq", None)
        image_seq = _image_ref_capture_seq(image_ref)
        if latest_seq is not None and image_seq is not None and image_seq != latest_seq:
            return _failure(
                "stale_image_ref",
                "image_ref is not from the latest overlay panorama",
                image_ref=image_ref,
                target_alias=target_alias,
            )
        return _failure(
            "invalid_image_ref",
            "image_ref is not valid for this session",
            image_ref=image_ref,
            target_alias=target_alias,
        )
    latest_seq = getattr(session, "latest_visual_capture_seq", None)
    if latest_seq is not None and image_meta.get("capture_seq") != latest_seq:
        return _failure(
            "stale_image_ref",
            "image_ref is not from the latest overlay panorama",
            image_ref=image_ref,
            target_alias=target_alias,
        )

    alias_registry = getattr(session, "last_visual_overlay_aliases", {}) or {}
    aliases = alias_registry.get(image_ref) if isinstance(alias_registry, Mapping) else None
    alias_row = aliases.get(target_alias) if isinstance(aliases, Mapping) else None
    if not isinstance(alias_row, Mapping):
        return _failure(
            "unknown_target_alias",
            "target_alias is not present on the latest overlay image",
            image_ref=image_ref,
            target_alias=target_alias,
        )
    if latest_seq is not None and alias_row.get("capture_seq") != latest_seq:
        return _failure(
            "stale_image_ref",
            "target_alias is not from the latest overlay panorama",
            image_ref=image_ref,
            target_alias=target_alias,
        )

    target_ref = str(alias_row.get("target_ref") or "").strip()
    if not target_ref:
        return _failure(
            "unknown_target_alias",
            "target_alias does not resolve to a current target",
            image_ref=image_ref,
            target_alias=target_alias,
        )
    latest_refs = getattr(session, "last_visible_nav_target_refs", None)
    if target_ref not in set(str(ref) for ref in (latest_refs or set())):
        return _failure(
            "target_not_currently_visible",
            "target_alias is no longer in the latest visible overlay set",
            image_ref=image_ref,
            target_alias=target_alias,
        )

    projected = _projected_row_for_target(session, target_ref, alias_row)
    turn_result = _turn_toward_projected_direction(adapter, session, projected)
    nav_payload = {
        "target_ref": target_ref,
        "max_steps": int(payload.get("max_steps", 40)),
        "horizon_m": _positive_float(
            payload.get("horizon_m"),
            default=_DEFAULT_OVERLAY_HORIZON_M,
        ),
        "goal_radius": _positive_float(payload.get("goal_radius"), default=0.6),
        "standoff_m": _positive_float(payload.get("standoff_m"), default=0.7),
        "output_dir": str(payload.get("output_dir") or str(workspace_root() / "data" / "runs" / "artifacts")),
    }
    result = navigate_oracle_local(adapter, session.session_id, nav_payload)
    body = dict(result) if isinstance(result, Mapping) else {}
    if body.get("ok") is False:
        status = str(body.get("status") or "unresolved_local_target")
        return _failure(
            status,
            str(body.get("error") or status),
            image_ref=image_ref,
            target_alias=target_alias,
        )

    status = str(body.get("status") or "")
    if status in {"blocked", "unreachable", "error"}:
        body["ok"] = False
        body["status"] = "navigation_blocked"
        body["nav_status"] = "navigation_blocked"

    body["backend"] = "visual_overlay_navmesh"
    body["target_source"] = "visual_overlay_alias"
    body["image_ref"] = image_ref
    body["target_alias"] = target_alias
    body["direction"] = projected.get("direction")
    body["selected_overlay_alias"] = {
        "image_ref": image_ref,
        "target_alias": target_alias,
        "direction": projected.get("direction"),
        "capture_seq": image_meta.get("capture_seq"),
    }
    if turn_result is not None:
        body["pre_navigation_turn"] = {
            key: value
            for key, value in turn_result.items()
            if key in {"direction", "action", "degrees", "steps_taken", "collided"}
        }
    for key in (
        "matched_target_ref",
        "matched_label",
        "query",
        "visual_grounding",
        "visual_detections",
        "_debug",
    ):
        body.pop(key, None)
    return {key: value for key, value in body.items() if value is not None}


def _latest_image_meta(session: NavigationSession, image_ref: str) -> Mapping[str, Any] | None:
    registry = getattr(session, "last_visual_image_refs", {}) or {}
    image_meta = registry.get(image_ref) if isinstance(registry, Mapping) else None
    return image_meta if isinstance(image_meta, Mapping) else None


def _projected_row_for_target(
    session: NavigationSession,
    target_ref: str,
    alias_row: Mapping[str, Any],
) -> Mapping[str, Any]:
    for row in getattr(session, "last_visible_nav_targets", []) or []:
        if isinstance(row, Mapping) and str(row.get("target_ref") or "") == target_ref:
            return row
    return {
        "target_ref": target_ref,
        "direction": str(alias_row.get("direction") or "front"),
    }


def _image_ref_capture_seq(image_ref: str) -> int | None:
    match = _IMAGE_REF_RE.match(image_ref)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _failure(
    status: str,
    error: str,
    *,
    image_ref: str | None = None,
    target_alias: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ok": False,
        "backend": "visual_overlay_navmesh",
        "status": status,
        "error": error,
    }
    if image_ref:
        body["image_ref"] = image_ref
    if target_alias:
        body["target_alias"] = target_alias
    return body
