"""Visually grounded local navigation handler."""

from __future__ import annotations

from supernav.paths import workspace_root

import os
import re
from typing import Any, Mapping

from habitat_contract.navigation import NavigationBackend, NavigationSession

from supernav.methods.navigation.oracle_local_nav import _call_grounding_service, navigate_oracle_local
from supernav.methods.navigation.oracle_local_nav.config import (
    _DEFAULT_GROUNDING_SCORE_THRESHOLD,
    _DEFAULT_GROUNDING_TIMEOUT_S,
    _DEFAULT_GROUNDING_TOP_K,
    _DEFAULT_GROUNDING_URL,
    _STOPWORDS,
    _VISUAL_PHRASE_SYNONYMS,
)
from supernav.methods.navigation.oracle_local_nav.grounding_client import _ground_detections_for_panorama
from supernav.methods.navigation.oracle_local_nav.match_evaluation import (
    _best_detection_projection_match,
    _visual_candidate_table,
    _write_visual_grounding_candidate_artifact,
)
from supernav.methods.navigation.oracle_local_nav.scene_graph import _label_match_score
from supernav.methods.navigation.oracle_local_nav.utils import _nonnegative_int, _positive_float, _public_detection


def navigate_visual_local(
    adapter: NavigationBackend,
    session_id: str | None,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Bridge action handler for visually grounded local navigation.

    The caller supplies only a visual phrase. By default this handler only uses
    the current front view, so a side/back panorama object cannot pull the agent
    away from the area it is facing. If the instruction explicitly asks for a
    relative side/back direction, non-front panorama views are allowed. The
    handler first tries to bind that phrase to an internally projected visible
    scene-graph object. If no conservative label match exists, it falls back to
    Grounding DINO, binds the detection box to a projected object, then delegates
    to the bounded oracle local navigator with the matched private target_ref.
    """

    session = adapter.navigation_session(session_id)
    adapter.navigation_pathfinder(session)

    instruction = str(payload.get("instruction") or "").strip()
    if not instruction:
        return _visual_grounding_failure(
            status="no_visual_grounding",
            error="visual local navigation requires a non-empty instruction",
            query=instruction,
        )

    panorama = _latest_or_fresh_panorama(adapter, session, payload)
    if not panorama:
        return _visual_grounding_failure(
            status="no_visual_grounding",
            error="no panorama image is available for visual local navigation",
            query=instruction,
        )

    projected_rows = _projected_rows(getattr(session, "last_visible_nav_targets", []) or [])
    if not projected_rows:
        return _visual_grounding_failure(
            status="not_in_front_view",
            error=(
                "no projected scene-graph objects are visible in the current front view; "
                "turn toward the target and retry"
            ),
            query=instruction,
        )
    allow_non_front = _instruction_allows_non_front(instruction)
    candidate_rows = projected_rows if allow_non_front else _front_projected_rows(projected_rows)
    candidate_panorama = panorama
    if not allow_non_front:
        front_image = _front_panorama_image(panorama)
        candidate_panorama = [front_image] if front_image is not None else []
    if not candidate_rows:
        return _visual_grounding_failure(
            status="not_in_front_view",
            error=(
                "no projected scene-graph objects matching the requested direction are "
                "visible in the current front view; turn toward the target and retry"
            ),
            query=instruction,
        )
    if not candidate_panorama:
        return _visual_grounding_failure(
            status="no_visual_grounding",
            error="no front panorama image is available for visual local navigation",
            query=instruction,
        )
    scene_graph = getattr(session, "scene_graph", None)
    if not isinstance(scene_graph, Mapping):
        scene_graph = None

    grounding_url = str(
        payload.get("grounding_url")
        or os.environ.get("NAV_GROUNDING_URL")
        or _DEFAULT_GROUNDING_URL
    )
    output_dir = str(payload.get("output_dir") or str(workspace_root() / "data" / "runs" / "artifacts"))
    score_threshold = _positive_float(
        payload.get("grounding_score_threshold")
        or os.environ.get("NAV_GROUNDING_SCORE_THRESHOLD"),
        default=_DEFAULT_GROUNDING_SCORE_THRESHOLD,
    )
    timeout_s = _positive_float(
        payload.get("grounding_timeout_s")
        or os.environ.get("NAV_GROUNDING_TIMEOUT_S"),
        default=_DEFAULT_GROUNDING_TIMEOUT_S,
    )
    top_k = _nonnegative_int(
        payload.get("grounding_top_k") or os.environ.get("NAV_GROUNDING_TOP_K"),
        default=_DEFAULT_GROUNDING_TOP_K,
    )
    phrases = _visual_grounding_phrases(instruction)
    visible_match = _best_visible_object_label_match(
        instruction=instruction,
        projected_rows=candidate_rows,
        phrases=phrases,
    )
    if visible_match is not None:
        target_ref = str(visible_match.get("target_ref") or "").strip()
        if target_ref:
            return _navigate_to_projected_target(
                adapter=adapter,
                session=session,
                payload=payload,
                output_dir=output_dir,
                target_ref=target_ref,
                projected=visible_match,
                target_source="visible_object_label_match",
                debug_extra={
                    "visible_object_match_score": visible_match.get("_label_match_score"),
                    "visible_object_match_phrase": visible_match.get("_matched_phrase"),
                },
            )

    detections = _ground_detections_for_panorama(
        images=candidate_panorama,
        phrases=phrases,
        grounding_url=grounding_url,
        score_threshold=score_threshold,
        timeout_s=timeout_s,
    )
    direction_count = len({
        str(image.get("direction") or "").lower()
        for image in candidate_panorama
        if isinstance(image, Mapping) and str(image.get("direction") or "").strip()
    })
    match_top_k = top_k * max(1, direction_count) if top_k > 0 else top_k
    match = _best_detection_projection_match(
        detections,
        candidate_rows,
        query=instruction,
        top_k=match_top_k,
        scene_graph=scene_graph,
    )

    candidate_table = match[3] if match is not None and len(match) > 3 else []
    if match is None and detections:
        candidate_table = _visual_candidate_table(
            detections=detections,
            projected_rows=candidate_rows,
            query=instruction,
            top_k=match_top_k,
            scene_graph=scene_graph,
        )
    debug_artifact = _write_visual_grounding_candidate_artifact(
        output_dir=output_dir,
        query=instruction,
        detections=detections,
        candidate_table=candidate_table,
    )

    if not detections:
        return _visual_grounding_failure(
            status="no_visual_grounding",
            error=(
                "Grounding DINO returned no valid detection in the selected view; "
                "turn toward the target and retry"
            ),
            query=instruction,
            debug_artifact=debug_artifact,
        )

    if match is None:
        return _visual_grounding_failure(
            status="not_in_front_view",
            error=(
                "visual detections in the selected view did not overlap any "
                "projected scene-graph object; turn toward the target and retry"
            ),
            query=instruction,
            detections=[_public_detection(det) for det in detections],
            debug_artifact=debug_artifact,
        )

    detection, projected, _match_score, _candidate_table = match
    target_ref = str(projected.get("target_ref") or "").strip()
    if not target_ref:
        return _visual_grounding_failure(
            status="no_scene_graph_projection_match",
            error="matched projected object did not provide an internal target ref",
            query=instruction,
            debug_artifact=debug_artifact,
        )

    body = _navigate_to_projected_target(
        adapter=adapter,
        session=session,
        payload=payload,
        output_dir=output_dir,
        target_ref=target_ref,
        projected=projected,
        target_source="visual_grounding",
        debug_extra={
            "visual_match_score": _match_score,
            "grounding_source": detection.get("grounding_source"),
            "grounding_query_phrase": detection.get("query_phrase"),
            "visual_grounding_candidate_count": len(candidate_table),
        },
    )
    body["visual_grounding"] = _public_detection(detection)
    body.setdefault("_debug", {})
    if isinstance(body["_debug"], dict) and debug_artifact:
        body["_debug"]["visual_grounding_candidates_path"] = debug_artifact
    return body


_NON_FRONT_DIRECTION_RE = re.compile(
    r"\b(?:to|on|at|toward|towards|from|off)\s+"
    r"(?:my\s+|the\s+)?(?:left|right|back|rear)\b"
    r"|\bbehind(?:\s+me)?\b"
    r"|\b(?:left|right|back|rear)\s+(?:side|view)\b",
    re.IGNORECASE,
)


def _instruction_allows_non_front(instruction: str) -> bool:
    """Return whether a visual-local instruction explicitly asks off-front."""
    return bool(_NON_FRONT_DIRECTION_RE.search(str(instruction or "")))


def _navigate_to_projected_target(
    *,
    adapter: NavigationBackend,
    session: NavigationSession,
    payload: Mapping[str, Any],
    output_dir: str,
    target_ref: str,
    projected: Mapping[str, Any],
    target_source: str,
    debug_extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    turn_result = _turn_toward_projected_direction(adapter, session, projected)
    nav_payload = {
        "target_ref": target_ref,
        "max_steps": int(payload.get("max_steps", 40)),
        "horizon_m": _positive_float(payload.get("horizon_m"), default=3.0),
        "goal_radius": _positive_float(payload.get("goal_radius"), default=0.6),
        "standoff_m": _positive_float(payload.get("standoff_m"), default=0.7),
        "output_dir": output_dir,
    }
    result = navigate_oracle_local(adapter, session.session_id, nav_payload)
    body = dict(result) if isinstance(result, Mapping) else {}
    body["backend"] = "visual_local_grounded"
    body["target_source"] = target_source
    body.pop("matched_target_ref", None)
    body.pop("matched_label", None)
    body.setdefault("_debug", {})
    if isinstance(body["_debug"], dict):
        body["_debug"]["matched_target_ref"] = target_ref
        body["_debug"]["matched_projection"] = projected
        if turn_result is not None:
            body["_debug"]["pre_navigation_turn"] = turn_result
        for key, value in dict(debug_extra or {}).items():
            if value is not None:
                body["_debug"][key] = value
    return body


def _visual_grounding_failure(
    *,
    status: str,
    error: str,
    query: str,
    detections: list[dict[str, Any]] | None = None,
    debug_artifact: str | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "ok": False,
        "backend": "visual_local_grounded",
        "status": status,
        "target_source": "visual_grounding",
        "query": query,
        "error": error,
    }
    if detections is not None:
        body["visual_detections"] = detections
    if debug_artifact:
        body["_debug"] = {"visual_grounding_candidates_path": debug_artifact}
    return body


def _front_panorama_image(
    panorama: list[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Return the front-direction row from a panorama capture, if present."""
    for row in panorama:
        if isinstance(row, Mapping) and str(row.get("direction") or "").lower() == "front":
            return row
    return None


def _front_projected_rows(
    projected_rows: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Keep only scene-graph projections from the front view."""
    return [
        row
        for row in projected_rows
        if isinstance(row, Mapping)
        and str(row.get("direction") or "").lower() == "front"
    ]


def _projected_rows(
    projected_rows: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    return [
        row
        for row in projected_rows
        if isinstance(row, Mapping) and str(row.get("target_ref") or "").strip()
    ]


def _best_visible_object_label_match(
    *,
    instruction: str,
    projected_rows: list[Mapping[str, Any]],
    phrases: list[str],
) -> Mapping[str, Any] | None:
    candidates: list[dict[str, Any]] = []
    query_phrases = [instruction, *phrases]
    for row in projected_rows:
        label = str(row.get("label") or row.get("category") or "").strip()
        if not label:
            continue
        best_score = 0.0
        best_phrase = ""
        for phrase in query_phrases:
            score = _label_match_score(label, target_label=phrase, instruction="")
            if score > best_score:
                best_score = score
                best_phrase = phrase
        if best_score < 2.0:
            continue
        item = dict(row)
        item["_label_match_score"] = best_score
        item["_matched_phrase"] = best_phrase
        candidates.append(item)
    if not candidates:
        return None
    candidates.sort(
        key=lambda row: (
            float(row.get("_label_match_score") or 0.0),
            _direction_priority(row.get("direction")),
            float(row.get("visible_fraction") or 0.0),
            -_depth_sort_value(row.get("depth_m")),
        ),
        reverse=True,
    )
    return candidates[0]


def _direction_priority(direction: Any) -> float:
    direction_text = str(direction or "").lower()
    if direction_text == "front":
        return 4.0
    if direction_text in {"right", "left"}:
        return 3.0
    if direction_text == "back":
        return 2.0
    return 0.0


def _depth_sort_value(value: Any) -> float:
    try:
        depth = float(value)
    except (TypeError, ValueError):
        return 1_000_000.0
    if depth <= 0.0:
        return 1_000_000.0
    return depth


def _turn_toward_projected_direction(
    adapter: NavigationBackend,
    session: NavigationSession,
    projected: Mapping[str, Any],
) -> dict[str, Any] | None:
    direction = str(projected.get("direction") or "").lower()
    turn: tuple[str, float] | None
    if direction in ("", "front"):
        turn = None
    elif direction == "right":
        turn = ("turn_right", 90.0)
    elif direction == "back":
        turn = ("turn_right", 180.0)
    elif direction == "left":
        turn = ("turn_left", 90.0)
    else:
        turn = None
    if turn is None:
        return None
    action, degrees = turn
    result = adapter.step_action(
        session.session_id,
        {"action": action, "degrees": degrees, "include_observation_data": False},
    )
    return {
        "direction": direction,
        "action": action,
        "degrees": degrees,
        "steps_taken": result.get("steps_taken") if isinstance(result, Mapping) else None,
        "collided": result.get("collided") if isinstance(result, Mapping) else None,
    }


def _latest_or_fresh_panorama(
    adapter: NavigationBackend,
    session: NavigationSession,
    payload: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    latest = getattr(session, "last_panorama_images", None)
    if isinstance(latest, list) and any(
        isinstance(row, Mapping) and row.get("path") for row in latest
    ):
        return [row for row in latest if isinstance(row, Mapping)]
    result = adapter.get_panorama(
        session.session_id,
        {
            "include_depth_analysis": False,
            "output_dir": str(payload.get("output_dir") or str(workspace_root() / "data" / "runs" / "artifacts")),
            "agent_image_max_size": int(payload.get("agent_image_max_size", 256)),
        },
    )
    images = result.get("images") if isinstance(result, Mapping) else None
    return [row for row in images or [] if isinstance(row, Mapping)]


def _visual_grounding_phrases(instruction: str) -> list[str]:
    raw = str(instruction or "").strip()
    cleaned = _clean_visual_phrase(raw)
    phrases: list[str] = []

    def add(value: str) -> None:
        phrase = " ".join(str(value or "").strip().split())
        if not phrase:
            return
        if phrase.lower() not in {item.lower() for item in phrases}:
            phrases.append(phrase)

    add(cleaned or raw)
    if cleaned:
        add(_article_phrase(cleaned))
    lowered = f" {cleaned.lower() or raw.lower()} "
    for key, synonyms in _VISUAL_PHRASE_SYNONYMS.items():
        if f" {key} " not in lowered:
            continue
        for synonym in synonyms:
            add(synonym)
            add(_article_phrase(synonym))
    if raw and len(raw.split()) <= 5:
        add(raw)
    return phrases[:10]


def _clean_visual_phrase(value: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", value.lower())
    kept = [word for word in words if word not in _STOPWORDS]
    return " ".join(kept[:4])


def _article_phrase(value: str) -> str:
    phrase = str(value or "").strip()
    if not phrase:
        return ""
    if phrase.endswith("."):
        return phrase
    lowered = phrase.lower()
    if lowered.startswith(("a ", "an ", "the ")):
        return f"{phrase}."
    article = "an" if lowered[0] in "aeiou" else "a"
    return f"{article} {phrase}."
