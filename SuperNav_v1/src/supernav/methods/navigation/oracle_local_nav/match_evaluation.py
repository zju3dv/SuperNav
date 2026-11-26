"""Detection-to-scene-graph-projection matching and scoring."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Mapping

from supernav.methods.navigation.oracle_local_nav.config import (
    _DEFAULT_GROUNDING_TOP_K,
    _SEMANTIC_GROUP_TERMS,
    _SEATING_HARD_REJECT_GROUPS,
)
from supernav.methods.navigation.oracle_local_nav.coords import _object_coordinate_audit
from supernav.methods.navigation.oracle_local_nav.grounding_client import _detection_score, _detection_xyxy, _json_safe_detection
from supernav.methods.navigation.oracle_local_nav.scene_graph import _normalize_text
from supernav.methods.navigation.oracle_local_nav.types import _VisualMatchEvaluation
from supernav.methods.navigation.oracle_local_nav.utils import (
    _bbox_center,
    _bbox_iou,
    _clamp01,
    _debug_bbox,
    _json_safe_value,
    _point_in_bbox,
    _positive_float,
    _projected_row_xyxy,
)

def _best_detection_projection_match(
    detections: list[Mapping[str, Any]],
    projected_rows: list[Mapping[str, Any]],
    *,
    query: str = "",
    top_k: int = _DEFAULT_GROUNDING_TOP_K,
    scene_graph: Mapping[str, Any] | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any], float, list[dict[str, Any]]] | None:
    evaluations = _visual_match_evaluations(
        detections=detections,
        projected_rows=projected_rows,
        query=query,
        top_k=top_k,
    )
    table = _visual_match_candidate_table(evaluations, scene_graph=scene_graph)
    viable = [item for item in evaluations if item.final_score is not None]
    if not viable:
        return None
    best = max(
        viable,
        key=lambda item: (
            float(item.final_score or 0.0),
            float(item.detection.get("score") or 0.0),
            item.geometry_score,
        ),
    )
    for row in table:
        row["chosen"] = row.get("candidate_index") == id(best)
    return best.detection, best.projected, float(best.final_score or 0.0), table

def _visual_candidate_table(
    *,
    detections: list[Mapping[str, Any]],
    projected_rows: list[Mapping[str, Any]],
    query: str,
    top_k: int,
    scene_graph: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    return _visual_match_candidate_table(
        _visual_match_evaluations(
            detections=detections,
            projected_rows=projected_rows,
            query=query,
            top_k=top_k,
        ),
        scene_graph=scene_graph,
    )

def _visual_match_evaluations(
    *,
    detections: list[Mapping[str, Any]],
    projected_rows: list[Mapping[str, Any]],
    query: str,
    top_k: int,
) -> list[_VisualMatchEvaluation]:
    evaluations: list[_VisualMatchEvaluation] = []
    for detection in _top_visual_detections(detections, top_k=top_k):
        det_box = detection.get("box_xyxy")
        if not isinstance(det_box, tuple):
            continue
        det_direction = str(detection.get("direction") or "").lower()
        for row in projected_rows:
            if str(row.get("direction") or "").lower() != det_direction:
                continue
            proj_box = _projected_row_xyxy(row)
            if proj_box is None:
                continue
            geometry_score = _visual_geometry_score(det_box, proj_box)
            if geometry_score <= 0.0:
                continue
            semantic_score, semantic_reject = _visual_semantic_score(
                query=query,
                detection=detection,
                projected=row,
            )
            bbox_penalty, bbox_reject = _visual_bbox_penalty(
                query=query,
                detection=detection,
            )
            reject_reason = semantic_reject or bbox_reject
            final_score: float | None = None
            if reject_reason is None:
                final_score = (
                    0.45 * _clamp01(float(detection.get("score") or 0.0))
                    + 0.35 * geometry_score
                    + 0.20 * semantic_score
                    - bbox_penalty
                )
            evaluations.append(
                _VisualMatchEvaluation(
                    detection=detection,
                    projected=row,
                    geometry_score=geometry_score,
                    semantic_score=semantic_score,
                    bbox_penalty=bbox_penalty,
                    final_score=final_score,
                    reject_reason=reject_reason,
                )
            )
    return evaluations

def _top_visual_detections(
    detections: list[Mapping[str, Any]],
    *,
    top_k: int,
) -> list[Mapping[str, Any]]:
    sorted_detections = sorted(
        detections,
        key=lambda item: float(item.get("score") or 0.0),
        reverse=True,
    )
    if top_k <= 0:
        return sorted_detections
    return sorted_detections[:top_k]

def _visual_geometry_score(
    det_box: tuple[float, float, float, float],
    proj_box: tuple[float, float, float, float],
) -> float:
    iou = _bbox_iou(det_box, proj_box)
    det_center = _bbox_center(det_box)
    proj_center = _bbox_center(proj_box)
    det_inside_proj = _point_in_bbox(det_center, proj_box)
    proj_inside_det = _point_in_bbox(proj_center, det_box)
    if iou <= 0.0 and not det_inside_proj and not proj_inside_det:
        return 0.0
    score = iou
    if det_inside_proj:
        score += 0.35
    if proj_inside_det:
        score += 0.20
    return _clamp01(score)

def _visual_semantic_score(
    *,
    query: str,
    detection: Mapping[str, Any],
    projected: Mapping[str, Any],
) -> tuple[float, str | None]:
    query_text = " ".join(
        str(value or "")
        for value in (
            query,
            detection.get("query_phrase"),
            detection.get("label"),
        )
    )
    label = str(projected.get("label") or projected.get("category") or "")
    query_group = _semantic_group(query_text)
    label_group = _semantic_group(label)
    if query_group is None or label_group is None:
        return 0.0, None
    if query_group == label_group:
        return 1.0, None
    if query_group == "seating" and label_group in _SEATING_HARD_REJECT_GROUPS:
        return 0.0, f"semantic_incompatible:{query_group}_vs_{label_group}"
    if query_group == "opening" and label_group in {"structure", "opening"}:
        return 0.45, None
    if label_group == "opening" and query_group == "structure":
        return 0.35, None
    if query_group == "surface" and label_group in {"decor", "lighting", "small_object"}:
        return 0.0, f"semantic_incompatible:{query_group}_vs_{label_group}"
    return 0.0, None

def _semantic_group(text: str) -> str | None:
    normalized = _normalize_text(text)
    if not normalized:
        return None
    tokens = set(normalized.split())
    for group, terms in _SEMANTIC_GROUP_TERMS.items():
        for term in terms:
            term_tokens = set(_normalize_text(term).split())
            if term_tokens and term_tokens <= tokens:
                return group
    return None

def _visual_bbox_penalty(
    *,
    query: str,
    detection: Mapping[str, Any],
) -> tuple[float, str | None]:
    box = detection.get("box_xyxy")
    if not isinstance(box, tuple):
        return 0.0, None
    image_width = _positive_float(detection.get("image_width"), default=0.0)
    image_height = _positive_float(detection.get("image_height"), default=0.0)
    if image_width <= 0.0 or image_height <= 0.0:
        return 0.0, None
    group = _semantic_group(
        " ".join(str(value or "") for value in (query, detection.get("query_phrase")))
    )
    x0, y0, x1, y1 = box
    width_ratio = max(0.0, x1 - x0) / image_width
    height_ratio = max(0.0, y1 - y0) / image_height
    area_ratio = width_ratio * height_ratio
    edge_margin_x = image_width * 0.02
    edge_margin_y = image_height * 0.02
    touches_vertical_edges = y0 <= edge_margin_y and y1 >= image_height - edge_margin_y
    touches_any_edge = (
        x0 <= edge_margin_x
        or y0 <= edge_margin_y
        or x1 >= image_width - edge_margin_x
        or y1 >= image_height - edge_margin_y
    )
    if (
        group == "seating"
        and touches_vertical_edges
        and height_ratio >= 0.92
        and width_ratio <= 0.75
    ):
        return 1.0, "bbox_reject:full_height_vertical_stripe_for_seating"
    if group == "small_object" and (area_ratio >= 0.35 or touches_any_edge):
        return 0.45, "bbox_reject:oversized_or_edge_touching_small_object"
    if group not in {"opening", None} and touches_any_edge and area_ratio >= 0.65:
        return 0.25, None
    return 0.0, None

def _visual_match_candidate_table(
    evaluations: list[_VisualMatchEvaluation],
    *,
    scene_graph: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in evaluations:
        coordinate_audit = _object_coordinate_audit(
            scene_graph,
            str(item.projected.get("target_ref") or ""),
        )
        row = {
            "candidate_index": id(item),
            "query_phrase": item.detection.get("query_phrase"),
            "direction": item.detection.get("direction"),
            "dino_score": round(float(item.detection.get("score") or 0.0), 4),
            "dino_bbox_px": _debug_bbox(item.detection.get("box_xyxy")),
            "grounding_source": item.detection.get("grounding_source"),
            "scene_target_ref": item.projected.get("target_ref"),
            "scene_label": item.projected.get("label"),
            "scene_bbox_px": item.projected.get("bbox_px"),
            "scene_coordinate_audit": coordinate_audit or None,
            "scene_position_raw_xyz": coordinate_audit.get("raw_xyz"),
            "scene_position_habitat_xyz": coordinate_audit.get("habitat_xyz"),
            "coordinate_transform_applied": coordinate_audit.get(
                "semantic_xy_floor_y_flipped"
            ),
            "geometry_score": round(item.geometry_score, 4),
            "semantic_score": round(item.semantic_score, 4),
            "bbox_penalty": round(item.bbox_penalty, 4),
            "final_score": (
                round(item.final_score, 4)
                if item.final_score is not None
                else None
            ),
            "chosen": False,
            "reject_reason": item.reject_reason,
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            row["final_score"] is not None,
            float(row["final_score"] or -999.0),
            float(row["dino_score"] or 0.0),
        ),
        reverse=True,
    )
    return rows

def _write_visual_grounding_candidate_artifact(
    *,
    output_dir: str,
    query: str,
    detections: list[Mapping[str, Any]],
    candidate_table: list[dict[str, Any]],
) -> str | None:
    try:
        root = Path(output_dir) / "visual_grounding"
        root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            suffix=".json",
            prefix="candidates_",
            dir=str(root),
            mode="w",
            encoding="utf-8",
            delete=False,
        ) as handle:
            json.dump(
                _json_safe_value({
                    "query": query,
                    "detections": [_json_safe_detection(row) for row in detections],
                    "candidate_count": len(candidate_table),
                    "candidates": candidate_table,
                }),
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
            return handle.name
    except Exception:
        return None

