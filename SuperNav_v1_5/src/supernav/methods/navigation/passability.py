"""Pure depth evidence judgments; MCP session state stays in the server."""
from __future__ import annotations
from typing import Any, Dict, List, Optional

def _mcp_float_or_none(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mcp_grid_cells(evidence: Dict[str, Any]) -> List[Dict[str, Any]]:
    body = evidence.get("body")
    if not isinstance(body, dict):
        return []
    grid = body.get("grid")
    if not isinstance(grid, list):
        return []
    cells: List[Dict[str, Any]] = []
    for row in grid:
        if not isinstance(row, list):
            continue
        for cell in row:
            if isinstance(cell, dict):
                cells.append(cell)
    return cells


def _mcp_depth_grid_judgment(
    evidence: Dict[str, Any],
    *,
    distance_m: float,
) -> Dict[str, Any]:
    body = evidence.get("body") if isinstance(evidence, dict) else None
    if not isinstance(body, dict):
        return {
            "status": "uncertain",
            "risk_reason": ["missing_depth_grid"],
            "suggested_next_tool": "hab_depth_grid",
        }
    rows = int(body.get("rows") or 0)
    cols = int(body.get("cols") or 0)
    cells = _mcp_grid_cells(evidence)
    if rows <= 0 or cols <= 0 or not cells:
        return {
            "status": "uncertain",
            "risk_reason": ["invalid_depth_grid"],
            "suggested_next_tool": "hab_depth_grid",
        }

    center = cols // 2
    corridor_cols = {center}
    if cols >= 5:
        corridor_cols.update({max(0, center - 1), min(cols - 1, center + 1)})
    bottom_start = max(0, rows - 2)
    corridor = [
        cell
        for cell in cells
        if int(cell.get("col", -1)) in corridor_cols
        and int(cell.get("row", -1)) >= bottom_start
    ]
    if not corridor:
        return {
            "status": "uncertain",
            "risk_reason": ["missing_bottom_corridor_cells"],
            "suggested_next_tool": "hab_depth_grid",
        }

    min_values = [
        value
        for value in (_mcp_float_or_none(cell.get("min_dist")) for cell in corridor)
        if value is not None
    ]
    if not min_values:
        return {
            "status": "uncertain",
            "risk_reason": ["no_valid_depth_in_corridor"],
            "suggested_next_tool": "hab_depth_grid",
        }

    clearance = _mcp_float_or_none(body.get("clearance_threshold")) or 0.5
    required_clearance = max(clearance, float(distance_m) + 0.1)
    blocked_cells = [
        cell
        for cell in corridor
        if cell.get("clear") is False
        or (
            _mcp_float_or_none(cell.get("min_dist")) is not None
            and float(_mcp_float_or_none(cell.get("min_dist")) or 0.0)
            < required_clearance
        )
    ]
    min_corridor = min(min_values)
    evidence_summary = {
        "center_corridor_min_m": round(min_corridor, 3),
        "required_clearance_m": round(required_clearance, 3),
        "corridor_cell_count": len(corridor),
        "blocked_corridor_cell_count": len(blocked_cells),
    }
    if blocked_cells:
        return {
            "status": "blocked",
            "risk_reason": ["bottom_corridor_blocked", "insufficient_clearance"],
            "suggested_next_tool": "hab_side_depth_grid",
            "evidence_summary": evidence_summary,
        }
    return {
        "status": "passable_hint",
        "risk_reason": [],
        "suggested_next_tool": "hab_forward",
        "evidence_summary": evidence_summary,
    }


def _mcp_depth_analyze_judgment(
    evidence: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    body = evidence.get("body") if isinstance(evidence, dict) else None
    if not isinstance(body, dict):
        return {
            "status": "uncertain",
            "risk_reason": ["missing_depth_grid"],
            "suggested_next_tool": "hab_depth_grid",
        }
    center = body.get("front_center")
    if not isinstance(center, dict):
        return {
            "status": "uncertain",
            "risk_reason": ["missing_front_center_depth"],
            "suggested_next_tool": "hab_depth_grid",
        }
    if center.get("clear") is not True:
        return {
            "status": "blocked",
            "risk_reason": ["front_center_blocked"],
            "suggested_next_tool": "hab_side_depth_grid",
        }
    return {
        "status": "uncertain",
        "risk_reason": ["coarse_depth_only"],
        "suggested_next_tool": "hab_depth_grid",
    }
