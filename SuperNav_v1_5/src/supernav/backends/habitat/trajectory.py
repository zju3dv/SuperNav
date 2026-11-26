"""Automatic MCP trajectory sidecar logging.

The MCP server is a thin public control surface over the bridge. This
module keeps a durable JSON sidecar for each MCP simulator session so
metric scripts can read start pose, pose samples, and final pose after
the session has been closed.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_SCHEMA_VERSION = 2
_MOVING_TOOLS = frozenset(
    {
        "backward",
        "forward",
        "turn",
        "navigate",
        "visual_local_navigate",
        "visual_ground_preview",
        "visual_overlay_navigate",
        "visual_point_navigate",
        "oracle_local_navigate",
        "reset_agent_pose",
        "look",
        "look_vertical",
        "panorama",
    }
)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _pose_vec(
    pose: Optional[Mapping[str, Any]],
    key: str,
    length: int,
) -> Optional[list[float]]:
    if not pose:
        return None
    return _coerce_vec(pose.get(key), length)


def _coerce_vec(value: Any, length: int) -> Optional[list[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return None
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def extract_pose(body: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract a pose from a bridge/tool response.

    Bridge movement responses usually expose pose under
    ``metrics.agent_state``. Some responses expose
    ``agent_state`` directly, and navmesh-only summaries expose
    ``state_summary.position``. Return ``None`` when the response does
    not carry an absolute position, which is expected in mapless mode.
    """
    candidates: list[Mapping[str, Any]] = []
    heading_candidates: list[Any] = []
    metrics = body.get("metrics")
    if isinstance(metrics, Mapping):
        agent_state = metrics.get("agent_state")
        if isinstance(agent_state, Mapping):
            candidates.append(agent_state)
        state_summary = metrics.get("state_summary")
        if isinstance(state_summary, Mapping):
            candidates.append(state_summary)
            heading_candidates.append(state_summary.get("heading_deg"))
    agent_state = body.get("agent_state")
    if isinstance(agent_state, Mapping):
        candidates.append(agent_state)
    state_summary = body.get("state_summary")
    if isinstance(state_summary, Mapping):
        candidates.append(state_summary)
        heading_candidates.append(state_summary.get("heading_deg"))

    final_position = _coerce_vec(body.get("final_position"), 3)
    if final_position is not None:
        final_pose: Dict[str, Any] = {"position": final_position}
        final_rotation = _coerce_vec(body.get("final_quat_wxyz"), 4)
        if final_rotation is not None:
            final_pose["rotation"] = final_rotation
        heading = body.get("final_heading_deg", body.get("final_yaw_deg"))
        if heading is not None:
            try:
                final_pose["heading_deg"] = float(heading)
            except (TypeError, ValueError):
                pass
        candidates.append(final_pose)

    for candidate in candidates:
        position = _coerce_vec(candidate.get("position"), 3)
        if position is None:
            continue
        pose: Dict[str, Any] = {"position": position}
        rotation = _coerce_vec(candidate.get("rotation"), 4)
        if rotation is not None:
            pose["rotation"] = rotation
        heading = candidate.get("heading_deg")
        if heading is None:
            heading = next(
                (value for value in heading_candidates if value is not None), None
            )
        if heading is not None:
            try:
                pose["heading_deg"] = float(heading)
            except (TypeError, ValueError):
                pass
        return pose
    return None


def extract_step_count(body: Mapping[str, Any]) -> Optional[int]:
    """Return the authoritative cumulative simulator step count when present."""
    candidates: list[Any] = [body.get("step_count")]
    metrics = body.get("metrics")
    if isinstance(metrics, Mapping):
        candidates.append(metrics.get("step_count"))
    for value in candidates:
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return None


def extract_tool_steps(body: Mapping[str, Any]) -> Optional[int]:
    for key in ("steps_executed", "steps_taken", "steps"):
        value = body.get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return int(value)
    return None


def _pose_json(pose: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    if not pose:
        return None
    position = _pose_vec(pose, "position", 3)
    if position is None:
        return None
    result: Dict[str, Any] = {"position": position}
    rotation = _pose_vec(pose, "rotation", 4)
    if rotation is not None:
        # Keep ``rotation`` for schema-v1 readers while naming the convention.
        result["rotation"] = rotation
        result["rotation_wxyz"] = rotation
    heading = pose.get("heading_deg")
    if heading is not None:
        try:
            result["heading_deg"] = float(heading)
        except (TypeError, ValueError):
            pass
    return result


def extract_trajectory_points(body: Mapping[str, Any]) -> list[list[float]]:
    raw = body.get("trajectory_points")
    if not isinstance(raw, list):
        return []
    points: list[list[float]] = []
    for item in raw:
        point = _coerce_vec(item, 3)
        if point is not None:
            points.append(point)
    return points


class McpTrajectoryLog:
    """Read/modify/write helper for one MCP session trajectory JSON."""

    def __init__(self, artifacts_dir: str | Path) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self._paths_by_session: Dict[str, Path] = {}

    def path_for_session(self, session_id: str) -> Path:
        path = self._paths_by_session.get(session_id)
        if path is not None:
            return path
        safe_session = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in session_id
        )
        path = self.artifacts_dir / f"{safe_session}.trajectory.json"
        self._paths_by_session[session_id] = path
        return path

    def start(
        self,
        *,
        session_id: str,
        scene: Optional[str],
        pose: Optional[Mapping[str, Any]],
        source_tool: str,
        source_response: Optional[Mapping[str, Any]] = None,
        tool_seq: Optional[int] = None,
        captured_at: Optional[str] = None,
        step_count: Optional[int] = None,
        trajectory_points: Optional[list[list[float]]] = None,
        diagnostic: Optional[str] = None,
    ) -> Optional[Path]:
        if not session_id:
            return None
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        path = self.path_for_session(session_id)
        started_at = captured_at or now_utc()
        safe_pose = _pose_json(pose)
        dense_points = _json_safe(trajectory_points or [])
        dense_end = len(dense_points)
        doc: Dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "session_id": session_id,
            "scene": scene,
            "scene_id": scene,
            "started_at": started_at,
            "ended_at": None,
            "status": "running",
            "coordinate_system": "habitat_world",
            "capture_status": "running",
            "capture_issues": [],
            "capture_diagnostics": [],
            "start_pose": safe_pose,
            "start_position": _json_safe(_pose_vec(safe_pose, "position", 3)),
            "start_rotation": _json_safe(_pose_vec(safe_pose, "rotation", 4)),
            "end_pose": None,
            "final_position": None,
            "final_rotation": None,
            "trajectory": [],
            "events": [],
            "endpoint_samples": [],
            "dense_trajectory": {
                "points": dense_points,
                "reveals": (
                    [
                        {
                            "start_index": 0,
                            "end_index": dense_end,
                            "tool_seq": tool_seq,
                            "captured_at": started_at,
                            "sim_step_count": step_count,
                            "source": "init_audit",
                        }
                    ]
                    if dense_end
                    else []
                ),
                "integrity": "running",
            },
        }
        if diagnostic:
            doc["capture_diagnostics"].append(
                {"phase": "start", "captured_at": started_at, "detail": diagnostic}
            )
            if "failed" in diagnostic or "unavailable" in diagnostic:
                doc["capture_issues"].append("start_dense_audit_unavailable")
        if safe_pose:
            sample = {
                "index": 0,
                "event": "start",
                "ts": started_at,
                "tool": source_tool,
                "tool_seq": tool_seq,
                "step_count": step_count,
                "dense_range": [0, dense_end],
                "pose": safe_pose,
            }
            doc["trajectory"].append(sample)
            doc["events"].append(sample)
            doc["start_sample"] = dict(sample)
        else:
            doc["events"].append(
                {
                    "index": 0,
                    "event": "start_pose_unavailable",
                    "ts": started_at,
                    "tool": source_tool,
                    "tool_seq": tool_seq,
                    "diagnostic": diagnostic,
                }
            )
            doc["capture_issues"].append("start_pose_unavailable")
        if source_response is not None:
            for key in ("episode_id", "task_type", "scene_id"):
                value = source_response.get(key)
                if value is not None:
                    doc[key] = _json_safe(value)
            doc["source"] = {
                "init_scene": {
                    "is_gaussian": bool(source_response.get("is_gaussian", False)),
                }
            }
        self._write(path, doc)
        return path

    def sample(
        self,
        *,
        session_id: str,
        tool_name: str,
        pose: Optional[Mapping[str, Any]],
        body: Mapping[str, Any],
        tool_seq: Optional[int] = None,
        captured_at: Optional[str] = None,
        trajectory_points: Optional[list[list[float]]] = None,
        step_count: Optional[int] = None,
        diagnostic: Optional[str] = None,
    ) -> Optional[Path]:
        if not session_id or tool_name not in _MOVING_TOOLS:
            return None
        path = self.path_for_session(session_id)
        if not path.is_file():
            return None
        doc = self._read(path, session_id=session_id)
        safe_pose = _pose_json(pose)
        event = "pose_sample" if safe_pose else "pose_unavailable"
        ts = captured_at or now_utc()
        dense_range = self._merge_dense_prefix(
            doc,
            trajectory_points or [],
            tool_seq=tool_seq,
            captured_at=ts,
            sim_step_count=(
                step_count if step_count is not None else extract_step_count(body)
            ),
            source="endpoint_audit",
        )
        sample: Dict[str, Any] = {
            "index": len(doc.get("events", [])),
            "event": event,
            "ts": ts,
            "tool": tool_name,
            "tool_seq": tool_seq,
            "step_count": (
                step_count if step_count is not None else extract_step_count(body)
            ),
            "tool_steps": extract_tool_steps(body),
            "dense_range": list(dense_range),
            "result_status": body.get("status"),
            "action": body.get("action"),
            "collided": body.get("collided"),
            "diagnostic": diagnostic,
        }
        if safe_pose:
            sample["pose"] = safe_pose
            doc.setdefault("trajectory", []).append(sample)
        else:
            doc.setdefault("capture_issues", []).append(
                f"endpoint_pose_unavailable:{tool_seq}:{tool_name}"
            )
        doc.setdefault("endpoint_samples", []).append(sample)
        if diagnostic:
            doc.setdefault("capture_diagnostics", []).append(
                {
                    "phase": "endpoint",
                    "tool_seq": tool_seq,
                    "tool": tool_name,
                    "captured_at": ts,
                    "detail": diagnostic,
                }
            )
            if "failed" in diagnostic or "unavailable" in diagnostic:
                doc.setdefault("capture_issues", []).append(
                    f"dense_audit_unavailable:{tool_seq}:{tool_name}"
                )
        doc.setdefault("events", []).append(sample)
        self._write(path, doc)
        return path

    def finish(
        self,
        *,
        session_id: str,
        pose: Optional[Mapping[str, Any]],
        reason: str,
        status: str = "closed",
        trajectory_points: Optional[list[list[float]]] = None,
        tool_seq: Optional[int] = None,
        captured_at: Optional[str] = None,
        step_count: Optional[int] = None,
        diagnostic: Optional[str] = None,
    ) -> Optional[Path]:
        if not session_id:
            return None
        path = self.path_for_session(session_id)
        if not path.is_file():
            return None
        doc = self._read(path, session_id=session_id)
        ts = captured_at or now_utc()
        safe_pose = _pose_json(pose)
        dense_range = self._merge_dense_prefix(
            doc,
            trajectory_points or [],
            tool_seq=tool_seq,
            captured_at=ts,
            sim_step_count=step_count,
            source="close_audit",
        )
        event: Dict[str, Any] = {
            "index": len(doc.get("events", [])),
            "event": "end" if safe_pose else "end_pose_unavailable",
            "ts": ts,
            "tool": "close_session",
            "tool_seq": tool_seq,
            "step_count": step_count,
            "dense_range": list(dense_range),
            "reason": reason,
            "diagnostic": diagnostic,
        }
        if safe_pose:
            event["pose"] = safe_pose
            doc["end_pose"] = safe_pose
            doc["final_position"] = _json_safe(_pose_vec(pose, "position", 3))
            doc["final_rotation"] = _json_safe(_pose_vec(pose, "rotation", 4))
            doc.setdefault("trajectory", []).append(event)
        doc["ended_at"] = ts
        doc["status"] = status
        doc["end_sample"] = dict(event)
        doc["trajectory_points"] = list(
            doc.get("dense_trajectory", {}).get("points", [])
        )
        dense = doc.setdefault("dense_trajectory", {})
        dense["close_point_count"] = len(doc["trajectory_points"])
        issues = list(dict.fromkeys(str(x) for x in doc.get("capture_issues", [])))
        if not safe_pose:
            issues.append("end_pose_unavailable")
        if diagnostic:
            doc.setdefault("capture_diagnostics", []).append(
                {"phase": "close", "captured_at": ts, "detail": diagnostic}
            )
            if "failed" in diagnostic or "unavailable" in diagnostic:
                issues.append("dense_audit_unavailable:close")
        doc["capture_issues"] = list(dict.fromkeys(issues))
        doc["capture_status"] = "complete" if not doc["capture_issues"] else "partial"
        dense["integrity"] = (
            "complete"
            if not any("dense" in issue for issue in doc["capture_issues"])
            else "partial"
        )
        doc.setdefault("events", []).append(event)
        self._write(path, doc)
        return path

    @staticmethod
    def _merge_dense_prefix(
        doc: Dict[str, Any],
        points: list[list[float]],
        *,
        tool_seq: Optional[int],
        captured_at: str,
        sim_step_count: Optional[int],
        source: str,
    ) -> tuple[int, int]:
        dense = doc.setdefault(
            "dense_trajectory", {"points": [], "reveals": [], "integrity": "running"}
        )
        old = list(dense.get("points") or [])
        new = [_coerce_vec(point, 3) for point in points]
        new = [point for point in new if point is not None]
        if not new:
            return len(old), len(old)
        prefix_matches = len(new) >= len(old) and all(
            all(abs(float(a) - float(b)) <= 1e-5 for a, b in zip(left, right))
            for left, right in zip(old, new[: len(old)])
        )
        if not prefix_matches:
            doc.setdefault("capture_issues", []).append(
                f"dense_prefix_discontinuity:{tool_seq}"
            )
            dense["integrity"] = "partial"
            # Do not fabricate a segment across a reset or teleport.
            return len(old), len(old)
        start_index = len(old)
        end_index = len(new)
        if end_index > start_index:
            dense["points"] = _json_safe(new)
            dense.setdefault("reveals", []).append(
                {
                    "start_index": start_index,
                    "end_index": end_index,
                    "tool_seq": tool_seq,
                    "captured_at": captured_at,
                    "sim_step_count": sim_step_count,
                    "source": source,
                }
            )
        return start_index, end_index

    def mark_completion(
        self,
        *,
        session_id: str,
        status: str,
        reason: str = "",
        confidence: Optional[float] = None,
        pose: Optional[Mapping[str, Any]] = None,
        body: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Path]:
        """Record the agent's explicit task-completion declaration.

        This is intentionally separate from ``finish()``: a benchmark
        agent can say "I am done" before the session is closed, and the
        harness can evaluate the pose at that declaration point.
        """
        if not session_id:
            return None
        path = self.path_for_session(session_id)
        if not path.is_file():
            return None
        doc = self._read(path, session_id=session_id)
        ts = now_utc()
        event: Dict[str, Any] = {
            "index": len(doc.get("events", [])),
            "event": "completion",
            "ts": ts,
            "tool": "mark_completion",
            "completion_status": status,
            "reason": reason,
        }
        if confidence is not None:
            event["confidence"] = float(confidence)
        if pose:
            safe_pose = _json_safe(dict(pose))
            event["pose"] = safe_pose
            doc["completion_pose"] = safe_pose
            doc["completion_position"] = _json_safe(_pose_vec(pose, "position", 3))
            doc["completion_rotation"] = _json_safe(_pose_vec(pose, "rotation", 4))
            doc.setdefault("trajectory", []).append(event)
        if body:
            event["body"] = _json_safe(dict(body))
        doc["completion_status"] = status
        doc["completion_reason"] = reason
        doc["completion_confidence"] = confidence
        doc["completion_at"] = ts
        doc.setdefault("events", []).append(event)
        self._write(path, doc)
        return path

    def mark_goal(
        self,
        *,
        session_id: str,
        target_index: int,
        pose: Optional[Mapping[str, Any]] = None,
        body: Optional[Mapping[str, Any]] = None,
    ) -> Optional[Path]:
        """Record a multi-goal episode's per-goal arrival declaration.

        Analogous to ``mark_completion()`` but fired once per goal: the
        agent declares "goal N reached" mid-episode and offline scoring
        evaluates the pose at each declaration point against the
        manifest's per-goal ground truth.
        """
        if not session_id:
            return None
        path = self.path_for_session(session_id)
        if not path.is_file():
            return None
        doc = self._read(path, session_id=session_id)
        ts = now_utc()
        event: Dict[str, Any] = {
            "index": len(doc.get("events", [])),
            "event": "goal_marked",
            "ts": ts,
            "tool": "nav_goals",
            "target_index": int(target_index),
        }
        if pose:
            safe_pose = _json_safe(dict(pose))
            event["pose"] = safe_pose
        if body:
            event["body"] = _json_safe(dict(body))
        mark_entry: Dict[str, Any] = {"target_index": int(target_index), "ts": ts}
        if pose:
            mark_entry["pose"] = _json_safe(dict(pose))
            position = _pose_vec(pose, "position", 3)
            if position is not None:
                mark_entry["position"] = _json_safe(position)
        doc.setdefault("goal_marks", []).append(mark_entry)
        doc.setdefault("events", []).append(event)
        self._write(path, doc)
        return path

    def annotate_body(
        self, body: Dict[str, Any], *, session_id: Optional[str]
    ) -> Dict[str, Any]:
        if session_id:
            body.setdefault("trajectory_log", str(self.path_for_session(session_id)))
        return body

    def _read(self, path: Path, *, session_id: str) -> Dict[str, Any]:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
            except (OSError, json.JSONDecodeError):
                pass
        return {
            "schema_version": _SCHEMA_VERSION,
            "session_id": session_id,
            "scene": None,
            "started_at": None,
            "ended_at": None,
            "status": "running",
            "start_pose": None,
            "end_pose": None,
            "trajectory": [],
            "events": [],
            "endpoint_samples": [],
            "dense_trajectory": {"points": [], "reveals": [], "integrity": "running"},
            "capture_status": "running",
            "capture_issues": [],
            "capture_diagnostics": [],
        }

    @staticmethod
    def _write(path: Path, doc: Mapping[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(
            json.dumps(_json_safe(doc), ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        tmp.replace(path)


__all__ = [
    "McpTrajectoryLog",
    "extract_pose",
    "extract_step_count",
    "extract_tool_steps",
    "extract_trajectory_points",
]
