"""Guardrails for model-visible GT leakage in raw LLM call logs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


EVAL_ONLY_KEYS = frozenset(
    {
        "_debug",
        "answer",
        "answer_key",
        "answer_keys",
        "eval_goal_position",
        "eval_goal_positions",
        "failure_classification",
        "final_position",
        "goal_position",
        "goal_positions",
        "ground_truth",
        "oracle",
        "oracle_pass",
        "oracle_reason",
        "oracle_score",
        "oracle_success",
        "reference_path",
        "reference_paths",
        "score",
        "score_metrics",
        "score_table",
        "scores",
        "strict_task_success",
        "target_position",
        "target_positions",
    }
)


def strip_model_visible_eval_fields(
    value: Any, *, allow_pointnav_goal_position: bool = False
) -> Any:
    """Remove eval-only fields before serializing content back to a model."""

    if isinstance(value, dict):
        child_allow_pointnav_goal_position = (
            allow_pointnav_goal_position or value.get("task_type") == "pointnav"
        )
        return {
            key: strip_model_visible_eval_fields(
                child,
                allow_pointnav_goal_position=child_allow_pointnav_goal_position,
            )
            for key, child in value.items()
            if not _is_model_visible_eval_key(str(key))
            or _is_agent_visible_pointnav_goal(
                value,
                str(key),
                allow_nested=allow_pointnav_goal_position,
            )
        }
    if isinstance(value, list):
        return [
            strip_model_visible_eval_fields(
                item,
                allow_pointnav_goal_position=allow_pointnav_goal_position,
            )
            for item in value
        ]
    return value


def model_visible_tool_result_json(
    result: Any, *, include_failure_body: bool = True
) -> str:
    """Serialize a ToolResult for model-visible conversation history."""

    if bool(getattr(result, "ok", False)):
        body = getattr(result, "body", {})
        return json.dumps(
            strip_model_visible_eval_fields(body),
            ensure_ascii=False,
            default=str,
        )

    payload: dict[str, Any] = {}
    body = getattr(result, "body", None)
    if include_failure_body:
        if isinstance(body, dict):
            payload.update(strip_model_visible_eval_fields(body))
        elif body:
            payload["body"] = strip_model_visible_eval_fields(body)
    payload["error"] = (
        getattr(result, "error", None) or payload.get("error") or "unknown error"
    )
    return json.dumps(payload, ensure_ascii=False, default=str)


def _is_model_visible_eval_key(key: str) -> bool:
    return (
        key in EVAL_ONLY_KEYS
        or key.startswith("gt_")
        or key.startswith("oracle_")
        or key.endswith("_geodesic_distance")
    )


def _is_agent_visible_pointnav_goal(
    container: dict[str, Any], key: str, *, allow_nested: bool
) -> bool:
    return key == "goal_position" and (
        (
            container.get("task_type") == "pointnav"
            and isinstance(container.get(key), list)
        )
        or (allow_nested and isinstance(container.get(key), list))
    )


@dataclass(frozen=True)
class GtLeakHit:
    path: str
    key: str
    line_no: int
    value_preview: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "key": self.key,
            "line_no": self.line_no,
            "value_preview": self.value_preview,
        }


@dataclass(frozen=True)
class GtLeakScanResult:
    path: str
    hits: tuple[GtLeakHit, ...]
    lines_scanned: int

    @property
    def hit_count(self) -> int:
        return len(self.hits)

    @property
    def ok(self) -> bool:
        return not self.hits

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "ok": self.ok,
            "hit_count": self.hit_count,
            "lines_scanned": self.lines_scanned,
            "hits": [hit.as_dict() for hit in self.hits],
        }


def scan_raw_llm_calls(path: str | Path) -> GtLeakScanResult:
    raw_path = Path(path)
    hits: list[GtLeakHit] = []
    lines_scanned = 0
    for line_no, line in enumerate(
        raw_path.read_text(encoding="utf-8", errors="replace").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        lines_scanned += 1
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        hits.extend(_scan_value(payload, line_no=line_no, path=f"line{line_no}"))
    return GtLeakScanResult(
        path=str(raw_path),
        hits=tuple(hits),
        lines_scanned=lines_scanned,
    )


def scan_raw_llm_call_paths(paths: Iterable[str | Path]) -> dict[str, Any]:
    results = [scan_raw_llm_calls(path) for path in paths]
    return {
        "ok": all(result.ok for result in results),
        "hit_count": sum(result.hit_count for result in results),
        "files_scanned": len(results),
        "results": [result.as_dict() for result in results],
    }


def _scan_value(
    value: Any,
    *,
    line_no: int,
    path: str,
    seen: set[int] | None = None,
    suppress_agent_authored_keys: bool = False,
    allow_pointnav_goal_position: bool = False,
) -> list[GtLeakHit]:
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return []
    seen.add(value_id)

    hits: list[GtLeakHit] = []
    if isinstance(value, dict):
        child_allow_pointnav_goal_position = (
            allow_pointnav_goal_position or value.get("task_type") == "pointnav"
        )
        if _is_tool_result_payload(value):
            child_suppress_agent_authored_keys = False
        else:
            child_suppress_agent_authored_keys = (
                suppress_agent_authored_keys
                or _is_direct_model_response_payload(path, value)
                or _is_agent_authored_payload(value)
            )
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if _is_eval_only_key(
                str(key),
                value,
                suppress_agent_authored_keys=child_suppress_agent_authored_keys,
                allow_pointnav_goal_position=allow_pointnav_goal_position,
            ):
                hits.append(
                    GtLeakHit(
                        path=child_path,
                        key=str(key),
                        line_no=line_no,
                        value_preview=repr(child)[:200],
                    )
                )
            hits.extend(
                _scan_value(
                    child,
                    line_no=line_no,
                    path=child_path,
                    seen=seen,
                    suppress_agent_authored_keys=child_suppress_agent_authored_keys,
                    allow_pointnav_goal_position=child_allow_pointnav_goal_position,
                )
            )
        return hits
    if isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(
                _scan_value(
                    child,
                    line_no=line_no,
                    path=f"{path}[{index}]",
                    seen=seen,
                    suppress_agent_authored_keys=suppress_agent_authored_keys,
                    allow_pointnav_goal_position=allow_pointnav_goal_position,
                )
            )
        return hits
    if isinstance(value, str):
        decoded = _maybe_json(value)
        if decoded is not None:
            hits.extend(
                _scan_value(
                    decoded,
                    line_no=line_no,
                    path=f"{path}<json>",
                    seen=seen,
                    suppress_agent_authored_keys=suppress_agent_authored_keys,
                    allow_pointnav_goal_position=allow_pointnav_goal_position,
                )
            )
    return hits


def _maybe_json(text: str) -> Any | None:
    stripped = text.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return None


def _is_eval_only_key(
    key: str,
    container: dict[str, Any],
    *,
    suppress_agent_authored_keys: bool,
    allow_pointnav_goal_position: bool,
) -> bool:
    if suppress_agent_authored_keys and key in {
        "answer",
        "goal_position",
        "goal_positions",
    }:
        return False
    if key == "goal_position" and (
        container.get("task_type") == "pointnav" or allow_pointnav_goal_position
    ):
        return False
    return (
        key in EVAL_ONLY_KEYS
        or key.startswith("gt_")
        or key.startswith("oracle_")
        or key.endswith("_geodesic_distance")
    )


def _is_direct_model_response_payload(path: str, value: dict[str, Any]) -> bool:
    parts = path.split(".", maxsplit=2)
    return (
        len(parts) == 2
        and parts[0].startswith("line")
        and parts[1] == "response"
        and "output" in value
    )


def _is_agent_authored_payload(value: dict[str, Any]) -> bool:
    if value.get("role") == "assistant":
        return True
    return value.get("type") == "agent_message"


def _is_tool_result_payload(value: dict[str, Any]) -> bool:
    return value.get("type") in {"function_call_output", "mcp_tool_call_end"}


__all__ = [
    "EVAL_ONLY_KEYS",
    "GtLeakHit",
    "GtLeakScanResult",
    "model_visible_tool_result_json",
    "scan_raw_llm_call_paths",
    "scan_raw_llm_calls",
    "strip_model_visible_eval_fields",
]
