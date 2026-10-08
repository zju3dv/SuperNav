from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from supernav.runtime.streams import iter_json_lines, tool_result_payload


def compute_metrics(
    canonical_path: str | Path,
    *,
    arm: str,
    slug: str,
    rep: Optional[str] = None,
    benchmark_profile: Optional[str] = None,
) -> Dict[str, Any]:
    tool_calls: Dict[str, int] = {}
    llm_turns = 0
    nav_steps_total = 0
    nav_legs = 0
    collisions = 0
    close_called = False
    session_closed = False
    terminal_claim: Optional[Dict[str, str]] = None
    blocked_with_unentered_branch = False
    blocked_frontier_audited = False
    blocked_close_challenges = 0
    pending_nav_name: Optional[str] = None
    for event in iter_json_lines(canonical_path):
        typ = event.get("type")
        if typ == "tool_call":
            name = str(event.get("name") or "unknown")
            tool_calls[name] = tool_calls.get(name, 0) + 1
            llm_turns += 1
            if name in {
                "hab_navigate_wam",
                "hab_visual_overlay_navigate",
                "hab_visual_local_navigate",
                "hab_oracle_local_navigate",
            }:
                nav_legs += 1
                pending_nav_name = name
            elif name in {"hab_visual_point_navigate", "hab_visual_ground_preview"}:
                pending_nav_name = name
            if name == "hab_close_session":
                close_called = True
        elif typ == "tool_result":
            content = str(event.get("content") or "")
            if event.get("name") == "hab_close_session":
                payload = tool_result_payload(event)
                closed_now = payload.get("closed") is True
                session_closed = session_closed or closed_now
                if payload.get("status") == "blocked_close_audit_required":
                    blocked_close_challenges += 1
                result_claim = payload.get("terminal_claim") if closed_now else None
                if isinstance(result_claim, dict) and closed_now:
                    outcome = str(result_claim.get("outcome") or "").strip().lower()
                    if outcome in {"achieved", "blocked"}:
                        terminal_claim = {
                            "outcome": outcome,
                            "reason": str(result_claim.get("reason") or ""),
                        }
                diagnostic = payload.get("terminal_diagnostic")
                blocked_with_unentered_branch = blocked_with_unentered_branch or bool(
                    isinstance(diagnostic, dict)
                    and diagnostic.get("code") == "blocked_not_exhausted"
                )
                terminal_audit = payload.get("terminal_audit")
                memory = payload.get("spatial_memory")
                memory_events = (
                    memory.get("new_events", []) if isinstance(memory, dict) else []
                )
                blocked_frontier_audited = blocked_frontier_audited or bool(
                    (
                        isinstance(terminal_audit, dict)
                        and terminal_audit.get("code") == "blocked_frontier_audited"
                    )
                    or any(
                        isinstance(row, dict)
                        and row.get("action") == "blocked_frontier_audited"
                        for row in memory_events
                    )
                )
            if "collision" in content.lower():
                collisions += 1
            if pending_nav_name and "steps" in content:
                parsed_steps = _steps_from_tool_result_content(content)
                if parsed_steps is not None:
                    nav_steps_total += parsed_steps
                    if pending_nav_name in {
                        "hab_visual_point_navigate",
                        "hab_visual_ground_preview",
                    }:
                        status = _status_from_tool_result_content(content)
                        if status != "preview_ready":
                            nav_legs += 1
                pending_nav_name = None
            elif pending_nav_name:
                status = _status_from_tool_result_content(content)
                if (
                    pending_nav_name
                    in {"hab_visual_point_navigate", "hab_visual_ground_preview"}
                    and status == "preview_ready"
                ):
                    pending_nav_name = None
    forward_calls = tool_calls.get("hab_forward", 0)
    backward_calls = tool_calls.get("hab_backward", 0)
    turn_calls = tool_calls.get("hab_turn", 0)
    safety_calls = (
        tool_calls.get("hab_passability_check", 0)
        + tool_calls.get("hab_depth_analyze", 0)
        + tool_calls.get("hab_depth_grid", 0)
    )
    look_cost = (
        tool_calls.get("hab_see", 0)
        + tool_calls.get("hab_panorama", 0)
        + tool_calls.get("hab_turn", 0)
        + tool_calls.get("hab_look_around", 0)
    )
    profile = str(benchmark_profile or "")
    if terminal_claim is not None:
        formal_task_outcome = terminal_claim["outcome"]
        outcome_basis = "structured_close_outcome"
    elif session_closed:
        formal_task_outcome = "unstructured_close"
        outcome_basis = "close_result_without_outcome"
    elif close_called:
        formal_task_outcome = "close_failed"
        outcome_basis = "close_call_without_successful_result"
    else:
        formal_task_outcome = "no_close"
        outcome_basis = "no_close_call"
    if profile == "global_task":
        success = bool(
            session_closed
            and terminal_claim is not None
            and terminal_claim.get("outcome") == "achieved"
        )
        success_semantics = "closed_and_structured_achieved"
    else:
        success = close_called
        success_semantics = "legacy_close_call"
    return {
        "arm": arm,
        "slug": slug,
        "rep": rep,
        "success": success,
        "success_semantics": success_semantics,
        "close_called": close_called,
        "session_closed": session_closed,
        "agent_terminal_claim": terminal_claim,
        "formal_task_outcome": formal_task_outcome,
        "outcome_basis": outcome_basis,
        "blocked_with_unentered_branch": blocked_with_unentered_branch,
        "blocked_frontier_audited": blocked_frontier_audited,
        "blocked_close_challenges": blocked_close_challenges,
        "llm_turns": llm_turns,
        "total_tool_calls": sum(tool_calls.values()),
        "look_cost": look_cost,
        "nav_legs": nav_legs,
        "nav_steps_total": nav_steps_total,
        "forward_calls": forward_calls,
        "backward_calls": backward_calls,
        "turn_calls": turn_calls,
        "safety_calls": safety_calls,
        "movement_tool_calls": forward_calls + backward_calls + turn_calls + nav_legs,
        "collision_mentions": collisions,
        "tool_calls": tool_calls,
    }


def _steps_from_tool_result_content(content: str) -> Optional[int]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        for key in ("steps_executed", "steps_taken", "steps"):
            value = parsed.get(key)
            if isinstance(value, (int, float)):
                return int(value)
    match = re.search(r"steps[^0-9]{1,16}(\d+)", content)
    if match:
        return int(match.group(1))
    return None


def _status_from_tool_result_content(content: str) -> Optional[str]:
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        status = parsed.get("status")
        if isinstance(status, str):
            return status
    return None


def append_jsonl(path: str | Path, row: Mapping[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
