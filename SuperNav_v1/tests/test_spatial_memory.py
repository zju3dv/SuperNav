from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from PIL import Image

from supernav.methods.navigation.spatial_memory import SpatialMemoryLedger
from supernav.methods.navigation.tools.navigation_oracle import VisualGroundPreviewTool
from supernav.methods.navigation.tools.session import CloseSessionTool

REPO_ROOT = Path(__file__).resolve().parents[1]

from supernav.evaluation.metrics import compute_metrics  # noqa: E402
from supernav.evaluation.replay_manifest import build_manifest  # noqa: E402
from supernav.runtime.streams import augment_canonical_events, write_canonical_events
from supernav.evaluation.video.render import synthetic_trace_events, trace_events  # noqa: E402


def _panorama(capture: int) -> list[dict[str, str]]:
    return [
        {
            "view": view,
            "direction": view,
            "image_ref": f"pano:session-1:{capture}:{view}",
            "path": f"/private/pano_{view}_{capture}.png",
        }
        for view in ("front", "right", "back", "left")
    ]


def _branches(prefix: str = "") -> list[dict[str, object]]:
    return [
        {
            "view": "front",
            "phrase": f"{prefix}glass and stone doorway".strip(),
            "point": [0.25, 0.62],
        },
        {
            "view": "right",
            "phrase": f"{prefix}open doorway beside cabinets".strip(),
            "point": [0.71, 0.61],
        },
    ]


def _statuses(memory: dict) -> dict[str, str]:
    return {
        branch["branch_id"]: branch["status"]
        for junction in memory["recent_junctions"]
        for branch in junction["branches"]
    }


def test_public_tool_schemas_add_optional_branch_and_structured_close() -> None:
    grounding = VisualGroundPreviewTool.metadata.parameters_schema["properties"]
    close = CloseSessionTool.metadata.parameters_schema["properties"]
    assert grounding["branch_id"]["type"] == "string"
    assert "branch_id" not in VisualGroundPreviewTool.metadata.parameters_schema.get(
        "required", []
    )
    assert close["outcome"]["enum"] == ["achieved", "blocked"]
    assert close["close_audit_token"]["type"] == "string"
    assert close["frontier_audit"]["type"] == "array"
    assert close["post_entry_exception"]["type"] == "object"
    assert "outcome" not in CloseSessionTool.metadata.parameters_schema.get(
        "required", []
    )


def test_global_result_redaction_keeps_visual_refs_but_hides_pose_and_paths() -> None:
    from supernav.methods.navigation.mcp_server import _redact_global_task_result

    redacted = _redact_global_task_result(
        {
            "backend": "visual_point_navmesh",
            "position": [1.0, 0.2, 3.0],
            "metrics": {"agent_state": {"rotation": [0, 0, 0, 1]}},
            "panorama_images": [
                {
                    "image_ref": "pano:session-1:1:front",
                    "direction": "front",
                    "heading_deg": -90.0,
                    "path": "/private/front.png",
                    "original_path": "/private/original-front.png",
                }
            ],
            "overlay_image": "/private/overlay.png",
            "spatial_memory": {
                "frontier": {
                    "unentered_branches": [
                        {"branch_id": "branch_0001", "point": [0.2, 0.7]}
                    ]
                }
            },
        }
    )
    assert "position" not in redacted
    assert "metrics" not in redacted
    assert "backend" not in redacted
    assert "navmesh" not in json.dumps(redacted)
    assert redacted["overlay_available"] is True
    assert redacted["panorama_images"] == [
        {
            "image_ref": "pano:session-1:1:front",
            "direction": "front",
        }
    ]
    branch = redacted["spatial_memory"]["frontier"]["unentered_branches"][0]
    assert branch == {"branch_id": "branch_0001", "point": [0.2, 0.7]}


def test_mcp_registration_binds_formal_grounding_without_forwarding_branch(
    monkeypatch,
) -> None:
    from supernav.methods.navigation import mcp_server

    monkeypatch.setenv("HAB_MCP_SPATIAL_MEMORY", "1")
    monkeypatch.delenv("HAB_MCP_GLOBAL_TASK", raising=False)
    monkeypatch.setattr(mcp_server._bridge, "session_id", "session-1")
    mcp_server._spatial_memory.reset("session-1")
    mcp_server._cache_mcp_panorama_images(_panorama(1))
    registered = json.loads(
        mcp_server.hab_register_spatial_junction(branches=_branches())
    )
    branch_id = registered["spatial_memory"]["frontier"]["unentered_branch_ids"][0]
    navigation_payloads = []

    def fake_call(action, payload=None, **_kwargs):
        if action == "navigate_visual_ground_preview":
            navigation_payloads.append(dict(payload or {}))
            return {
                "result": {
                    "status": "reached_visual_ground_candidate",
                    "auto_confirmed": True,
                    "selected_candidate_id": 0,
                    "steps_executed": 3,
                    "displacement_m": 0.6,
                    "planned_path_m": 0.9,
                }
            }
        return {}

    monkeypatch.setattr(mcp_server._bridge, "call", fake_call)
    grounded = json.loads(
        mcp_server.hab_visual_ground_preview(
            view="front",
            phrase="glass and stone doorway",
            branch_id=branch_id,
            include_images=False,
        )
    )
    assert navigation_payloads
    assert "branch_id" not in navigation_payloads[0]
    statuses = _statuses(grounded["spatial_memory"])
    assert statuses[branch_id] == "entered"


def test_global_init_uses_private_defaults_redacts_result_and_writes_audit(
    monkeypatch, tmp_path: Path
) -> None:
    from supernav.methods.navigation import mcp_server

    defaults = {
        "scene": "private-scene",
        "scene_dataset_config_file": "/private/dataset.json",
        "depth": True,
        "start_position": [1.0, 0.2, 3.0],
        "start_rotation": [0.0, 1.0, 0.0, 0.0],
        "sensor_height": 1.2,
    }
    monkeypatch.setenv("HAB_MCP_GLOBAL_TASK", "1")
    monkeypatch.setenv("HAB_MCP_INIT_DEFAULTS_JSON", json.dumps(defaults))
    monkeypatch.setenv("NAV_ARTIFACTS_DIR", str(tmp_path))
    monkeypatch.setattr(mcp_server._bridge, "session_id", None)
    init_payloads = []

    def fake_call(action, payload=None, **_kwargs):
        if action == "init_scene":
            init_payloads.append(dict(payload or {}))
            return {
                "session_id": "session-private",
                "scene": "private-scene",
                "scene_dataset_config_file": "/private/dataset.json",
                "agent_state": {"position": [1.0, 0.2, 3.0]},
                "artifact_path": "/private/frame.png",
            }
        return {}

    monkeypatch.setattr(mcp_server._bridge, "call", fake_call)
    visible = json.loads(
        mcp_server.hab_init_scene(
            scene="agent-override",
            depth=False,
            start_position=[9.0, 9.0, 9.0],
        )
    )
    payload = init_payloads[0]
    assert payload["scene"] == "private-scene"
    assert payload["scene_dataset_config_file"] == "/private/dataset.json"
    assert payload["start_position"] == [1.0, 0.2, 3.0]
    assert payload["start_rotation"] == [0.0, 1.0, 0.0, 0.0]
    assert payload["sensor"]["depth_sensor"] is True
    assert payload["sensor"]["sensor_height"] == 1.2
    visible_text = json.dumps(visible)
    assert "/private/" not in visible_text
    assert "position" not in visible_text

    sidecar = tmp_path / "session-private.benchmark_audit.jsonl"
    records = [json.loads(line) for line in sidecar.read_text().splitlines()]
    assert records[-1]["tool_name"] == "hab_init_scene"
    assert "/private/dataset.json" in json.dumps(records[-1]["result"])
    assert mcp_server._is_private_sidecar_artifact(sidecar.name) is True


def test_sibling_branch_facts_are_stable_and_only_formal_motion_enters() -> None:
    ledger = SpatialMemoryLedger(max_junctions=3, long_move_m=3.0)
    ledger.reset("session-1")
    registered = ledger.register_junction(_branches(), _panorama(1))
    ids = registered["frontier"]["unentered_branch_ids"]

    repeated = ledger.register_junction(_branches(), _panorama(1))
    assert repeated["frontier"]["unentered_branch_ids"] == ids
    assert repeated["new_events"] == []

    preview = ledger.process_tool_result(
        "visual_ground_preview",
        {"branch_id": ids[0], "view": "front", "phrase": "glass doorway"},
        {
            "status": "preview_ready",
            "confirm_token": "token-a",
            "image_ref": "pano:session-1:1:front",
        },
    )
    assert _statuses(preview)[ids[0]] == "observed"

    reached = ledger.process_tool_result(
        "visual_ground_preview",
        {
            "branch_id": ids[0],
            "confirm_token": "token-a",
            "candidate_id": 2,
        },
        {
            "status": "reached_visual_ground_candidate",
            "selected_candidate_id": 2,
            "steps_executed": 4,
            "displacement_m": 0.8,
            "planned_path_m": 4.1,
        },
    )
    statuses = _statuses(reached)
    assert statuses[ids[0]] == "entered"
    assert statuses[ids[1]] == "observed"
    assert any(event["action"] == "branch_entered" for event in reached["new_events"])
    assert all(
        event["action"] != "long_move_with_local_frontier"
        for event in reached["new_events"]
    )


def test_abandoned_expired_and_fresh_observation_tokens_preserve_history() -> None:
    ledger = SpatialMemoryLedger()
    ledger.reset("session-1")
    memory = ledger.register_junction(_branches(), _panorama(1))
    branch_id = memory["frontier"]["unentered_branch_ids"][0]
    ledger.process_tool_result(
        "visual_ground_preview",
        {"branch_id": branch_id},
        {"status": "preview_ready", "confirm_token": "token-a"},
    )
    abandoned = ledger.process_tool_result(
        "visual_ground_preview",
        {},
        {"status": "no_detections"},
    )
    assert _statuses(abandoned)[branch_id] == "observed"
    assert any(
        event["action"] == "confirm_abandoned" for event in abandoned["new_events"]
    )

    ledger.process_tool_result(
        "visual_ground_preview",
        {"branch_id": branch_id},
        {"status": "preview_ready", "confirm_token": "token-b"},
    )
    expired = ledger.process_tool_result(
        "visual_ground_preview",
        {"branch_id": branch_id, "confirm_token": "token-b"},
        {"status": "expired_confirm_token", "steps_executed": 0},
    )
    assert _statuses(expired)[branch_id] == "observed"
    assert any(
        event["action"] == "confirm_invalidated" for event in expired["new_events"]
    )

    ledger.process_tool_result(
        "visual_ground_preview",
        {"branch_id": branch_id},
        {"status": "preview_ready", "confirm_token": "token-c"},
    )
    refreshed = ledger.process_tool_result("turn", {}, {"status": "ok"})
    assert _statuses(refreshed)[branch_id] == "observed"
    assert refreshed["frontier"]["unentered_branch_ids"]
    assert any(
        event.get("reason") == "fresh_observation" for event in refreshed["new_events"]
    )


def test_recent_junction_bound_blocked_close_and_long_return_diagnostics() -> None:
    ledger = SpatialMemoryLedger(max_junctions=3, long_move_m=3.0)
    ledger.reset("session-1")
    for capture in range(1, 5):
        ledger.register_junction(_branches(f"{capture} "), _panorama(capture))
    current = ledger.result([])
    assert [row["junction_id"] for row in current["recent_junctions"]] == [
        "junction_0002",
        "junction_0003",
        "junction_0004",
    ]
    assert len(current["frontier"]["unentered_branch_ids"]) == 6

    reminder = ledger.process_tool_result(
        "visual_ground_preview",
        {"phrase": "corridor opening"},
        {
            "status": "preview_ready",
            "candidates": [{"planned_path_m": 3.4}],
        },
    )
    assert any(
        event["action"] == "frontier_reminder" for event in reminder["new_events"]
    )
    moved = ledger.process_tool_result(
        "visual_point_navigate",
        {},
        {"status": "reached_visual_point", "planned_path_m": 3.2},
    )
    assert any(
        event["action"] == "long_move_with_local_frontier"
        for event in moved["new_events"]
    )

    blocked = ledger.process_tool_result(
        "close_session", {"outcome": "blocked"}, {"closed": True}
    )
    assert any(
        event["action"] == "blocked_not_exhausted" for event in blocked["new_events"]
    )

    achieved_ledger = SpatialMemoryLedger()
    achieved_ledger.reset("session-2")
    achieved_ledger.register_junction(_branches(), _panorama(1))
    achieved = achieved_ledger.process_tool_result(
        "close_session", {"outcome": "achieved"}, {"closed": True}
    )
    assert all(
        event["action"] != "blocked_not_exhausted" for event in achieved["new_events"]
    )


def _valid_frontier_audit(memory: dict) -> list[dict[str, str]]:
    return [
        {
            "branch_id": branch_id,
            "disposition": "attempted_unreachable",
            "evidence": f"bounded attempt for {branch_id} did not reach the opening",
        }
        for branch_id in memory["frontier"]["unentered_branch_ids"]
    ]


def test_blocked_close_audit_challenges_then_accepts_exact_frontier() -> None:
    ledger = SpatialMemoryLedger()
    ledger.reset("session-1")
    registered = ledger.register_junction(_branches(), _panorama(1))

    challenge = ledger.evaluate_blocked_close({"outcome": "blocked"})
    assert challenge is not None
    assert challenge["status"] == "blocked_close_audit_required"
    assert challenge["closed"] is False
    assert challenge["blocked_close_gate"]["post_entry_followup_required"] is False
    assert any(
        row["action"] == "blocked_close_audit_requested"
        for row in challenge["spatial_memory"]["new_events"]
    )

    allowed = ledger.evaluate_blocked_close(
        {
            "outcome": "blocked",
            "close_audit_token": challenge["close_audit_token"],
            "frontier_audit": _valid_frontier_audit(registered),
        }
    )
    assert allowed is None
    closed = ledger.process_tool_result(
        "close_session", {"outcome": "blocked"}, {"closed": True}
    )
    actions = {row["action"] for row in closed["new_events"]}
    assert "blocked_frontier_audited" in actions
    assert "blocked_not_exhausted" not in actions


def test_blocked_close_audit_rejects_incomplete_duplicate_unknown_and_bad_evidence() -> (
    None
):
    def attempt(frontier_audit):
        ledger = SpatialMemoryLedger()
        ledger.reset("session-1")
        registered = ledger.register_junction(_branches(), _panorama(1))
        challenge = ledger.evaluate_blocked_close({"outcome": "blocked"})
        assert challenge is not None
        result = ledger.evaluate_blocked_close(
            {
                "outcome": "blocked",
                "close_audit_token": challenge["close_audit_token"],
                "frontier_audit": frontier_audit(registered),
            }
        )
        assert result is not None
        assert result["status"] == "invalid_blocked_close_audit"
        return result["error"]

    assert "exactly cover" in attempt(lambda memory: _valid_frontier_audit(memory)[:1])
    assert "duplicate" in attempt(
        lambda memory: [
            _valid_frontier_audit(memory)[0],
            _valid_frontier_audit(memory)[0],
        ]
    )
    assert "unknown" in attempt(
        lambda memory: _valid_frontier_audit(memory)[:-1]
        + [
            {
                "branch_id": "branch_unknown",
                "disposition": "unsafe",
                "evidence": "visible hazard",
            }
        ]
    )
    assert "disposition" in attempt(
        lambda memory: [
            {**row, "disposition": "probably_irrelevant"}
            for row in _valid_frontier_audit(memory)
        ]
    )
    assert "non-empty evidence" in attempt(
        lambda memory: [
            {**row, "evidence": ""} for row in _valid_frontier_audit(memory)
        ]
    )


def test_blocked_close_audit_token_is_state_bound_and_intervening_tools_invalidate() -> (
    None
):
    ledger = SpatialMemoryLedger()
    ledger.reset("session-1")
    registered = ledger.register_junction(_branches(), _panorama(1))
    challenge = ledger.evaluate_blocked_close({"outcome": "blocked"})
    assert challenge is not None

    ledger.begin_tool("turn")
    turn = ledger.process_tool_result("turn", {}, {"status": "ok"})
    assert any(
        row["action"] == "blocked_close_audit_invalidated" for row in turn["new_events"]
    )
    stale = ledger.evaluate_blocked_close(
        {
            "outcome": "blocked",
            "close_audit_token": challenge["close_audit_token"],
            "frontier_audit": _valid_frontier_audit(registered),
        }
    )
    assert stale is not None
    assert stale["status"] == "invalid_blocked_close_audit"

    fresh = ledger.evaluate_blocked_close({"outcome": "blocked"})
    assert fresh is not None
    ledger.register_junction(_branches("new "), _panorama(2))
    changed = ledger.evaluate_blocked_close(
        {
            "outcome": "blocked",
            "close_audit_token": fresh["close_audit_token"],
            "frontier_audit": _valid_frontier_audit(ledger.result([])),
        }
    )
    assert changed is not None
    assert "no longer matches" in changed["error"]


def test_post_entry_followup_gate_requires_explicit_visual_exception() -> None:
    ledger = SpatialMemoryLedger()
    ledger.reset("session-1")
    registered = ledger.register_junction(_branches(), _panorama(1))
    branch_id = registered["frontier"]["unentered_branch_ids"][0]
    ledger.process_tool_result(
        "visual_ground_preview",
        {"branch_id": branch_id},
        {
            "status": "reached_visual_ground_candidate",
            "auto_confirmed": True,
            "steps_executed": 4,
            "displacement_m": 0.8,
            "panorama_images": _panorama(2),
        },
    )
    current = ledger.result([])
    challenge = ledger.evaluate_blocked_close({"outcome": "blocked"})
    assert challenge is not None
    assert challenge["blocked_close_gate"]["post_entry_followup_required"] is True
    missing = ledger.evaluate_blocked_close(
        {
            "outcome": "blocked",
            "close_audit_token": challenge["close_audit_token"],
            "frontier_audit": _valid_frontier_audit(current),
        }
    )
    assert missing is not None
    assert "post_entry_exception" in missing["error"]
    allowed = ledger.evaluate_blocked_close(
        {
            "outcome": "blocked",
            "close_audit_token": challenge["close_audit_token"],
            "frontier_audit": _valid_frontier_audit(current),
            "post_entry_exception": {
                "disposition": "no_safe_passable_interior",
                "evidence": "fresh surround shows a sealed wall beyond the threshold",
            },
        }
    )
    assert allowed is None


def test_mcp_blocked_close_challenge_never_calls_bridge_and_valid_retry_closes_once(
    monkeypatch,
) -> None:
    from supernav.methods.navigation import mcp_server

    monkeypatch.setenv("HAB_MCP_GLOBAL_TASK", "1")
    monkeypatch.setenv("HAB_MCP_SPATIAL_MEMORY", "1")
    monkeypatch.setattr(mcp_server._bridge, "session_id", "session-1")
    mcp_server._spatial_memory.reset("session-1")
    registered = mcp_server._spatial_memory.register_junction(_branches(), _panorama(1))
    calls = []

    def fake_call(action, payload=None, **_kwargs):
        calls.append(action)
        if action == "get_audit_metrics":
            return {}
        if action == "close_session":
            return {"closed": True}
        raise AssertionError(action)

    monkeypatch.setattr(mcp_server._bridge, "call", fake_call)
    challenged = json.loads(
        mcp_server.hab_close_session(outcome="blocked", reason="all routes failed")
    )
    assert challenged["status"] == "blocked_close_audit_required"
    assert challenged["closed"] is False
    assert calls == []

    closed = json.loads(
        mcp_server.hab_close_session(
            outcome="blocked",
            reason="all routes failed",
            close_audit_token=challenged["close_audit_token"],
            frontier_audit=_valid_frontier_audit(registered),
        )
    )
    assert closed["closed"] is True
    assert closed["terminal_claim"]["outcome"] == "blocked"
    assert closed["terminal_audit"]["code"] == "blocked_frontier_audited"
    assert calls.count("close_session") == 1


def test_achieved_and_empty_frontier_blocked_close_bypass_audit() -> None:
    ledger = SpatialMemoryLedger()
    ledger.reset("session-1")
    assert ledger.evaluate_blocked_close({"outcome": "blocked"}) is None
    ledger.register_junction(_branches(), _panorama(1))
    assert ledger.evaluate_blocked_close({"outcome": "achieved"}) is None


def test_canonical_audit_merge_is_exact_deduplicated_and_sidecar_optional(
    tmp_path: Path,
) -> None:
    model_content = json.dumps(
        {
            "status": "junction_registered",
            "spatial_memory": {
                "new_events": [
                    {
                        "event_id": "spatial_memory_000001",
                        "action": "junction_registered",
                        "revision": 1,
                        "junction_id": "junction_0001",
                    }
                ]
            },
        }
    )
    events = [
        {
            "type": "tool_call",
            "name": "hab_register_spatial_junction",
            "input": {"branches": _branches()},
            "timestamp": "2026-07-13T01:00:00Z",
        },
        {
            "type": "tool_result",
            "name": "hab_register_spatial_junction",
            "content": model_content,
            "timestamp": "2026-07-13T01:00:01Z",
        },
    ]
    audit = tmp_path / "session-1.benchmark_audit.jsonl"
    audit.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "tool_seq": 2,
                "tool_name": "hab_register_spatial_junction",
                "model_content_sha256": hashlib.sha256(
                    model_content.encode("utf-8")
                ).hexdigest(),
                "result": {
                    "private_path": "/audit/only",
                    "spatial_memory": json.loads(model_content)["spatial_memory"],
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    once = augment_canonical_events(events, audit_path=audit)
    twice = augment_canonical_events(once, audit_path=audit)
    assert sum(event.get("type") == "spatial_memory" for event in twice) == 1
    result = next(event for event in twice if event.get("type") == "tool_result")
    assert result["audit"]["private_path"] == "/audit/only"
    assert result["audit_tool_seq"] == 2

    old = augment_canonical_events(
        [{"type": "tool_result", "name": "hab_turn", "content": '{"status":"ok"}'}],
        audit_path=tmp_path / "missing.jsonl",
    )
    assert old == [
        {"type": "tool_result", "name": "hab_turn", "content": '{"status":"ok"}'}
    ]


def test_global_metrics_require_structured_achieved_and_report_blocked_warning(
    tmp_path: Path,
) -> None:
    canonical = tmp_path / "canonical.jsonl"
    achieved_rows = [
        {
            "type": "tool_call",
            "name": "hab_close_session",
            "input": {"outcome": "achieved", "reason": "target visible"},
        },
        {
            "type": "tool_result",
            "name": "hab_close_session",
            "content": json.dumps(
                {
                    "closed": True,
                    "terminal_claim": {
                        "outcome": "achieved",
                        "reason": "target visible",
                    },
                }
            ),
        },
    ]
    write_canonical_events(canonical, achieved_rows)
    achieved = compute_metrics(
        canonical,
        arm="global",
        slug="task003",
        benchmark_profile="global_task",
    )
    assert achieved["success"] is True
    assert achieved["formal_task_outcome"] == "achieved"
    assert achieved["blocked_with_unentered_branch"] is False

    blocked_rows = [
        {
            "type": "tool_call",
            "name": "hab_close_session",
            "input": {"outcome": "blocked"},
        },
        {
            "type": "tool_result",
            "name": "hab_close_session",
            "content": json.dumps(
                {
                    "closed": True,
                    "terminal_claim": {"outcome": "blocked", "reason": ""},
                    "terminal_diagnostic": {"code": "blocked_not_exhausted"},
                }
            ),
        },
    ]
    write_canonical_events(canonical, blocked_rows)
    blocked = compute_metrics(
        canonical,
        arm="global",
        slug="task003",
        benchmark_profile="global_task",
    )
    assert blocked["success"] is False
    assert blocked["formal_task_outcome"] == "blocked"
    assert blocked["blocked_with_unentered_branch"] is True

    challenged_rows = [
        {
            "type": "tool_call",
            "name": "hab_close_session",
            "input": {"outcome": "blocked"},
        },
        {
            "type": "tool_result",
            "name": "hab_close_session",
            "content": json.dumps(
                {
                    "status": "blocked_close_audit_required",
                    "closed": False,
                    "close_audit_token": "audit-token",
                }
            ),
        },
    ]
    write_canonical_events(canonical, challenged_rows)
    challenged = compute_metrics(
        canonical,
        arm="global",
        slug="task003",
        benchmark_profile="global_task",
    )
    assert challenged["formal_task_outcome"] == "close_failed"
    assert challenged["agent_terminal_claim"] is None
    assert challenged["blocked_close_challenges"] == 1

    audited_rows = challenged_rows + [
        {
            "type": "tool_call",
            "name": "hab_close_session",
            "input": {
                "outcome": "blocked",
                "close_audit_token": "audit-token",
            },
        },
        {
            "type": "tool_result",
            "name": "hab_close_session",
            "content": json.dumps(
                {
                    "closed": True,
                    "terminal_claim": {"outcome": "blocked", "reason": "audited"},
                    "terminal_audit": {"code": "blocked_frontier_audited"},
                }
            ),
        },
    ]
    write_canonical_events(canonical, audited_rows)
    audited = compute_metrics(
        canonical,
        arm="global",
        slug="task003",
        benchmark_profile="global_task",
    )
    assert audited["formal_task_outcome"] == "blocked"
    assert audited["blocked_frontier_audited"] is True
    assert audited["blocked_close_challenges"] == 1


def test_replay_uses_audit_paths_and_traces_memory_in_both_timelines(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    visuals_root = tmp_path / "visuals"
    session_dir = visuals_root / "session-1"
    session_dir.mkdir(parents=True)
    overlay = session_dir / "ground_overlay.png"
    Image.new("RGB", (32, 32), (30, 40, 50)).save(overlay)
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps({"session_id": "session-1"}), encoding="utf-8"
    )
    events = [
        {
            "type": "tool_call",
            "name": "hab_visual_ground_preview",
            "input": {
                "view": "front",
                "phrase": "glass doorway",
                "branch_id": "branch_0001",
            },
            "timestamp": "2026-07-13T01:00:00Z",
        },
        {
            "type": "tool_result",
            "name": "hab_visual_ground_preview",
            "content": '{"status":"preview_ready","overlay_available":true}',
            "audit": {
                "status": "preview_ready",
                "overlay_image": str(overlay),
                "candidate_count": 2,
            },
            "timestamp": "2026-07-13T01:00:01Z",
        },
        {
            "type": "spatial_memory",
            "event_id": "spatial_memory_000001",
            "action": "ground_preview_bound",
            "branch_id": "branch_0001",
            "timestamp": "2026-07-13T01:00:01Z",
            "time_start": "2026-07-13T01:00:01Z",
            "time_end": "2026-07-13T01:00:01Z",
        },
        {
            "type": "assistant_text",
            "text": "Checking the branch.",
            "timestamp": "2026-07-13T01:00:02Z",
        },
    ]
    write_canonical_events(run_dir / "canonical.jsonl", events)
    manifest = build_manifest(run_dir, visuals_root)
    assert manifest["entries"][0]["frames"] == [str(overlay)]
    assert manifest["entries"][0]["target_annotation"]["branch_id"] == "branch_0001"
    assert any(kind == "memory" for _, kind, _, _ in trace_events(events))

    untimed = [
        {
            key: value
            for key, value in event.items()
            if key not in {"timestamp", "time_start", "time_end"}
        }
        for event in events
    ]
    assert any(
        kind == "memory"
        for _, kind, _, _ in synthetic_trace_events(untimed, start_ts=0.0, end_ts=10.0)
    )


def _valid_arrival_confirmation() -> dict:
    return {
        "final_approach_done": True,
        "target_base_cut": True,
        "estimated_distance_m": 0.8,
        "evidence": "target base cut by frame bottom edge; body dominates front view",
    }


def test_achieved_close_audit_challenges_then_accepts_valid_confirmation() -> None:
    ledger = SpatialMemoryLedger()
    ledger.reset("session-1")

    challenge = ledger.evaluate_achieved_close({"outcome": "achieved"})
    assert challenge is not None
    assert challenge["status"] == "achieved_close_audit_required"
    assert challenge["closed"] is False
    assert any(
        row["action"] == "achieved_close_audit_requested"
        for row in challenge["spatial_memory"]["new_events"]
    )

    allowed = ledger.evaluate_achieved_close(
        {
            "outcome": "achieved",
            "close_audit_token": challenge["close_audit_token"],
            "arrival_confirmation": _valid_arrival_confirmation(),
        }
    )
    assert allowed is None
    closed = ledger.process_tool_result(
        "close_session", {"outcome": "achieved"}, {"closed": True}
    )
    actions = {row["action"] for row in closed["new_events"]}
    assert "achieved_arrival_confirmed" in actions


def test_achieved_close_audit_rejects_missing_and_bad_confirmation() -> None:
    ledger = SpatialMemoryLedger()
    ledger.reset("session-1")

    challenge = ledger.evaluate_achieved_close({"outcome": "achieved"})
    token = challenge["close_audit_token"]

    missing = ledger.evaluate_achieved_close(
        {"outcome": "achieved", "close_audit_token": token}
    )
    assert missing["status"] == "invalid_achieved_close_audit"
    assert missing["closed"] is False

    bad_distance = ledger.evaluate_achieved_close(
        {
            "outcome": "achieved",
            "close_audit_token": token,
            "arrival_confirmation": {
                "final_approach_done": True,
                "target_base_cut": True,
                "estimated_distance_m": "far",
                "evidence": "x",
            },
        }
    )
    assert bad_distance["status"] == "invalid_achieved_close_audit"

    # failed retries keep the binding alive: a valid retry still passes
    allowed = ledger.evaluate_achieved_close(
        {
            "outcome": "achieved",
            "close_audit_token": token,
            "arrival_confirmation": _valid_arrival_confirmation(),
        }
    )
    assert allowed is None


def test_achieved_close_audit_rejects_unmet_criteria() -> None:
    ledger = SpatialMemoryLedger()
    ledger.reset("session-1")
    token = ledger.evaluate_achieved_close({"outcome": "achieved"})[
        "close_audit_token"
    ]

    too_far = _valid_arrival_confirmation()
    too_far["estimated_distance_m"] = 1.6
    result = ledger.evaluate_achieved_close(
        {
            "outcome": "achieved",
            "close_audit_token": token,
            "arrival_confirmation": too_far,
        }
    )
    assert result["status"] == "invalid_achieved_close_audit"
    assert "criteria not met" in result["error"]

    not_approached = _valid_arrival_confirmation()
    not_approached["final_approach_done"] = False
    result = ledger.evaluate_achieved_close(
        {
            "outcome": "achieved",
            "close_audit_token": token,
            "arrival_confirmation": not_approached,
        }
    )
    assert result["status"] == "invalid_achieved_close_audit"


def test_achieved_close_audit_token_invalidated_by_intervening_tool() -> None:
    ledger = SpatialMemoryLedger()
    ledger.reset("session-1")
    token = ledger.evaluate_achieved_close({"outcome": "achieved"})[
        "close_audit_token"
    ]

    ledger.begin_tool("hab_local_navigate")
    result = ledger.evaluate_achieved_close(
        {
            "outcome": "achieved",
            "close_audit_token": token,
            "arrival_confirmation": _valid_arrival_confirmation(),
        }
    )
    assert result["status"] == "invalid_achieved_close_audit"
    assert "stale or unknown" in result["error"]

    # close_session itself does not invalidate: challenge -> immediate retry works
    rechallenge = ledger.evaluate_achieved_close({"outcome": "achieved"})
    assert rechallenge["status"] == "achieved_close_audit_required"
    allowed = ledger.evaluate_achieved_close(
        {
            "outcome": "achieved",
            "close_audit_token": rechallenge["close_audit_token"],
            "arrival_confirmation": _valid_arrival_confirmation(),
        }
    )
    assert allowed is None
