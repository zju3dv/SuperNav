# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json

import pytest

from dimos.agents.mcp.external_harness_authority_proxy import (
    ExternalHarnessAuthorityProxy,
    MovementAuthorityEnvelope,
)
from dimos.agents.mcp.external_harness_codex import (
    GO2_GLOBAL_NAVIGATION_PIXELGOAL_REFERENCES,
    GO2_PIXELGOAL_SKILL_TEXT,
    codex_goal_prompt,
)
from dimos.agents.mcp.test_external_harness_authority_proxy import PixelGoalMcpAdapter


@pytest.mark.parametrize(
    "reason", ["exploration_complete", "exploration_failed", "no_progress",
               "blocked_pixel_target", "deadline_exceeded"]
)
@pytest.mark.parametrize("terminal_images", [True, False])
def test_pixelgoal_handoff_returns_reason_and_views_without_retry(
    reason: str,
    terminal_images: bool,
) -> None:
    class HandoffAdapter(PixelGoalMcpAdapter):
        def call_tool(self, name, arguments=None, *, meta=None):
            result = super().call_tool(name, arguments, meta=meta)
            if name == "pixelgoal_status":
                payload = json.loads(result["content"][0]["text"])
                terminal = {
                    "goal_id": "goal-1",
                    "ok": False,
                    "status": "timeout" if reason == "deadline_exceeded" else "needs_agent",
                    "code": reason,
                    "needs_agent_reason": reason,
                }
                payload.update(status=terminal, last_result=terminal)
                result["content"][0]["text"] = json.dumps(payload)
            return result

    upstream = HandoffAdapter(terminal_images=terminal_images)
    envelope = MovementAuthorityEnvelope.create(
        task_goal="inspect doorway",
        allowed_movement_tools=("pixelgoal_navigate",),
        duration_s=600.0,
    )
    proxy = ExternalHarnessAuthorityProxy(upstream, envelope=envelope)
    result = proxy.call_tool("pixelgoal_navigate", {"point": [0.3, 0.55], "direction": "right"})
    payload = json.loads(result["content"][0]["text"])
    assert payload["last_result"]["code"] == reason
    assert payload["last_result"]["ok"] is False
    assert payload["hop_running"] is False
    assert sum(b["type"] == "image" for b in result["content"]) == 4
    calls = [c[0] for c in upstream.calls]
    assert calls.count("pixelgoal_navigate") == 1
    assert calls.count("pixelgoal_status") == 2
    assert "pixelgoal_stop" not in calls
    assert calls.count("observe_surround") == (0 if terminal_images else 1)
    assert envelope.movement_calls_signed == 1


@pytest.mark.parametrize("changed", ["goal_id", "hop_running"])
def test_pixelgoal_rechecks_terminal_identity_before_returning_images(changed, monkeypatch) -> None:
    monkeypatch.setattr(
        "dimos.agents.mcp.external_harness_authority_proxy.PIXELGOAL_POLL_INTERVAL_S",
        0.0,
    )

    class ChangedAdapter(PixelGoalMcpAdapter):
        image_polls = 0

        def call_tool(self, name, arguments=None, *, meta=None):
            result = super().call_tool(name, arguments, meta=meta)
            if name == "pixelgoal_status" and arguments.get("include_images"):
                self.image_polls += 1
                if self.image_polls == 1:
                    payload = json.loads(result["content"][0]["text"])
                    if changed == "goal_id":
                        payload["last_result"]["goal_id"] = "another-goal"
                        payload["status"]["goal_id"] = "another-goal"
                    else:
                        payload["hop_running"] = True
                    result["content"][0]["text"] = json.dumps(payload)
            return result

    upstream = ChangedAdapter()
    envelope = MovementAuthorityEnvelope.create(
        task_goal="inspect doorway",
        allowed_movement_tools=("pixelgoal_navigate",),
        duration_s=600.0,
    )
    proxy = ExternalHarnessAuthorityProxy(upstream, envelope=envelope)
    result = proxy.call_tool("pixelgoal_navigate", {"point": [0.3, 0.55], "direction": "right"})
    payload = json.loads(result["content"][0]["text"])
    assert payload["last_result"]["goal_id"] == "goal-1"
    assert payload["hop_running"] is False
    assert upstream.image_polls == 2
    assert [c[0] for c in upstream.calls].count("pixelgoal_navigate") == 1


def test_pixelgoal_fallback_uses_fresh_observation_health() -> None:
    class HealthAdapter(PixelGoalMcpAdapter):
        def call_tool(self, name, arguments=None, *, meta=None):
            result = super().call_tool(name, arguments, meta=meta)
            if name in {"pixelgoal_status", "observe_surround"}:
                payload = json.loads(result["content"][0]["text"])
                if name == "pixelgoal_status":
                    payload["camera_health"] = {"ok": False, "stale": ["left"]}
                else:
                    payload.update(
                        ok=True,
                        stale=[],
                        missing=[],
                        fresh=["front", "right", "back", "left"],
                    )
                result["content"][0]["text"] = json.dumps(payload)
            return result

    upstream = HealthAdapter(terminal_images=False)
    envelope = MovementAuthorityEnvelope.create(
        task_goal="inspect doorway",
        allowed_movement_tools=("pixelgoal_navigate",),
        duration_s=600.0,
    )
    proxy = ExternalHarnessAuthorityProxy(upstream, envelope=envelope)
    result = proxy.call_tool("pixelgoal_navigate", {"point": [0.3, 0.55], "direction": "right"})
    payload = json.loads(result["content"][0]["text"])
    assert payload["camera_health"]["ok"] is True
    assert payload["camera_health"]["stale"] == []
    assert payload["last_result"]["goal_id"] == "goal-1"


def test_generated_handoff_instructions_prioritize_safety():
    recovery = GO2_GLOBAL_NAVIGATION_PIXELGOAL_REFERENCES["recovery-and-stop.md"]
    prompt = codex_goal_prompt(
        task="inspect doorway", mode="authority-proxy", pixelgoal_enabled=True
    )
    for reason in (
        "exploration_complete",
        "exploration_failed",
        "no_progress",
        "deadline_exceeded",
    ):
        assert reason in GO2_PIXELGOAL_SKILL_TEXT
        assert reason in recovery
        assert reason in prompt
    assert "precedence" in recovery
    assert "same point or doorway" in prompt
    assert "60 seconds for the whole hop" in GO2_PIXELGOAL_SKILL_TEXT
    assert "successive exploration legs" in GO2_PIXELGOAL_SKILL_TEXT
