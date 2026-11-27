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

"""Runtime-policy prompt construction for the native DimOS agent."""

from __future__ import annotations

from dimos.agents.runtime_policy import AgentRuntimePolicy, is_task_authority_mode


def build_native_policy_system_prompt(
    base_prompt: str | None,
    *,
    policy: AgentRuntimePolicy,
) -> str | None:
    """Append runtime policy instructions to a native-agent system prompt."""

    if base_prompt is None:
        return None

    return (
        f"{base_prompt}\n\n# HARNESS OPERATING RULES\n"
        f"{_harness_operating_rules()}\n\n"
        f"# RUNTIME POLICY\n"
        f"- {_speech_instruction(policy)}\n"
        f"- {_movement_instruction(policy)}\n"
        f"- {_tool_scope_instruction(policy)}\n"
        f"- {_navigation_instruction()}\n"
        f"- {_failure_recovery_instruction()}\n"
        f"- {_dangerous_skill_instruction(policy)}\n"
        f"- VLM provider: {policy.vlm_provider}.\n"
        f"- Spatial memory mode: {policy.spatial_memory_mode}.\n"
        f"- Tool policy admin override: "
        f"{'enabled' if policy.tool_policy_admin_override else 'disabled'}.\n"
    )


def _harness_operating_rules() -> str:
    return (
        "- The runtime policy below overrides any conflicting base prompt, tool "
        "description, or older robot demo instruction.\n"
        "- Treat server-side policy and structured tool/status results as "
        "authoritative; hidden or refused tools are unavailable even if the base "
        "prompt mentions them.\n"
        "- Text responses are visible in the operator UI. Do not assume the "
        "operator only hears speaker audio."
    )


def _speech_instruction(policy: AgentRuntimePolicy) -> str:
    if policy.speak_mode == "off":
        return (
            "Runtime speech policy is off. Do not call the `speak` tool. "
            "Use concise text responses instead."
        )
    if policy.speak_mode == "on_request":
        return (
            "Runtime speech policy is on_request. Only call `speak` when the "
            "user explicitly asks for audible speech. Otherwise respond in text "
            "because the operator UI is visible."
        )
    if policy.speak_mode == "status_only":
        return (
            "Runtime speech policy is status_only. Use `speak` only for short "
            "start, completion, or failure status messages."
        )
    return "Runtime speech policy is always. Voice-first responses are allowed."


def _movement_instruction(policy: AgentRuntimePolicy) -> str:
    if policy.movement_mode == "confirm":
        return (
            "Runtime movement policy is confirm. For real robot movement, propose "
            "the action and wait for explicit user confirmation before moving."
        )
    if policy.movement_mode == "deny":
        return (
            "Runtime movement policy is deny. Do not start movement; stop/cancel "
            "safety tools may still be used."
        )
    if is_task_authority_mode(policy.movement_mode):
        return (
            f"Runtime movement policy is {policy.movement_mode}. A user's explicit "
            "movement, navigation, exploration, patrol, follow, or posture command "
            "grants one bounded task authority. Use movement tools within that "
            "authority without asking for confirmation before every step or before "
            "starting the requested motion. Ask again only when leaving the task "
            "envelope, using a dangerous skill, or extending beyond the task "
            "budget. Stop if the authority is refused, exhausted, or the user "
            "interrupts."
        )
    return f"Runtime movement policy is {policy.movement_mode}."


def _tool_scope_instruction(policy: AgentRuntimePolicy) -> str:
    if policy.tool_scope == "full":
        return "Runtime tool scope is full. Use available tools subject to policy and task intent."
    if policy.tool_scope == "no_frontier":
        return (
            "Runtime tool scope is no_frontier. Do not call frontier exploration; "
            "prefer explicit navigation, inspection, or local movement tools."
        )
    if policy.tool_scope == "no_map":
        return (
            "Runtime tool scope is no_map. Do not use map-backed navigation or "
            "exploration; rely on current visual/lidar evidence and local motion."
        )
    if policy.tool_scope == "semantic_memory_off":
        return (
            "Runtime tool scope is semantic_memory_off. Do not use semantic/tagged "
            "spatial memory; rely on current perception, explicit coordinates, or local motion."
        )
    if policy.tool_scope == "vision_lidar_only":
        return (
            "Runtime tool scope is vision_lidar_only. Do not use prebuilt map, "
            "frontier, semantic memory, semantic navigation, tagged locations, or "
            "map-backed navigation. Rely on camera/lidar diagnostics, explicit "
            "range/protection status, and bounded local motion only."
        )
    return (
        "Runtime tool scope is readonly. Use inspection/status tools only; do not "
        "move the robot, speak, tag memory, or mutate state."
    )


def _navigation_instruction() -> str:
    return (
        "Navigation lifecycle is structured. If a navigation tool reports STARTED "
        "or FOLLOWING_PATH, call `navigation_status(request_id)` before claiming "
        "arrival, completion, or failure. Use `return_to_task_start()` only when "
        "a navigation task-start breadcrumb has been recorded."
    )


def _failure_recovery_instruction() -> str:
    return (
        "If movement or navigation is refused, exhausted, blocked, stale, or "
        "preempted, report the concrete policy/status reason, inspect diagnostics "
        "or status tools, and do not retry blindly."
    )


def _dangerous_skill_instruction(policy: AgentRuntimePolicy) -> str:
    if policy.dangerous_skills == "deny":
        return "Runtime dangerous-skill policy is deny. Do not call dynamic stunt skills."
    if policy.dangerous_skills == "confirm":
        return (
            "Runtime dangerous-skill policy is confirm. Explain the risk and wait for "
            "explicit confirmation before any dynamic stunt skill."
        )
    return "Runtime dangerous-skill policy is allow. Still prefer low-risk skills unless asked."
