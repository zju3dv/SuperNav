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

"""Validation helpers for exposing DimOS MCP to external harness clients."""

from __future__ import annotations

import json
from typing import Any

from dimos.agents.harness.go2_harness_profiles import go2_harness_profile_for_policy
from dimos.agents.harness.image_blocks import is_image_block
from dimos.agents.mcp.external_harness_authority import (
    external_harness_probe_meta,
    untrusted_movement_confirmation_probe_meta,
)
from dimos.agents.mcp.external_harness_scope_readiness import scope_readiness_check
from dimos.agents.mcp.external_harness_types import (
    ExternalHarnessMcpAdapter,
    HarnessValidationCheck,
    HarnessValidationReport,
    fail_check as _fail,
    pass_check as _pass,
)
from dimos.agents.runtime_policy import AgentRuntimePolicy, tool_exposed_by_policy


def validate_external_harness_surface(
    adapter: ExternalHarnessMcpAdapter,
    *,
    expected_tool_scope: str | None = "readonly",
    artifact_id: str | None = None,
    require_visual_readback: bool = False,
    require_movement_authority_refusal: bool = False,
    require_hidden_movement_refusal: bool = False,
    require_idle_runtime: bool = False,
) -> HarnessValidationReport:
    """Validate the MCP surface before attaching a mature external harness.

    This function intentionally uses only MCP-visible operations. The optional
    movement-authority and hidden-movement refusal probes use zero-degree
    rotation and expect the server to refuse before dispatch; they must not be
    used as live motion proof.
    """

    checks: list[HarnessValidationCheck] = []
    tools: list[dict[str, Any]] = []
    status: dict[str, Any] = {}

    try:
        adapter.initialize()
        checks.append(_pass("initialize", "MCP server responded to initialize."))
    except Exception as e:
        checks.append(_fail("initialize", f"MCP initialize failed: {e}"))
        return HarnessValidationReport(url=adapter.url, checks=checks)

    try:
        tools = adapter.list_tools()
        checks.append(_pass("tools_list", f"MCP exposed {len(tools)} tools."))
    except Exception as e:
        checks.append(_fail("tools_list", f"MCP tools/list failed: {e}"))

    try:
        status_text = adapter.call_tool_text("server_status")
        parsed = json.loads(status_text)
        if isinstance(parsed, dict):
            status = parsed
            checks.append(_pass("server_status", "server_status returned structured JSON."))
        else:
            checks.append(_fail("server_status", "server_status did not return a JSON object."))
    except Exception as e:
        checks.append(_fail("server_status", f"server_status failed: {e}"))

    policy = _policy_from_status(status)
    if policy is None:
        checks.append(_fail("runtime_policy", "server_status has no valid agent_runtime_policy."))
    else:
        checks.append(
            _pass(
                "runtime_policy",
                f"Runtime policy visible: profile={policy.profile}, scope={policy.tool_scope}.",
            )
        )
        checks.append(_admin_override_check(policy))
        checks.append(_go2_harness_profile_check(status, policy))
        checks.append(_movement_authority_surface_check(status, policy, require_idle_runtime))
        checks.append(_capability_surface_check(status, policy, require_idle_runtime))
        if expected_tool_scope is not None:
            if policy.tool_scope == expected_tool_scope:
                checks.append(_pass("tool_scope", f"Tool scope is {expected_tool_scope}."))
            else:
                checks.append(
                    _fail(
                        "tool_scope",
                        f"Expected tool scope {expected_tool_scope}, got {policy.tool_scope}.",
                    )
                )
        checks.append(_tool_list_policy_check(tools, policy))
        scope_check = scope_readiness_check(tools, policy)
        if scope_check is not None:
            checks.append(scope_check)

    tool_names = {str(tool.get("name") or "") for tool in tools}
    if "server_status" in tool_names and "read_image" in tool_names:
        checks.append(_pass("inspection_tools", "server_status and read_image are exposed."))
    else:
        checks.append(
            _fail(
                "inspection_tools",
                "server_status and read_image must be exposed for external inspection.",
                missing=sorted({"server_status", "read_image"} - tool_names),
            )
        )

    if require_visual_readback and not artifact_id:
        checks.append(
            _fail(
                "visual_readback",
                "Visual readback was required but no artifact id was provided.",
            )
        )
    elif artifact_id:
        checks.append(_read_image_check(adapter, artifact_id))

    if require_movement_authority_refusal:
        checks.append(_movement_authority_refusal_check(adapter, tool_names))
    if require_hidden_movement_refusal:
        checks.append(_hidden_movement_refusal_check(adapter, policy))

    return HarnessValidationReport(
        url=adapter.url, checks=checks, server_status=status, tools=tools
    )


def _policy_from_status(status: dict[str, Any]) -> AgentRuntimePolicy | None:
    payload = status.get("agent_runtime_policy")
    if not isinstance(payload, dict):
        return None
    try:
        return AgentRuntimePolicy(**payload)
    except TypeError:
        return None


def _tool_list_policy_check(
    tools: list[dict[str, Any]], policy: AgentRuntimePolicy
) -> HarnessValidationCheck:
    violations: list[str] = []
    for tool in tools:
        name = str(tool.get("name") or "")
        meta = tool.get("_meta") if isinstance(tool.get("_meta"), dict) else {}
        uses = meta.get("dimos/uses", []) if isinstance(meta, dict) else []
        if not tool_exposed_by_policy(name, uses, policy=policy):
            violations.append(name)
    if violations:
        return _fail(
            "tools_list_policy",
            "tools/list exposes tools hidden by the runtime policy.",
            tools=violations,
        )
    return _pass("tools_list_policy", "tools/list is consistent with runtime policy.")


def _go2_harness_profile_check(
    status: dict[str, Any],
    policy: AgentRuntimePolicy,
) -> HarnessValidationCheck:
    payload = status.get("go2_harness_profile")
    if not isinstance(payload, dict):
        return _fail(
            "go2_harness_profile",
            "server_status is missing the Go2 harness profile audit label.",
        )
    expected = go2_harness_profile_for_policy(policy)
    if payload.get("name") != expected.name or payload.get("tool_scope") != expected.tool_scope:
        return _fail(
            "go2_harness_profile",
            "Go2 harness profile does not match the runtime tool scope.",
            expected=expected.model_dump(),
            actual=payload,
        )
    return _pass(
        "go2_harness_profile",
        f"Go2 harness profile is {expected.name} for scope {expected.tool_scope}.",
    )


def _admin_override_check(policy: AgentRuntimePolicy) -> HarnessValidationCheck:
    if policy.tool_policy_admin_override:
        return _fail(
            "admin_override",
            "Tool policy admin override is enabled; disable it before external harness bring-up.",
            tool_policy_admin_override=True,
        )
    return _pass(
        "admin_override",
        "Tool policy admin override is disabled.",
        tool_policy_admin_override=False,
    )


def _movement_authority_surface_check(
    status: dict[str, Any],
    policy: AgentRuntimePolicy,
    require_idle_runtime: bool,
) -> HarnessValidationCheck:
    active = status.get("active_movement_authorities") or status.get("movement_leases") or {}
    if not isinstance(active, dict):
        return _fail(
            "movement_authority_surface",
            "server_status returned malformed movement authority state.",
            active_movement_authorities=active,
        )
    if active and (policy.tool_scope == "readonly" or require_idle_runtime):
        reason = (
            "external harness validation requires no active movement authority."
            if require_idle_runtime
            else "readonly external harness surface has active movement authority."
        )
        return _fail(
            "movement_authority_surface",
            reason,
            active_clients=sorted(str(client_id) for client_id in active),
            require_idle_runtime=require_idle_runtime,
        )
    return _pass(
        "movement_authority_surface",
        "Movement authority state is compatible with the external harness profile.",
        active_clients=sorted(str(client_id) for client_id in active),
        require_idle_runtime=require_idle_runtime,
    )


def _capability_surface_check(
    status: dict[str, Any],
    policy: AgentRuntimePolicy,
    require_idle_runtime: bool,
) -> HarnessValidationCheck:
    holders = status.get("capability_holders", {})
    if holders is None:
        holders = {}
    if not isinstance(holders, dict):
        return _fail(
            "capability_surface",
            "server_status returned malformed capability holder state.",
            capability_holders=holders,
        )
    if holders and (policy.tool_scope == "readonly" or require_idle_runtime):
        reason = (
            "external harness validation requires no active capability holders."
            if require_idle_runtime
            else "readonly external harness surface has active capability holders."
        )
        return _fail(
            "capability_surface",
            reason,
            capability_holders=_string_map(holders),
            require_idle_runtime=require_idle_runtime,
        )
    return _pass(
        "capability_surface",
        "Capability holder state is compatible with the external harness profile.",
        capability_holders=_string_map(holders),
        require_idle_runtime=require_idle_runtime,
    )


def _string_map(payload: dict[Any, Any]) -> dict[str, str]:
    return dict(sorted((str(key), str(value)) for key, value in payload.items()))


def _read_image_check(
    adapter: ExternalHarnessMcpAdapter, artifact_id: str
) -> HarnessValidationCheck:
    try:
        result = adapter.call_tool("read_image", {"artifact_id": artifact_id})
    except Exception as e:
        return _fail("read_image", f"read_image failed: {e}", artifact_id=artifact_id)

    content = result.get("content", []) if isinstance(result, dict) else []
    if not isinstance(content, list):
        return _fail(
            "read_image",
            f"read_image returned malformed content for {artifact_id}.",
            artifact_id=artifact_id,
            content=content,
        )

    image_blocks = [item for item in content if isinstance(item, dict) and is_image_block(item)]
    image_blocks_with_payload = [block for block in image_blocks if _image_block_has_payload(block)]
    artifact_metadata = _read_image_artifact_metadata(content)
    metadata_artifact_id = artifact_metadata.get("artifact_id")
    if image_blocks_with_payload and metadata_artifact_id == artifact_id:
        return _pass(
            "read_image",
            f"read_image returned metadata and image content for {artifact_id}.",
            artifact_id=artifact_id,
            image_block_types=[str(block.get("type") or "") for block in image_blocks],
            metadata=artifact_metadata,
        )
    if image_blocks_with_payload:
        return _fail(
            "read_image",
            f"read_image returned image content but no matching metadata for {artifact_id}.",
            artifact_id=artifact_id,
            image_block_types=[str(block.get("type") or "") for block in image_blocks],
            metadata=artifact_metadata,
        )
    return _fail(
        "read_image",
        f"read_image did not return image content for {artifact_id}.",
        artifact_id=artifact_id,
        content_types=[item.get("type") for item in content if isinstance(item, dict)],
    )


def _read_image_artifact_metadata(content: list[Any]) -> dict[str, Any]:
    for item in content:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        text = str(item.get("text") or "")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        artifact = payload.get("artifact")
        if isinstance(artifact, dict):
            return artifact
    return {}


def _image_block_has_payload(block: dict[str, Any]) -> bool:
    block_type = str(block.get("type") or "")
    if block_type == "image_url":
        image_url = block.get("image_url")
        return isinstance(image_url, dict) and bool(image_url.get("url"))
    if block_type == "input_image":
        return bool(block.get("image_url"))
    if block_type == "image":
        source = block.get("source")
        if not isinstance(source, dict):
            return False
        return bool(source.get("data") or source.get("url"))
    return False


def _movement_authority_refusal_check(
    adapter: ExternalHarnessMcpAdapter,
    tool_names: set[str],
) -> HarnessValidationCheck:
    """Probe that a tokenless external confirmation cannot authorize movement."""

    if "rotate_in_place" not in tool_names:
        return _fail(
            "movement_authority_refusal",
            "Cannot probe movement authority refusal because rotate_in_place is not exposed.",
            missing_tool="rotate_in_place",
        )
    try:
        text = adapter.call_tool_text(
            "rotate_in_place",
            {"degrees": 0.0},
            meta=untrusted_movement_confirmation_probe_meta(),
        )
    except Exception as e:
        return _fail(
            "movement_authority_refusal",
            f"Movement authority refusal probe failed: {e}",
        )

    if "requires a one-shot DimOS movement authority token" in text:
        return _pass(
            "movement_authority_refusal",
            "Tokenless external movement confirmation is refused before dispatch.",
        )
    return _fail(
        "movement_authority_refusal",
        "Tokenless external movement confirmation was not refused as expected.",
        response=text[:500],
    )


def _hidden_movement_refusal_check(
    adapter: ExternalHarnessMcpAdapter,
    policy: AgentRuntimePolicy | None,
) -> HarnessValidationCheck:
    """Probe that a hidden movement tool is still refused at dispatch."""

    probe_tool = "rotate_in_place"
    probe_args = {"degrees": 0.0}
    if policy is None:
        return _fail(
            "hidden_movement_refusal",
            "Cannot probe hidden movement dispatch without a valid runtime policy.",
            probe_tool=probe_tool,
        )
    if policy.tool_policy_admin_override:
        return _fail(
            "hidden_movement_refusal",
            "Cannot safely probe hidden movement dispatch while admin override is enabled.",
            probe_tool=probe_tool,
        )
    if policy.tool_scope != "readonly":
        return _fail(
            "hidden_movement_refusal",
            "Hidden movement dispatch probe requires readonly tool scope.",
            probe_tool=probe_tool,
            tool_scope=policy.tool_scope,
        )
    try:
        text = adapter.call_tool_text(
            probe_tool,
            probe_args,
            meta=external_harness_probe_meta(),
        )
    except Exception as e:
        return _fail(
            "hidden_movement_refusal",
            f"Hidden movement dispatch probe failed: {e}",
            probe_tool=probe_tool,
        )

    if _is_hidden_movement_refusal(text):
        return _pass(
            "hidden_movement_refusal",
            "Hidden movement tool dispatch is refused for an external harness client.",
            probe_tool=probe_tool,
        )
    return _fail(
        "hidden_movement_refusal",
        "Hidden movement tool dispatch was not refused as expected.",
        probe_tool=probe_tool,
        response=text[:500],
    )


def _is_hidden_movement_refusal(text: str) -> bool:
    normalized = text.lower()
    return (
        "runtime tool scope is readonly" in normalized
        or "runtime movement policy is deny" in normalized
        or "tool not found" in normalized
    )
