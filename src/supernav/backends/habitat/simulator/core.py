# Copyright (c) Meta Platforms, Inc. and its affiliates.
# Licensed under the MIT license; see THIRD_PARTY_NOTICES.md in the repository root.

from __future__ import annotations

import logging
import time
from collections import deque
from functools import partial
from importlib import import_module
from typing import Any, Callable, Dict, Mapping, Optional

from supernav.backends.habitat.simulator import sdk as habitat_sim

from supernav.backends.habitat.simulator.types import _API_VERSION, _Session


# Actions that originate from the rerun viewer or internal tooling — excluded
# from the action ring so only genuine agent tool calls appear in the log.
_VIEWER_ACTIONS: frozenset = frozenset({
    "get_visuals",
    "get_topdown_map",
    "analyze_depth",
    "get_runtime_status",
    "get_audit_metrics",
    "get_collision_profile",
    "describe_api",
    "preload_localnav",
})


_NAVIGATION_MODULES = {
    "navigate_with_localnav": "supernav.methods.localnav.navigation",
    "preload_localnav": "supernav.methods.localnav.navigation",
    **{action: "supernav.methods.navigation.oracle_local_nav" for action in (
        "navigate_oracle_local", "navigate_visual_local", "navigate_visual_ground_preview",
        "navigate_visual_overlay", "navigate_visual_point", "get_oracle_local_map",
    )},
}


def _dispatch_navigation(adapter, module_name, action, session_id, payload):
    return getattr(import_module(module_name), action)(adapter, session_id, payload)


class HabitatAdapterCoreMixin:
    """Core request lifecycle and generic response handling."""

    def __init__(
        self,
        simulator_factory: Optional[
            Callable[[Mapping[str, Any]], habitat_sim.Simulator]
        ] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._simulator_factory = simulator_factory or self._default_simulator_factory
        self._logger = logger or logging.getLogger("habitat_gs.adapter")
        self._started_at_s = time.monotonic()
        self._sessions: Dict[str, _Session] = {}
        # Ring buffer of recent agent tool calls (viewer polling excluded).
        self._action_ring: deque = deque(maxlen=200)
        self._handlers: Dict[
            str, Callable[[Optional[str], Mapping[str, Any]], Dict[str, Any]]
        ] = {
            "describe_api": self._describe_api,
            "init_scene": self._init_scene,
            "get_scene_info": self._get_scene_info,
            "set_agent_state": self._set_agent_state,
            "reset_agent_pose": self._reset_agent_pose,
            "sample_navigable_point": self._sample_navigable_point,
            "find_shortest_path": self._find_shortest_path,
            "get_topdown_map": self._get_topdown_map,
            "navigate_step": self._navigate_step,
            "step_action": self._step_action,
            "step_and_capture": self._step_and_capture,
            "get_observation": self._get_observation,
            "get_visuals": self._get_visuals,
            "export_video_trace": self._export_video_trace,
            "get_metrics": self._get_metrics,
            "get_audit_metrics": self._get_audit_metrics,
            "get_runtime_status": self._get_runtime_status_action,
            "get_collision_profile": self._get_collision_profile,
            "get_scene_graph": self._get_scene_graph,
            "close_session": self._close_session,
            "get_panorama": self._get_panorama,
            "analyze_depth": self._analyze_depth,
            "query_depth": self._query_depth,
            "depth_grid": self._depth_grid,
            "side_depth_grid": self._side_depth_grid,
        }
        for action, module_name in _NAVIGATION_MODULES.items():
            self._handlers[action] = partial(_dispatch_navigation, self, module_name, action)

    def handle_request(self, request: Mapping[str, Any]) -> Dict[str, Any]:
        request_id = request.get("request_id")
        action = request.get("action")
        session_id = request.get("session_id")
        payload = request.get("payload", {})

        if not isinstance(action, str):
            return self._error_response(
                request_id=request_id,
                action="unknown",
                session_id=session_id,
                error_type="ValueError",
                error_message='Missing or invalid "action" field',
            )

        if not isinstance(payload, Mapping):
            return self._error_response(
                request_id=request_id,
                action=action,
                session_id=session_id,
                error_type="ValueError",
                error_message='Field "payload" must be a JSON object',
            )

        handler = self._handlers.get(action)
        if handler is None:
            return self._error_response(
                request_id=request_id,
                action=action,
                session_id=session_id,
                error_type="ValueError",
                error_message=f"Unsupported action: {action}",
            )
        if action == "get_audit_metrics" and not bool(
            request.get("_audit_internal_authorized", False)
        ):
            return self._error_response(
                request_id=request_id,
                action=action,
                session_id=session_id,
                error_type="PermissionError",
                error_message="get_audit_metrics is bridge-internal only",
            )

        try:
            result = handler(session_id, payload)
            response_session_id = result.get("session_id", session_id)
            if isinstance(response_session_id, str):
                session = self._sessions.get(response_session_id)
                if session is not None:
                    session.last_activity_s = time.monotonic()
            if action not in _VIEWER_ACTIONS:
                self._action_ring.append({
                    "t": round(time.time(), 3),
                    "action": action,
                    "ok": True,
                    "session_id": response_session_id or session_id,
                })
            return self._ok_response(
                request_id=request_id,
                action=action,
                session_id=response_session_id,
                result=result,
            )
        except Exception as exc:  # noqa: BLE001
            if action not in _VIEWER_ACTIONS:
                self._action_ring.append({
                    "t": round(time.time(), 3),
                    "action": action,
                    "ok": False,
                    "session_id": session_id,
                    "error": str(exc)[:120],
                })
            return self._error_response(
                request_id=request_id,
                action=action,
                session_id=session_id,
                error_type=type(exc).__name__,
                error_message=str(exc),
            )

    def close_all(self) -> None:
        for session_id in list(self._sessions):
            self._dispose_session(self._sessions.pop(session_id), reason="close_all")

    def get_runtime_status(self) -> Dict[str, Any]:
        now = time.monotonic()
        return {
            "api_version": _API_VERSION,
            "active_sessions": len(self._sessions),
            "uptime_s": round(time.monotonic() - self._started_at_s, 3),
            "sessions": [
                {
                    "session_id": session.session_id,
                    "scene": session.scene,
                    "step_count": session.step_count,
                    "age_s": round(now - session.created_at_s, 3),
                    "idle_s": round(now - session.last_activity_s, 3),
                }
                for session in self._sessions.values()
            ],
            # Last 50 agent tool calls (viewer polling excluded).
            "action_ring": list(self._action_ring)[-50:],
        }

    def _get_runtime_status_action(
        self, session_id: Optional[str], payload: Mapping[str, Any]
    ) -> Dict[str, Any]:
        return self.get_runtime_status()

    def reap_idle_sessions(self, idle_timeout_s: float) -> list[str]:
        if idle_timeout_s <= 0:
            return []

        now = time.monotonic()
        expired_session_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if (now - session.last_activity_s) >= idle_timeout_s
        ]
        for session_id in expired_session_ids:
            session = self._sessions.pop(session_id)
            self._dispose_session(session, reason="idle_timeout")
        return expired_session_ids

    @staticmethod
    def _ok_response(
        request_id: Any,
        action: str,
        session_id: Optional[str],
        result: Mapping[str, Any],
    ) -> Dict[str, Any]:
        return {
            "ok": True,
            "action": action,
            "request_id": request_id,
            "session_id": session_id,
            "result": dict(result),
        }

    @staticmethod
    def _error_response(
        request_id: Any,
        action: str,
        session_id: Any,
        error_type: str,
        error_message: str,
    ) -> Dict[str, Any]:
        return {
            "ok": False,
            "action": action,
            "request_id": request_id,
            "session_id": session_id,
            "error": {
                "type": error_type,
                "message": error_message,
            },
        }
