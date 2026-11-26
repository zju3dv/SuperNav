"""Small, model-visible spatial fact ledger for benchmark MCP sessions.

The ledger deliberately records only agent-registered image observations and
formal tool outcomes.  It does not inspect simulator coordinates, match places
across images, or assign room semantics.
"""

from __future__ import annotations

import hashlib
import secrets
from collections import deque
from copy import deepcopy
from typing import Any, Mapping, Sequence

_MOTION_TOOLS = {
    "forward",
    "backward",
    "turn",
    "look_vertical",
    "panorama",
    "navigate",
    "visual_local_navigate",
    "visual_overlay_navigate",
    "visual_point_navigate",
    "oracle_local_navigate",
}

_BLOCKED_FRONTIER_DISPOSITIONS = {
    "freshly_verified_duplicate",
    "unsafe",
    "visibly_impassable",
    "attempted_unreachable",
}


class SpatialMemoryError(ValueError):
    """A user-facing spatial-memory registration error."""


class SpatialMemoryLedger:
    """Bounded junction/branch facts owned by one MCP process."""

    def __init__(self, *, max_junctions: int = 3, long_move_m: float = 3.0) -> None:
        self.max_junctions = max(1, int(max_junctions))
        self.long_move_m = max(0.0, float(long_move_m))
        self.reset("")

    def reset(self, session_id: str) -> None:
        self.session_id = str(session_id or "")
        self.revision = 0
        self._junction_seq = 0
        self._branch_seq = 0
        self._event_seq = 0
        self._junctions: deque[dict[str, Any]] = deque()
        self._branches: dict[str, dict[str, Any]] = {}
        self._capture_registrations: dict[str, dict[str, Any]] = {}
        self._pending_tokens: dict[str, dict[str, Any]] = {}
        self._blocked_close_audit: dict[str, Any] | None = None
        self._approved_blocked_close_audit: dict[str, Any] | None = None
        self._achieved_close_audit: dict[str, Any] | None = None
        self._approved_achieved_close_audit: dict[str, Any] | None = None
        self._post_entry_followup_branch_id: str | None = None
        self._deferred_events: list[dict[str, Any]] = []

    def begin_tool(self, tool_name: str) -> None:
        """Invalidate a challenged close audit after any intervening tool."""

        if str(tool_name or "") == "close_session":
            return
        if self._blocked_close_audit is None and self._achieved_close_audit is None:
            return
        if self._blocked_close_audit is not None:
            self._blocked_close_audit = None
            self._deferred_events.append(
                self._event(
                    "blocked_close_audit_invalidated",
                    reason="intervening_non_close_tool",
                )
            )
        if self._achieved_close_audit is not None:
            self._achieved_close_audit = None
            self._deferred_events.append(
                self._event(
                    "achieved_close_audit_invalidated",
                    reason="intervening_non_close_tool",
                )
            )

    def evaluate_blocked_close(self, args: Mapping[str, Any]) -> dict[str, Any] | None:
        """Challenge or validate a blocked close before bridge dispatch.

        ``None`` authorizes normal close dispatch. A returned body is a
        model-visible non-closing result.
        """

        outcome = str(args.get("outcome") or "").strip().lower()
        if outcome != "blocked":
            return None
        frontier = self.frontier()
        branch_ids = list(frontier["unentered_branch_ids"])
        followup_required = self._post_entry_followup_branch_id is not None
        if not branch_ids and not followup_required:
            return None

        token = str(args.get("close_audit_token") or "").strip()
        if not token:
            events = self._drain_deferred_events()
            token = secrets.token_urlsafe(18)
            self._blocked_close_audit = {
                "token": token,
                "session_id": self.session_id,
                "frontier_state": self._frontier_state(),
                "post_entry_followup_branch_id": self._post_entry_followup_branch_id,
            }
            events.append(
                self._event(
                    "blocked_close_audit_requested",
                    branch_ids=branch_ids,
                    post_entry_followup_required=followup_required,
                )
            )
            return {
                "ok": False,
                "status": "blocked_close_audit_required",
                "closed": False,
                "close_audit_token": token,
                "blocked_close_gate": {
                    "frontier": frontier["unentered_branches"],
                    "post_entry_followup_required": followup_required,
                },
                "spatial_memory": self.result(events),
            }

        binding = self._blocked_close_audit
        if binding is None or not secrets.compare_digest(
            str(binding.get("token") or ""), token
        ):
            return self._blocked_close_error("stale or unknown close_audit_token")
        if (
            binding.get("session_id") != self.session_id
            or binding.get("frontier_state") != self._frontier_state()
            or binding.get("post_entry_followup_branch_id")
            != self._post_entry_followup_branch_id
        ):
            self._blocked_close_audit = None
            events = self._drain_deferred_events()
            events.append(
                self._event(
                    "blocked_close_audit_invalidated",
                    reason="spatial_memory_state_changed",
                )
            )
            return self._blocked_close_error(
                "close audit no longer matches current spatial memory",
                events=events,
            )

        try:
            frontier_audit = self._validate_frontier_audit(
                args.get("frontier_audit"), branch_ids
            )
            post_entry_exception = self._validate_post_entry_exception(
                args.get("post_entry_exception"), required=followup_required
            )
        except SpatialMemoryError as exc:
            return self._blocked_close_error(str(exc))

        self._blocked_close_audit = None
        self._approved_blocked_close_audit = {
            "frontier_audit": frontier_audit,
            "post_entry_exception": post_entry_exception,
        }
        return None

    def evaluate_achieved_close(self, args: Mapping[str, Any]) -> dict[str, Any] | None:
        """Challenge or validate an achieved close before bridge dispatch.

        ``None`` authorizes normal close dispatch. A returned body is a
        model-visible non-closing result. Every achieved close must carry a
        one-shot arrival confirmation so a "looks close" stop cannot slip
        through without explicit verification.
        """

        outcome = str(args.get("outcome") or "").strip().lower()
        if outcome != "achieved":
            return None

        token = str(args.get("close_audit_token") or "").strip()
        if not token:
            events = self._drain_deferred_events()
            token = secrets.token_urlsafe(18)
            self._achieved_close_audit = {
                "token": token,
                "session_id": self.session_id,
            }
            events.append(self._event("achieved_close_audit_requested"))
            return {
                "ok": False,
                "status": "achieved_close_audit_required",
                "closed": False,
                "close_audit_token": token,
                "achieved_close_gate": {
                    "instruction": (
                        "An achieved close requires one explicit arrival "
                        "confirmation. Re-inspect your latest surround and verify "
                        "ALL of: (1) you spent one hab_local_navigate hop marked "
                        "on the target's body or base (or that hop returned "
                        "blocked/no_progress at touch distance); (2) the "
                        "target's base is cut by the frame's bottom edge and its "
                        "body dominates the view; (3) your estimated distance to "
                        "the target is at most 1.0 m. If every criterion holds, "
                        "retry hab_close_session immediately with "
                        "outcome='achieved', this close_audit_token, and "
                        "arrival_confirmation={final_approach_done, "
                        "target_base_cut, estimated_distance_m, evidence}. If "
                        "any criterion fails, do NOT close achieved: resume the "
                        "final-approach procedure, or close blocked when "
                        "recovery is exhausted. Any intervening tool call "
                        "invalidates this token."
                    ),
                },
                "spatial_memory": self.result(events),
            }

        binding = self._achieved_close_audit
        if binding is None or not secrets.compare_digest(
            str(binding.get("token") or ""), token
        ):
            return self._achieved_close_error("stale or unknown close_audit_token")
        if binding.get("session_id") != self.session_id:
            self._achieved_close_audit = None
            return self._achieved_close_error(
                "close audit no longer matches the active session"
            )

        try:
            confirmation = self._validate_arrival_confirmation(
                args.get("arrival_confirmation")
            )
        except SpatialMemoryError as exc:
            return self._achieved_close_error(str(exc))

        self._achieved_close_audit = None
        self._approved_achieved_close_audit = confirmation
        return None

    def register_junction(
        self,
        branches: Any,
        panorama_rows: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        normalized = self._normalize_registration(branches, panorama_rows)
        signature = tuple(
            f"{row['image_ref']}|{row['view']}|{row['point']}|{row['agent_phrase']}"
            for row in normalized
        )
        capture_keys = {_capture_key(str(row["image_ref"])) for row in normalized}
        if len(capture_keys) != 1:
            raise SpatialMemoryError(
                "all branches must refer to views from the same latest panorama"
            )
        capture_key = capture_keys.pop()
        existing = self._capture_registrations.get(capture_key)
        if existing is not None:
            if existing["signature"] != signature:
                raise SpatialMemoryError(
                    "latest panorama already has a different registered junction; "
                    "take a fresh panorama before correcting it"
                )
            return self.result([])

        self._junction_seq += 1
        junction_id = f"junction_{self._junction_seq:04d}"
        junction = {
            "junction_id": junction_id,
            "source_capture": _capture_label(normalized[0]["image_ref"]),
            "branches": [],
        }
        events = [self._event("junction_registered", junction_id=junction_id)]
        for observation in normalized:
            self._branch_seq += 1
            branch_id = f"branch_{self._branch_seq:04d}"
            branch = {
                "branch_id": branch_id,
                "status": "observed",
                "observation": observation,
            }
            junction["branches"].append(branch)
            self._branches[branch_id] = branch
            events.append(
                self._event(
                    "branch_observed",
                    junction_id=junction_id,
                    branch_id=branch_id,
                    status_after="observed",
                )
            )

        self._junctions.append(junction)
        self._capture_registrations[capture_key] = {
            "signature": signature,
            "junction_id": junction_id,
        }
        if len(self._junctions) > self.max_junctions:
            evicted = self._junctions.popleft()
            for branch in evicted["branches"]:
                self._branches.pop(str(branch["branch_id"]), None)
            stale_keys = [
                key
                for key, value in self._capture_registrations.items()
                if value.get("junction_id") == evicted["junction_id"]
            ]
            for key in stale_keys:
                self._capture_registrations.pop(key, None)
            events.append(
                self._event(
                    "junction_evicted",
                    junction_id=str(evicted["junction_id"]),
                )
            )
        return self.result(events)

    def validate_grounding_arguments(self, args: Mapping[str, Any]) -> None:
        """Reject branch references that cannot be audited before movement."""

        branch_id = str(args.get("branch_id") or "").strip()
        if not branch_id:
            return
        if branch_id not in self._branches:
            raise SpatialMemoryError(f"unknown or evicted branch_id: {branch_id}")
        token = str(args.get("confirm_token") or "").strip()
        if not token:
            return
        binding = self._pending_tokens.get(_token_ref(token))
        if binding is None:
            raise SpatialMemoryError(
                "confirm_token is not bound to an active spatial-memory branch"
            )
        if binding.get("branch_id") != branch_id:
            raise SpatialMemoryError(
                "confirm_token is bound to a different spatial-memory branch"
            )

    def process_tool_result(
        self,
        tool_name: str,
        args: Mapping[str, Any],
        body: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Update facts from one formal result and return newly emitted events."""

        name = str(tool_name or "")
        inputs = dict(args)
        payload = dict(body)
        events: list[dict[str, Any]] = self._drain_deferred_events()
        pending_followup_before = self._post_entry_followup_branch_id

        if name == "init_scene":
            self.reset(str(payload.get("session_id") or self.session_id))
            return self.result([])

        if name == "visual_ground_preview":
            events.extend(self._process_grounding(inputs, payload))
        elif name in _MOTION_TOOLS:
            events.extend(self._invalidate_pending("fresh_observation"))
            events.extend(self._long_move_events(inputs, payload))
        elif name == "close_session":
            events.extend(self._invalidate_pending("session_closed"))
            outcome = str(inputs.get("outcome") or "unknown").strip().lower()
            if outcome not in {"achieved", "blocked"}:
                outcome = "unknown"
            if outcome == "achieved":
                approved = self._approved_achieved_close_audit
                self._approved_achieved_close_audit = None
                if approved is not None:
                    events.append(
                        self._event(
                            "achieved_arrival_confirmed",
                            estimated_distance_m=approved["estimated_distance_m"],
                            evidence=approved["evidence"],
                            diagnostic={
                                "code": "achieved_arrival_confirmed",
                                "severity": "info",
                            },
                        )
                    )
            if outcome == "blocked":
                approved = self._approved_blocked_close_audit
                self._approved_blocked_close_audit = None
                if approved is not None:
                    events.append(
                        self._event(
                            "blocked_frontier_audited",
                            frontier_audit=approved["frontier_audit"],
                            post_entry_exception=approved["post_entry_exception"],
                            diagnostic={
                                "code": "blocked_frontier_audited",
                                "severity": "info",
                            },
                        )
                    )
                else:
                    frontier = self.frontier()
                    if frontier["unentered_branch_ids"]:
                        events.append(
                            self._event(
                                "blocked_not_exhausted",
                                branch_ids=list(frontier["unentered_branch_ids"]),
                                diagnostic={
                                    "code": "blocked_not_exhausted",
                                    "severity": "warning",
                                    "unentered_branches": frontier[
                                        "unentered_branches"
                                    ],
                                },
                            )
                        )
        if (
            pending_followup_before is not None
            and self._post_entry_followup_branch_id == pending_followup_before
            and self._is_followup_observation(name, payload)
        ):
            self._post_entry_followup_branch_id = None
            events.append(
                self._event(
                    "post_entry_followup_observed",
                    branch_id=pending_followup_before,
                )
            )
        return self.result(events)

    def result(self, events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        event_rows = [dict(event) for event in events]
        return {
            "schema_version": 1,
            "revision": self.revision,
            "current_junction_id": (
                self._junctions[-1]["junction_id"] if self._junctions else None
            ),
            "recent_junctions": deepcopy(list(self._junctions)),
            "frontier": self.frontier(),
            "new_events": event_rows,
        }

    def frontier(self) -> dict[str, Any]:
        unentered: list[dict[str, Any]] = []
        for junction in reversed(self._junctions):
            if len(junction["branches"]) < 2:
                continue
            for branch in junction["branches"]:
                if branch.get("status") == "entered":
                    continue
                observation = branch.get("observation", {})
                unentered.append(
                    {
                        "junction_id": junction["junction_id"],
                        "branch_id": branch["branch_id"],
                        "status": branch.get("status"),
                        "image_ref": observation.get("image_ref"),
                        "view": observation.get("view"),
                        "point": observation.get("point"),
                        "agent_phrase": observation.get("agent_phrase"),
                    }
                )
        return {
            "unentered_branch_ids": [row["branch_id"] for row in unentered],
            "unentered_branches": unentered,
        }

    def _normalize_registration(
        self,
        branches: Any,
        panorama_rows: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        if not isinstance(branches, list) or not 2 <= len(branches) <= 4:
            raise SpatialMemoryError("branches must contain 2 to 4 opening candidates")
        by_view = {
            str(row.get("direction") or row.get("view") or "").strip().lower(): row
            for row in panorama_rows
            if isinstance(row, Mapping)
        }
        normalized: list[dict[str, Any]] = []
        seen: set[tuple[str, str, tuple[float, float], str]] = set()
        for raw in branches:
            if not isinstance(raw, Mapping):
                raise SpatialMemoryError("each branch must be an object")
            view = str(raw.get("view") or "").strip().lower()
            row = by_view.get(view)
            if row is None:
                raise SpatialMemoryError(
                    "each branch view must refer to the latest front/right/back/left panorama"
                )
            image_ref = str(row.get("image_ref") or "").strip()
            if not image_ref:
                raise SpatialMemoryError("latest panorama row has no image_ref")
            phrase = str(raw.get("phrase") or "").strip()
            if not phrase:
                raise SpatialMemoryError("each branch requires a non-empty phrase")
            point = raw.get("point")
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise SpatialMemoryError("each branch point must be normalized [x, y]")
            try:
                xy = (round(float(point[0]), 6), round(float(point[1]), 6))
            except (TypeError, ValueError) as exc:
                raise SpatialMemoryError(
                    "branch point coordinates must be numbers"
                ) from exc
            if not all(0.0 <= value <= 1.0 for value in xy):
                raise SpatialMemoryError(
                    "branch point coordinates must be between 0 and 1"
                )
            key = (image_ref, view, xy, phrase)
            if key in seen:
                raise SpatialMemoryError("duplicate branch registration")
            seen.add(key)
            normalized.append(
                {
                    "image_ref": image_ref,
                    "view": view,
                    "point": [xy[0], xy[1]],
                    "agent_phrase": phrase,
                    "evidence_kind": "agent_registered_visual_candidate",
                }
            )
        normalized.sort(
            key=lambda row: (
                ("front", "right", "back", "left").index(row["view"]),
                row["point"][0],
                row["point"][1],
                row["agent_phrase"],
            )
        )
        return normalized

    def _process_grounding(
        self, args: Mapping[str, Any], body: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        branch_id = str(args.get("branch_id") or "").strip()
        token = str(args.get("confirm_token") or "").strip()
        token_ref = _token_ref(token) if token else ""

        if token:
            binding = self._pending_tokens.pop(token_ref, None)
            if (
                not branch_id
                or binding is None
                or binding.get("branch_id") != branch_id
            ):
                return events
            branch = self._branches.get(branch_id)
            if branch is None:
                return events
            selected_unentered = branch.get("status") != "entered"
            status = str(body.get("status") or "")
            if status in {
                "expired_confirm_token",
                "stale_confirm_token",
                "invalid_confirm_token",
                "confirm_mismatch",
            }:
                events.append(
                    self._event(
                        "confirm_invalidated",
                        branch_id=branch_id,
                        confirm_token_ref=token_ref,
                        reason=status,
                    )
                )
                return events
            events.extend(
                self._apply_movement(
                    branch_id,
                    body,
                    selection_mode="explicit_confirm",
                    confirm_token_ref=token_ref,
                )
            )
            if not selected_unentered:
                events.extend(self._long_move_events(args, body))
            return events

        events.extend(self._invalidate_pending("superseded_by_new_preview"))
        if not branch_id:
            events.extend(self._long_move_events(args, body))
            return events
        branch = self._branches.get(branch_id)
        if branch is None:
            return events
        selected_unentered = branch.get("status") != "entered"
        image_ref = str(body.get("image_ref") or args.get("image_ref") or "")
        view = str(body.get("direction") or args.get("view") or "")
        phrase = str(body.get("phrase") or args.get("phrase") or "")
        branch["latest_reobservation"] = {
            "image_ref": image_ref,
            "view": view,
            "agent_phrase": phrase,
            "evidence_kind": "agent_asserted_reobservation",
        }
        status = str(body.get("status") or "")
        confirm_token = str(body.get("confirm_token") or "")
        if status == "preview_ready" and confirm_token:
            ref = _token_ref(confirm_token)
            self._pending_tokens[ref] = {
                "branch_id": branch_id,
                "image_ref": image_ref,
            }
            events.append(
                self._event(
                    "ground_preview_bound",
                    branch_id=branch_id,
                    confirm_token_ref=ref,
                )
            )
        elif body.get("auto_confirmed") or status in {
            "reached_visual_ground_candidate",
            "en_route_visual_ground_candidate",
            "navigation_blocked",
            "self_check_required",
        }:
            events.extend(
                self._apply_movement(
                    branch_id,
                    body,
                    selection_mode="auto_confirmed_single",
                )
            )
        if not selected_unentered:
            events.extend(self._long_move_events(args, body))
        return events

    def _apply_movement(
        self,
        branch_id: str,
        body: Mapping[str, Any],
        *,
        selection_mode: str,
        confirm_token_ref: str | None = None,
    ) -> list[dict[str, Any]]:
        branch = self._branches.get(branch_id)
        if branch is None:
            return []
        old_status = str(branch.get("status") or "observed")
        steps = _number(body.get("steps_executed"), default=0.0)
        displacement = _number(body.get("displacement_m"), default=0.0)
        nav_status = str(body.get("status") or body.get("nav_status") or "")
        new_status = old_status
        action = "selection_failed"
        if steps > 0 and displacement >= 0.15:
            if nav_status == "reached_visual_ground_candidate":
                new_status = "entered"
                action = "branch_entered"
                self._post_entry_followup_branch_id = branch_id
            else:
                new_status = "approached"
                action = "branch_approached"
        branch["status"] = new_status
        branch["movement"] = {
            "selection_mode": selection_mode,
            "source_candidate_id": body.get("selected_candidate_id"),
            "nav_status": nav_status,
            "planned_path_m": body.get("planned_path_m"),
            "displacement_m": body.get("displacement_m"),
            "steps_executed": body.get("steps_executed"),
        }
        return [
            self._event(
                action,
                branch_id=branch_id,
                status_before=old_status,
                status_after=new_status,
                source_candidate_id=body.get("selected_candidate_id"),
                planned_path_m=body.get("planned_path_m"),
                displacement_m=body.get("displacement_m"),
                confirm_token_ref=confirm_token_ref,
            )
        ]

    def _invalidate_pending(self, reason: str) -> list[dict[str, Any]]:
        events = [
            self._event(
                (
                    "confirm_abandoned"
                    if "superseded" in reason
                    else "confirm_invalidated"
                ),
                branch_id=str(binding.get("branch_id") or ""),
                confirm_token_ref=token_ref,
                reason=reason,
            )
            for token_ref, binding in self._pending_tokens.items()
        ]
        self._pending_tokens.clear()
        return events

    def _long_move_events(
        self, args: Mapping[str, Any], body: Mapping[str, Any]
    ) -> list[dict[str, Any]]:
        frontier = self.frontier()
        ids = list(frontier["unentered_branch_ids"])
        if not ids:
            return []
        branch_id = str(args.get("branch_id") or "")
        if branch_id and branch_id in ids:
            return []
        planned = body.get("planned_path_m")
        if str(body.get("status") or "") == "preview_ready":
            values = [
                _number(candidate.get("planned_path_m"), default=0.0)
                for candidate in body.get("candidates", [])
                if isinstance(candidate, Mapping)
            ]
            planned = max(values, default=0.0)
        distance = _number(planned, default=0.0)
        if distance < self.long_move_m:
            return []
        action = (
            "frontier_reminder"
            if str(body.get("status") or "") == "preview_ready"
            else "long_move_with_local_frontier"
        )
        return [
            self._event(
                action,
                branch_ids=ids,
                planned_path_m=round(distance, 4),
                diagnostic={
                    "code": action,
                    "severity": "warning",
                    "planned_path_m": round(distance, 4),
                    "unentered_branch_ids": ids,
                },
            )
        ]

    def _event(self, action: str, **fields: Any) -> dict[str, Any]:
        self._event_seq += 1
        self.revision += 1
        event = {
            "event_id": f"spatial_memory_{self._event_seq:06d}",
            "action": action,
            "revision": self.revision,
        }
        event.update({key: value for key, value in fields.items() if value is not None})
        return event

    def _drain_deferred_events(self) -> list[dict[str, Any]]:
        events = self._deferred_events
        self._deferred_events = []
        return events

    def _frontier_state(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (str(row["branch_id"]), str(row.get("status") or ""))
            for row in self.frontier()["unentered_branches"]
        )

    def _blocked_close_error(
        self,
        error: str,
        *,
        events: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "status": "invalid_blocked_close_audit",
            "closed": False,
            "error": error,
            "spatial_memory": self.result(events),
        }

    def _achieved_close_error(
        self,
        error: str,
        *,
        events: Sequence[Mapping[str, Any]] = (),
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "status": "invalid_achieved_close_audit",
            "closed": False,
            "error": error,
            "spatial_memory": self.result(events),
        }

    def _validate_arrival_confirmation(self, value: Any) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise SpatialMemoryError(
                "arrival_confirmation is required for the audited achieved-close retry"
            )

        def _flag(name: str) -> bool:
            raw = value.get(name)
            if not isinstance(raw, bool):
                raise SpatialMemoryError(
                    f"arrival_confirmation.{name} must be a boolean"
                )
            return raw

        final_approach_done = _flag("final_approach_done")
        target_base_cut = _flag("target_base_cut")
        try:
            estimated = float(value.get("estimated_distance_m"))
        except (TypeError, ValueError):
            raise SpatialMemoryError(
                "arrival_confirmation.estimated_distance_m must be a number"
            ) from None
        evidence = str(value.get("evidence") or "").strip()
        if not evidence:
            raise SpatialMemoryError(
                "arrival_confirmation.evidence must describe the visible proof"
            )
        if not final_approach_done or not target_base_cut or estimated > 1.0:
            raise SpatialMemoryError(
                "arrival criteria not met: do not close achieved; resume the "
                "final-approach procedure, or close blocked when recovery is "
                "exhausted"
            )
        return {
            "final_approach_done": True,
            "target_base_cut": True,
            "estimated_distance_m": estimated,
            "evidence": evidence,
        }

    def _validate_frontier_audit(
        self, value: Any, expected_branch_ids: Sequence[str]
    ) -> list[dict[str, str]]:
        if not expected_branch_ids:
            if value not in (None, []):
                raise SpatialMemoryError(
                    "frontier_audit must be omitted when frontier is empty"
                )
            return []
        if not isinstance(value, list):
            raise SpatialMemoryError("frontier_audit must audit every active branch")
        normalized: list[dict[str, str]] = []
        seen: set[str] = set()
        for row in value:
            if not isinstance(row, Mapping):
                raise SpatialMemoryError("each frontier_audit entry must be an object")
            branch_id = str(row.get("branch_id") or "").strip()
            disposition = str(row.get("disposition") or "").strip()
            evidence = str(row.get("evidence") or "").strip()
            if not branch_id or branch_id in seen:
                raise SpatialMemoryError(
                    "frontier_audit has a missing or duplicate branch_id"
                )
            if disposition not in _BLOCKED_FRONTIER_DISPOSITIONS:
                raise SpatialMemoryError(
                    "frontier_audit disposition must be freshly_verified_duplicate, "
                    "unsafe, visibly_impassable, or attempted_unreachable"
                )
            if not evidence:
                raise SpatialMemoryError(
                    "every frontier_audit entry requires non-empty evidence"
                )
            seen.add(branch_id)
            normalized.append(
                {
                    "branch_id": branch_id,
                    "disposition": disposition,
                    "evidence": evidence,
                }
            )
        expected = set(expected_branch_ids)
        if seen != expected:
            missing = sorted(expected - seen)
            unknown = sorted(seen - expected)
            raise SpatialMemoryError(
                f"frontier_audit must exactly cover active frontier; "
                f"missing={missing}, unknown={unknown}"
            )
        normalized.sort(key=lambda row: row["branch_id"])
        return normalized

    def _validate_post_entry_exception(
        self, value: Any, *, required: bool
    ) -> dict[str, str] | None:
        if not required:
            if value is not None:
                raise SpatialMemoryError(
                    "post_entry_exception must be omitted when no follow-up is pending"
                )
            return None
        if not isinstance(value, Mapping):
            raise SpatialMemoryError(
                "post_entry_exception is required after a just-entered branch"
            )
        disposition = str(value.get("disposition") or "").strip()
        evidence = str(value.get("evidence") or "").strip()
        if disposition != "no_safe_passable_interior" or not evidence:
            raise SpatialMemoryError(
                "post_entry_exception requires disposition=no_safe_passable_interior "
                "and non-empty evidence"
            )
        return {"disposition": disposition, "evidence": evidence}

    @staticmethod
    def _is_followup_observation(tool_name: str, body: Mapping[str, Any]) -> bool:
        if tool_name in {"turn", "panorama", "look_vertical"}:
            return True
        if tool_name not in _MOTION_TOOLS and tool_name != "visual_ground_preview":
            return False
        if str(body.get("status") or "") == "preview_ready":
            return False
        return bool(body.get("panorama_images")) or (
            _number(body.get("steps_executed"), default=0.0) > 0
            or _number(body.get("displacement_m"), default=0.0) > 0
        )


def _token_ref(token: str) -> str:
    return "token:" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:12]


def _capture_label(image_ref: str) -> str:
    parts = str(image_ref).split(":")
    return parts[-2] if len(parts) >= 2 else str(image_ref)


def _capture_key(image_ref: str) -> str:
    parts = str(image_ref).rsplit(":", 1)
    return parts[0] if len(parts) == 2 else str(image_ref)


def _number(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


__all__ = ["SpatialMemoryError", "SpatialMemoryLedger"]
