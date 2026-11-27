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

"""Server-side movement lease accounting.

A movement lease is a task-level authorization budget. It replaces per-tool
confirmation in live sessions without turning robot movement into an unbounded
``allow_all`` mode.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import math
import threading
import time
from typing import Any
import uuid

_MAX_CONSUMED_GRANT_TOKENS = 4096


@dataclass(frozen=True)
class MovementCost:
    """Estimated movement budget consumed by a tool call."""

    tool_calls: int = 1
    linear_m: float = 0.0
    rotation_deg: float = 0.0

    def model_dump(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MovementLeaseTicket:
    """Generation token proving a movement call was authorized by a live lease."""

    client_id: str
    lease_id: str
    epoch: int
    tool_name: str
    cost: MovementCost
    ticket_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def model_dump(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cost"] = self.cost.model_dump()
        return payload


@dataclass(frozen=True)
class MovementLeaseAuthorization:
    """Result of checking and consuming a movement lease."""

    allowed: bool
    reason: str | None
    lease: MovementLease | None
    cost: MovementCost
    ticket: MovementLeaseTicket | None = None


@dataclass
class MovementLease:
    """Task-level authorization to run bounded movement tools."""

    lease_id: str
    epoch: int
    client_id: str
    task: str
    created_at: float
    expires_at: float
    max_duration_s: float = 60.0
    max_linear_m: float = 4.0
    max_rotation_deg: float = 240.0
    max_tool_calls: int = 8
    used_linear_m: float = 0.0
    used_rotation_deg: float = 0.0
    used_tool_calls: int = 0
    last_tool_name: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def expired(self, now: float | None = None) -> bool:
        return (now if now is not None else time.monotonic()) >= self.expires_at

    def remaining(self, now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.monotonic()
        return {
            "duration_s": max(0.0, self.expires_at - now),
            "linear_m": max(0.0, self.max_linear_m - self.used_linear_m),
            "rotation_deg": max(0.0, self.max_rotation_deg - self.used_rotation_deg),
            "tool_calls": max(0, self.max_tool_calls - self.used_tool_calls),
        }

    def model_dump(self, now: float | None = None) -> dict[str, Any]:
        payload = asdict(self)
        payload["remaining"] = self.remaining(now)
        return payload


class MovementLeaseManager:
    """Thread-safe movement lease registry keyed by MCP client/session id."""

    def __init__(self) -> None:
        self._leases: dict[str, MovementLease] = {}
        self._epochs: dict[str, int] = {}
        self._consumed_grant_tokens: set[tuple[str, str]] = set()
        self._consumed_grant_token_order: deque[tuple[str, str]] = deque()
        self._lock = threading.Lock()

    def consume_grant_token(self, *, client_id: str, token: str) -> bool:
        """Consume a one-shot token used to create task movement authority.

        The token is client-generated, so it is not a security boundary. It is a
        replay guard that prevents an old native tool-call payload from granting
        fresh movement authority again after stop/preemption.
        """

        normalized = token.strip()
        if not normalized:
            return False
        key = (client_id, normalized)
        with self._lock:
            if key in self._consumed_grant_tokens:
                return False
            self._consumed_grant_tokens.add(key)
            self._consumed_grant_token_order.append(key)
            while len(self._consumed_grant_token_order) > _MAX_CONSUMED_GRANT_TOKENS:
                expired = self._consumed_grant_token_order.popleft()
                self._consumed_grant_tokens.discard(expired)
            return True

    def grant(
        self,
        *,
        client_id: str,
        task: str,
        now: float | None = None,
        max_duration_s: float = 60.0,
        max_linear_m: float = 4.0,
        max_rotation_deg: float = 240.0,
        max_tool_calls: int = 8,
    ) -> MovementLease:
        now = now if now is not None else time.monotonic()
        with self._lock:
            epoch = self._bump_epoch_locked(client_id)
            lease = MovementLease(
                lease_id=uuid.uuid4().hex,
                epoch=epoch,
                client_id=client_id,
                task=task,
                created_at=now,
                expires_at=now + max_duration_s,
                max_duration_s=max_duration_s,
                max_linear_m=max_linear_m,
                max_rotation_deg=max_rotation_deg,
                max_tool_calls=max_tool_calls,
            )
            self._leases[client_id] = lease
        return lease

    def active(self, client_id: str, now: float | None = None) -> MovementLease | None:
        now = now if now is not None else time.monotonic()
        with self._lock:
            lease = self._leases.get(client_id)
            if lease is None:
                return None
            if lease.expired(now):
                # Leave removal to pop_expired(), so the MCP watchdog can issue
                # the required stop/preemption trace instead of losing the lease.
                return None
            return lease

    def consume(
        self,
        *,
        client_id: str,
        tool_name: str,
        args: dict[str, Any],
        now: float | None = None,
    ) -> tuple[bool, str | None, MovementLease | None, MovementCost]:
        authorization = self.consume_authorization(
            client_id=client_id,
            tool_name=tool_name,
            args=args,
            now=now,
        )
        return (
            authorization.allowed,
            authorization.reason,
            authorization.lease,
            authorization.cost,
        )

    def consume_authorization(
        self,
        *,
        client_id: str,
        tool_name: str,
        args: dict[str, Any],
        now: float | None = None,
    ) -> MovementLeaseAuthorization:
        """Consume movement budget and return a ticket for execution-time recheck."""

        now = now if now is not None else time.monotonic()
        cost = estimate_movement_cost(tool_name, args)
        with self._lock:
            lease = self._leases.get(client_id)
            if lease is None or lease.expired(now):
                # Expired leases are refused here but retained for the watchdog's
                # active preemption path.
                return MovementLeaseAuthorization(False, "No active movement lease.", None, cost)

            if lease.used_tool_calls + cost.tool_calls > lease.max_tool_calls:
                return MovementLeaseAuthorization(
                    False,
                    "Movement lease tool-call budget is exhausted.",
                    lease,
                    cost,
                )
            if lease.used_linear_m + cost.linear_m > lease.max_linear_m:
                return MovementLeaseAuthorization(
                    False,
                    "Movement lease linear-distance budget is exhausted.",
                    lease,
                    cost,
                )
            if lease.used_rotation_deg + cost.rotation_deg > lease.max_rotation_deg:
                return MovementLeaseAuthorization(
                    False,
                    "Movement lease rotation budget is exhausted.",
                    lease,
                    cost,
                )

            ticket = MovementLeaseTicket(
                client_id=client_id,
                lease_id=lease.lease_id,
                epoch=lease.epoch,
                tool_name=tool_name,
                cost=cost,
            )
            lease.used_tool_calls += cost.tool_calls
            lease.used_linear_m += cost.linear_m
            lease.used_rotation_deg += cost.rotation_deg
            lease.last_tool_name = tool_name
            lease.history.append(
                {
                    "tool": tool_name,
                    "cost": cost.model_dump(),
                    "ts": now,
                    "ticket_id": ticket.ticket_id,
                }
            )
            return MovementLeaseAuthorization(True, None, lease, cost, ticket)

    def validate_ticket(
        self,
        ticket: MovementLeaseTicket,
        *,
        now: float | None = None,
    ) -> tuple[bool, str | None, MovementLease | None]:
        """Verify a consumed movement ticket is still current before executing."""

        now = now if now is not None else time.monotonic()
        with self._lock:
            lease = self._leases.get(ticket.client_id)
            if lease is None:
                return (False, "Movement lease was preempted before execution.", None)
            if lease.expired(now):
                # Refuse the stale ticket, but keep the lease available for
                # watchdog preemption.
                return (False, "Movement lease expired before execution.", None)
            if lease.lease_id != ticket.lease_id or lease.epoch != ticket.epoch:
                return (False, "Movement lease was superseded before execution.", lease)
            return (True, None, lease)

    def refund_authorization(
        self,
        ticket: MovementLeaseTicket,
        *,
        reason: str,
        now: float | None = None,
    ) -> bool:
        """Refund a consumed movement ticket when execution is refused pre-RPC."""

        now = now if now is not None else time.monotonic()
        with self._lock:
            lease = self._leases.get(ticket.client_id)
            if (
                lease is None
                or lease.expired(now)
                or lease.lease_id != ticket.lease_id
                or lease.epoch != ticket.epoch
            ):
                return False

            charged_entry = None
            for entry in reversed(lease.history):
                if entry.get("ticket_id") == ticket.ticket_id:
                    charged_entry = entry
                    break
            if charged_entry is None or charged_entry.get("refunded"):
                return False

            charged_entry["refunded"] = True
            charged_entry["refund_reason"] = reason
            charged_entry["refund_ts"] = now
            lease.used_tool_calls = max(0, lease.used_tool_calls - ticket.cost.tool_calls)
            lease.used_linear_m = max(0.0, lease.used_linear_m - ticket.cost.linear_m)
            lease.used_rotation_deg = max(
                0.0,
                lease.used_rotation_deg - ticket.cost.rotation_deg,
            )
            lease.last_tool_name = next(
                (
                    str(entry.get("tool"))
                    for entry in reversed(lease.history)
                    if not entry.get("refunded")
                ),
                None,
            )
            return True

    def clear(self, client_id: str) -> None:
        with self._lock:
            self._leases.pop(client_id, None)
            self._bump_epoch_locked(client_id)

    def preempt(self, client_id: str) -> MovementLease | None:
        """Remove and return the active lease for *client_id*, if present."""

        with self._lock:
            lease = self._leases.pop(client_id, None)
            if lease is not None:
                self._bump_epoch_locked(client_id)
            return lease

    def preempt_expired(
        self,
        client_id: str,
        *,
        now: float | None = None,
    ) -> MovementLease | None:
        """Remove and return an expired lease for one client, if present."""

        now = now if now is not None else time.monotonic()
        with self._lock:
            lease = self._leases.get(client_id)
            if lease is None or not lease.expired(now):
                return None
            self._leases.pop(client_id, None)
            self._bump_epoch_locked(client_id)
            return lease

    def pop_expired(self, now: float | None = None) -> list[MovementLease]:
        """Remove and return leases whose deadline has passed."""

        now = now if now is not None else time.monotonic()
        with self._lock:
            expired = [lease for lease in self._leases.values() if lease.expired(now)]
            for lease in expired:
                self._leases.pop(lease.client_id, None)
                self._bump_epoch_locked(lease.client_id)
            return expired

    def snapshot(self, now: float | None = None) -> dict[str, Any]:
        now = now if now is not None else time.monotonic()
        with self._lock:
            return {
                client_id: lease.model_dump(now)
                for client_id, lease in sorted(self._leases.items())
                if not lease.expired(now)
            }

    def client_ids(self) -> list[str]:
        """Return client ids with retained movement authority records."""

        with self._lock:
            return sorted(self._leases)

    def _bump_epoch_locked(self, client_id: str) -> int:
        epoch = self._epochs.get(client_id, 0) + 1
        self._epochs[client_id] = epoch
        return epoch


def estimate_movement_cost(tool_name: str, args: dict[str, Any]) -> MovementCost:
    """Estimate lease budget cost from a movement tool call's public args."""

    if tool_name in {"relative_move", "drive"}:
        forward = _float_arg(args, "forward")
        left = _float_arg(args, "left")
        degrees = _float_arg(args, "degrees")
        return MovementCost(
            linear_m=math.hypot(forward, left),
            rotation_deg=abs(degrees),
        )
    if tool_name == "rotate_in_place":
        return MovementCost(rotation_deg=abs(_float_arg(args, "degrees")))
    return MovementCost()


def _float_arg(args: dict[str, Any], key: str) -> float:
    value = args.get(key, 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
