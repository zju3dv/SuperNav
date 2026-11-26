"""HTTP client for the habitat bridge (POST /v1/request).

`BridgeClient` forwards MCP tool calls to the external simulator bridge.

"""

from __future__ import annotations

from http.client import HTTPException

import json
import urllib.error
import urllib.request
from urllib.parse import urlparse
from typing import Any, Dict, Optional

from supernav.runtime.support.audit_token import read_audit_token


class BridgeClient:
    """Minimal HTTP client for the habitat bridge (POST /v1/request)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 18911):
        self._request_counter = 0
        self.session_id: Optional[str] = None
        self.configure(host=host, port=port)

    def configure(self, *, host: str, port: int) -> None:
        self.host = host
        self.port = int(port)
        self.base_url = f"http://{self.host}:{self.port}"

    def _next_id(self) -> str:
        self._request_counter += 1
        return f"hab-{self._request_counter}"

    def _audit_token_endpoint(self) -> tuple[str, int]:
        parsed = urlparse(self.base_url)
        host = parsed.hostname or self.host
        port = parsed.port
        if port is None:
            port = 443 if parsed.scheme == "https" else 80
        return host, int(port)

    def healthz(self) -> bool:
        try:
            req = urllib.request.Request(f"{self.base_url}/healthz", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except (OSError, ValueError, HTTPException):
            return False

    def _open(
        self,
        req: urllib.request.Request,
        *,
        timeout: float,
        proxy_free: bool,
    ) -> Any:
        if proxy_free:
            # Audit calls are bridge-internal localhost traffic.  A process may
            # already have a proxy-backed global urllib opener cached before
            # NO_PROXY is applied, so use a dedicated direct opener here.
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            return opener.open(req, timeout=timeout)
        return urllib.request.urlopen(req, timeout=timeout)

    def call(
        self,
        action: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        timeout: Optional[float] = None,
        audit_internal: bool = False,
    ) -> Dict[str, Any]:
        """Issue a single POST to the bridge HTTP API.

        ``timeout`` — optional per-call override (seconds). Defaults to 60s
        for long-running actions. Shutdown-critical callers
        (mark_terminal_status, signal handlers) should pass a short value
        (~3s) so a hung bridge handler cannot block process termination
        for a full minute.
        """
        envelope: Dict[str, Any] = {
            "request_id": self._next_id(),
            "action": action,
            "payload": payload or {},
        }
        if self.session_id:
            envelope["session_id"] = self.session_id
        body = json.dumps(envelope, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if audit_internal:
            token_host, token_port = self._audit_token_endpoint()
            token = read_audit_token(token_host, token_port)
            if token:
                headers["X-Habitat-Audit-Token"] = token
        req = urllib.request.Request(
            f"{self.base_url}/v1/request",
            data=body,
            headers=headers,
            method="POST",
        )
        effective_timeout = 60 if timeout is None else float(timeout)
        try:
            with self._open(
                req,
                timeout=effective_timeout,
                proxy_free=audit_internal,
            ) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise RuntimeError(f"Bridge HTTP {e.code}: {error_body}") from e
        if not result.get("ok", True):
            err = result.get("error", {})
            raise RuntimeError(f"Bridge error: {err.get('message', result)}")
        return result.get("result", result)


__all__ = ["BridgeClient"]
