from __future__ import annotations

import base64
import hashlib
import json
import mimetypes
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Mapping
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _service_url(value: object) -> str:
    url = str(value or "").strip().rstrip("/")
    parsed = urllib_parse.urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "grounding warm-up requires NAV_LOCATE_ANYTHING_URL with an http(s) URL"
        )
    if not url.endswith("/ground"):
        url += "/ground"
    return url


def _report_url(url: str) -> str:
    parsed = urllib_parse.urlsplit(url)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parsed.port is not None:
        host = f"{host}:{parsed.port}"
    return urllib_parse.urlunsplit((parsed.scheme, host, parsed.path, "", ""))


def perform_grounding_warmup(
    *,
    config: Mapping[str, Any],
    environment: Mapping[str, str],
    workspace_root: Path,
    dry_run: bool,
) -> dict[str, Any]:
    warmup_cfg = config.get("grounding_warmup")
    if not isinstance(warmup_cfg, Mapping):
        raise ValueError("grounding_warmup must be an object for enabled arms")

    image_value = str(warmup_cfg.get("image") or "").strip()
    if not image_value:
        raise ValueError("grounding_warmup.image is required")
    image_path = Path(image_value).expanduser()
    if not image_path.is_absolute():
        image_path = (workspace_root / image_path).resolve()
    if not image_path.is_file():
        raise ValueError(f"grounding warm-up image not found: {image_path}")

    phrase = str(warmup_cfg.get("phrase") or "").strip()
    if not phrase:
        raise ValueError("grounding_warmup.phrase is required")
    mode = str(warmup_cfg.get("mode") or "box").strip().lower()
    if mode not in {"box", "point"}:
        raise ValueError("grounding_warmup.mode must be 'box' or 'point'")
    timeout_s = float(warmup_cfg.get("timeout_s", 120))
    if timeout_s <= 0:
        raise ValueError("grounding_warmup.timeout_s must be positive")

    url = _service_url(environment.get("NAV_LOCATE_ANYTHING_URL"))
    image_bytes = image_path.read_bytes()
    report: dict[str, Any] = {
        "status": "not_executed_dry_run" if dry_run else "running",
        "url": _report_url(url),
        "image": str(image_path),
        "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "phrase": phrase,
        "mode": mode,
        "timeout_s": timeout_s,
    }
    if dry_run:
        return report

    body = json.dumps(
        {
            "image_base64": base64.b64encode(image_bytes).decode("ascii"),
            "image_mime_type": mimetypes.guess_type(image_path.name)[0] or "",
            "phrase": phrase,
            "mode": mode,
        }
    ).encode("utf-8")
    req = urllib_request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = _utc_now()
    started = monotonic()
    report["started_at"] = started_at
    try:
        # Benchmark service endpoints are local/private. Never let ambient proxy
        # variables redirect this readiness request through an HTTP gateway.
        opener = urllib_request.build_opener(urllib_request.ProxyHandler({}))
        with opener.open(req, timeout=timeout_s) as response:
            status_code = int(response.getcode())
            payload = json.loads(response.read().decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("LocateAnything warm-up returned a non-object response")
        if payload.get("ok") is not True:
            raise RuntimeError(
                f"LocateAnything warm-up returned ok={payload.get('ok')!r}: "
                f"{payload.get('error') or 'no error detail'}"
            )
        detections = payload.get("detections")
        report.update(
            {
                "status": "passed",
                "http_status": status_code,
                "detection_count": (
                    len(detections) if isinstance(detections, list) else 0
                ),
            }
        )
    except urllib_error.HTTPError as exc:
        report.update(
            {
                "status": "failed",
                "http_status": int(exc.code),
                "error": f"HTTPError: {exc}",
            }
        )
    except (OSError, ValueError, json.JSONDecodeError, urllib_error.URLError) as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    except Exception as exc:
        report.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        report["finished_at"] = _utc_now()
        report["duration_s"] = round(monotonic() - started, 3)
    return report
