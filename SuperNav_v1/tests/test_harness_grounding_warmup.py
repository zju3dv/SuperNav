from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib import error as urllib_error

REPO_ROOT = Path(__file__).resolve().parents[1]

from supernav.backends.habitat.grounding_warmup import perform_grounding_warmup
import supernav.experiments.episode as run_one_module


class _Response:
    def __init__(self, payload: object, status: int = 200) -> None:
        self._body = json.dumps(payload).encode("utf-8")
        self._status = status

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def getcode(self) -> int:
        return self._status

    def read(self) -> bytes:
        return self._body


class _Opener:
    def __init__(self, response: _Response) -> None:
        self.response = response
        self.requests = []

    def open(self, request, timeout):  # noqa: ANN001
        self.requests.append((request, timeout))
        return self.response


def _config(image: Path) -> dict:
    return {
        "grounding_warmup": {
            "image": str(image),
            "phrase": "bed",
            "mode": "box",
            "timeout_s": 120,
        }
    }


def test_grounding_warmup_posts_base64_without_proxy(
    monkeypatch, tmp_path: Path
) -> None:
    image = tmp_path / "fixture.png"
    image.write_bytes(b"image bytes")
    opener = _Opener(_Response({"ok": True, "detections": []}))
    proxy_handlers = []

    def fake_build_opener(handler):  # noqa: ANN001
        proxy_handlers.append(handler)
        return opener

    monkeypatch.setattr(
        "supernav.backends.habitat.grounding_warmup.urllib_request.build_opener", fake_build_opener
    )

    report = perform_grounding_warmup(
        config=_config(image),
        environment={"NAV_LOCATE_ANYTHING_URL": "http://grounding.example:8915/ground"},
        workspace_root=tmp_path,
        dry_run=False,
    )

    assert report["status"] == "passed"
    assert report["detection_count"] == 0
    assert opener.requests[0][1] == 120
    request = opener.requests[0][0]
    assert request.full_url == "http://grounding.example:8915/ground"
    assert json.loads(request.data)["image_base64"]
    assert proxy_handlers[0].proxies == {}


def test_grounding_warmup_dry_run_does_not_open_network(
    monkeypatch, tmp_path: Path
) -> None:
    image = tmp_path / "fixture.png"
    image.write_bytes(b"image bytes")

    def unexpected_open(*_args, **_kwargs):
        raise AssertionError("dry-run must not build a network opener")

    monkeypatch.setattr(
        "supernav.backends.habitat.grounding_warmup.urllib_request.build_opener", unexpected_open
    )

    report = perform_grounding_warmup(
        config=_config(image),
        environment={"NAV_LOCATE_ANYTHING_URL": "http://127.0.0.1:8915"},
        workspace_root=tmp_path,
        dry_run=True,
    )

    assert report["status"] == "not_executed_dry_run"
    assert report["url"].endswith("/ground")


def test_grounding_warmup_records_protocol_failure(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "fixture.png"
    image.write_bytes(b"image bytes")
    opener = _Opener(_Response({"ok": False, "error": "inference failed"}))
    monkeypatch.setattr(
        "supernav.backends.habitat.grounding_warmup.urllib_request.build_opener",
        lambda *_args: opener,
    )

    report = perform_grounding_warmup(
        config=_config(image),
        environment={"NAV_LOCATE_ANYTHING_URL": "http://127.0.0.1:8915/ground"},
        workspace_root=tmp_path,
        dry_run=False,
    )

    assert report["status"] == "failed"
    assert "inference failed" in report["error"]
    assert report["duration_s"] >= 0


def test_grounding_warmup_requires_configured_service_url(tmp_path: Path) -> None:
    image = tmp_path / "fixture.png"
    image.write_bytes(b"image bytes")

    try:
        perform_grounding_warmup(
            config=_config(image),
            environment={},
            workspace_root=tmp_path,
            dry_run=True,
        )
    except ValueError as exc:
        assert "NAV_LOCATE_ANYTHING_URL" in str(exc)
    else:
        raise AssertionError("missing service URL must fail closed")


def test_grounding_warmup_records_http_failure(monkeypatch, tmp_path: Path) -> None:
    image = tmp_path / "fixture.png"
    image.write_bytes(b"image bytes")

    class FailedOpener:
        def open(self, request, timeout):  # noqa: ANN001
            raise urllib_error.HTTPError(request.full_url, 503, "busy", {}, None)

    monkeypatch.setattr(
        "supernav.backends.habitat.grounding_warmup.urllib_request.build_opener",
        lambda *_args: FailedOpener(),
    )

    report = perform_grounding_warmup(
        config=_config(image),
        environment={"NAV_LOCATE_ANYTHING_URL": "http://127.0.0.1:8915/ground"},
        workspace_root=tmp_path,
        dry_run=False,
    )

    assert report["status"] == "failed"
    assert report["http_status"] == 503


def test_run_one_warmup_failure_blocks_before_project_preparation(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "config.json"
    output_dir = tmp_path / "runs"
    config_path.write_text(
        json.dumps(
            {"environment": {"backend": "habitat"}, "prompts_dir": str(Path(__file__).resolve().parents[1] / "configs/benchmarks/main/prompts"),
                "workspace_root": str(REPO_ROOT),
                "output_dir": str(output_dir),
                "scene": "scene",
                "scene_dataset_config_file": "/tmp/scene.json",
                "spawn": {"x": 0, "z": 0, "yaw": 0},
                "bridge": {"host": "127.0.0.1", "port": 18911},
                "mcp": {
                    "environment": {
                        "NAV_LOCATE_ANYTHING_URL": "http://127.0.0.1:8915/ground"
                    },
                },
                "grounding_warmup": {
                    "image": "data/test_assets/hbao_tests/van-gogh-room.color.png",
                    "phrase": "bed",
                },
                "agent": "fake",
                "model": {},
                "agents": {"fake": {}},
                "arms": {
                    "locate": {
                        "movement": "visual_ground_preview_locate",
                        "grounding_warmup": True,
                        "tool_whitelist": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    class FakeAgent:
        name = "fake"

        def prepare_project(self, **_kwargs):
            raise AssertionError(
                "project preparation must not run after warm-up failure"
            )

    monkeypatch.setattr(run_one_module, "get_agent_backend", lambda _name: FakeAgent())
    monkeypatch.setattr(
        run_one_module,
        "perform_grounding_warmup",
        lambda **_kwargs: {"status": "failed", "error": "service timed out"},
    )

    try:
        run_one_module.run_one(
            SimpleNamespace(
                config=str(config_path),
                arm="locate",
                slug="bed",
                instruction="Go to the bed.",
                rep=None,
                run_id=None,
                task_id=None,
                scene=None,
                scene_dataset_config_file=None,
                spawn=None,
                metadata=None,
                ground_truth=None,
                timeout_s=None,
                overwrite=False,
                dry_run=False,
            )
        )
    except RuntimeError as exc:
        assert "warm-up blocked" in str(exc)
    else:
        raise AssertionError("failed warm-up must block the run")

    run_payload = json.loads(
        (output_dir / "locate_bed" / "run.json").read_text(encoding="utf-8")
    )
    assert run_payload["terminal_status"] == "locate_anything_warmup_blocked"
