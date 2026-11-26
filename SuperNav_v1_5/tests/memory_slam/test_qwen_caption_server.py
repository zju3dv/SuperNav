from __future__ import annotations

import io
import importlib.util
import json
import sys
import time
import types
from pathlib import Path

from PIL import Image

from supernav.backends.habitat.memory_slam.qwen_caption_server import QwenCaptionServer


_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_HABITAT_SIM = str(_REPO_ROOT / "src_python" / "habitat_sim")


class _FakeInputs(dict):
    def to(self, _device: object) -> "_FakeInputs":
        return self


class _FakeProcessor:
    def apply_chat_template(
        self,
        messages: object,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return str(messages)

    def __call__(
        self,
        *,
        text: list[str],
        images: list[Image.Image],
        return_tensors: str,
        padding: bool = False,
    ) -> _FakeInputs:
        assert text
        assert images
        assert return_tensors == "pt"
        assert padding is True
        return _FakeInputs({"input_ids": [[1, 2, 3] for _ in images]})

    def batch_decode(
        self,
        ids: object,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> list[str]:
        assert skip_special_tokens is True
        assert clean_up_tokenization_spaces is False
        return [
            f"A sofa and a doorway are visible from view {index}.\n"
            for index, _row in enumerate(ids)
        ]


class _FakeModel:
    eval_called = False

    def eval(self) -> None:
        self.eval_called = True

    def parameters(self) -> object:
        return iter(())

    def generate(self, **kwargs: object) -> list[list[int]]:
        assert kwargs["max_new_tokens"] == 48
        assert kwargs["do_sample"] is False
        input_ids = kwargs["input_ids"]
        return [list(row) + [4, 5] for row in input_ids]


def _loaded_server(tmp_path: Path) -> tuple[QwenCaptionServer, _FakeModel]:
    model = _FakeModel()

    def _loader(_server: QwenCaptionServer) -> tuple[_FakeProcessor, _FakeModel]:
        return _FakeProcessor(), model

    server = QwenCaptionServer(model_loader=_loader, warmup=False)
    server.load()
    assert model.eval_called is True
    return server, model


def _call_wsgi(
    app: QwenCaptionServer,
    method: str,
    path: str,
    payload: dict[str, object] | None = None,
) -> tuple[str, dict[str, object]]:
    body = b"" if payload is None else json.dumps(payload).encode("utf-8")
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": io.BytesIO(body),
    }
    status_holder: list[str] = []

    def _start_response(status: str, _headers: list[tuple[str, str]]) -> None:
        status_holder.append(status)

    response = b"".join(app(environ, _start_response))
    return status_holder[0], json.loads(response.decode("utf-8"))


def test_qwen_caption_server_preloads_model_before_serving(tmp_path: Path) -> None:
    server, _model = _loaded_server(tmp_path)

    status, body = _call_wsgi(server, "GET", "/healthz")

    assert status == "200 OK"
    assert body["loaded"] is True
    assert body["model"] == "Qwen/Qwen3.5-2B"


def test_qwen_caption_server_rejects_missing_image_path(tmp_path: Path) -> None:
    server, _model = _loaded_server(tmp_path)

    status, body = _call_wsgi(server, "POST", "/caption", {})

    assert status == "400 Bad Request"
    assert "image_path" in str(body["error"])


def test_qwen_caption_server_returns_clean_caption(tmp_path: Path) -> None:
    server, _model = _loaded_server(tmp_path)
    image_path = tmp_path / "view.png"
    Image.new("RGB", (12, 8), color=(32, 64, 96)).save(image_path)

    status, body = _call_wsgi(
        server,
        "POST",
        "/caption",
        {
            "job_id": "kf_000001:front",
            "keyframe_id": "kf_000001",
            "view": "front",
            "image_path": str(image_path),
        },
    )

    assert status == "200 OK"
    assert body == {
        "caption": "A sofa and a doorway are visible from view 0.",
        "model": "Qwen/Qwen3.5-2B",
    }


def test_qwen_caption_server_returns_batch_captions(tmp_path: Path) -> None:
    server, _model = _loaded_server(tmp_path)
    image_paths = []
    for index in range(4):
        image_path = tmp_path / f"view{index}.png"
        Image.new("RGB", (12, 8), color=(32, 64, 96)).save(image_path)
        image_paths.append(image_path)

    status, body = _call_wsgi(
        server,
        "POST",
        "/caption_batch",
        {
            "items": [
                {
                    "job_id": f"kf_000001:{view}",
                    "keyframe_id": "kf_000001",
                    "view": view,
                    "image_path": str(image_path),
                }
                for view, image_path in zip(
                    ("front", "right", "back", "left"),
                    image_paths,
                    strict=True,
                )
            ]
        },
    )

    assert status == "200 OK"
    assert body["model"] == "Qwen/Qwen3.5-2B"
    assert [row["job_id"] for row in body["results"]] == [
        "kf_000001:front",
        "kf_000001:right",
        "kf_000001:back",
        "kf_000001:left",
    ]
    assert [row["caption"] for row in body["results"]] == [
        "A sofa and a doorway are visible from view 0.",
        "A sofa and a doorway are visible from view 1.",
        "A sofa and a doorway are visible from view 2.",
        "A sofa and a doorway are visible from view 3.",
    ]
