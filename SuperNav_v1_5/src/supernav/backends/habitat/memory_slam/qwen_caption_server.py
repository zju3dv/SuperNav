#!/usr/bin/env python3
"""Lightweight HTTP service wrapping Qwen3.5-2B for SLAM area captions.

The service preloads the model before binding a socket and exposes:

    GET /healthz
    POST /caption
    POST /caption_batch

The /caption request accepts the payload emitted by SlamAreaCaptionWorker and
returns a JSON object containing a single ``caption`` string.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import os
import re
from pathlib import Path
from typing import Any, Callable

from PIL import Image


DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_PROMPT = (
    "Describe the visible indoor scene in one short factual sentence. "
    "Mention major objects, surfaces, openings, and layout. "
    "Do not guess room type, hidden objects, goals, or navigation actions."
)
DEFAULT_MAX_REQUEST_BYTES = 8 * 1024 * 1024


class QwenCaptionServer:
    """Simple WSGI service for local Qwen image captioning."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL,
        device_map: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 48,
        image_max_size: int = 768,
        max_caption_chars: int = 300,
        prompt: str = DEFAULT_PROMPT,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        model_loader: Callable[["QwenCaptionServer"], tuple[Any, Any]] | None = None,
        warmup: bool = True,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.dtype = dtype
        self.max_new_tokens = max(1, int(max_new_tokens))
        self.image_max_size = max(1, int(image_max_size))
        self.max_caption_chars = max(1, int(max_caption_chars))
        self.prompt = prompt
        self.max_request_bytes = max(1, int(max_request_bytes))
        self._model_loader = model_loader
        self.processor: Any | None = None
        self.model: Any | None = None
        self.loaded = False
        self.load_error: str | None = None
        self._warmup_enabled = warmup

    def load(self) -> None:
        if self.loaded:
            return
        print(f"Loading SLAM area caption model: {self.model_name}", flush=True)
        try:
            if self._model_loader is None:
                processor, model = _load_qwen_model(
                    self.model_name,
                    device_map=self.device_map,
                    dtype=self.dtype,
                )
            else:
                processor, model = self._model_loader(self)
            if hasattr(model, "eval"):
                model.eval()
            self.processor = processor
            self.model = model
            self.loaded = True
            if self._warmup_enabled:
                self.caption_image(Image.new("RGB", (8, 8), color=(128, 128, 128)))
            print("SLAM area caption model loaded.", flush=True)
        except Exception as exc:  # pragma: no cover - startup reports exact failure.
            self.load_error = f"{type(exc).__name__}: {exc}"
            raise

    @property
    def device_label(self) -> str:
        model = self.model
        hf_device_map = getattr(model, "hf_device_map", None)
        if hf_device_map:
            return str(hf_device_map)
        try:
            return str(next(model.parameters()).device)
        except Exception:
            return str(self.device_map)

    def health(self) -> dict[str, Any]:
        return {
            "protocol": "supernav-caption/v1",
            "version": 1,
            "ok": self.loaded and self.load_error is None,
            "status": "healthy" if self.loaded and self.load_error is None else "loading_failed",
            "model": self.model_name,
            "device": self.device_label,
            "loaded": self.loaded,
        }

    def caption_image(self, image: Image.Image) -> str:
        return self.caption_images([image])[0]

    def caption_images(self, images: list[Image.Image]) -> list[str]:
        if self.processor is None or self.model is None:
            raise RuntimeError("model is not loaded")
        resized = [
            _resize_image(image.convert("RGB"), self.image_max_size)
            for image in images
        ]
        raw = _generate_captions(
            processor=self.processor,
            model=self.model,
            images=resized,
            prompt=self.prompt,
            max_new_tokens=self.max_new_tokens,
        )
        return [
            _clean_caption(caption, self.prompt, self.max_caption_chars)
            for caption in raw
        ]

    def caption_path(self, image_path: str) -> dict[str, Any]:
        path = Path(image_path)
        if not path.is_file():
            raise ValueError(f"image not found: {image_path}")
        try:
            with Image.open(path) as opened:
                opened.load()
                caption = self.caption_image(opened.convert("RGB"))
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"inference failed: {exc}") from exc
        return {"caption": caption, "model": self.model_name}

    def caption_bytes(self, image_bytes: bytes) -> dict[str, Any]:
        """Caption encoded image bytes supplied by a remote robot."""
        try:
            with Image.open(io.BytesIO(image_bytes)) as opened:
                opened.load()
                caption = self.caption_image(opened.convert("RGB"))
        except Exception as exc:
            raise ValueError(f"invalid image bytes: {exc}") from exc
        return {"caption": caption, "model": self.model_name}

    def caption_batch(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if not items:
            raise ValueError("items must be a non-empty list")
        images: list[Image.Image] = []
        prepared: list[dict[str, Any]] = []
        try:
            for item in items:
                if not isinstance(item, dict):
                    raise ValueError("each item must be an object")
                image_data = item.get("image_data")
                if isinstance(image_data, str) and image_data:
                    try:
                        raw = base64.b64decode(image_data, validate=True)
                    except (binascii.Error, ValueError) as exc:
                        raise ValueError(f"invalid image_data: {exc}") from exc
                    try:
                        with Image.open(io.BytesIO(raw)) as opened:
                            opened.load()
                            images.append(opened.convert("RGB"))
                    except Exception as exc:
                        raise ValueError(f"invalid image_data image: {exc}") from exc
                else:
                    image_path = str(item.get("image_path", "")).strip()
                    if not image_path:
                        raise ValueError("image_path or image_data is required")
                    path = Path(image_path)
                    if not path.is_file():
                        raise ValueError(f"image not found: {image_path}")
                    with Image.open(path) as opened:
                        opened.load()
                        images.append(opened.convert("RGB"))
                prepared.append(item)
            captions = self.caption_images(images)
        except ValueError:
            raise
        except Exception as exc:
            raise RuntimeError(f"inference failed: {exc}") from exc
        return {
            "model": self.model_name,
            "results": [
                {
                    "job_id": item.get("job_id"),
                    "keyframe_id": item.get("keyframe_id"),
                    "view": item.get("view"),
                    "caption": caption,
                }
                for item, caption in zip(prepared, captions, strict=False)
            ],
        }

    def __call__(self, environ: dict[str, Any], start_response: Any) -> list[bytes]:
        method = environ.get("REQUEST_METHOD", "GET")
        path = environ.get("PATH_INFO", "/")

        if method == "GET" and path in ("/", "/healthz"):
            return self._respond(start_response, "200 OK", self.health())

        if method != "POST" or path not in {"/caption", "/caption_batch"}:
            return self._respond(
                start_response,
                "404 Not Found",
                {"ok": False, "error": "not found"},
            )

        try:
            content_length = int(environ.get("CONTENT_LENGTH", 0))
        except (TypeError, ValueError):
            return self._respond(
                start_response,
                "400 Bad Request",
                {"ok": False, "error": "invalid Content-Length"},
            )
        if content_length < 0:
            return self._respond(
                start_response,
                "400 Bad Request",
                {"ok": False, "error": "invalid Content-Length"},
            )
        if content_length > self.max_request_bytes:
            return self._respond(
                start_response,
                "413 Payload Too Large",
                {"ok": False, "error": "request body too large"},
            )

        content_type = str(environ.get("CONTENT_TYPE", "")).split(";", 1)[0].strip().lower()
        raw_bytes = environ["wsgi.input"].read(content_length)
        if path == "/caption" and content_type in {
            "application/octet-stream",
            "image/jpeg",
            "image/png",
            "image/webp",
        }:
            try:
                result = self.caption_bytes(raw_bytes)
            except ValueError as exc:
                return self._respond(start_response, "400 Bad Request", {"ok": False, "error": str(exc)})
            except Exception as exc:
                return self._respond(start_response, "500 Internal Server Error", {"ok": False, "error": str(exc)[:200]})
            return self._respond(start_response, "200 OK", result)

        try:
            raw_body = raw_bytes.decode("utf-8")
            payload = json.loads(raw_body)
            if not isinstance(payload, dict):
                raise ValueError("request JSON must be an object")
        except Exception as exc:
            return self._respond(
                start_response,
                "400 Bad Request",
                {"ok": False, "error": f"bad request: {exc}"},
            )

        if path == "/caption_batch":
            items = payload.get("items")
            if not isinstance(items, list):
                return self._respond(
                    start_response,
                    "400 Bad Request",
                    {"ok": False, "error": "items must be a list"},
                )
            try:
                result = self.caption_batch(items)
            except ValueError as exc:
                return self._respond(
                    start_response,
                    "400 Bad Request",
                    {"ok": False, "error": str(exc)},
                )
            except Exception as exc:
                return self._respond(
                    start_response,
                    "500 Internal Server Error",
                    {"ok": False, "error": str(exc)[:200]},
                )
            return self._respond(start_response, "200 OK", result)

        try:
            image_data = payload.get("image_data")
            if isinstance(image_data, str) and image_data:
                try:
                    raw = base64.b64decode(image_data, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError(f"invalid image_data: {exc}") from exc
                result = self.caption_bytes(raw)
            else:
                image_path = str(payload.get("image_path", "")).strip()
                if not image_path:
                    raise ValueError("image_path or image_data is required")
                result = self.caption_path(image_path)
        except ValueError as exc:
            return self._respond(
                start_response,
                "400 Bad Request",
                {"ok": False, "error": str(exc)},
            )
        except Exception as exc:
            return self._respond(
                start_response,
                "500 Internal Server Error",
                {"ok": False, "error": str(exc)[:200]},
            )
        return self._respond(start_response, "200 OK", result)

    @staticmethod
    def _respond(start_response: Any, status: str, payload: dict[str, Any]) -> list[bytes]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        start_response(
            status,
            [
                ("Content-Type", "application/json"),
                ("Content-Length", str(len(body))),
            ],
        )
        return [body]


def _load_qwen_model(model_name: str, *, device_map: str, dtype: str) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForMultimodalLM
    except ImportError:
        from transformers import AutoModelForImageTextToText as AutoModelForMultimodalLM

    torch_dtype: Any = "auto"
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    elif dtype == "fp32":
        torch_dtype = torch.float32

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForMultimodalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    return processor, model


def _generate_caption(
    *,
    processor: Any,
    model: Any,
    image: Image.Image,
    prompt: str,
    max_new_tokens: int,
) -> str:
    captions = _generate_captions(
        processor=processor,
        model=model,
        images=[image],
        prompt=prompt,
        max_new_tokens=max_new_tokens,
    )
    return captions[0] if captions else ""


def _generate_captions(
    *,
    processor: Any,
    model: Any,
    images: list[Image.Image],
    prompt: str,
    max_new_tokens: int,
) -> list[str]:
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        for image in images
    ]
    if hasattr(processor, "apply_chat_template"):
        texts = [
            processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=True,
            )
            for message in messages
        ]
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
        )
    else:
        inputs = processor(
            text=[prompt for _ in images],
            images=images,
            return_tensors="pt",
            padding=True,
        )

    device = _first_model_device(model)
    if device is not None and hasattr(inputs, "to"):
        inputs = inputs.to(device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    input_ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    if input_ids is not None:
        trimmed = [
            output[len(input_row) :]
            for input_row, output in zip(input_ids, output_ids, strict=False)
        ]
    else:
        trimmed = output_ids
    if hasattr(processor, "batch_decode"):
        decoded = processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [str(row) for row in decoded]
    return [str(row) for row in trimmed]


def _first_model_device(model: Any) -> Any | None:
    try:
        return next(model.parameters()).device
    except Exception:
        return None


def _resize_image(image: Image.Image, max_size: int) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= max_size:
        return image
    scale = max_size / float(longest)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def _clean_caption(raw: str, prompt: str, max_chars: int) -> str:
    text = str(raw or "").strip()
    if text.startswith(prompt):
        text = text[len(prompt) :].strip()
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def _env_int(name: str, default: int) -> int:
    try:
        value = int(os.environ.get(name, ""))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen SLAM area caption HTTP service")
    parser.add_argument(
        "--model",
        default=os.environ.get("SLAM_AREA_CAPTION_MODEL", DEFAULT_MODEL),
        help="Model name or local path",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("SLAM_AREA_CAPTION_HOST", "127.0.0.1"),
        help="Bind host",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SLAM_AREA_CAPTION_PORT", "8926")),
        help="Bind port",
    )
    parser.add_argument(
        "--device-map",
        default=os.environ.get("SLAM_AREA_CAPTION_DEVICE_MAP", "auto"),
        help="Transformers device_map value",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "bf16", "fp16", "fp32"),
        default=os.environ.get("SLAM_AREA_CAPTION_DTYPE", "auto"),
        help="Model dtype",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=_env_int("SLAM_AREA_CAPTION_MAX_NEW_TOKENS", 48),
        help="Caption generation token budget",
    )
    parser.add_argument(
        "--image-max-size",
        type=int,
        default=_env_int("SLAM_AREA_CAPTION_IMAGE_MAX_SIZE", 768),
        help="Resize image longest side before inference",
    )
    parser.add_argument(
        "--max-caption-chars",
        type=int,
        default=_env_int("SLAM_AREA_CAPTION_MAX_CAPTION_CHARS", 300),
        help="Maximum returned caption length",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        default=os.environ.get("SLAM_AREA_CAPTION_NO_WARMUP", "").lower()
        in {"1", "true", "yes"},
        help="Skip startup warmup inference",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    app = QwenCaptionServer(
        model_name=args.model,
        device_map=args.device_map,
        dtype=args.dtype,
        max_new_tokens=args.max_new_tokens,
        image_max_size=args.image_max_size,
        max_caption_chars=args.max_caption_chars,
        warmup=not args.no_warmup,
    )
    app.load()

    from wsgiref.simple_server import make_server

    server = make_server(args.host, args.port, app)
    print(
        f"SLAM area caption service listening on http://{args.host}:{args.port}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down.", flush=True)


if __name__ == "__main__":
    main()
