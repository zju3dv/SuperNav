#!/usr/bin/env python
"""Persistent open-vocab detection service (Grounding DINO tiny) for the agent
loop. Loaded once so per-run MCP subprocesses do not pay
the model cold-start. Plain stdlib HTTP, one endpoint:

  POST /ground   {"image_path": "/abs/path.png", "phrases": "a bed."}
  -> {"ok": true, "detections": [{"label","score","box","area_ratio",
                                  "angle_off_deg"}, ...]}   # sorted by score

angle_off_deg: horizontal offset of the box center as a turn angle (deg, +right)
assuming HFOV_DEG field of view — lets callers convert "seen in this view" into
an exact hab_turn. Start:  python grounding_server.py --port 18913
"""
import argparse, json, math, os, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
HFOV_DEG = 90.0
SCORE_T = 0.30          # service-side floor; callers can filter higher
TEXT_T = 0.25

_PROC = _MODEL = _DEV = None


def _nonnegative_float(value, default):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else default


def _load():
    global _PROC, _MODEL, _DEV
    from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
    _DEV = "cuda" if torch.cuda.is_available() else "cpu"
    _PROC = AutoProcessor.from_pretrained(MODEL_ID)
    _MODEL = (AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID)
              .to(_DEV).eval())
    print(f"[grounding] {MODEL_ID} resident on {_DEV}", file=sys.stderr, flush=True)


@torch.no_grad()
def _detect(img: Image.Image, phrases: str):
    inputs = _PROC(images=img, text=phrases, return_tensors="pt").to(_DEV)
    out = _MODEL(**inputs)
    kw = dict(input_ids=inputs.input_ids, target_sizes=[img.size[::-1]])
    try:
        res = _PROC.post_process_grounded_object_detection(
            out, box_threshold=SCORE_T, text_threshold=TEXT_T, **kw)[0]
    except TypeError:
        res = _PROC.post_process_grounded_object_detection(
            out, threshold=SCORE_T, text_threshold=TEXT_T, **kw)[0]
    dets = []
    labels = res.get("text_labels", res.get("labels", []))
    for score, label, box in zip(res["scores"], labels, res["boxes"]):
        x0, y0, x1, y1 = [float(v) for v in box]
        W, H = img.size
        dets.append({
            "label": str(label), "score": round(float(score), 3),
            "box": [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)],
            "area_ratio": round((x1 - x0) * (y1 - y0) / (W * H), 3),
            "angle_off_deg": round(math.degrees(math.atan(
                ((x0 + x1) / 2 - W / 2) / (W / 2)
                * math.tan(math.radians(HFOV_DEG / 2)))), 1),
        })
    return sorted(dets, key=lambda d: -d["score"])


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(
            200,
            {
                "ok": True,
                "model": MODEL_ID,
                "thresholds": {"box": SCORE_T, "text": TEXT_T},
            },
        )

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n))
            img = Image.open(req["image_path"]).convert("RGB")
            dets = _detect(img, req.get("phrases", "an object."))
            self._send(
                200,
                {
                    "ok": True,
                    "detections": dets,
                    "thresholds": {"box": SCORE_T, "text": TEXT_T},
                },
            )
        except Exception as e:  # noqa: BLE001
            self._send(500, {"ok": False, "error": f"{type(e).__name__}: {e}"})


def main():
    global SCORE_T, TEXT_T
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18913)
    ap.add_argument(
        "--box-threshold",
        type=float,
        default=_nonnegative_float(os.environ.get("GROUNDING_BOX_THRESHOLD"), SCORE_T),
    )
    ap.add_argument(
        "--text-threshold",
        type=float,
        default=_nonnegative_float(os.environ.get("GROUNDING_TEXT_THRESHOLD"), TEXT_T),
    )
    args = ap.parse_args()
    SCORE_T = _nonnegative_float(args.box_threshold, SCORE_T)
    TEXT_T = _nonnegative_float(args.text_threshold, TEXT_T)
    _load()
    srv = ThreadingHTTPServer((args.host, args.port), H)
    print(
        f"[grounding] serving on {args.host}:{args.port} "
        f"box_threshold={SCORE_T:.3f} text_threshold={TEXT_T:.3f}",
        file=sys.stderr,
        flush=True,
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()
