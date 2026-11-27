"""Optional spectator telemetry using the policy's existing RGB, never extra views."""
from __future__ import annotations

from io import BytesIO
import json
import logging
import math
import os
from pathlib import Path
import uuid

from supernav.runtime.live import LivePublisher, atomic_write

LOGGER = logging.getLogger(__name__)


class NativeLiveView:
    """Translate Unity coordinates and isolate observer failures from task execution."""

    def __init__(self, episode):
        self.publisher = None
        self.position = None
        self.heading = 0.0
        self.path_length = 0.0
        self.capture = None
        directory = os.environ.get("SUPERNAV_LIVE_DIR")
        if not directory:
            return
        try:
            context = json.loads(os.environ.get("SUPERNAV_LIVE_CONTEXT", "{}"))
            publisher = LivePublisher(Path(directory), "ai2thor-" + uuid.uuid4().hex,
                                      scene=f"AI2-THOR · {episode['scene_id']}",
                                      fps=float(os.environ.get("SUPERNAV_LIVE_FPS", "5")))
            publisher.state.update(
                backend="ai2thor", instruction=episode["instruction"],
                task_id=episode["episode_id"], agent="Demand-driven navigation",
                evaluation=None, agent_claim=None, success_scoring="withheld",
                observation_permissions="front RGB only",
            )
            publisher.state.update({key: str(context[key]) for key in ("agent", "run_id")
                                    if key in context})
            self.publisher = publisher
            publisher.publish()
        except Exception:
            self._disable()

    def _disable(self):
        self.publisher = None
        LOGGER.warning("AI2-THOR live view disabled; task execution is unchanged", exc_info=True)

    def _call(self, callback):
        if self.publisher is not None:
            try:
                callback(self.publisher)
            except Exception:
                self._disable()

    def sample(self, session, metadata, *, force=False):
        if self.publisher is None:
            return

        def publish(publisher):
            agent = metadata["agent"]
            position = [float(agent["position"][axis]) for axis in "xyz"]
            if self.position is not None:
                self.path_length += math.hypot(position[0] - self.position[0],
                                               position[2] - self.position[2])
            self.position = position
            # Unity yaw 0 faces +Z, yaw 90 faces +X. The viewer uses +X = 0,
            # +Z = 90 on its X-right/Z-down canvas.
            self.heading = (90 - float(agent["rotation"]["y"])) % 360
            if force or publisher.due():
                self._frame(publisher, session)

        self._call(publish)

    def _image(self, publisher, rgb, callback):
        from PIL import Image

        buffer = BytesIO()
        Image.fromarray(rgb).save(buffer, format="PNG")
        path = publisher.directory / f".{publisher.session_id}.source.png"
        try:
            atomic_write(path, buffer.getvalue())
            callback(path)
        finally:
            path.unlink(missing_ok=True)

    def _frame(self, publisher, session):
        if session.last_rgb is None or self.position is None:
            return
        publisher.state["observation_id"] = session.observation_id
        views = session.observation_views
        publisher.state["observation_permissions"] = "front RGB only" if len(views) == 1 else "four directional RGB"
        publisher.sample(position=self.position, heading=self.heading, step=session.actions,
                         path_length=self.path_length, capture=lambda: session.last_rgb, force=True)
        frames = getattr(session, "view_frames", {"front": session.last_rgb})
        if self.capture != session.capture and all(view in frames for view in views):
            # Publish a whole capture together; never mix old side views with a
            # new front frame or create two groups for the same capture number.
            from PIL import Image

            rows, paths = [], []
            offsets = dict(front=0, right=90, back=180, left=-90)
            try:
                for view in views:
                    buffer = BytesIO()
                    Image.fromarray(frames[view]).save(buffer, format="PNG")
                    path = publisher.directory / f".{publisher.session_id}.{view}.png"
                    paths.append(path)
                    atomic_write(path, buffer.getvalue())
                    rows.append(dict(direction=view, path=str(path),
                                     image_ref=self._view_ref(session, view),
                                     heading_deg=(self.heading - offsets[view]) % 360))
                publisher.panorama(rows, session.capture)
            finally:
                for path in paths:
                    path.unlink(missing_ok=True)
            self.capture = session.capture

    @staticmethod
    def _view_ref(session, view):
        return session.observation_id + (":" + view if len(session.observation_views) > 1 else "")

    def refresh(self, session):
        # A tool response/close flushes the newest frame even between FPS ticks.
        # Repeated observations reuse the same pixels; no controller call occurs.
        if self.capture != session.capture:
            self._call(lambda publisher: self._frame(publisher, session))

    def event(self, session, action, phase, *, ok=None):
        def publish(publisher):
            publisher.state["step"] = session.actions
            publisher.event(action, phase, ok=ok)
        self._call(publish)

    def point(self, session, pixel, tool, *, description=None, view="front"):
        def publish(publisher):
            import numpy as np
            from PIL import Image, ImageDraw

            self.refresh(session)
            frames = getattr(session, "view_frames", {"front": session.last_rgb})
            marked = Image.fromarray(frames[view]).copy()
            draw = ImageDraw.Draw(marked)
            x, y = (float(pixel[0]) * (marked.width - 1), float(pixel[1]) * (marked.height - 1))
            radius = max(5, round(min(marked.size) / 45))
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         outline="#ff7f00", width=3)
            draw.line((x - radius - 4, y, x + radius + 4, y), fill="#ff7f00", width=2)
            draw.line((x, y - radius - 4, x, y + radius + 4), fill="#ff7f00", width=2)
            self._image(publisher, np.asarray(marked), lambda path: publisher.overlay(str(path), dict(
                direction=view, capture_seq=session.capture, image_ref=self._view_ref(session, view),
                point=list(pixel), tool=tool, description=description,
                success_scoring="withheld")))
            publisher.event(tool, "recorded", ok=True)
        self._call(publish)

    def finish(self, session, reason):
        def publish(publisher):
            self.refresh(session)
            publisher.state.update(step=session.actions, stop_called=reason == "agent_stop", reason=reason)
            publisher.event("STOP" if reason == "agent_stop" else reason, "finished",
                            ok=not session.infra_failed)
            interrupted = session.infra_failed or reason in ("operator_interrupt", "wall_timeout", "infrastructure_error")
            publisher.finish("interrupted" if interrupted else "closed")
        self._call(publish)
