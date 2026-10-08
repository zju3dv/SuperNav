"""Opt-in spectator hooks over the external adapter, on its original GL thread."""
from __future__ import annotations

import logging
import json
import os
from pathlib import Path

from supernav.runtime.live import LivePublisher

LOGGER = logging.getLogger(__name__)
_INTERNAL_ACTIONS = {"get_audit_metrics", "get_metrics", "get_runtime_status",
                     "describe_api"}


class SuperNavLiveMixin:
    def __init__(self, *args, **kwargs):
        self._live_publishers = {}
        self._live_failed = set()
        self._live_directory = os.environ.get("SUPERNAV_LIVE_DIR", "")
        self._live_fps = 5.0
        self._live_context = {}
        if self._live_directory:
            try:
                self._live_fps = float(os.environ.get("SUPERNAV_LIVE_FPS", "5"))
                context = json.loads(os.environ.get("SUPERNAV_LIVE_CONTEXT", "{}"))
                if isinstance(context, dict):
                    self._live_context = {key: str(context[key]) for key in
                                          ("instruction", "agent", "run_id", "task_id") if key in context}
            except (ValueError, TypeError):
                LOGGER.warning("Invalid live viewer settings; live capture disabled")
                self._live_directory = ""
        super().__init__(*args, **kwargs)

    def _live_call(self, session_id, callback):
        if not self._live_directory or not session_id or session_id in self._live_failed:
            return
        try:
            return callback()
        except Exception:
            self._live_failed.add(session_id)
            LOGGER.warning("Live viewer capture disabled for session %s", session_id,
                           exc_info=True)

    def _live_sample(self, session, *, force=False):
        def sample():
            publisher = self._live_publishers.get(session.session_id)
            if publisher is None:
                publisher = LivePublisher(Path(self._live_directory), session.session_id,
                                          scene=str(session.scene), fps=self._live_fps)
                self._live_publishers[session.session_id] = publisher
                publisher.state.update(self._live_context)
            if force or publisher.due():
                publisher.sample(
                    position=[float(v) for v in self._current_position(session)],
                    heading=float(self._heading_degrees(session)), step=session.step_count,
                    path_length=float(session.cumulative_path_length), force=force,
                    # Never use last_sensor_obs here: _record_pose can precede its update.
                    capture=lambda: self._capture_sensor_observations(session).get("color_sensor"),
                )
        self._live_call(session.session_id, sample)

    def _record_pose(self, session):
        result = super()._record_pose(session)
        self._live_sample(session)
        return result

    def _get_panorama(self, session_id, payload):
        result = super()._get_panorama(session_id, payload)
        if self._live_directory:
            session = self._sessions.get(session_id)
            if session is not None:
                self._live_sample(session)
                self._live_call(session_id, lambda: self._live_publishers[session_id].panorama(
                    result.get("images", []), session.latest_visual_capture_seq))
        return result

    def _publish_live_overlay(self, session, path, **metadata):
        if not path or not self._live_directory:
            return
        self._live_sample(session)
        self._live_call(session.session_id, lambda: self._live_publishers[session.session_id].overlay(path, metadata))

    def handle_request(self, request):
        action = request.get("action", "unknown")
        if not isinstance(action, str):
            return super().handle_request(request)
        sid = request.get("session_id")
        session = self._sessions.get(sid) if isinstance(sid, str) else None
        if session is not None and action not in _INTERNAL_ACTIONS:
            self._live_sample(session, force=action == "close_session")
            self._live_call(sid, lambda: self._live_publishers[sid].event(action, "started"))
        # Return exactly the original outcome, including errors.
        response = super().handle_request(request)
        sid = response.get("session_id") or sid
        session = self._sessions.get(sid) if isinstance(sid, str) else None
        if session is not None and action not in _INTERNAL_ACTIONS:
            self._live_sample(session, force=True)
        if isinstance(sid, str) and sid in self._live_publishers and action not in _INTERNAL_ACTIONS:
            self._live_call(sid, lambda: self._live_publishers[sid].event(
                action, "finished", ok=response.get("ok") is True))
            if action == "close_session" and response.get("ok") is True:
                self._live_call(sid, lambda: self._live_publishers[sid].finish())
                self._live_publishers.pop(sid, None)
                self._live_failed.discard(sid)
        return response

    def _dispose_session(self, session, reason):
        result = super()._dispose_session(session, reason)
        sid = session.session_id
        if sid in self._live_publishers:
            status = "closed" if reason == "close_session" else "interrupted"
            self._live_call(sid, lambda: self._live_publishers[sid].finish(status))
            if reason != "close_session":
                self._live_publishers.pop(sid, None)
                self._live_failed.discard(sid)
        return result

    def close_all(self):
        try:
            return super().close_all()
        finally:
            for sid, publisher in self._live_publishers.items():
                if publisher.state["status"] == "running":
                    self._live_call(sid, lambda p=publisher: p.finish("interrupted"))
