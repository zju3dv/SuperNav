"""Exercise live capture, actual HTTP/SSE transport and portable evidence reading."""
from __future__ import annotations

from contextlib import contextmanager
import http.client
import json
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from supernav.backends.habitat.live import SuperNavLiveMixin
from supernav.runtime.live import LivePublisher
from supernav.web.server import ViewerServer
from supernav.web.store import SessionStore
from supernav.web.trace import TraceReader


def sample(publisher, step=0, *, force=True):
    publisher.sample(position=[float(step), 0, 0], heading=90, step=step,
                     path_length=float(step), capture=lambda: np.zeros((12, 16, 3), dtype=np.uint8),
                     force=force)


def test_publisher_throttles_capture_and_bounds_sidecar(tmp_path):
    publisher = LivePublisher(tmp_path, "session-1", scene="room", fps=1)
    sample(publisher)
    publisher.sample(position=[1, 0, 0], heading=0, step=1, path_length=1,
                     capture=lambda: pytest.fail("capture was not throttled"))
    assert json.loads((tmp_path / "session-1.json").read_text())["step"] == 0
    publisher.state["trajectory"] = [[i, 0, 0] for i in range(4001)]
    sample(publisher, 4002)
    for _ in range(101):
        publisher.event("navigate", "started")
    doc = json.loads((tmp_path / "session-1.json").read_text())
    assert len(doc["trajectory"]) == 4000 and doc["trail_truncated"]
    assert len(doc["events"]) == 100
    assert {p.name for p in tmp_path.iterdir()} == {"session-1.json", "session-1.jpg"}
    assert (tmp_path / "session-1.jpg").read_bytes().startswith(b"\xff\xd8")
    publisher.finish()
    assert json.loads((tmp_path / "session-1.json").read_text())["status"] == "closed"


class FakeAdapter:
    """Its step bookkeeping follows the adapter's before-capture ordering."""
    def __init__(self):
        self.session = SimpleNamespace(session_id="fake-session", scene="room", step_count=0,
                                       cumulative_path_length=0, position=[0, 0, 0])
        self._sessions = {self.session.session_id: self.session}
        self.bookkeeping = []
        self.reads = 0

    def _current_position(self, session):
        return session.position

    def _heading_degrees(self, session):
        return 90

    def _capture_sensor_observations(self, session):
        self.reads += 1
        return {"color_sensor": np.full((12, 16, 3), session.step_count, dtype=np.uint8)}

    def _record_pose(self, session):
        self.bookkeeping.append(list(session.position))

    def handle_request(self, request):
        session = self.session
        if request["action"] == "step":
            session.step_count += 1
            session.position[0] += 1
            session.cumulative_path_length += 1
            self._record_pose(session)
        elif request["action"] == "close_session":
            self._sessions.pop(session.session_id)
            self._dispose_session(session, "close_session")
        return {"ok": request["action"] != "error", "session_id": session.session_id,
                "result": {"step_count": session.step_count, "original": "preserved"}}

    def _dispose_session(self, session, reason):
        pass

    def close_all(self):
        self._sessions.clear()


class Adapter(SuperNavLiveMixin, FakeAdapter):
    pass


@pytest.mark.parametrize("enabled", [True, False])
def test_spectator_preserves_outcomes_bookkeeping_and_close(enabled, monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERNAV_LIVE_DIR", str(tmp_path) if enabled else "")
    adapter = Adapter()
    request = {"action": "step", "session_id": "fake-session"}
    original = FakeAdapter()
    assert adapter.handle_request(request) == original.handle_request(request)
    assert adapter.bookkeeping == original.bookkeeping == [[1, 0, 0]]
    error = {"action": "error", "session_id": "fake-session"}
    assert adapter.handle_request(error) == original.handle_request(error)
    if enabled:
        state = json.loads((tmp_path / "fake-session.json").read_text())
        assert state["pose"]["position"] == [1, 0, 0] and state["step"] == 1
        assert state["events"][-1]["ok"] is False
    else:
        assert not list(tmp_path.iterdir()) and adapter.reads == 0
    close = {"action": "close_session", "session_id": "fake-session"}
    assert adapter.handle_request(close) == original.handle_request(close)
    if enabled:
        assert json.loads((tmp_path / "fake-session.json").read_text())["status"] == "closed"


def test_disk_failure_never_retries_or_hides_a_successful_move(monkeypatch, tmp_path):
    target = tmp_path / "not-a-directory"
    target.write_text("occupied")
    monkeypatch.setenv("SUPERNAV_LIVE_DIR", str(target))
    adapter = Adapter()
    outcome = adapter.handle_request({"action": "step", "session_id": "fake-session"})
    assert outcome["ok"] is True and outcome["result"]["step_count"] == 1
    assert adapter.bookkeeping == [[1, 0, 0]]


def test_idle_disposal_marks_interrupted_and_releases_publisher(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERNAV_LIVE_DIR", str(tmp_path))
    adapter = Adapter()
    adapter.handle_request({"action": "step", "session_id": "fake-session"})
    adapter._dispose_session(adapter.session, "idle_timeout")
    assert json.loads((tmp_path / "fake-session.json").read_text())["status"] == "interrupted"
    assert not adapter._live_publishers


def test_render_failure_retains_last_frame_and_new_pose(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERNAV_LIVE_DIR", str(tmp_path))
    adapter = Adapter()
    adapter.handle_request({"action": "step", "session_id": "fake-session"})
    previous = (tmp_path / "fake-session.jpg").read_bytes()
    def broken_capture(session):
        raise RuntimeError("renderer unavailable")
    monkeypatch.setattr(adapter, "_capture_sensor_observations", broken_capture)
    outcome = adapter.handle_request({"action": "step", "session_id": "fake-session"})
    assert outcome["ok"] and outcome["result"]["step_count"] == 2
    state = json.loads((tmp_path / "fake-session.json").read_text())
    assert state["pose"]["position"] == [2, 0, 0] and state["capture_error"] == "RuntimeError"
    assert (tmp_path / "fake-session.jpg").read_bytes() == previous


@contextmanager
def running_server(store):
    server = ViewerServer(("127.0.0.1", 0), store)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.stopping.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def get(port, path):
    client = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    client.request("GET", path)
    response = client.getresponse()
    status, headers, body = response.status, dict(response.getheaders()), response.read()
    client.close()
    return status, headers, body


def next_snapshot(response):
    for _ in range(100):
        line = response.readline()
        if line.startswith(b"data: "):
            return json.loads(line[6:])
    pytest.fail("No SSE snapshot")


def test_http_stream_updates_before_producer_finishes_and_reconnects(tmp_path):
    publisher = LivePublisher(tmp_path, "session-1", scene="room")
    sample(publisher)
    runs = tmp_path / "runs"
    store = SessionStore(tmp_path, runs_root=runs)
    with running_server(store) as port:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        connection.request("GET", "/api/sessions/session-1/events")
        response = connection.getresponse()
        assert response.getheader("Content-Type").startswith("text/event-stream")
        assert next_snapshot(response)["step"] == 0
        sample(publisher, 7)
        active = next_snapshot(response)
        assert active["status"] == "running" and active["step"] == 7
        status, _, image = get(port, active["image_url"])
        assert status == 200 and image.startswith(b"\xff\xd8")
        # A later metadata write must stream even when telemetry revision is unchanged.
        runs.mkdir()
        (runs / "run.json").write_text(json.dumps({"instruction": "Find the chair."}))
        (runs / "metrics.json").write_text(json.dumps({"session_id": "session-1"}))
        store._next_scan = 0
        assert next_snapshot(response)["instruction"] == "Find the chair."
        connection.close()
        publisher.finish()
        reconnect = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        reconnect.request("GET", "/api/sessions/session-1/events", headers={"Last-Event-ID": "1"})
        assert next_snapshot(reconnect.getresponse())["status"] == "closed"
        reconnect.close()


def test_http_asset_allowlist_and_symlink_escape(tmp_path):
    publisher = LivePublisher(tmp_path / "live", "session-1", scene="room")
    sample(publisher)
    private = tmp_path / "auth.json"
    private.write_text('{"secret":"never serve"}')
    frame = tmp_path / "live/session-1.jpg"
    frame.unlink()
    frame.symlink_to(private)
    (tmp_path / "live/escape.json").symlink_to(private)
    with running_server(SessionStore(tmp_path / "live")) as port:
        for path in ("/auth.json", "/../auth.json", "/%2e%2e/auth.json", "/api/sessions/escape",
                     "/api/sessions/session-1/frame", "/api/sessions/session-1/frame?index=-1"):
            assert get(port, path)[0] == 404
        for path in ("/", "/app.js", "/app.css", "/fonts/space-grotesk.ttf"):
            status, headers, body = get(port, path)
            assert status == 200 and body and "nosniff" == headers["X-Content-Type-Options"]


def test_archive_discovery_skips_partial_files_and_never_treats_claim_as_gt(tmp_path):
    visuals, runs = tmp_path / "visuals", tmp_path / "runs"
    visuals.mkdir(); runs.mkdir()
    (visuals / "broken.trajectory.json").write_text('{"session_id":')
    (visuals / "s.trajectory.json").write_text(json.dumps({
        "session_id": "s", "status": "closed", "scene": "room",
        "dense_trajectory": {"points": [[0, 0, 0], [3, 0, 4]]},
        "end_pose": {"position": [3, 0, 4]}, "trajectory": [{"step_count": 6}],
    }))
    (visuals / "s").mkdir()
    (visuals / "s/step000001_color_sensor.png").write_bytes(b"test frame")
    (visuals / "s/step000001_depth_sensor.png").write_bytes(b"depth must not appear")
    (runs / "run.json").write_text(json.dumps({"instruction": "<script>alert(1)</script>", "agent": "codex"}))
    (runs / "metrics.json").write_text(json.dumps({"session_id": "s", "success": True,
                                                  "agent_terminal_claim": {"outcome": "achieved"}}))
    store = SessionStore(tmp_path / "live", visuals_roots=[visuals], runs_root=runs)
    rows = store.sessions()
    assert len(rows) == 1
    doc = store.snapshot(rows[0]["id"])
    assert doc["path_length_m"] == 5 and len(doc["frames"]) == 1
    assert doc["evaluation"] is None and doc["agent_claim"]["outcome"] == "achieved"
    assert doc["source"] == "recording" and doc["instruction"].startswith("<script>")
    assert store.frame(rows[0]["id"], -1) is None


def test_dead_producer_is_interrupted_without_rewriting_evidence(monkeypatch, tmp_path):
    publisher = LivePublisher(tmp_path, "session-1", scene="room")
    sample(publisher)
    path = tmp_path / "session-1.json"
    before = path.read_bytes()
    def dead(*args):
        raise ProcessLookupError
    monkeypatch.setattr("supernav.web.store.os.kill", dead)
    assert SessionStore(tmp_path).snapshot("session-1")["status"] == "interrupted"
    assert path.read_bytes() == before


def test_viewer_imports_without_simulator_sdk():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'habitat_sim' or fullname.startswith('habitat_sim.'):
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
from supernav.web.server import ViewerServer
from supernav.runtime.live import LivePublisher
from supernav.backends.habitat.live import SuperNavLiveMixin
"""
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr


def test_four_views_and_overlay_are_exact_bounded_and_linked(tmp_path):
    from PIL import Image
    source = tmp_path / "native.png"
    Image.new("RGB", (64, 48), "orange").save(source)
    native = source.read_bytes()
    publisher = LivePublisher(tmp_path / "live", "session-1", scene="room")
    sample(publisher)
    images = [{"direction": d, "path": str(source), "image_ref": f"pano:session-1:1:{d}"}
              for d in ("front", "right", "back", "left")]
    publisher.panorama(images, 1)
    publisher.overlay(str(source), {"tool": "hab_visual_point_navigate", "capture_seq": 1,
                                    "direction": "left", "point": [.4, .7], "image_ref": images[-1]["image_ref"]})
    store = SessionStore(tmp_path / "live")
    doc = store.snapshot("session-1")
    with running_server(store) as port:
        for view in doc["panoramas"][0]["views"].values():
            assert get(port, view["image_url"])[2] == native
        assert get(port, doc["overlays"][0]["image_url"])[2] == native
        assert get(port, "/api/sessions/session-1/frame?kind=panorama&index=1&view=../../native")[0] == 404
    # A front-only capture must not accidentally reuse the other previous views.
    publisher.panorama(images[:1], 2)
    assert set(store.snapshot("session-1")["panoramas"][-1]["views"]) == {"front"}
    for i in range(3, 27):
        publisher.panorama(images, i)
        publisher.overlay(str(source), {"capture_seq": i})
    assert len(publisher.state["panoramas"]) == 12
    assert len(publisher.state["overlays"]) == 24
    assert len(list((tmp_path / "live").iterdir())) == 2 + 48 + 24
    assert store.frame("session-1", 1, kind="panorama", view="front") is None
    assert store.frame("session-1", 1, kind="overlay") is None
    assert source.read_bytes() == native


def test_adapter_publishes_original_panorama_and_point_immediately(monkeypatch, tmp_path):
    monkeypatch.setenv("SUPERNAV_LIVE_DIR", str(tmp_path / "live"))
    source = tmp_path / "original.png"
    source.write_bytes(b"original native image bytes")
    outcome = {"images": [{"direction": "front", "path": str(source), "image_ref": "pano:fake-session:7:front"}]}

    class ObservingFake(FakeAdapter):
        def _get_panorama(self, session_id, payload):
            self.session.latest_visual_capture_seq = 7
            return outcome

    class ObservingAdapter(SuperNavLiveMixin, ObservingFake):
        pass

    adapter = ObservingAdapter()
    assert adapter._get_panorama("fake-session", {}) is outcome
    adapter._publish_live_overlay(adapter.session, str(source), capture_seq=7, direction="front")
    # Visible before the enclosing tool returns / starts its movement.
    doc = SessionStore(tmp_path / "live").snapshot("fake-session")
    assert doc["panoramas"][0]["capture_seq"] == doc["overlays"][0]["capture_seq"] == 7
    assert adapter.bookkeeping == []


def append_events(path, *rows):
    with path.open("a") as stream:
        for row in rows:
            stream.write(json.dumps(row) + "\n")


def test_trace_reads_complete_lines_incrementally_and_pairs_tool_updates(tmp_path):
    path = tmp_path / "raw.jsonl"
    append_events(path, {"type": "item.completed", "item": {"id": "a", "type": "agent_message", "text": "I will inspect the left doorway."}},
                  {"type": "item.started", "item": {"id": "b", "type": "mcp_tool_call", "tool": "hab_visual_point_navigate", "arguments": {"view": "left", "point": [.4, .7]}}})
    reader = TraceReader(tmp_path)
    first = reader.snapshot()
    assert [e["kind"] for e in first["events"]] == ["thought", "tool"]
    assert first["events"][-1]["status"] == "running"
    incomplete = {"type": "item.completed", "item": {"id": "b", "type": "mcp_tool_call", "tool": "hab_visual_point_navigate", "arguments": {"view": "left", "point": [.4, .7]}, "result": {"ok": True}}}
    with path.open("a") as f:
        f.write(json.dumps(incomplete))
    assert reader.snapshot()["revision"] == first["revision"]
    with path.open("a") as f:
        f.write("\n")
    events = reader.snapshot()["events"]
    assert len(events) == 2 and events[-1]["status"] == "completed"
    assert events[-1]["line"] == 2 and events[-1]["result_line"] == 3
    assert reader.snapshot()["events"] == events


def test_trace_merges_actual_codex_summaries_chronologically_without_encrypted_content(tmp_path):
    path = tmp_path / "raw.jsonl"
    args = {"view": "left", "point": [.4, .7]}
    append_events(path, {"type": "item.completed", "item": {"id": "a", "type": "agent_message", "text": "Inspect the door."}},
                  {"type": "item.completed", "item": {"id": "b", "type": "mcp_tool_call", "tool": "hab_visual_point_navigate", "arguments": args, "result": {"ok": True}}})
    session = tmp_path / "codex_project/.codex_home/sessions/2026/10/02/native.jsonl"
    session.parent.mkdir(parents=True)
    append_events(session,
        {"type": "response_item", "timestamp": "2026-10-02T01:00:00Z", "payload": {"type": "message", "role": "assistant", "content": [{"text": "Inspect the door."}]}},
        {"type": "response_item", "timestamp": "2026-10-02T01:00:01Z", "payload": {"type": "reasoning", "summary": [{"text": "The left doorway is reachable."}], "encrypted_content": "must never expose"}},
        {"type": "event_msg", "timestamp": "2026-10-02T01:00:02Z", "payload": {"type": "item_completed", "item": {"type": "McpToolCall", "tool": "hab_visual_point_navigate", "arguments": args}}},
        {"type": "response_item", "timestamp": "2026-10-02T01:00:03Z", "payload": {"type": "reasoning", "summary": [], "encrypted_content": "must never expose"}})
    snapshot = TraceReader(tmp_path).snapshot()
    assert [e["kind"] for e in snapshot["events"]] == ["thought", "thought", "tool"]
    assert snapshot["events"][-1]["timestamp_kind"] == "returned"
    assert "must never expose" not in json.dumps(snapshot)
    assert len(snapshot["events"]) == 3


def test_trace_switches_to_canonical_and_retains_result_provenance(tmp_path):
    append_events(tmp_path / "raw.jsonl", {"role": "assistant", "content": "Native message"})
    reader = TraceReader(tmp_path)
    assert reader.snapshot()["events"][0]["text"] == "Native message"
    append_events(tmp_path / "canonical.jsonl",
        {"type": "assistant_reasoning", "text": "I will inspect this point."},
        {"type": "tool_call", "name": "hab_visual_point_navigate", "input": {"point": [.4, .7]}},
        {"type": "tool_result", "name": "hab_visual_point_navigate", "audit_tool_seq": 2,
         "audit": {"selected_anchor": {"image_ref": "pano:s:1:left"}}, "content": json.dumps({"ok": False, "error": "unreachable", "api_key": "never", "image": {"type": "image", "data": "base64"}})})
    result = reader.snapshot()
    assert len(result["events"]) == 2
    tool = result["events"][-1]
    assert tool["status"] == "error" and tool["audit_tool_seq"] == 2
    assert tool["image_ref"] == "pano:s:1:left" and tool["source"] == "canonical.jsonl"
    assert tool["result"]["api_key"] == "[redacted]"
    assert "base64" not in json.dumps(result)


def test_native_kimi_and_opencode_trace_and_bounded_history(tmp_path):
    path = tmp_path / "raw.jsonl"
    append_events(path,
        {"role": "assistant", "content": [{"think": "Look toward the kitchen."}], "tool_calls": [{"id": "call-1", "function": {"name": "hab_turn", "arguments": '{"degrees": 90}'}}]},
        {"role": "tool", "tool_call_id": "call-1", "content": '{"ok":true}'},
        {"type": "tool_use", "part": {"callID": "call-2", "tool": "hab_forward", "state": {"status": "completed", "input": {"distance": 1}, "output": "Moved"}}})
    reader = TraceReader(tmp_path)
    events = reader.snapshot()["events"]
    assert len(events) == 3 and events[0]["channel"] == "recorded reasoning"
    assert events[1]["arguments"] == {"degrees": 90} and events[1]["status"] == "completed"
    assert events[2]["name"] == "hab_forward"
    append_events(path, *[{"type": "tool_call", "name": "hab_turn", "input": {}} for _ in range(350)])
    assert len(reader.snapshot()["events"]) == 300 and len(reader.pending["hab_turn"]) == 300


def test_trace_links_running_episode_before_metrics_are_written(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "run.json").write_text(json.dumps({"run_id": "episode-1"}))
    append_events(runs / "raw.jsonl", {"role": "assistant", "content": "I will inspect the room."})
    publisher = LivePublisher(tmp_path / "live", "s", scene="room")
    publisher.state["run_id"] = "episode-1"
    sample(publisher)
    store = SessionStore(tmp_path / "live", runs_root=runs)
    with running_server(store) as port:
        status, _, body = get(port, "/api/sessions/s/trace")
    assert status == 200 and json.loads(body)["events"][0]["text"] == "I will inspect the room."
    assert not (runs / "canonical.jsonl").exists()


def test_live_kimi_wire_contains_thinking_and_pending_tools(tmp_path):
    wire = tmp_path / "kimi_project/kimi_home/.kimi-code/sessions/workspace/session/agents/main/wire.jsonl"
    wire.parent.mkdir(parents=True)
    append_events(wire,
        {"type": "context.append_loop_event", "time": 1790821793148, "event": {"type": "content.part", "part": {"think": "Check the doorway before moving."}}},
        {"type": "context.append_loop_event", "time": 1790821793149, "event": {"type": "tool.call", "toolCallId": "a", "name": "hab_turn", "args": {"degrees": 90}}})
    reader = TraceReader(tmp_path)
    events = reader.snapshot()["events"]
    assert events[0]["text"] == "Check the doorway before moving."
    assert events[1]["status"] == "running"
    append_events(wire, {"type": "context.append_loop_event", "time": 1790821794000, "event": {"type": "tool.result", "toolCallId": "a", "result": {"isError": True, "output": "Session unavailable"}}})
    events = reader.snapshot()["events"]
    assert len(events) == 2 and events[1]["status"] == "error"
    assert events[1]["result_line"] == 3


def test_trace_recovers_after_oversized_line_and_never_follows_external_session(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "raw.jsonl").write_bytes(b"x" * (8 * 1024 * 1024 + 5) + b"\n")
    append_events(run / "raw.jsonl", {"role": "assistant", "content": "Resumed after the oversized record."})
    outside = tmp_path / "private.jsonl"
    append_events(outside, {"type": "response_item", "payload": {"type": "reasoning", "summary": "unrelated private session"}})
    (run / "codex_session.jsonl").symlink_to(outside)
    reader = TraceReader(run)
    reader.snapshot()
    events = reader.snapshot()["events"]
    assert len(events) == 1 and events[0]["text"] == "Resumed after the oversized record."


def test_archive_media_pairs_capture_and_exact_audit_overlay(tmp_path):
    visuals = tmp_path / "visuals"
    folder = visuals / "s"
    folder.mkdir(parents=True)
    (visuals / "s.trajectory.json").write_text(json.dumps({"session_id": "s", "status": "closed"}))
    for direction in ("front", "right", "back", "left"):
        (folder / f"pano_{direction}_step000007_color_sensor.png").write_bytes(b"raw")
        (folder / f"pano_{direction}_step000007_color_sensor_agent.png").write_bytes(b"agent-visible")
    overlay = visuals / "visual_point.png"
    overlay.write_bytes(b"native-overlay")
    private = tmp_path / "outside.png"
    private.write_bytes(b"unrelated")
    append_events(visuals / "s.benchmark_audit.jsonl",
        {"tool_seq": 2, "tool_name": "hab_visual_point_navigate", "result": {"overlay_image": str(overlay), "selected_anchor": {"image_ref": "pano:s:7:left", "point": [.4, .7]}}},
        {"tool_seq": 3, "tool_name": "hab_visual_point_navigate", "result": {"overlay_image": str(private)}})
    store = SessionStore(tmp_path / "live", visuals_roots=[visuals])
    key = store.sessions()[0]["id"]
    doc = store.snapshot(key)
    assert len(doc["panoramas"]) == 1 and len(doc["panoramas"][0]["views"]) == 4
    assert len(doc["overlays"]) == 1 and doc["overlays"][0]["capture_seq"] == 7
    assert store.frame(key, 7, kind="panorama", view="left").read_bytes() == b"agent-visible"
    assert store.frame(key, 2, kind="overlay").read_bytes() == b"native-overlay"
    assert store.frame(key, 3, kind="overlay") is None
