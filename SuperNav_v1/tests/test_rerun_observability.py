"""Evidence replay and live frame polling."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from supernav.evaluation.session_trace_adapters import (
    CodexSessionTraceAdapter,
    KimiSessionTraceAdapter,
    ObservabilityRecord,
    ObservabilityTokens,
    ObservabilityToolCall,
    write_observability_jsonl,
)
from supernav.web import rerun_viewer


def test_offline_replay_preserves_common_records_and_raw_evidence(tmp_path, monkeypatch):
    records = (
        ObservabilityRecord(
            kind="llm_call", seq=0, call_id="codex-call-000001",
            reasoning="Inspect the doorway.",
            tool_calls=(ObservabilityToolCall("hab_look", {"session_id": "synthetic"}),),
            tokens=ObservabilityTokens(input_tokens=11, output_tokens=7, total_tokens=18),
        ),
        ObservabilityRecord(
            kind="frame", seq=1, call_id="codex-call-000001",
            image_path="frame.png", pose={"position": [1.0, 0.0, 2.0]}, action="hab_look",
        ),
    )
    source = write_observability_jsonl(records, tmp_path / "observability.jsonl")
    raw = {"call_id": records[0].call_id, "request": {"prompt": "synthetic"},
           "response": {"output": "original tool response"}}
    (tmp_path / "raw_llm_calls.jsonl").write_text(json.dumps(raw) + "\n")
    logged = {}
    frames = []
    monkeypatch.setattr(rerun_viewer, "_try_log_text", logged.__setitem__)
    monkeypatch.setattr(rerun_viewer, "rr", SimpleNamespace())
    replay = rerun_viewer.OfflineNavReplay(SimpleNamespace(offline_bundle=tmp_path, offline_fps=30))
    monkeypatch.setattr(replay, "_log_frame", lambda record, **kwargs: frames.append(record))

    assert replay._load_records() == records
    replay.run()
    assert frames == [records[1]]
    assert logged["agent/llm_reasoning"] == records[0].reasoning
    assert '"total_tokens": 18' in logged["agent/llm_tokens"]
    assert '"name": "hab_look"' in logged["agent/tool_call_decisions"]
    assert '"prompt": "synthetic"' in logged["agent/llm_raw_input"]
    assert "original tool response" in logged["agent/llm_raw_output"]
    assert source.read_text() == "".join(json.dumps(row.to_json(), sort_keys=True) + "\n" for row in records)
    assert (tmp_path / "raw_llm_calls.jsonl").read_text() == json.dumps(raw) + "\n"


def test_offline_raw_calls_remain_replayable_without_common_records(tmp_path, monkeypatch):
    raw = {"call_id": "kimi-call-000001", "seq": 0,
           "request": {"prompt": "synthetic"}, "response": {"text": "Move ahead."}}
    (tmp_path / "raw_llm_calls.jsonl").write_text(json.dumps(raw) + "\n")
    logged = {}
    monkeypatch.setattr(rerun_viewer, "_try_log_text", logged.__setitem__)
    monkeypatch.setattr(rerun_viewer, "rr", SimpleNamespace())
    rerun_viewer.OfflineNavReplay(SimpleNamespace(offline_bundle=tmp_path, offline_fps=30)).run()
    assert "Move ahead." in logged["agent/llm_raw_output"]


def test_live_polling_accepts_current_frame_snapshots(monkeypatch):
    args = SimpleNamespace(bridge_host="127.0.0.1", bridge_port=18911,
                           session_id="synthetic", tail_frames_dir=None, poll_interval_ms=1)
    viewer = rerun_viewer.NavViewer(args)
    snapshot = rerun_viewer.FrameSnapshot(
        wall_time_s=1.0, session_id="synthetic", step_count=4,
        metrics={"steps": 4}, visuals={}, topdown=None, depth_analysis=None,
    )
    logged, errors = [], []
    captures = iter((snapshot, snapshot))

    def capture():
        try:
            return next(captures)
        except StopIteration:
            raise KeyboardInterrupt

    monkeypatch.setattr(viewer, "_drain_tail_frames", lambda: 0)
    monkeypatch.setattr(viewer, "_capture", capture)
    monkeypatch.setattr(viewer, "_log_frame", logged.append)
    monkeypatch.setattr(rerun_viewer, "_try_log_text", lambda *args: errors.append(args))
    monkeypatch.setattr(rerun_viewer.time, "sleep", lambda _: None)
    viewer.run()
    assert logged == [snapshot, snapshot]
    assert errors == []


@pytest.mark.parametrize("backend", ["codex", "kimi"])
def test_native_adapters_preserve_current_reasoning_and_tools(tmp_path, backend):
    if backend == "codex":
        source = tmp_path / "codex_session.jsonl"
        rows = [
            {"type": "response_item", "payload": {"type": "message", "role": "assistant",
                "content": [{"type": "output_text", "text": "Inspect the doorway."}]}},
            {"type": "response_item", "payload": {"type": "function_call", "call_id": "call-1",
                "name": "hab_look", "arguments": '{"session_id":"synthetic"}'}},
        ]
        adapter = CodexSessionTraceAdapter()
    else:
        source = tmp_path / "agents" / "main" / "wire.jsonl"
        source.parent.mkdir(parents=True)
        (tmp_path / "manifest.json").write_text('{}')
        rows = [{"type": "assistant_message", "content": "Inspect the doorway.",
                 "tool_calls": [{"name": "hab_look", "arguments": {"session_id": "synthetic"}}]}]
        adapter = KimiSessionTraceAdapter()
    content = "".join(json.dumps(row) + "\n" for row in rows)
    source.write_text(content)
    records = adapter.parse(source)
    assert len(records) == 1
    assert records[0].reasoning == "Inspect the doorway."
    assert records[0].tool_calls == (ObservabilityToolCall("hab_look", {"session_id": "synthetic"}),)
    assert source.read_text() == content
