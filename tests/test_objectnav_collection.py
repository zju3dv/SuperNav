"""Completed-result collection must not bias scores or publish partial updates."""

import json
import os
import sys
from pathlib import Path

import pytest


import supernav.evaluation.objectnav.collection as collect


def make_result(root, name, *, stopped=False, trajectory=False):
    run = root / name
    run.mkdir(parents=True)
    (run / "run.json").write_text(json.dumps({"task_id": name}))
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "task_id": name,
                "close_called": stopped,
                "session_id": name,
                "process_completed": False,
                "returncode": 1,
            }
        )
    )
    visuals = root / "visuals"
    visuals.mkdir(exist_ok=True)
    if trajectory:
        (visuals / f"{name}.trajectory.json").write_text("{}")
    return run, visuals


def test_inventory_excludes_pending_nested_and_credential_artifacts(tmp_path):
    run, visuals = make_result(tmp_path, "finished")
    os.utime(run / "metrics.json", (1, 1))
    (run / "auth.json").write_text("private")
    (run / "codex_project").mkdir()
    (run / "codex_project" / "auth.json").write_text("private")
    make_result(tmp_path, "fresh")
    make_result(tmp_path / "_preflight", "not-formal")
    inventory = collect.inventory(tmp_path, visuals, 60)
    assert [row["task_id"] for row in inventory["runs"]] == ["finished"]
    assert inventory["pending"] == 1
    assert inventory["invalid"] == []
    assert "auth.json" not in inventory["runs"][0]["files"]
    # A finalized no-STOP/error result is still in the completed denominator.
    assert inventory["runs"][0]["trajectory"] is None


def test_inventory_reports_missing_stopped_trajectory(tmp_path):
    _, visuals = make_result(tmp_path, "stopped", stopped=True)
    result = collect.inventory(tmp_path, visuals, 0)
    assert not result["runs"]
    assert "missing its trajectory" in result["invalid"][0]["error"]


def test_select_latest_deduplicates_by_task_and_rejects_foreign_tasks():
    older = {
        "task_id": "task",
        "metrics_mtime": 10,
        "source": "local",
        "directory": "old",
    }
    newer = {
        "task_id": "task",
        "metrics_mtime": 20,
        "source": "remote",
        "directory": "new",
    }
    assert collect.select_latest([newer, older], {"task"}) == [newer]
    with pytest.raises(ValueError, match="outside the experiment"):
        collect.select_latest([newer], {"other"})


def test_collection_failure_preserves_previous_report_and_releases_lock(
    tmp_path, monkeypatch
):
    previous = tmp_path / "history" / "previous"
    previous.mkdir(parents=True)
    (previous / "summary.json").write_text('{"completed": 5}')
    latest = tmp_path / "latest"
    latest.symlink_to(previous, target_is_directory=True)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"output_dir": str(tmp_path)}))
    monkeypatch.setattr(sys, "argv", ["collect", "--config", str(config)])

    def fail(*args):
        raise RuntimeError("remote unavailable")

    monkeypatch.setattr(collect, "collect_and_score", fail)
    assert collect.main() == 1
    assert latest.resolve() == previous
    assert json.loads((latest / "summary.json").read_text()) == {"completed": 5}
    assert "remote unavailable" in (tmp_path / "last_error.json").read_text()
    monkeypatch.setattr(collect, "collect_and_score", lambda *args: previous)
    assert collect.main() == 0


def test_score_failure_does_not_publish_one_of_two_scores(tmp_path, monkeypatch):
    output = tmp_path / "output"
    previous = output / "history" / "previous"
    previous.mkdir(parents=True)
    (output / "latest").symlink_to(previous, target_is_directory=True)
    manifest = tmp_path / "instructions.json"
    manifest.write_text('{"instructions": []}')
    monkeypatch.setattr(collect, "collect_sources", lambda *args: ({}, []))

    def fail_after_first_score(config, selected, snapshot):
        (snapshot / "score_0.2m.json").write_text("{}")
        raise RuntimeError("second score failed")

    monkeypatch.setattr(collect, "score_snapshot", fail_after_first_score)
    with pytest.raises(RuntimeError, match="second score"):
        collect.collect_and_score({"instructions": str(manifest)}, output)
    assert (output / "latest").resolve() == previous
