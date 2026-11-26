"""Frozen dataset integrity and archive-path safety checks."""

import hashlib
import json
import zipfile

import pytest

from supernav.evaluation.demand_driven.dataset import audit_archive, load_episode, validate_dataset, write_json


@pytest.fixture
def archive(tmp_path):
    house = json.dumps({"objects": [{"id": "Table|1", "children": [{"id": "Mug|1"}]}]}).encode()
    episode = dict(episode_id="e", scene_id="train.jsonl_1", instruction="Prepare a drink",
                   start_position=dict(x=0,y=.95,z=0), start_rotation_y=0, start_horizon=0,
                   reproducibility=dict(house_data_sha256=hashlib.sha256(house).hexdigest()),
                   stage_plan=[dict(allowed_target_candidates=[dict(object_id="Mug|1")])])
    path = tmp_path / "test.zip"
    with zipfile.ZipFile(path, "w") as output:
        output.writestr("test/episodes/001_e.json", json.dumps(episode))
        output.writestr("test/reproducibility/e/house_data.json", house)
        output.writestr("test/result_replacement_manifest.json", json.dumps(dict(model_conditioned=True, replacement_count=25)))
    return path


def test_all_nested_object_ids_and_selection_provenance_are_checked(archive):
    audit = audit_archive(archive, expected_count=1)
    assert audit["episode_count"] == audit["scene_count"] == 1
    assert audit["model_conditioned_selection"] is True
    assert audit["replacement_count"] == 25
    assert audit["episodes"][0]["episode_sha256"]
    assert audit["success_scoring"] == "withheld"


def test_unexpected_episode_count_is_rejected(archive):
    with pytest.raises(ValueError, match="Expected 200"):
        audit_archive(archive)


@pytest.mark.parametrize("name", ["../escape.json", "/absolute.json", "test/../../escape.json", "test\\escape.json"])
def test_archive_path_traversal_is_rejected(archive, name):
    with zipfile.ZipFile(archive, "a") as output:
        output.writestr(name, "{}")
    with pytest.raises(ValueError, match="Unsafe"):
        audit_archive(archive, expected_count=1)


def test_episode_start_and_instruction_cannot_change_after_freeze(archive, tmp_path):
    audit = audit_archive(archive, expected_count=1)
    dataset = tmp_path / "materialized"
    with zipfile.ZipFile(archive) as source:
        source.extractall(dataset)
    write_json(dataset / "audit.json", audit)
    original, _ = load_episode(dataset, "e")
    assert original["instruction"] == "Prepare a drink"
    path = dataset / audit["episodes"][0]["episode_path"]
    modified = json.loads(path.read_text())
    modified["start_position"]["x"] += .1
    write_json(path, modified)
    with pytest.raises(ValueError, match="Instruction/start/GT"):
        load_episode(dataset, "e")


def test_house_changes_are_rejected(archive, tmp_path):
    audit = audit_archive(archive, expected_count=1)
    dataset = tmp_path / "materialized"
    with zipfile.ZipFile(archive) as source:
        source.extractall(dataset)
    write_json(dataset / "audit.json", audit)
    write_json(dataset / audit["episodes"][0]["house_path"], {"objects": []})
    with pytest.raises(ValueError, match="House fingerprint"):
        load_episode(dataset, "e")


def test_materialized_validation_is_read_only_and_preserves_provenance(archive, tmp_path):
    audit = audit_archive(archive, expected_count=1)
    dataset = tmp_path / "materialized"
    with zipfile.ZipFile(archive) as source:
        source.extractall(dataset)
    write_json(dataset / "audit.json", audit)
    before = (dataset / "audit.json").stat().st_mtime_ns
    result = validate_dataset(dataset)
    assert result["valid"] and result["model_conditioned_selection"]
    assert result["episode_count"] == result["scene_count"] == 1
    assert (dataset / "audit.json").stat().st_mtime_ns == before
    audit["scene_count"] = 2
    write_json(dataset / "audit.json", audit)
    with pytest.raises(ValueError, match="scene count"):
        validate_dataset(dataset)
