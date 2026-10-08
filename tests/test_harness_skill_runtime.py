from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from supernav.runtime.skill_runtime import create_skill_snapshot, lint_skill_root, load_skill_snapshot


def _skill(root: Path, name: str = "room-sweep") -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Sweep rooms methodically.\n---\n# Steps\n",
        encoding="utf-8",
    )
    return directory


def test_snapshot_copies_text_bundle_and_records_stable_hash(tmp_path: Path) -> None:
    source = tmp_path / "source"
    skill = _skill(source)
    (skill / "references").mkdir()
    (skill / "references" / "notes.md").write_text("notes\n", encoding="utf-8")
    (skill / "scripts").mkdir()
    (skill / "scripts" / "probe.py").write_text("print('ok')\n", encoding="utf-8")

    first = create_skill_snapshot(
        source, run_root=tmp_path / "run-a", skill_set_id="set"
    )
    second = create_skill_snapshot(
        source, run_root=tmp_path / "run-b", skill_set_id="set"
    )

    assert first.snapshot_sha256 == second.snapshot_sha256
    assert (first.skills_root / "room-sweep/scripts/probe.py").is_file()
    assert first.manifest["excluded_user_skills"] is True
    assert first.manifest["skills"] == [
        {"name": "room-sweep", "description": "Sweep rooms methodically."}
    ]
    assert all(
        {"path", "size", "sha256"} == set(row) for row in first.manifest["files"]
    )
    assert (
        load_skill_snapshot(first.snapshot_root).snapshot_sha256
        == first.snapshot_sha256
    )


def test_snapshot_does_not_copy_ignored_root_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _skill(source)
    (source / ".local-notes").write_text("not bundle content", encoding="utf-8")
    snapshot = create_skill_snapshot(
        source, run_root=tmp_path / "run", skill_set_id="set"
    )
    assert not (snapshot.skills_root / ".local-notes").exists()


def test_content_change_changes_snapshot_hash(tmp_path: Path) -> None:
    source = tmp_path / "source"
    skill = _skill(source)
    first = create_skill_snapshot(
        source, run_root=tmp_path / "run-a", skill_set_id="set"
    )
    (skill / "SKILL.md").write_text(
        "---\nname: room-sweep\ndescription: Changed.\n---\nbody\n", encoding="utf-8"
    )
    second = create_skill_snapshot(
        source, run_root=tmp_path / "run-b", skill_set_id="set"
    )
    assert first.snapshot_sha256 != second.snapshot_sha256


@pytest.mark.parametrize(
    ("frontmatter", "message"),
    [
        ("name: wrong\ndescription: okay", "name must match"),
        ("name: room-sweep", "description must be"),
        ("name: [broken\ndescription: okay", "invalid YAML"),
    ],
)
def test_invalid_frontmatter_fails(
    tmp_path: Path, frontmatter: str, message: str
) -> None:
    root = tmp_path / "skills"
    skill = root / "room-sweep"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(f"---\n{frontmatter}\n---\n", encoding="utf-8")
    report = lint_skill_root(root)
    assert not report.ok
    assert message in " ".join(report.errors)


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("image.png", "not really png", "file type"),
        ("api_key.txt", "placeholder", "secret-like"),
        ("notes.txt", "api_key=abcdefghijklmnopqrstuvwxyz", "possible secret"),
    ],
)
def test_unsafe_files_fail(
    tmp_path: Path, filename: str, content: str, message: str
) -> None:
    root = tmp_path / "skills"
    skill = _skill(root)
    (skill / filename).write_text(content, encoding="utf-8")
    report = lint_skill_root(root)
    assert not report.ok
    assert message in " ".join(report.errors)


def test_binary_and_symlink_fail(tmp_path: Path) -> None:
    root = tmp_path / "skills"
    skill = _skill(root)
    (skill / "binary.txt").write_bytes(b"text\0binary")
    outside = tmp_path / "outside.md"
    outside.write_text("private", encoding="utf-8")
    os.symlink(outside, skill / "linked.md")
    errors = " ".join(lint_skill_root(root).errors)
    assert "binary content" in errors
    assert "symlinks are not allowed" in errors


def test_tampered_snapshot_content_fails_verification(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _skill(source)
    snapshot = create_skill_snapshot(
        source, run_root=tmp_path / "run", skill_set_id="set"
    )
    (snapshot.skills_root / "room-sweep/SKILL.md").write_text(
        "tampered", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="does not match manifest"):
        load_skill_snapshot(snapshot.snapshot_root)


def test_tampered_manifest_hash_fails_verification(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _skill(source)
    snapshot = create_skill_snapshot(
        source, run_root=tmp_path / "run", skill_set_id="set"
    )
    manifest = dict(snapshot.manifest)
    manifest["snapshot_sha256"] = "0" * 64
    snapshot.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA256"):
        load_skill_snapshot(snapshot.snapshot_root)


def test_existing_snapshot_is_not_overwritten(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _skill(source)
    create_skill_snapshot(source, run_root=tmp_path / "run", skill_set_id="set")
    with pytest.raises(FileExistsError):
        create_skill_snapshot(source, run_root=tmp_path / "run", skill_set_id="set")


def test_selected_skill_installation_excludes_other_policies(tmp_path):
    from supernav.runtime.skill_runtime import create_skill_snapshot, copy_snapshot_skills

    source = tmp_path / "source"
    for name in ("chosen", "unrelated"):
        path = source / name
        path.mkdir(parents=True)
        (path / "SKILL.md").write_text(
            f"---\nname: {name}\ndescription: Test navigation skill.\n---\nInstructions.\n"
        )
    snapshot = create_skill_snapshot(
        source, run_root=tmp_path / "run", skill_set_id="test"
    )
    target = tmp_path / "installed"
    copy_snapshot_skills(snapshot.snapshot_root, target, names=["chosen"])
    assert [p.name for p in target.iterdir()] == ["chosen"]
