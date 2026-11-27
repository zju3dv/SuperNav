"""Validated, immutable skill bundles for benchmark runs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except ImportError as exc:  # pragma: no cover - exercised by deployment packaging
    raise RuntimeError(
        "PyYAML is required for benchmark skill frontmatter; "
        "install requirements-agent.txt"
    ) from exc


SKILL_FORMAT = "open-agent-skills-directory-v1"
BUNDLE_POLICY = "auditable-text-v1"
ALLOWED_SUFFIXES = {
    ".js",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}
MAX_FILE_BYTES = 1024 * 1024
MAX_BUNDLE_BYTES = 10 * 1024 * 1024
_NAME_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_SECRET_NAME_RE = re.compile(
    r"(?:^\.env(?:\.|$)|secret|credential|api[_-]?key|private[_-]?key|"
    r"(?:^|[_-])token(?:[_.-]|$))",
    re.IGNORECASE,
)
_SECRET_CONTENT_PATTERNS = (
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[opusr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret[_-]?key)\s*[:=]\s*"
        r"[\"']?[A-Za-z0-9_./+=-]{16,}"
    ),
)


@dataclass(frozen=True)
class SkillEntry:
    name: str
    description: str
    path: Path


@dataclass(frozen=True)
class SkillLintReport:
    root: Path
    ok: bool
    skills: tuple[SkillEntry, ...]
    errors: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "root": str(self.root),
            "ok": self.ok,
            "skills": [
                {
                    "name": skill.name,
                    "description": skill.description,
                    "path": str(skill.path),
                }
                for skill in self.skills
            ],
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class SkillSnapshot:
    snapshot_root: Path
    skills_root: Path
    manifest_path: Path
    manifest: Mapping[str, Any]

    @property
    def snapshot_sha256(self) -> str:
        return str(self.manifest["snapshot_sha256"])


def _name_error(name: str) -> str | None:
    if not name:
        return "name is required"
    if "--" in name:
        return "name must not contain consecutive hyphens"
    if not _NAME_RE.fullmatch(name):
        return "name must use lowercase letters, numbers, and single hyphens"
    return None


def _frontmatter(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None, "SKILL.md must be UTF-8 text"
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        return None, "frontmatter must start with ---"
    try:
        end = lines[1:].index("---") + 1
    except ValueError:
        return None, "frontmatter must end with ---"
    try:
        fields = yaml.safe_load("\n".join(lines[1:end]))
    except yaml.YAMLError as exc:
        return None, f"invalid YAML frontmatter: {exc}"
    if not isinstance(fields, dict):
        return None, "frontmatter must be a YAML mapping"
    if any(not isinstance(key, str) for key in fields):
        return None, "frontmatter keys must be strings"
    return fields, None


def _iter_tree(root: Path) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    errors: list[str] = []
    for current, dirs, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in list(dirs):
            path = current_path / name
            if path.is_symlink():
                errors.append(f"{path.relative_to(root)}: symlinks are not allowed")
                dirs.remove(name)
        for name in names:
            path = current_path / name
            if path.is_symlink():
                errors.append(f"{path.relative_to(root)}: symlinks are not allowed")
            elif path.is_file():
                files.append(path)
    return sorted(files), errors


def _validate_file(path: Path, root: Path) -> tuple[int, list[str]]:
    rel = path.relative_to(root).as_posix()
    errors: list[str] = []
    if path.name != "SKILL.md" and path.suffix.lower() not in ALLOWED_SUFFIXES:
        errors.append(f"{rel}: file type is not allowed")
    if _SECRET_NAME_RE.search(path.name):
        errors.append(f"{rel}: secret-like file name is not allowed")
    size = path.stat().st_size
    if size > MAX_FILE_BYTES:
        errors.append(f"{rel}: exceeds {MAX_FILE_BYTES} byte file limit")
        return size, errors
    data = path.read_bytes()
    if b"\0" in data:
        errors.append(f"{rel}: binary content is not allowed")
        return size, errors
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        errors.append(f"{rel}: files must be UTF-8 text")
        return size, errors
    if any(pattern.search(text) for pattern in _SECRET_CONTENT_PATTERNS):
        errors.append(f"{rel}: possible secret content is not allowed")
    return size, errors


def lint_skill_root(root: Path) -> SkillLintReport:
    """Validate a directory containing one or more directory-form skills."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        return SkillLintReport(
            root, False, (), (f"skills root does not exist: {root}",)
        )
    skills: list[SkillEntry] = []
    errors: list[str] = []
    total_size = 0
    children = sorted(root.iterdir(), key=lambda path: path.name)
    for child in children:
        if child.is_symlink():
            errors.append(f"{child.name}: symlinks are not allowed")
            continue
        if child.name.startswith("."):
            continue
        if not child.is_dir():
            errors.append(f"{child.name}: only directory-form skills are allowed")
            continue
        if message := _name_error(child.name):
            errors.append(f"{child.name}: {message}")
        files, tree_errors = _iter_tree(child)
        errors.extend(f"{child.name}/{message}" for message in tree_errors)
        for path in files:
            size, file_errors = _validate_file(path, root)
            total_size += size
            errors.extend(file_errors)
        skill_md = child / "SKILL.md"
        if not skill_md.is_file() or skill_md.is_symlink():
            errors.append(f"{child.name}: SKILL.md is required")
            continue
        fields, parse_error = _frontmatter(skill_md)
        if parse_error:
            errors.append(f"{child.name}/SKILL.md: {parse_error}")
            continue
        assert fields is not None
        name = fields.get("name")
        description = fields.get("description")
        if name != child.name:
            errors.append(f"{child.name}: frontmatter name must match directory name")
        if not isinstance(description, str) or not description.strip():
            errors.append(f"{child.name}: description must be a non-empty string")
        if isinstance(name, str) and (message := _name_error(name)):
            errors.append(f"{child.name}: {message}")
        if name == child.name and isinstance(description, str) and description.strip():
            skills.append(SkillEntry(name, description.strip(), skill_md))
    if total_size > MAX_BUNDLE_BYTES:
        errors.append(f"skill bundle exceeds {MAX_BUNDLE_BYTES} byte total limit")
    if not skills:
        errors.append("no valid directory-form skills found")
    return SkillLintReport(root, not errors, tuple(skills), tuple(errors))


def _file_records(skills_root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(p for p in skills_root.rglob("*") if p.is_file()):
        data = path.read_bytes()
        records.append(
            {
                "path": path.relative_to(skills_root).as_posix(),
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    return records


def _bundle_hash(records: list[dict[str, Any]]) -> str:
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()


def create_skill_snapshot(
    source_root: Path,
    *,
    run_root: Path,
    skill_set_id: str,
) -> SkillSnapshot:
    """Lint and copy a skill root into a run-local immutable snapshot."""
    report = lint_skill_root(source_root)
    if not report.ok:
        raise ValueError("skill lint failed: " + "; ".join(report.errors))
    snapshot_root = Path(run_root) / "skill_snapshot"
    if snapshot_root.exists():
        raise FileExistsError(f"skill snapshot already exists: {snapshot_root}")
    skills_root = snapshot_root / "skills"
    skills_root.parent.mkdir(parents=True, exist_ok=False)
    skills_root.mkdir()
    # Copy only directories represented in the successful lint report. Root-level
    # metadata and hidden user files are deliberately outside the bundle format.
    for skill in report.skills:
        shutil.copytree(report.root / skill.name, skills_root / skill.name)
    records = _file_records(skills_root)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "format": SKILL_FORMAT,
        "bundle_policy": BUNDLE_POLICY,
        "skill_set_id": skill_set_id,
        "source_root": str(report.root),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "excluded_user_skills": True,
        "skills": [
            {"name": skill.name, "description": skill.description}
            for skill in report.skills
        ],
        "files": records,
        "snapshot_sha256": _bundle_hash(records),
    }
    manifest_path = snapshot_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SkillSnapshot(snapshot_root, skills_root, manifest_path, manifest)


def load_skill_snapshot(snapshot_root: Path) -> SkillSnapshot:
    """Load a snapshot and fail if its manifest or copied files were changed."""
    snapshot_root = Path(snapshot_root).expanduser().resolve()
    manifest_path = snapshot_root / "manifest.json"
    skills_root = snapshot_root / "skills"
    if not manifest_path.is_file() or not skills_root.is_dir():
        raise FileNotFoundError(f"invalid skill snapshot layout: {snapshot_root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported skill snapshot manifest")
    files, symlink_errors = _iter_tree(skills_root)
    if symlink_errors:
        raise ValueError(
            "skill snapshot contains symlinks: " + "; ".join(symlink_errors)
        )
    actual = _file_records(skills_root)
    if actual != manifest.get("files"):
        raise ValueError(
            "skill snapshot file inventory or content does not match manifest"
        )
    if _bundle_hash(actual) != manifest.get("snapshot_sha256"):
        raise ValueError("skill snapshot SHA256 does not match manifest")
    return SkillSnapshot(snapshot_root, skills_root, manifest_path, manifest)


def copy_snapshot_skills(
    snapshot_root: Path, destination_root: Path, *, names: list[str] | None = None
) -> Path:
    """Verify a snapshot before installing its skills into a backend sandbox."""
    snapshot = load_skill_snapshot(snapshot_root)
    if names is not None:
        available = {row["name"] for row in snapshot.manifest["skills"]}
        if not names or not set(names) <= available:
            raise ValueError("selected skills must be present in the snapshot")
    destination_root = Path(destination_root)
    if destination_root.exists():
        shutil.rmtree(destination_root)
    if names is None:
        shutil.copytree(snapshot.skills_root, destination_root)
    else:
        destination_root.mkdir(parents=True)
        for name in names:
            shutil.copytree(snapshot.skills_root / name, destination_root / name)
    return destination_root
