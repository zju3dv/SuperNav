#!/usr/bin/env python3
"""Export the committed HEAD as a checked source archive, without Git history.

Usage: python scripts/export_release_source.py --output .cache/release/supernav-source.tar.gz
Only HEAD is exported; commit reviewed changes before producing a release.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile


PRIVATE_DIRS = {".codex", ".codex_home", ".kimi", ".claude", ".opencode", ".aws", ".ssh"}
PRIVATE_NAMES = {
    "auth.json", "auth.toml", "auth.yaml", "auth.yml", "credentials", "secrets.json", "secrets.toml",
    "secrets.yaml", "secrets.yml", ".claude.json", ".mcp.json", ".codex.toml", ".kimi.toml",
    "opencode.json", "opencode.jsonc", ".netrc", ".npmrc", ".pypirc", "settings.local.json",
}
PRIVATE_ROOTS = ("configs/local", "configs/auth", "data/runs", "data/nav_artifacts", "data/nav_artifacts_controller")
INTERNAL_ROOTS = ("AGENTS.md", "docs/development", "docs/README.md", ".cache", ".venv", "build", "dist")


def is_env_example(name: str) -> bool:
    return PurePosixPath(name).name.endswith(".env.example")


def is_benchmark_data(name: str) -> bool:
    """Exclude task datasets from source and installed distribution paths."""
    path = PurePosixPath(name)
    parts = path.parts
    for root in ("configs/benchmarks", "configs/tasks", "tests/fixtures/test_suite"):
        prefix = tuple(root.split("/"))
        for index in range(len(parts)):
            if parts[index:index + len(prefix)] != prefix:
                continue
            # Directories are also checked as wheel destinations. Keep the
            # documentation and generic prompt resources in those directories.
            if not path.suffix or path.suffix == ".md":
                return False
            if root == "configs/benchmarks" and "prompts" in parts[index + len(prefix):]:
                return path.suffix not in {".yaml", ".yml"}
            return True
    return False


def is_private_path(name: str, *, include_internal: bool = True, include_benchmarks: bool = True) -> bool:
    """Classify relative source or installed paths without reading their contents."""
    parts = PurePosixPath(name).parts
    basename = parts[-1] if parts else ""
    if include_benchmarks and is_benchmark_data(name):
        return True
    if PRIVATE_DIRS.intersection(parts):
        return True
    roots = PRIVATE_ROOTS + (INTERNAL_ROOTS if include_internal else ())
    if any(tuple(root.split("/")) == parts[i:i + len(root.split("/"))]
           for root in roots for i in range(len(parts))):
        return True
    if is_env_example(basename):
        return False
    return (
        basename in PRIVATE_NAMES
        or basename == ".env"
        or basename.startswith((".env.", "credentials.", "id_rsa", "id_ed25519"))
        or basename.endswith((".env", ".pem", ".key"))
        or ".env." in basename
    )


def check_empty_env_example(name: str, content: bytes) -> None:
    """Allow comments and empty assignments; never print a credential value."""
    for number, line in enumerate(content.decode("utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line or line.split("=", 1)[1].strip() not in {"", "''", '""'}:
            raise ValueError(f"Environment example contains a value or invalid assignment: {name}:{number}")


def export_source(repo: Path, output: Path) -> None:
    """Reject tracked secrets, then export and validate HEAD with export-ignore."""
    def git(*args: str) -> bytes:
        return subprocess.check_output(["git", "-C", str(repo), *args])

    if git("status", "--porcelain", "-z", "--untracked-files=normal"):
        raise ValueError("Commit reviewed changes before exporting HEAD; the working tree has uncommitted files")
    commit = git("rev-parse", "HEAD").decode().strip()
    entries = git("ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0")
    for entry in filter(None, entries):
        metadata, raw_name = entry.split(b"\t", 1)
        name = os.fsdecode(raw_name)
        mode = metadata.split(b" ", 1)[0]
        if is_private_path(name, include_internal=False, include_benchmarks=False):
            raise ValueError(f"Sensitive path is tracked in HEAD: {name}")
        if mode != b"100644" and mode != b"100755":
            raise ValueError(f"Source export requires regular files: {name}")
        if is_env_example(name):
            check_empty_env_example(name, git("show", f"{commit}:{name}"))

    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite archive: {output}")
    with tempfile.TemporaryDirectory(prefix="source-export-", dir=output.parent) as directory:
        archive = Path(directory) / "source.tar.gz"
        subprocess.run(["git", "-C", str(repo), "archive", "--format=tar.gz",
                        f"--output={archive}", commit], check=True)
        with tarfile.open(archive) as source:
            for member in source.getmembers():
                if is_private_path(member.name) or member.issym() or member.islnk():
                    raise ValueError(f"Private path or link entered source archive: {member.name}")
        # Create exclusively so a concurrently created output is also preserved.
        with archive.open("rb") as src, output.open("xb") as dst:
            shutil.copyfileobj(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Destination .tar.gz file (must not exist)")
    args = parser.parse_args()
    export_source(Path(__file__).resolve().parents[1], args.output)
    print(f"Exported committed HEAD without Git history: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
