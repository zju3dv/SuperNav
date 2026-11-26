"""Assemble a public DimOS checkout plus SuperNav into an isolated runtime tree."""

import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile

from supernav.registry import regenerate


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "supernav/manifest.json"
RECEIPT = ".supernav-build.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_relative(value: str) -> Path:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"invalid relative path: {value}")
    return Path(*path.parts)


def verify_inputs(root: Path = ROOT) -> tuple[dict, str]:
    manifest_path = root / "supernav/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["schema_version"] != 1:
        raise ValueError("unsupported manifest version")
    upstream = root / "dimos"
    actual = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if actual != manifest["upstream_commit"]:
        raise ValueError("wrong public DimOS commit; run git submodule update --init")
    dirty = subprocess.check_output(["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=normal"], text=True)
    if dirty.strip():
        raise ValueError("public DimOS submodule must be clean")
    for entry in manifest["overlay"]:
        path = root / "supernav/overlay" / safe_relative(entry["path"])
        if digest(path) != entry["sha256"]:
            raise ValueError(f"overlay hash mismatch: {entry['path']}")
    for entry in manifest["patches"]:
        path = root / "supernav" / safe_relative(entry["path"])
        if digest(path) != entry["sha256"]:
            raise ValueError(f"patch hash mismatch: {entry['path']}")
    expected = {e["path"] for e in manifest["overlay"]}
    actual_files = {p.relative_to(root / "supernav/overlay").as_posix() for p in (root / "supernav/overlay").rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    if expected != actual_files:
        raise ValueError("overlay file list differs from manifest")
    fingerprint = hashlib.sha256(manifest_path.read_bytes())
    for name in ["prepare.py", "registry.py"]:
        fingerprint.update((root / "supernav" / name).read_bytes())
    return manifest, fingerprint.hexdigest()


def verify_runtime(root: Path = ROOT) -> Path:
    _, fingerprint = verify_inputs(root)
    runtime = root / "runtime/dimos"
    receipt = json.loads((runtime / RECEIPT).read_text())
    if receipt["fingerprint"] != fingerprint:
        raise ValueError("runtime is outdated; stop the stack and run prepare --rebuild")
    for name, expected in receipt["source_hashes"].items():
        if digest(runtime / safe_relative(name)) != expected:
            raise ValueError(f"runtime source modified: {name}; review before rebuilding")
    return runtime


def prepare(root: Path = ROOT, rebuild: bool = False) -> Path:
    manifest, fingerprint = verify_inputs(root)
    runtime_parent = root / "runtime"
    runtime_parent.mkdir(exist_ok=True)
    output = runtime_parent / "dimos"
    with (runtime_parent / "stack.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("stop the running stack before preparing its sources") from exc
        if output.exists() and not rebuild:
            return verify_runtime(root)
        if output.exists() and not (output / RECEIPT).is_file():
            raise ValueError("refusing to replace an unmanaged runtime directory")
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=runtime_parent) as temporary:
            stage = Path(temporary) / "dimos"
            stage.mkdir()
            archive = subprocess.check_output(
                ["git", "-C", str(root / "dimos"),
                 "-c", "filter.lfs.process=", "-c", "filter.lfs.smudge=",
                 "-c", "filter.lfs.required=false", "archive", manifest["upstream_commit"]],
                env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"},
            )
            with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
                tar.extractall(stage, filter="data")
            # Prevent git apply from discovering the enclosing release repository.
            env = {**os.environ, "GIT_CEILING_DIRECTORIES": str(stage.parent)}
            for entry in manifest["patches"]:
                patch = root / "supernav" / safe_relative(entry["path"])
                subprocess.run(["git", "apply", "--check", str(patch)], cwd=stage, env=env, check=True)
                subprocess.run(["git", "apply", str(patch)], cwd=stage, env=env, check=True)
            for entry in manifest["overlay"]:
                relative = safe_relative(entry["path"])
                destination = stage / relative
                if destination.exists():
                    raise ValueError(f"overlay collides with upstream file: {relative}; use a patch")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / "supernav/overlay" / relative, destination)
                destination.chmod(0o755 if entry["executable"] else 0o644)
            regenerate(stage)
            source_hashes = {p.relative_to(stage).as_posix(): digest(p) for p in sorted(stage.rglob("*")) if p.is_file()}
            receipt = {"fingerprint": fingerprint, "upstream_commit": manifest["upstream_commit"], "source_hashes": source_hashes}
            (stage / RECEIPT).write_text(json.dumps(receipt, indent=2) + "\n")
            backup = runtime_parent / ".dimos-previous"
            if backup.exists():
                raise ValueError("previous build backup exists; inspect runtime/.dimos-previous first")
            if output.exists():
                output.rename(backup)
            try:
                stage.rename(output)
            except BaseException:
                if backup.exists():
                    backup.rename(output)
                raise
            if backup.exists():
                shutil.rmtree(backup)
    return verify_runtime(root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify an existing runtime without rebuilding")
    parser.add_argument("--rebuild", action="store_true", help="replace a previously generated runtime while stopped")
    parser.add_argument("--inputs-only", action="store_true", help="verify public source and integration inputs")
    args = parser.parse_args()
    try:
        if sum([args.check, args.rebuild, args.inputs_only]) > 1:
            parser.error("choose only one mode")
        if args.inputs_only:
            manifest, _ = verify_inputs()
            print(f"Public DimOS {manifest['upstream_commit']}: integration inputs verified")
        else:
            result = verify_runtime() if args.check else prepare(rebuild=args.rebuild)
            print(f"SuperNav runtime ready: {result}")
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"supernav: {exc}\n")


if __name__ == "__main__":
    main()
