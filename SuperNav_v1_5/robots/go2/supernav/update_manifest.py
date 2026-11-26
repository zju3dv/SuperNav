"""Maintainer tool: update the explicit overlay and patch hash manifest."""

import hashlib
import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["overlay"] = [
        {"path": p.relative_to(root / "overlay").as_posix(),
         "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
         "executable": bool(p.stat().st_mode & 0o111)}
        for p in sorted((root / "overlay").rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    ]
    manifest["patches"] = [
        {"path": p.relative_to(root).as_posix(), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in sorted((root / "patches").glob("*.patch"))
    ]
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Updated manifest; review its diff before preparing the runtime.")


if __name__ == "__main__":
    main()
