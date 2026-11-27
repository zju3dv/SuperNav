"""Maintainer tool: update the explicit overlay and patch hash manifest."""

import hashlib
import json
from pathlib import Path


AUDITED_REPLACEMENTS = {
    "dimos/agents/mcp/mcp_adapter.py",
    "dimos/agents/skills/test_unitree_skill_container.py",
    "dimos/core/coordination/module_coordinator.py",
    "dimos/mapping/relocalization/module.py",
    "dimos/mapping/relocalization/relocalize.py",
    "dimos/navigation/replanning_a_star/controllers.py",
    "dimos/navigation/replanning_a_star/global_planner.py",
    "dimos/navigation/replanning_a_star/position_tracker.py",
    "dimos/robot/cli/dimos.py",
    "dimos/robot/unitree/connection.py",
    "dimos/robot/unitree/go2/blueprints/smart/unitree_go2.py",
    "dimos/robot/unitree/test_connection.py",
    "dimos/robot/unitree/type/odometry.py",
    "dimos/robot/unitree/type/test_odometry.py",
    "dimos/robot/unitree/unitree_skill_container.py",
}


def main() -> None:
    root = Path(__file__).resolve().parent
    path = root / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["overlay"] = [
        {"path": relative,
         "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
         "executable": bool(p.stat().st_mode & 0o111),
         **({"replace": True} if relative in AUDITED_REPLACEMENTS else {})}
        for p in sorted((root / "overlay").rglob("*"))
        for relative in [p.relative_to(root / "overlay").as_posix()]
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
