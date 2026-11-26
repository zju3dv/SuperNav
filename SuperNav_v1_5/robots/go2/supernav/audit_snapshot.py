"""Compare an assembled release with a materialized, approved DimOS snapshot.

Unlike the packaging manifest, this checks source provenance, including shared
public dependencies that the overlay did not replace. Neither tree is modified.
"""

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REQUIRED_SOURCE_PATHS = [
    "dimos/agents/mcp/mcp_adapter.py",
    "dimos/agents/mcp/external_harness_authority_proxy.py",
    "dimos/agents/mcp/external_harness_codex.py",
    "dimos/agents/harness/go2_graph_memory.py",
    "dimos/agents/harness/go2_prior_run.py",
    "dimos/mapping/occupancy/agent_map.py",
    "dimos/mapping/occupancy/habitat_area_segmentation.py",
    "dimos/mapping/occupancy/room_segmentation.py",
    "dimos/mapping/relocalization/module.py",
    "dimos/mapping/relocalization/relocalize.py",
    "dimos/mapping/voxels.py",
    "dimos/core/coordination/module_coordinator.py",
    "dimos/navigation/replanning_a_star/global_planner.py",
    "dimos/navigation/replanning_a_star/position_tracker.py",
    "dimos/navigation/replanning_a_star/min_cost_astar.py",
    "dimos/navigation/replanning_a_star/min_cost_astar_cpp.cpp",
    "dimos/robot/unitree/unitree_skill_container.py",
]

REQUIRED_SKILL_PATHS = [
    "skills/go2-2d-slam/SKILL.md",
    "skills/go2-graph-memory-hierarchy/SKILL.md",
    "skills/go2-graph-memory-hierarchy/references/node-schema.md",
    "skills/go2-graph-memory-hierarchy/references/examples.md",
    "skills/go2-graph-memory-hierarchy/references/workflows.md",
    "skills/go2-graph-memory-hierarchy/references/tool-interface.md",
    "skills/go2-graph-slam-navigation/SKILL.md",
    "skills/go2-graph-slam-navigation/references/route-fusion.md",
]


class WithoutDocstrings(ast.NodeTransformer):
    def generic_visit(self, node):
        node = super().generic_visit(node)
        body = getattr(node, "body", None)
        if isinstance(body, list) and body and isinstance(body[0], ast.Expr):
            value = body[0].value
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                node.body = body[1:]
        return node


def structure(text):
    return ast.dump(
        WithoutDocstrings().visit(ast.parse(text)), include_attributes=False
    )


def definition(text, name):
    tree = ast.parse(text)
    for node in tree.body:
        if getattr(node, "name", None) == name:
            return ast.dump(WithoutDocstrings().visit(node), include_attributes=False)
    raise ValueError(f"missing definition: {name}")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def audit(snapshot, runtime):
    checks = []
    failures = []

    def compare(label, expected, actual, adaptation=None):
        equal = expected == actual
        checks.append(
            {
                "path": label,
                "equal": equal,
                "adaptation": adaptation,
                "expected_sha256": digest(
                    expected if isinstance(expected, bytes) else expected.encode()
                ),
                "actual_sha256": digest(
                    actual if isinstance(actual, bytes) else actual.encode()
                ),
            }
        )
        if not equal:
            failures.append(label)

    # Inspect shared production modules and their runtime assets, not just
    # files in the overlay. Test payloads are not execution configuration.
    for source in sorted((snapshot / "dimos").rglob("*")):
        rel = source.relative_to(snapshot)
        if not source.is_file() or source.suffix not in {
            ".py",
            ".cpp",
            ".h",
            ".json",
            ".toml",
            ".yaml",
            ".yml",
            ".sh",
            ".lua",
            ".md",
        }:
            continue
        if any(
            part.startswith("test")
            or part in {"conftest.py", "fixtures", "__pycache__"}
            for part in rel.parts
        ):
            continue
        target = runtime / rel
        if not target.is_file() or str(rel) == "dimos/robot/all_blueprints.py":
            continue
        expected, actual = source.read_bytes(), target.read_bytes()
        if str(rel) == "dimos/robot/unitree/go2/blueprints/basic/unitree_go2_basic.py":
            compare(
                str(rel),
                expected.decode().replace(
                    "dimos.robot.unitree.nomad.pixel_goal_map",
                    "dimos.robot.unitree.pixelgoal.pixel_goal_map",
                ),
                actual.decode(),
                "package rename",
            )
        elif str(rel) == "dimos/robot/unitree/go2/connection.py":
            compare(
                str(rel),
                structure(expected.decode()),
                structure(actual.decode()),
                "comments only",
            )
        else:
            compare(str(rel), expected, actual)

    for rel in REQUIRED_SOURCE_PATHS:
        if not (snapshot / rel).is_file():
            failures.append(f"missing approved source: {rel}")
        if not (runtime / rel).is_file():
            failures.append(f"missing required source: {rel}")

    for rel in REQUIRED_SKILL_PATHS:
        source, target = snapshot / rel, runtime / rel
        if not source.is_file() or not target.is_file():
            failures.append(f"missing canonical skill: {rel}")
        else:
            compare(rel, source.read_bytes(), target.read_bytes())

    source_root = snapshot / "dimos/robot/unitree/nomad"
    target_root = runtime / "dimos/robot/unitree/pixelgoal"
    for name in [
        "goal.py",
        "pixel_goal_map.py",
        "pixel_goal_map_module.py",
        "area_captioning.py",
        "live_state.py",
    ]:
        source, target = source_root / name, target_root / name
        expected, actual = source.read_text(), target.read_text()
        if name == "pixel_goal_map_module.py":
            for old in ["nav_module", "pixel_goal", "pixel_goal_module"]:
                expected = expected.replace(
                    f"from .{old} import", "from .pixel_goal_support import"
                )
            expected = expected.replace(
                'Path.home() / "code/dimos-live/runs/operator-console/.active-caption-dir"',
                'Path.home() / ".local/state/dimos/.active-caption-dir"',
            ).replace('Path.home() / "nomad-live"', 'Path.home() / "pixelgoal-live"')
        elif name == "area_captioning.py":
            # The only deployment adaptation is removing the private endpoint.
            lines = expected.splitlines(True)
            expected = "".join(
                'DEFAULT_CAPTION_SERVER_URL = ""\n'
                if line.startswith("DEFAULT_CAPTION_SERVER_URL = ")
                else line
                for line in lines
            )
        elif name == "live_state.py":
            expected = expected.replace('"nomad-live"', '"pixelgoal-live"').replace(
                'environ.get("NOMAD_LIVE_DIR")',
                'environ.get("DIMOS_PIXELGOAL_LIVE_DIR")',
            )
            expected, actual = structure(expected), structure(actual)
        compare(
            f"pixelgoal/{name}",
            expected,
            actual,
            "documented package/deployment adaptation",
        )

    for name in [
        "LocalNavError",
        "LocalNavInputError",
        "NormalizedPoint",
        "GoalSnapshot",
    ]:
        compare(
            f"pixelgoal/contracts.py:{name}",
            definition((source_root / "contracts.py").read_text(), name),
            definition((target_root / "contracts.py").read_text(), name),
            "retained contract",
        )
    support = (target_root / "pixel_goal_support.py").read_text()
    for source_file, names in [
        ("nav_module.py", ["UnitreeMotionSpec", "NavLiveSink"]),
        ("pixel_goal.py", ["PixelGoalHopResult"]),
        ("pixel_goal_module.py", ["NomadStatusSpec"]),
    ]:
        for name in names:
            compare(
                f"pixelgoal/pixel_goal_support.py:{name}",
                definition((source_root / source_file).read_text(), name),
                definition(support, name),
                "extracted support definition; docstrings may differ",
            )

    # The release exposes only the snapshot's map backend. Compare its actual
    # composition too: matching module files is insufficient if launch-time
    # overrides silently disable map persistence or change planner defaults.
    source_blueprint = (
        snapshot
        / "dimos/robot/unitree/go2/blueprints/nomad/unitree_go2_nomad_edge_mcp.py"
    ).read_text()
    target_blueprint = (
        runtime
        / "dimos/robot/unitree/go2/blueprints/pixelgoal/unitree_go2_pixelgoal.py"
    ).read_text()
    for name in [
        "_pixel_target_marker",
        "_pixel_projection_marker",
        "_pixel_stop_marker",
    ]:
        compare(
            f"pixelgoal/blueprint:{name}",
            definition(source_blueprint, name),
            definition(target_blueprint, name),
            "map-backend-only blueprint",
        )

    class MapBackendOnly(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            if (
                isinstance(node.func, ast.Name)
                and node.func.id == "pixelgoal_blueprint"
            ):
                return ast.parse(
                    "PixelGoalMapNavModule.blueprint(camera_info=GO2Connection.camera_info_static)",
                    mode="eval",
                ).body
            if isinstance(node.func, ast.Name) and node.func.id == "autoconnect":
                node.args = [
                    arg
                    for arg in node.args
                    if not (
                        isinstance(arg, ast.Call)
                        and isinstance(arg.func, ast.Attribute)
                        and isinstance(arg.func.value, ast.Name)
                        and arg.func.value.id
                        in {"NomadChunkExecutorModule", "NomadNavModule"}
                    )
                ]
            return node

    def blueprint_value(text, name, adapt=False):
        tree = ast.parse(text)
        node = next(
            node.value
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == name
                for target in node.targets
            )
        )
        if adapt:
            node = MapBackendOnly().visit(node)
        return ast.dump(node, include_attributes=False)

    compare(
        "pixelgoal/blueprint:composition",
        blueprint_value(source_blueprint, "unitree_go2_nomad_edge_mcp", True),
        blueprint_value(target_blueprint, "unitree_go2_pixelgoal"),
        "remove unused policy modules; select map backend",
    )
    return {"checks": checks, "failures": failures, "passed": not failures}


def run_probe(tree, runtime):
    env = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "DIMOS_SKIP_SYSTEM_CONFIG": "1",
        "LCM_DEFAULT_URL": "memq://",
        "USE_TF": "0",
        "USE_FLAX": "0",
    }
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("snapshot_probe.py")),
            "--tree",
            str(tree),
            "--runtime",
            str(runtime),
        ],
        env=env,
        text=True,
        capture_output=True,
        timeout=180,
    )
    if result.returncode:
        raise RuntimeError(
            f"Probe failed for {tree}:\n{result.stdout}\n{result.stderr}"
        )
    return json.loads(
        next(
            line.removeprefix("SNAPSHOT_RESULT=")
            for line in result.stdout.splitlines()
            if line.startswith("SNAPSHOT_RESULT=")
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="approved snapshot plus the manifest's audited backports",
    )
    parser.add_argument(
        "--runtime",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runtime/dimos",
    )
    parser.add_argument(
        "--behavior",
        action="store_true",
        help="also compare deterministic behavioral fixtures",
    )
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = audit(args.snapshot.resolve(), args.runtime.resolve())
    if args.behavior and report["passed"]:
        expected, actual = (
            run_probe(tree.resolve(), args.runtime.resolve())
            for tree in [args.snapshot, args.runtime]
        )
        differences = [
            key
            for key in expected.keys() | actual.keys()
            if expected.get(key) != actual.get(key)
        ]
        report["behavior"] = {
            "differences": differences,
            "snapshot": expected,
            "release": actual,
        }
        report["passed"] = not differences
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "passed": report["passed"],
                "source_checks": len(report["checks"]),
                "failures": report["failures"],
                "behavior_differences": report.get("behavior", {}).get("differences"),
            },
            indent=2,
        )
    )
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
