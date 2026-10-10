#!/usr/bin/env python3
"""Smoke-test an installed wheel using only its base dependencies.

Run with the Python interpreter of a clean virtual environment after installing
the wheel. The driver starts an isolated interpreter outside the checkout; no
editable install, simulator SDK, agent CLI, credentials, or scene data is needed.
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.abc
import importlib.util
from importlib.metadata import distribution
import io
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import sysconfig
import tempfile


SOURCE_ROOT = Path(__file__).resolve().parents[1]
SIMULATORS = {"habitat_sim", "habitat", "ai2thor"}


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _worker() -> None:
    _check(not Path.cwd().is_relative_to(SOURCE_ROOT), "Smoke working directory is inside the checkout")
    for name in SIMULATORS:
        _check(importlib.util.find_spec(name) is None, f"Simulator SDK is installed: {name}")

    simulator_imports: list[str] = []

    class NoSimulator(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".", 1)[0] in SIMULATORS:
                simulator_imports.append(fullname)
                raise AssertionError(f"Simulator SDK imported during wheel smoke: {fullname}")
            return None

    sys.meta_path.insert(0, NoSimulator())

    import supernav
    from supernav.experiments.config import list_experiments, load_experiment_config, resolve_experiment
    from supernav.paths import asset_root

    site_roots = {Path(sysconfig.get_path(key)).resolve() for key in ("purelib", "platlib")}

    def installed_origin(module) -> None:
        path = Path(module.__file__).resolve()
        _check(not path.is_relative_to(SOURCE_ROOT), f"Imported checkout module: {path}")
        _check(any(path.is_relative_to(root) for root in site_roots), f"Module is not installed in this environment: {path}")

    installed_origin(supernav)
    assets = asset_root().resolve()
    expected_assets = (Path(sysconfig.get_path("data")) / "share" / "supernav").resolve()
    _check(assets == expected_assets and assets.is_dir(), f"Unexpected installed asset root: {assets}")
    _check(not assets.is_relative_to(SOURCE_ROOT), "Assets came from the checkout")
    release_policy = runpy.run_path(str(Path(__file__).with_name("export_release_source.py")))
    package = distribution("supernav")
    for resource in package.files or ():
        name = resource.as_posix()
        _check(not release_policy["is_private_path"](name), f"Private file entered installed distribution: {name}")
        if release_policy["is_env_example"](name):
            release_policy["check_empty_env_example"](name, Path(package.locate_file(resource)).read_bytes())
    print("Installed distribution contains no private configuration paths; environment examples are empty: OK")
    for old in ("AGENTS.md", "bench", "data/runs", "tools", "config", "migration", "docs/migration", "docs/development", "docs/README.md"):
        _check(not (assets / old).exists(), f"Unsupported asset installed: {old}")
    print(f"Installed package: {Path(supernav.__file__).resolve()}")
    print(f"Installed assets: {assets}")

    entries = {e.name for e in distribution("supernav").entry_points if e.group == "console_scripts"}
    _check(entries == {"supernav"}, f"Unexpected console aliases: {entries}")
    for name in ("harness", "habitat_agent", "pluggable_harness", "analytics", "supernav.compat", "supernav.runtime.episode", "supernav.runtime.experiment", "supernav.runtime.nav_agent", "supernav.runtime.pluggable", "supernav.evaluation.nav_agent", "supernav.web.tui", "supernav.backends.habitat.launcher", "supernav.backends.habitat.bridge_process"):
        _check(importlib.util.find_spec(name) is None, f"Unsupported import is installed: {name}")
    console_script = Path(sysconfig.get_path("scripts")) / "supernav"
    _check(console_script.is_file(), f"Missing installed console script: {console_script}")

    def cli(*arguments: str, module: bool = False) -> str:
        output = io.StringIO()
        previous = sys.argv
        try:
            sys.argv = ["supernav", *arguments]
            with contextlib.redirect_stdout(output):
                try:
                    if module:
                        runpy.run_module("supernav", run_name="__main__")
                    else:
                        runpy.run_path(str(console_script), run_name="__main__")
                except SystemExit as exc:
                    _check(exc.code in (None, 0), f"CLI failed: {arguments!r}, exit={exc.code}")
        finally:
            sys.argv = previous
        return output.getvalue()

    _check("SuperNav" in cli("--help"), "Console help did not render")
    _check("SuperNav" in cli("--help", module=True), "python -m supernav help did not render")
    _check("--experiment" in cli("run", "--help"), "Sweep help is missing named experiments")
    _check("--experiment" in cli("run-one", "--help"), "Episode help is missing named experiments")
    _check("--port" in cli("habitat-bridge", "--help"), "Habitat HTTP help requires a simulator")
    _check("--request-json" in cli("habitat-adapter", "--help"), "Habitat stdio help requires a simulator")
    names = list_experiments()
    _check(set(names) == {"ai2thor-primitive", "habitat-geo-based-executor", "habitat-learned-executor"}, f"Unexpected installed recipes: {names}")
    listed = cli("config", "list")
    for name in names:
        _check(f"{name}\t{resolve_experiment(name)}" in listed, f"CLI did not list {name}")
    for name in names:
        loaded = load_experiment_config(resolve_experiment(name))
        shown = json.loads(cli("config", "show", "--experiment", name))
        _check(shown == loaded and "extends" not in shown, f"Config did not compose: {name}")

    from habitat_contract.task_schema import TaskCase
    _check(not hasattr(TaskCase, "to_task_record"), "TaskCase.to_task_record must not exist")
    from supernav.runtime.config import instruction_rows
    from supernav.paths import resolve_asset_path

    for name in ("habitat_contract", "habitat_contract.navigation", "habitat_contract.navigation_state", "supernav.runtime.config", "supernav.runtime.agents", "supernav.experiments.preparation.ovon_mtu3d", "supernav.evaluation.objectnav.score_hm3d", "supernav.evaluation.objectnav.score_multi"):
        installed_origin(importlib.import_module(name))
    _check((assets / "configs/viewers/rerun.yaml").is_file(), "Viewer configuration is missing")
    documentation = assets / "docs/release"
    guides = {
        "README.md", "configuration.md", "assets-and-skills.md", "habitat.md", "ai2thor.md",
        "web-viewer.md", "architecture.md", "layout.md", "tool-authoring-guide.md", "hm3d-and-ovon.md",
    }
    for language in ("en", "zh"):
        installed_guides = {path.name for path in (documentation / language).glob("*.md")}
        _check(installed_guides == guides, f"Incomplete {language} release documentation: {installed_guides}")
    for resource in ("README.md", "README.zh-CN.md", "configs/README.md", "configs/README.zh-CN.md",
                     "docs/release/assets/supernav-wordmark.svg", "docs/release/assets/SpaceGrotesk-OFL.txt"):
        _check((assets / resource).is_file(), f"Release documentation resource is missing: {resource}")
    _check(not list((assets / "docs").glob("*.md")), "Unclassified documentation entered the distribution")
    print("Bilingual release documentation and SVG assets; no development handoffs: OK")
    for old in ("bench/config/main/config.json", "tools/habitat_agent/prompts", "config/context_budget.yaml"):
        _check(not resolve_asset_path(old, base=Path.cwd()).exists(), f"Unsupported path resolves: {old}")
    habitat_recipes = [name for name in names if name != "ai2thor-primitive"]
    for name in habitat_recipes:
        config = load_experiment_config(resolve_experiment(name))
        _check(config.get("instructions_file") is None, f"Recipe embeds a benchmark manifest: {name}")
    _check(not list((assets / "configs/benchmarks").rglob("*.json")), "Benchmark task data entered the wheel")
    _check(not list((assets / "configs/tasks").glob("*.json")), "Benchmark task indexes entered the wheel")
    fixture_data = [path for path in (assets / "tests/fixtures/test_suite").rglob("*")
                    if path.suffix in {".json", ".jsonl", ".gz", ".png"}]
    _check(not fixture_data, "Real test-suite data entered the wheel")
    print("Canonical imports, three recipes and configuration assets; no bundled benchmark tasks: OK")
    for arguments in (("run", "--dry-run"), ("run-one", "--dry-run"), ("config", "show", "--experiment", "missing-example"), ("config", "show", "--experiment", "unsupported-example")):
        result = subprocess.run([sys.executable, "-I", "-m", "supernav", *arguments], capture_output=True, text=True, timeout=15)
        _check(result.returncode == 2 and "Traceback" not in result.stderr, f"Unsupported invocation accepted: {arguments}")
    print("Command selection validation: OK")

    instruction = "Find the red synthetic test cube."
    result = json.loads(cli(
        "run-one", "--experiment", "habitat-geo-based-executor", "--arm", "default",
        "--slug", "wheel-smoke", "--run-id", "wheel-smoke",
        "--scene", "synthetic_smoke_scene", "--spawn", '{"start_position": [0.0, 0.0, 0.0]}',
        "--instruction", instruction, "--output-dir", str(Path.cwd() / "runs"), "--dry-run",
    ))
    _check(result.get("dry_run") is True, "Episode did not report dry-run mode")
    run_dir = Path(result["run_dir"])
    _check(run_dir.is_relative_to(Path.cwd()), "Dry-run output escaped the temporary workspace")
    _check(instruction in (run_dir / "prompt.txt").read_text(), "Dry-run prompt is missing the instruction")
    run = json.loads((run_dir / "run.json").read_text())
    _check(run["arm"] == "default" and run["backend"] == "habitat", "Wrong dry-run experiment")
    _check(run["skill"]["mode"] == "native" and bool(run["skill"]["snapshot_sha256"]), "Native skill evidence is missing")
    _check(run["skill"]["visibility"]["status"] == "not_executed_dry_run", "Dry-run executed the skill visibility probe")
    project = Path(result["project_dir"])
    for name in ("global-navigation-geo-based-executor", "localnav-pointnav-geo-based-executor"):
        installed_skill = project / ".codex_home" / "skills" / name / "SKILL.md"
        _check(installed_skill.read_bytes() == (assets / "skills" / name / "SKILL.md").read_bytes(),
               f"Native skill content changed: {name}")
    _check((project / ".codex_home" / "config.toml").is_file(), "Codex project configuration is missing")
    _check(not (run_dir / "metrics.json").exists(), "Dry-run unexpectedly produced scored metrics")
    # Generate a tiny external task tree; all task content and coordinates are invented.
    task_root = Path.cwd() / "synthetic-tasks"
    task_root.mkdir()
    (task_root / "leaf.json").write_text(json.dumps({"instructions": [{
        "task_id": "synthetic-red-cube", "slug": "synthetic-red-cube", "text": instruction,
    }, {"task_id": "synthetic-blue-sphere", "slug": "synthetic-blue-sphere",
        "text": "Find the blue synthetic test sphere."}]}))
    (task_root / "middle.json").write_text(json.dumps({"includes": [{
        "path": "leaf.json", "defaults": {"scene": "synthetic_smoke_scene"},
    }]}))
    index = task_root / "index.json"
    index.write_text(json.dumps({"includes": [{"path": "middle.json", "defaults": {
        "spawn": {"start_position": [0.0, 0.0, 0.0]},
    }}]}))
    _check(len(instruction_rows(index)) == 2, "Synthetic nested task index did not expand")
    for name in habitat_recipes:
        config = load_experiment_config(resolve_experiment(name))
        arm = "default"
        output = Path.cwd() / "sweep" / name
        cli("run", "--experiment", name, "--arms", arm, "--instructions", str(index),
            "--task-ids", "synthetic-red-cube", "--output-dir", str(output), "--dry-run")
        runs = list(output.rglob("run.json"))
        _check(len(runs) == 1, f"Synthetic task selection failed: {name}")
        recorded = json.loads(runs[0].read_text())
        _check(recorded["task_id"] == "synthetic-red-cube" and recorded["scene"] == "synthetic_smoke_scene",
               f"External task overrides did not reach the episode: {name}")
        _check(len(list((output / "_sweeps").glob("*/instructions.resolved.json"))) == 1,
               f"Sweep did not freeze its selected, expanded task rows: {name}")
    _check(not simulator_imports, f"Simulator imports were attempted: {simulator_imports}")
    print("Synthetic external task dry-runs across two Habitat recipes, prompts and native skills: OK")
    print("No simulator SDKs installed or imported: OK")


def main() -> int:
    if sys.argv[1:] == ["--worker"]:
        _worker()
        return 0
    if sys.argv[1:]:
        raise SystemExit("Usage: <installed-venv>/bin/python scripts/check_installed_package.py")
    with tempfile.TemporaryDirectory(prefix="supernav-wheel-smoke-") as directory:
        root = Path(directory)
        codex_home = root / "empty_codex_home"
        codex_home.mkdir()
        env = {
            key: value for key, value in os.environ.items()
            if not key.startswith(("PYTHON", "SUPERNAV_", "NAV_", "HAB_", "HABITAT_", "AI2THOR_", "CODEX_"))
        }
        env["CODEX_HOME"] = str(codex_home)
        completed = subprocess.run(
            [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
            cwd=root, env=env, timeout=60, check=False,
        )
        return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
