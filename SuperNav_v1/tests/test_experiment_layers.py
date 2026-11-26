"""Canonical orchestration runs without simulator SDKs."""

import importlib
import subprocess
import sys

import pytest


@pytest.mark.parametrize("runner", ["sweep", "episode"])
def test_runner_module_entry_points_preserve_cli(runner, tmp_path):
    expected = "Run a portable benchmark sweep." if runner == "sweep" else "Run one portable benchmark episode."
    for module in [f"supernav.experiments.{runner}"]:
        result = subprocess.run(
            [sys.executable, "-m", module, "--help"],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=20,
        )
        assert result.returncode == 0, result.stderr
        assert expected in result.stdout
        assert "--dry-run" in result.stdout


def test_orchestration_imports_without_simulator_sdks(tmp_path):
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('habitat_sim', 'ai2thor'):
            raise AssertionError('Unexpected simulator SDK import: ' + fullname)
sys.meta_path.insert(0, Block())
from supernav.experiments import sweep, episode
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
