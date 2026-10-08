"""Canonical package imports, assets and command boundaries."""
from __future__ import annotations

import ast
import importlib.util
from importlib.metadata import distribution
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_canonical_package_import_boundaries(tmp_path):
    code = '''
import importlib, importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('habitat_agent', 'harness', 'pluggable_harness', 'analytics', 'habitat_sim'):
            raise AssertionError('unexpected dependency: '+fullname)
sys.meta_path.insert(0, Block())
from supernav.runtime import agents, support
from supernav.experiments import episode
from supernav.methods.navigation.tools.base import ToolRegistry
from supernav.backends.habitat import bridge_client
from habitat_contract.task_schema import TaskCase
assert ToolRegistry.get('depth_analyze') is not None
assert ToolRegistry.get('local_navigate') is not None
assert not any(type(f).__module__.startswith('supernav.compat') for f in sys.meta_path)
'''
    result = subprocess.run([sys.executable, '-c', code], cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(ROOT/'src')}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


def test_unsupported_package_names_are_not_importable():
    for name in ('habitat_agent', 'harness', 'pluggable_harness', 'supernav.compat',
                 'supernav.runtime.episode', 'supernav.runtime.experiment',
                 'supernav.runtime.nav_agent', 'supernav.runtime.pluggable', 'supernav.evaluation.nav_agent',
                 'supernav.web.tui', 'supernav.backends.habitat.launcher',
                 'supernav.backends.habitat.bridge_process'):
        assert importlib.util.find_spec(name) is None, name


def test_task_contract_excludes_to_task_record():
    from habitat_contract import task_schema

    assert not hasattr(task_schema.TaskCase, 'to_task_record')
    assert not hasattr(task_schema, 'OracleUnavailableError')


def test_distribution_publishes_only_the_supernav_command():
    commands = {entry.name for entry in distribution('supernav').entry_points
                if entry.group == 'console_scripts'}
    assert commands == {'supernav'}


def test_unsupported_scripts_and_recipe_names_are_absent():
    for path in (
        'tools/mcp_server.py', 'tools/nav_agent.py', 'tools/bridge_server.py',
        'bench/tool/run.py', 'bench/tool/run_one.py', 'bench/tool/build_prompt.py',
        'bench/tool/score_objectnav_hm3d.py', 'bench/tool/score_multi_objectnav.py',
        'configs/experiments/ai2thor.json', 'configs/experiments/habitat-ovon.json',
    ):
        assert not (ROOT / path).exists(), path


def test_bench_directory_and_symlink_are_absent():
    assert not (ROOT / 'bench').exists()
    assert not (ROOT / 'bench').is_symlink()


def test_canonical_sources_have_no_bench_path_defaults():
    violations = []
    for package in ('supernav', 'habitat_contract'):
        for path in (ROOT / 'src' / package).rglob('*.py'):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
            for node in ast.walk(tree):
                if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                    continue
                parent = parents.get(node)
                path_segment = node.value == 'bench' and (
                    isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Div)
                    or isinstance(parent, ast.Call) and (
                        isinstance(parent.func, ast.Name) and parent.func.id == 'Path'
                        or isinstance(parent.func, ast.Attribute) and parent.func.attr in ('join', 'joinpath')
                    )
                )
                if path_segment or re.search(r'(?<![\w-])bench[/\\]', node.value):
                    violations.append(f'{path.relative_to(ROOT)}:{node.lineno}')
    assert not violations, 'Unsupported bench paths in canonical sources: ' + ', '.join(violations)


@pytest.mark.parametrize('module', [
    'supernav.backends.habitat.stdio_adapter',
    'supernav.backends.habitat.http_server',
    'supernav.methods.localnav.server',
])
def test_canonical_module_help_outside_checkout(module, tmp_path):
    result = subprocess.run([sys.executable, '-m', module, '--help'], cwd=tmp_path,
        env={**os.environ, 'PYTHONPATH': str(ROOT/'src')}, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout.lower()
    assert 'RuntimeWarning' not in result.stderr
