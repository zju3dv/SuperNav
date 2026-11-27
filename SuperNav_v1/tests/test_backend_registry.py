"""The registry routes a backend without touching simulator SDKs."""
import sys
from types import SimpleNamespace

import pytest

from supernav.backends import BACKENDS, get_backend
from supernav.cli import mcp
from supernav.runtime.mcp import MCPServerSpec, resolve_mcp_spec


def test_registered_backend_routes_factory_cli_and_mcp(monkeypatch, tmp_path):
    calls = []
    spec = MCPServerSpec('test', 'server', (), {})
    monkeypatch.setitem(sys.modules, 'registry_test_backend', SimpleNamespace(
        factory=lambda: calls.append('factory') or 'instance',
        main=lambda: calls.append('mcp'),
        build=lambda **kwargs: spec,
    ))
    monkeypatch.setitem(BACKENDS, 'test', {
        'implementation': 'registry_test_backend:factory',
        'mcp_main': 'registry_test_backend:main',
        'mcp_spec': 'registry_test_backend:build',
    })
    assert get_backend({'environment': {'backend': 'test'}}) == 'instance'
    monkeypatch.setattr(sys, 'argv', ['supernav mcp', '--backend', 'test'])
    mcp()
    assert resolve_mcp_spec(workspace_root=tmp_path,
        config={'environment': {'backend': 'test'}}, arm_cfg={}) == spec
    assert calls == ['factory', 'mcp']


def test_unknown_backend_is_rejected_in_factory_and_mcp(tmp_path):
    config = {'environment': {'backend': 'missing'}}
    with pytest.raises(ValueError, match='Unknown environment backend'):
        get_backend(config)
    with pytest.raises(ValueError, match='Unknown environment backend'):
        resolve_mcp_spec(workspace_root=tmp_path, config=config, arm_cfg={})
