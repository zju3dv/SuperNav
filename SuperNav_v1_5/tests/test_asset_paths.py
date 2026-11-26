"""Asset lookup uses explicit paths and workspace precedence."""
from pathlib import Path

import pytest

from supernav.paths import resolve_asset_path
from supernav.runtime.config import resolve_path


@pytest.mark.parametrize('old,current', [
    ('bench/config/main/config.json', 'configs/benchmarks/main/config.json'),
    ('configs/pluggable_providers.toml', 'configs/providers/pluggable_providers.toml.example'),
])
def test_unavailable_asset_names_are_not_redirected(tmp_path, monkeypatch, old, current):
    assets, workspace = tmp_path/'installed-assets', tmp_path/'work'
    target=assets/current
    target.parent.mkdir(parents=True)
    target.write_text('preserved content')
    workspace.mkdir()
    monkeypatch.setenv('SUPERNAV_ASSET_ROOT', str(assets))
    assert resolve_asset_path(old, base=workspace) == workspace/old
    assert not resolve_asset_path(old, base=workspace).exists()
    assert resolve_asset_path(assets/old, base=workspace) == assets/old
    assert resolve_asset_path(current, base=workspace) == target


@pytest.mark.parametrize('unavailable', ['migration/source.json', 'docs/migration/source.json'])
def test_unavailable_archive_paths_are_not_redirected(tmp_path, monkeypatch, unavailable):
    assets, workspace = tmp_path / 'assets', tmp_path / 'work'
    monkeypatch.setenv('SUPERNAV_ASSET_ROOT', str(assets))
    resolved = resolve_asset_path(unavailable, base=workspace)
    assert resolved == workspace / unavailable
    assert not resolved.exists()
    assert not (assets / unavailable).exists()


def test_existing_user_asset_and_workspace_override_take_precedence(tmp_path, monkeypatch):
    assets, workspace = tmp_path/'installed', tmp_path/'work'
    relative=Path('configs/tasks/custom.json')
    for root in (assets,workspace):
        path=root/relative
        path.parent.mkdir(parents=True)
        path.write_text(str(root))
    monkeypatch.setenv('SUPERNAV_ASSET_ROOT', str(assets))
    assert resolve_asset_path(relative,base=workspace) == workspace/relative
    assert resolve_asset_path(workspace/relative,base=workspace) == workspace/relative


def test_unrelated_absolute_and_output_paths_are_not_remapped(tmp_path, monkeypatch):
    monkeypatch.setenv('SUPERNAV_ASSET_ROOT',str(tmp_path/'assets'))
    foreign=tmp_path/'another-project/bench/config/main/config.json'
    assert resolve_asset_path(foreign,base=tmp_path/'workspace') == foreign
    assert resolve_path('bench/runs/main',base=tmp_path) == tmp_path/'bench/runs/main'


def test_output_paths_are_resolved_as_requested(tmp_path):
    canonical = tmp_path / 'data/runs/main'
    canonical.mkdir(parents=True)
    assert resolve_path('data/runs/main', base=tmp_path) == canonical
    unavailable = resolve_path('bench/runs/main', base=tmp_path)
    assert unavailable == tmp_path / 'bench/runs/main'
    assert not unavailable.exists()
