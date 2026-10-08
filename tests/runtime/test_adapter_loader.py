from __future__ import annotations

import sys
from pathlib import Path

from supernav.backends.habitat.loader import _redirect_editable_habitat_sources


class _FakeEditableFinder:
    def __init__(self, source_root: Path) -> None:
        self.rebuild_flag = True
        self.known_source_files = {
            "habitat_sim": str(source_root / "__init__.py"),
            "habitat_sim.utils.settings": str(
                source_root / "utils" / "settings.py"
            ),
            "other_package": str(source_root.parent / "other_package.py"),
        }
        self.known_wheel_files = {
            "habitat_sim._ext.habitat_sim_bindings": "habitat_sim_bindings.so"
        }
        self.submodule_search_locations = {
            "habitat_sim": {str(source_root)},
            "habitat_sim.utils": {
                str(source_root / "utils")
            },
        }


def test_adapter_loader_retargets_editable_source_finder_to_worktree(
    monkeypatch, tmp_path
) -> None:
    installed_root = tmp_path / "installed" / "src_python" / "habitat_sim"
    local_root = tmp_path / "worktree" / "src_python" / "habitat_sim"
    for root in (installed_root, local_root):
        (root / "utils").mkdir(parents=True)
        (root / "__init__.py").write_text("", encoding="utf-8")
        (root / "utils" / "settings.py").write_text(
            "", encoding="utf-8"
        )
    finder = _FakeEditableFinder(installed_root)
    monkeypatch.setattr(sys, "meta_path", [finder])

    _redirect_editable_habitat_sources(local_root)

    assert finder.known_source_files["habitat_sim"] == str(local_root / "__init__.py")
    assert finder.known_source_files["habitat_sim.utils.settings"] == str(
        local_root / "utils" / "settings.py"
    )
    assert finder.known_source_files["other_package"].endswith("other_package.py")
    assert finder.known_wheel_files["habitat_sim._ext.habitat_sim_bindings"] == (
        "habitat_sim_bindings.so"
    )
    assert finder.submodule_search_locations["habitat_sim"] == {str(local_root)}
    assert finder.submodule_search_locations[
        "habitat_sim.utils"
    ] == {str(local_root / "utils")}

    assert finder.rebuild_flag is False


def test_checkout_validation_requires_only_public_sdk(monkeypatch, tmp_path):
    from supernav.backends.habitat.config import habitat_root
    package = tmp_path / "src_python" / "habitat_sim"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    monkeypatch.setenv("SUPERNAV_HABITAT_ROOT", str(tmp_path))
    assert habitat_root() == tmp_path


def test_explicit_sdk_does_not_expose_sibling_packages(monkeypatch, tmp_path):
    from supernav.backends.habitat.loader import _import_sdk_package
    package = tmp_path / "src_python" / "habitat_sim"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("SDK_SENTINEL = True\n")
    monkeypatch.delitem(sys.modules, "habitat_sim", raising=False)
    monkeypatch.setattr(sys, "path", list(sys.path))
    # Ensure monkeypatch removes the test import at teardown as well.
    monkeypatch.setitem(sys.modules, "habitat_sim", None)
    _import_sdk_package(package)
    assert sys.modules["habitat_sim"].SDK_SENTINEL
    assert str(package.parent) not in sys.path


def test_native_import_failure_is_not_masked(monkeypatch, tmp_path):
    import pytest
    from supernav.backends.habitat.loader import _import_sdk_package
    package = tmp_path / "habitat_sim"
    package.mkdir()
    (package / "__init__.py").write_text("raise ImportError('native bindings missing')\n")
    monkeypatch.setitem(sys.modules, "habitat_sim", None)
    with pytest.raises(ImportError, match="native bindings missing"):
        _import_sdk_package(package)
    assert "habitat_sim" not in sys.modules


def test_installed_sdk_needs_no_checkout(monkeypatch):
    from supernav.backends.habitat import loader
    monkeypatch.delenv("SUPERNAV_HABITAT_ROOT", raising=False)
    calls = []
    monkeypatch.setattr(loader.importlib, "import_module", calls.append)
    loader.prepare_simulator_import()
    assert calls == ["habitat_sim"]


def test_external_editable_sdk_cannot_shadow_owned_contract(monkeypatch, tmp_path):
    from supernav.backends.habitat.loader import prepare_contract_import

    finder = _FakeEditableFinder(tmp_path / "external" / "habitat_sim")
    finder.known_source_files["habitat_contract"] = "/external/habitat_contract/__init__.py"
    finder.known_source_files["habitat_contract.task_types"] = "/external/habitat_contract/task_types.py"
    finder.submodule_search_locations["habitat_contract"] = {"/external/habitat_contract"}
    monkeypatch.setattr(sys, "meta_path", [finder])
    prepare_contract_import()
    assert "habitat_sim" in finder.known_source_files
    assert "other_package" in finder.known_source_files
    assert not any(name.startswith("habitat_contract") for name in finder.known_source_files)
    assert "habitat_contract" not in finder.submodule_search_locations


def test_failed_sdk_import_removes_partial_children(monkeypatch, tmp_path):
    import pytest
    from supernav.backends.habitat.loader import _import_sdk_package

    package = tmp_path / "habitat_sim"
    package.mkdir()
    (package / "child.py").write_text("VALUE = 'partial'\n")
    (package / "__init__.py").write_text("from . import child\nraise ImportError('failed SDK')\n")
    monkeypatch.setitem(sys.modules, "habitat_sim", None)
    with pytest.raises(ImportError, match="failed SDK"):
        _import_sdk_package(package)
    assert "habitat_sim" not in sys.modules
    assert "habitat_sim.child" not in sys.modules


def test_source_selection_removes_old_only_python_modules(monkeypatch, tmp_path):
    installed = tmp_path / "installed" / "habitat_sim"
    public = tmp_path / "public" / "habitat_sim"
    public.mkdir(parents=True)
    (public / "__init__.py").write_text("")
    finder = _FakeEditableFinder(installed)
    monkeypatch.setattr(sys, "meta_path", [finder])
    _redirect_editable_habitat_sources(public)
    assert "habitat_sim.utils.settings" not in finder.known_source_files
    assert finder.submodule_search_locations["habitat_sim.utils"] == set()
    assert "habitat_sim._ext.habitat_sim_bindings" in finder.known_wheel_files


def test_source_selection_rejects_already_imported_different_sdk(monkeypatch, tmp_path):
    import pytest
    import types
    from supernav.backends.habitat.loader import _import_sdk_package

    existing = types.ModuleType("habitat_sim")
    existing.__file__ = str(tmp_path / "old" / "__init__.py")
    monkeypatch.setitem(sys.modules, "habitat_sim", existing)
    with pytest.raises(RuntimeError, match="fresh process"):
        _import_sdk_package(tmp_path / "public")
    assert sys.modules["habitat_sim"] is existing
