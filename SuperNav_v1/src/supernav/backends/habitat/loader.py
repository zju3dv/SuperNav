"""Bind SuperNav to an explicitly configured external Habitat SDK checkout.

The external checkout stays read-only, including editable build hooks.
"""
from __future__ import annotations

import importlib
import importlib.machinery
import importlib.metadata
import sys
import importlib.util
from pathlib import Path


def prepare_contract_import() -> None:
    """Prevent an external editable SDK from claiming SuperNav's protocol package.

    Some Habitat installations also registered application sources. They must
    not override the ``habitat_contract`` shipped beside SuperNav. This only
    removes the foreign finder's claims; ordinary import resolution then finds
    the installed canonical package.
    """
    existing = sys.modules.get("habitat_contract")
    origin = getattr(existing, "__file__", None)
    owned = Path(__file__).resolve().parents[3] / "habitat_contract"
    if origin and Path(origin).resolve().parent != owned:
        raise RuntimeError(
            "habitat_contract is already imported from an external installation; "
            "start a fresh process and import the SuperNav backend first"
        )
    for finder in sys.meta_path:
        sources = getattr(finder, "known_source_files", {})
        if not isinstance(sources, dict) or "habitat_sim" not in sources:
            continue
        for attribute in ("known_source_files", "known_wheel_files", "submodule_search_locations"):
            mapping = getattr(finder, attribute, {})
            if isinstance(mapping, dict):
                for name in list(mapping):
                    if name == "habitat_contract" or name.startswith("habitat_contract."):
                        del mapping[name]


def _redirect_editable_habitat_sources(package_dir: Path) -> None:
    """Make scikit-build editable imports honor the current worktree.

    The conda env can have a repo-root editable finder installed ahead of
    ``sys.path``.  Without retargeting that finder, bridge subprocesses started
    from a sibling git worktree silently import SDK modules from the installed
    checkout while the caller expects the worktree under test.
    """

    local_root = package_dir.resolve()
    package_name = "habitat_sim"

    for finder in sys.meta_path:
        known_source_files = getattr(finder, "known_source_files", None)
        if not isinstance(known_source_files, dict):
            continue

        root_file = known_source_files.get(package_name)
        if not isinstance(root_file, str):
            continue
        # An external simulator is a read-only backend. Importing its existing
        # binary must not trigger scikit-build's automatic build/install hooks.
        if hasattr(finder, "rebuild_flag"):
            finder.rebuild_flag = False
        old_root = Path(root_file).resolve().parent
        for module_name, source_file in list(known_source_files.items()):
            if module_name != package_name and not module_name.startswith(f"{package_name}."):
                continue
            source_path = Path(source_file)
            try:
                relative = source_path.resolve().relative_to(old_root)
            except (OSError, ValueError):
                continue
            candidate = local_root / relative
            if not candidate.exists():
                # Explicit source selection must not fall back to Python files
                # found only in the previous (possibly customized) checkout.
                del known_source_files[module_name]
                continue
            known_source_files[module_name] = str(candidate)

        search_locations = getattr(finder, "submodule_search_locations", None)
        if not isinstance(search_locations, dict):
            continue
        for module_name, locations in list(search_locations.items()):
            if module_name != package_name and not module_name.startswith(f"{package_name}."):
                continue
            updated: set[str] = set()
            for location in locations:
                location_path = Path(location)
                try:
                    relative = location_path.resolve().relative_to(old_root)
                except (OSError, ValueError):
                    updated.add(location)
                    continue
                candidate = local_root / relative
                if candidate.exists():
                    updated.add(str(candidate))
            search_locations[module_name] = updated


def _disable_editable_rebuilds() -> None:
    """Never rebuild an external simulator as a side effect of importing it."""
    for finder in sys.meta_path:
        sources = getattr(finder, "known_source_files", {})
        if isinstance(sources, dict) and "habitat_sim" in sources:
            if hasattr(finder, "rebuild_flag"):
                finder.rebuild_flag = False


def _import_sdk_package(package_dir: Path) -> None:
    """Load only the SDK package, without exposing checkout application packages."""
    package_dir = package_dir.resolve()
    existing = sys.modules.get("habitat_sim")
    if existing is not None:
        origin = getattr(existing, "__file__", None)
        if origin and Path(origin).resolve().parent == package_dir:
            return
        raise RuntimeError(
            "habitat_sim is already imported from another location; start a fresh "
            "process to select SUPERNAV_HABITAT_ROOT"
        )
    _redirect_editable_habitat_sources(package_dir)
    spec = importlib.util.spec_from_file_location(
        "habitat_sim", package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Habitat SDK from {package_dir}")
    previous_modules = {
        name for name in sys.modules
        if name == "habitat_sim" or name.startswith("habitat_sim.")
    }
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        # Editable builds install the native namespace separately from Python
        # sources. Bind only that namespace.
        native_roots = [str(package_dir)]
        for finder in sys.meta_path:
            locations = getattr(finder, "submodule_search_locations", {})
            if isinstance(locations, dict):
                native_roots.extend(
                    str(Path(path).parent)
                    for path in locations.get("habitat_sim._ext", ())
                )
        try:
            distribution = importlib.metadata.distribution("habitat_sim")
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            native_roots.append(str(distribution.locate_file("habitat_sim")))
        native_spec = importlib.machinery.PathFinder.find_spec(
            "habitat_sim._ext", list(dict.fromkeys(native_roots))
        )
        if native_spec is not None:
            native = importlib.util.module_from_spec(native_spec)
            sys.modules[native_spec.name] = native
            module._ext = native
            if native_spec.loader is not None:
                native_spec.loader.exec_module(native)
        spec.loader.exec_module(module)
    except BaseException:
        # Preserve the native import error. A fabricated package masks missing
        # compiled bindings and leaves a partially usable simulator in sys.modules.
        for name in list(sys.modules):
            if name not in previous_modules and (
                name == "habitat_sim" or name.startswith("habitat_sim.")
            ):
                del sys.modules[name]
        if sys.modules.get(spec.name) is module:
            sys.modules.pop(spec.name, None)
        raise


def prepare_simulator_import() -> None:
    """Load the installed SDK or an explicit checkout; never its agent adapter.

    SUPERNAV_HABITAT_ROOT is optional when Habitat-GS is installed normally.
    Explicit source selection retains installed native extension lookup, allowing
    an existing compatible build to be used without changing either checkout.
    """
    from supernav.backends.habitat.config import habitat_root

    sys.dont_write_bytecode = True
    prepare_contract_import()
    _disable_editable_rebuilds()
    root = habitat_root(required=False)
    if root is None:
        importlib.import_module("habitat_sim")
    else:
        _import_sdk_package(root / "src_python" / "habitat_sim")
