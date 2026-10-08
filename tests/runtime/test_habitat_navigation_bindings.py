from __future__ import annotations

import types

import pytest

from supernav.backends.habitat import http_server
from supernav.backends.habitat.simulator import core


_ACTIONS = [
    ("navigate_with_localnav", "supernav.methods.localnav.navigation"),
    ("preload_localnav", "supernav.methods.localnav.navigation"),
    ("navigate_oracle_local", "supernav.methods.navigation.oracle_local_nav"),
    ("navigate_visual_local", "supernav.methods.navigation.oracle_local_nav"),
    ("navigate_visual_ground_preview", "supernav.methods.navigation.oracle_local_nav"),
    ("navigate_visual_overlay", "supernav.methods.navigation.oracle_local_nav"),
    ("navigate_visual_point", "supernav.methods.navigation.oracle_local_nav"),
    ("get_oracle_local_map", "supernav.methods.navigation.oracle_local_nav"),
]


@pytest.fixture
def loaded_adapter(monkeypatch):
    """The real SuperNav-owned adapter constructs without any simulator SDK."""
    monkeypatch.setattr(http_server, "prepare_simulator_import", lambda: None)
    adapter_class = http_server._load_adapter()
    assert adapter_class.__module__ == "supernav.backends.habitat.adapter"
    assert all(not cls.__module__.startswith("habitat_sim") for cls in adapter_class.__mro__)
    return adapter_class()


@pytest.mark.parametrize(("action", "module_name"), _ACTIONS)
def test_loaded_adapter_dispatches_navigation_to_canonical_methods(
    monkeypatch, loaded_adapter, action, module_name
):
    imports = []
    payload = {"point": [0.5, 0.8]}
    response = {"ok": True, "movement_frames": ["native-frame.png"]}

    def handler(adapter, session_id, request):
        assert adapter is loaded_adapter
        assert session_id == "session-id"
        assert request is payload
        return response

    def canonical_import(name):
        assert name == module_name
        assert name.startswith("supernav.methods.")
        imports.append(name)
        return types.SimpleNamespace(**{action: handler})

    monkeypatch.setattr(core, "import_module", canonical_import)
    assert imports == []
    assert loaded_adapter._handlers[action]("session-id", payload) is response
    assert imports == [module_name]


def test_navigation_handler_exception_reaches_native_request_boundary(
    monkeypatch, loaded_adapter
):
    failure = ValueError("invalid session")

    def handler(*args):
        raise failure

    monkeypatch.setattr(
        core,
        "import_module",
        lambda name: types.SimpleNamespace(navigate_visual_point=handler),
    )
    with pytest.raises(ValueError) as caught:
        loaded_adapter._handlers["navigate_visual_point"]("session-id", {})
    assert caught.value is failure
