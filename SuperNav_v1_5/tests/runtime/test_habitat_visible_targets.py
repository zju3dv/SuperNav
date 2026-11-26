from __future__ import annotations

import builtins
from types import SimpleNamespace

import numpy as np
import pytest

from supernav.backends.habitat.visible_targets import SuperNavVisibleTargetsMixin


@pytest.fixture(autouse=True)
def scene_graph_helpers(monkeypatch):
    from supernav.methods import navigation

    helpers = SimpleNamespace(
        _scene_coordinate_system=lambda graph: "habitat",
        _object_records=lambda graph: graph["objects"],
        _as_vec3=lambda value, **kwargs: np.asarray(value, dtype=np.float32),
        _object_target_ref=lambda obj, index: obj["id"],
    )
    monkeypatch.setattr(navigation, "oracle_local_nav", helpers, raising=False)


class GeometryAdapter(SuperNavVisibleTargetsMixin):
    _PANORAMA_TURN_RIGHT_DEGREES = {"front": 0, "right": 90}

    def __init__(self, *, visible=True, depth_fraction=0.8, semantic_fraction=0.5):
        self.visible = visible
        self.depth_fraction = depth_fraction
        self.semantic_fraction = semantic_fraction

    def _camera_basis_from_state(self, **kwargs):
        return (np.zeros(3),) * 4

    def _aabb_corners(self, bbox_min, bbox_max):
        return np.array([bbox_min, bbox_max])

    def _project_aabb_to_image(self, **kwargs):
        return 10, 20, 29, 39, 1.0, 1.5, 2.0

    def _line_of_sight_reaches_aabb(self, **kwargs):
        return self.visible

    def _depth_visible_fraction(self, **kwargs):
        return self.depth_fraction, 1.23456

    def _semantic_visible_fraction(self, **kwargs):
        return self.semantic_fraction


def targets(adapter):
    session = SimpleNamespace(
        settings={"hfov": 90},
        scene_graph={"objects": [{
            "id": "chair-1", "label": "chair",
            "bbox_min_xyz": [0, 0, -2], "bbox_max_xyz": [1, 1, -1],
        }]},
    )
    return adapter._visible_nav_targets_for_view(
        session=session, agent_state=None, direction="right",
        width=100, height=100, depth=np.ones((100, 100)),
    )


def test_visible_targets_use_canonical_scene_graph(monkeypatch):
    original_import = builtins.__import__
    imports = []

    def guarded_import(name, *args, **kwargs):
        imports.append(name)
        assert name.split(".")[0] != "habitat_agent"
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    rows = targets(GeometryAdapter())
    assert "supernav.methods.navigation" in imports
    assert rows == [{
        "target_ref": "chair-1", "label": "chair", "direction": "right",
        "turn_right_deg": 90,
        "bbox_px": {"x": 10, "y": 20, "width": 20, "height": 20},
        "center_px": [20, 30], "visible_fraction": 0.8,
        "visibility_basis": "projection_depth_semantic",
        "_sort_area": 400.0, "_sort_depth": 1.5,
        "depth_m": 1.235, "semantic_overlap": 0.5,
    }]


@pytest.mark.parametrize("settings", [
    {"visible": False}, {"depth_fraction": 0.09}, {"semantic_fraction": 0.0},
])
def test_occluded_targets_still_fail_existing_visibility_filters(settings):
    assert targets(GeometryAdapter(**settings)) == []
