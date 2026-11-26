"""Implement method-layer navigation capabilities with public Habitat SDK APIs."""
from __future__ import annotations

from typing import Any

import numpy as np

from habitat_contract.navigation import NavigationPath
from supernav.backends.habitat.simulator import sdk as habitat_sim


class HabitatPathfinder:
    def __init__(self, pathfinder: Any):
        self._pathfinder = pathfinder

    @property
    def is_loaded(self):
        return bool(self._pathfinder.is_loaded)

    def shortest_path(self, start, end):
        path = habitat_sim.ShortestPath()
        path.requested_start = np.asarray(start, dtype=np.float32)
        path.requested_end = np.asarray(end, dtype=np.float32)
        if not self._pathfinder.find_path(path):
            return None
        return NavigationPath(
            geodesic_distance=float(path.geodesic_distance),
            points=[list(map(float, point)) for point in path.points],
        )

    def snap_point(self, point):
        return self._pathfinder.snap_point(point)

    def get_topdown_view(self, meters_per_pixel, height):
        return self._pathfinder.get_topdown_view(meters_per_pixel, height)

    def get_bounds(self):
        return self._pathfinder.get_bounds()

    def get_random_navigable_point(self):
        return self._pathfinder.get_random_navigable_point()

    def is_navigable(self, point):
        return bool(self._pathfinder.is_navigable(point))

    def seed(self, seed):
        self._pathfinder.seed(int(seed))


class HabitatRaycastScene:
    def __init__(self, simulator):
        self._simulator = simulator

    def cast_ray(self, origin, direction, *, max_distance):
        ray = habitat_sim.geo.Ray(
            np.asarray(origin, dtype=np.float32), np.asarray(direction, dtype=np.float32)
        )
        hits = self._simulator.cast_ray(ray, max_distance=float(max_distance))
        return sorted(float(hit.ray_distance) for hit in hits.hits)


class NavigationAccessMixin:
    """Explicit SDK-free surface consumed by ``supernav.methods``."""

    def navigation_session(self, session_id):
        return self._require_session(session_id)


    def navigation_pathfinder(self, session):
        return HabitatPathfinder(self._require_pathfinder_loaded(session))

    def navigation_scene(self, session):
        return HabitatRaycastScene(session.simulator)

    def agent_position(self, session):
        return self._current_position(session)

    def agent_rotation_wxyz(self, session):
        return self._current_rotation(session)

    def agent_forward(self, session):
        return self._forward_vector(session)

    def agent_heading_degrees(self, session):
        return self._heading_degrees(session)

    def sensor_observations(self, session):
        return session.simulator.get_sensor_observations(agent_ids=session.agent_id)

    def observations_at(self, session, position, rotation_xyzw):
        agent = session.simulator.get_agent(session.agent_id)
        saved_state = agent.get_state()
        try:
            state = habitat_sim.AgentState()
            state.position = np.asarray(position, dtype=np.float32)
            state.rotation = rotation_xyzw
            agent.set_state(state, infer_sensor_states=True)
            return session.simulator.get_sensor_observations(agent_ids=session.agent_id)
        finally:
            agent.set_state(saved_state, infer_sensor_states=True)

    def navigate_step(self, session_id, payload):
        return self._navigate_step(session_id, payload)

    def step_action(self, session_id, payload):
        return self._step_action(session_id, payload)

    def get_panorama(self, session_id, payload):
        return self._get_panorama(session_id, payload)

    def step_and_capture(self, session_id, payload):
        return self._step_and_capture(session_id, payload)

    def set_agent_state(self, session_id, payload):
        return self._set_agent_state(session_id, payload)

    def publish_live_overlay(self, session, overlay, **metadata):
        return self._publish_live_overlay(session, overlay, **metadata)

    def session_output_dir(self, output_dir, session_id):
        return self._session_output_dir(output_dir, session_id)

    def write_png_image(self, **kwargs):
        return self._write_png_image(**kwargs)

    def to_rgb_topdown(self, raw_map):
        return self._to_rgb_topdown(raw_map)

    def draw_topdown_marker(self, *args, **kwargs):
        return self._draw_topdown_marker(*args, **kwargs)

    def draw_topdown_arrow(self, *args, **kwargs):
        return self._draw_topdown_arrow(*args, **kwargs)

    def topdown_xy(self, pathfinder, point, meters_per_pixel):
        return self._topdown_xy(pathfinder, point, meters_per_pixel)
