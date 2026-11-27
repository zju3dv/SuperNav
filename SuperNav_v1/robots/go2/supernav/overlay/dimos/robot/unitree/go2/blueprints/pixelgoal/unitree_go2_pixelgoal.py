#!/usr/bin/env python3
# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Go2 map-backed PixelGoal with surround cameras, local planning and MCP."""

from typing import Any

from dimos.agents.mcp.mcp_server import McpServer
from dimos.constants import DEFAULT_CAPACITY_COLOR_IMAGE
from dimos.core.coordination.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.core.transport import pSHMTransport
from dimos.mapping.voxels import VoxelGridMapper
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.unitree.go2.blueprints.basic.unitree_go2_basic import rerun_config
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2 import unitree_go2
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.go2.surround_camera import (
    CAMERA_DIRECTIONS,
    SURROUND_SHM_TOPICS,
    Go2SurroundCameraModule,
)
from dimos.robot.unitree.pixelgoal.pixel_goal_map_module import PixelGoalMapNavModule
from dimos.robot.unitree.unitree_skill_container import UnitreeSkillContainer
from dimos.visualization.vis_module import vis_module







def _pixel_target_marker(point: Any) -> Any:
    import rerun as rr

    return rr.Points3D(
        positions=[[point.x, point.y, point.z]],
        colors=[[255, 55, 45]],
        radii=[0.18],
        labels=["navigation target"],
    )


def _pixel_projection_marker(point: Any) -> Any:
    import rerun as rr

    return rr.Points3D(
        positions=[[point.x, point.y, point.z]],
        colors=[[255, 155, 35]],
        radii=[0.12],
        labels=["raw lidar projection"],
    )


def _pixel_stop_marker(point: Any) -> Any:
    import rerun as rr

    return rr.Points3D(
        positions=[[point.x, point.y, point.z]],
        colors=[[0, 220, 210]],
        radii=[0.14],
        labels=["robot center stop"],
    )


_pixel_goal_rerun_config = {
    **rerun_config,
    "max_hz": {
        **rerun_config["max_hz"],
        "world/global_map": 0.5,
        "world/merged_map": 0.5,
        "world/navigation_costmap": 2.0,
        "world/global_costmap": 2.0,
    },
    "visual_override": {
        **rerun_config["visual_override"],
        "world/pixel_click": None,
        "world/pixel_goal_projection": _pixel_projection_marker,
        "world/pixel_goal_target": _pixel_target_marker,
        "world/pixel_goal_stop": _pixel_stop_marker,
    },
}


unitree_go2_pixelgoal = (
    autoconnect(
        unitree_go2,
        GO2Connection.blueprint(enable_video=False),
        VoxelGridMapper.blueprint(emit_every=1, load_on_start="none", save_on_stop=False),
        vis_module(global_config.viewer, rerun_config=_pixel_goal_rerun_config),
        McpServer.blueprint(),
        UnitreeSkillContainer.blueprint(),
        Go2SurroundCameraModule.blueprint(),
        PixelGoalMapNavModule.blueprint(camera_info=GO2Connection.camera_info_static),
    )
    .transports(
        {
            (f"surround_{direction}_image", Image): pSHMTransport(
                SURROUND_SHM_TOPICS[direction],
                default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE,
            )
            for direction in CAMERA_DIRECTIONS
        }
    )
    .global_config(n_workers=16, robot_model="unitree_go2")
)

__all__ = ["unitree_go2_pixelgoal"]
