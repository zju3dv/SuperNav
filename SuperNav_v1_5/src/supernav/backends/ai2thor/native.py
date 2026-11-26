"""Native house rendering, auditable primitive execution and a GT-free policy boundary."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
import time

from supernav.evaluation.demand_driven.dataset import policy_task, write_json
from supernav.backends.ai2thor.live import NativeLiveView
from supernav.backends.ai2thor.protocol import BUILD_COMMIT, NativeConfig


def make_controller(house, config: NativeConfig, runtime: Path, releases: Path, gpu: int,
                    *, server_timeout_s=120, server_start_timeout_s=180):
    # Import after the launcher redirects all writable caches and temporary files.
    import ai2thor
    from ai2thor.build import Build
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    if ai2thor.__version__ != "5.0.0":
        raise RuntimeError("Expected the pinned AI2-THOR 5.0.0 client")
    runtime.mkdir(parents=True, exist_ok=True)
    runtime = runtime.resolve()

    class ReadOnlyBuild(Build):
        # Like AI2-THOR's ExternalBuild, an explicitly supplied installation is
        # owned by its operator. Never download, prune or create locks beside it.
        def download(self):
            if not Path(self.executable_path).is_file() or not Path(self.metadata_path).is_file():
                raise RuntimeError("Pinned CloudRendering build missing; refusing implicit download")

        def lock_sh(self):
            pass

        def unlock(self):
            pass

    class PinnedController(Controller):
        @property
        def base_dir(self):
            return str(runtime.resolve())

        def find_build(self, *args, **kwargs):
            build = ReadOnlyBuild(CloudRendering, BUILD_COMMIT, False, str(releases.resolve()))
            build.download()
            return build

        def unity_command(self, width, height, headless):
            if headless:
                raise ValueError("CloudRendering needs graphics; do not pass -nographics")
            return super().unity_command(width, height, False) + [
                "-batchmode", "-logFile", str(runtime / "Player.log")
            ]

    controller = PinnedController.__new__(PinnedController)
    try:
        controller.__init__(scene=house, platform=CloudRendering, gpu_device=gpu,
                            server_timeout=server_timeout_s, server_start_timeout=server_start_timeout_s,
                            **config.controller_kwargs())
        return controller
    except BaseException:
        # The constructor may have launched Unity before a CreateHouse failure.
        try:
            controller.stop()
        except Exception:
            pass
        raise


def pose(metadata: dict) -> dict:
    return {key: metadata["agent"][key] for key in ("position", "rotation", "cameraHorizon")}


class NativeSession:
    """One task. Full simulator metadata stays in private evaluator files only."""

    observation_views = ("front",)

    def __init__(self, controller, episode: dict, config: NativeConfig, output: Path, house: dict | None = None):
        self.controller = controller
        self.episode = episode
        self.config = config
        self.output = output
        self.output.mkdir(parents=True, exist_ok=False)
        (output / "frames").mkdir()
        self.private = output / "evaluator"
        self.private.mkdir(mode=0o700)
        self.actions = 0
        self.capture = 0
        self.terminal = False
        self.infra_failed = False
        self.last_rgb = None
        self.observation_id = None
        self.history = []
        self.camera_index = None
        self.floor_y = None
        if config.camera_height_m is not None:
            if house is None:
                raise ValueError("Height-controlled rendering requires the frozen house floor definition")
            levels = {round(float(p["y"]), 6) for room in house["rooms"] for p in room["floorPolygon"]}
            if len(levels) != 1:
                raise ValueError("The height-controlled adapter only supports single-level houses")
            self.floor_y = levels.pop()
        self.started = time.monotonic()
        write_json(self.private / "episode.json", episode)
        write_json(output / "protocol.json", config.receipt())
        self.live = NativeLiveView(episode)

    def _journal(self, filename, value):
        with (self.private / filename).open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")

    def _save(self, event, action: dict, elapsed: float, setup=False):
        import numpy as np
        from PIL import Image

        metadata = event.metadata
        render_metadata = None
        render_elapsed = 0.0
        if self.config.camera_height_m is not None:
            agent = metadata["agent"]
            position = dict(x=agent["position"]["x"], y=self.floor_y + self.config.camera_height_m,
                            z=agent["position"]["z"])
            rotation = dict(x=agent["cameraHorizon"], y=agent["rotation"]["y"], z=0)
            camera_action = dict(position=position, rotation=rotation, fieldOfView=self.config.vertical_fov_deg)
            if self.camera_index is None:
                self.camera_index = len(metadata.get("thirdPartyCameras") or [])
                camera_action["action"] = "AddThirdPartyCamera"
            else:
                camera_action.update(action="UpdateThirdPartyCamera", thirdPartyCameraId=self.camera_index)
            began = time.monotonic()
            render = self.controller.step(**camera_action, raise_for_failure=True)
            render_elapsed = time.monotonic() - began
            render_metadata = render.metadata["thirdPartyCameras"][self.camera_index]
            actual = render_metadata["position"]
            if any(abs(actual[k] - position[k]) > .001 for k in "xyz"):
                raise RuntimeError("Rendered camera pose does not match the requested egocentric mount")
            if any(abs((render_metadata["rotation"][k] - rotation[k] + 180) % 360 - 180) > .01 for k in "xyz"):
                raise RuntimeError("Rendered camera orientation does not follow the physical agent")
            if abs(render_metadata["fieldOfView"] - self.config.vertical_fov_deg) > .01:
                raise RuntimeError("Rendered camera FOV differs from the frozen protocol")
            # Updating the observer must not move the physical agent.
            after = render.metadata["agent"]["position"]
            if any(abs(after[k] - agent["position"][k]) > .001 for k in "xyz"):
                raise RuntimeError("Camera synchronization unexpectedly moved the agent")
            rgb = np.asarray(render.third_party_camera_frames[self.camera_index]).copy()
            if rgb.shape == (self.config.height, self.config.width, 4):
                # Unity's third-party render texture is RGBA; policy inputs are RGB.
                rgb = rgb[:, :, :3].copy()
            self._journal("camera_sync.jsonl", dict(action_index=self.actions, requested=camera_action,
                          returned=render_metadata, elapsed_s=render_elapsed, budget_cost=0,
                          observation_permissions="same egocentric front view, not an additional viewpoint"))
        else:
            rgb = np.asarray(event.frame).copy()
        if rgb.shape != (self.config.height, self.config.width, 3) or rgb.std() < 1:
            raise RuntimeError(f"Invalid RGB frame: {rgb.shape}")
        self.capture += 1
        self.observation_id = f"frame_{self.capture:06d}"
        self.last_rgb = rgb
        self.history.append(rgb)
        del self.history[:-6]
        Image.fromarray(rgb).save(self.output / "frames" / (self.observation_id + ".jpg"), quality=90)
        self._journal("events.jsonl", dict(capture=self.capture, action_index=self.actions,
                      action=action, setup=setup, elapsed_s=elapsed, wall_time=time.time(),
                      observation_id=self.observation_id, metadata=metadata,
                      policy_camera=render_metadata, camera_sync_elapsed_s=render_elapsed))
        self._publish_live(metadata, setup=setup)
        return dict(observation_id=self.observation_id, action_index=self.actions,
                    action_success=bool(metadata["lastActionSuccess"]),
                    collided=bool(metadata.get("collided", False)),
                    terminal=self.terminal, success_scoring="withheld")

    def _publish_live(self, metadata, *, setup=False):
        self.live.sample(self, metadata, force=setup)

    def initialize(self):
        start = dict(action="TeleportFull", position=self.episode["start_position"],
                     rotation=dict(x=0, y=self.episode["start_rotation_y"], z=0),
                     horizon=self.episode["start_horizon"], standing=True, forceAction=False)
        began = time.monotonic()
        event = self.controller.step(**start, raise_for_failure=True)
        self._save(event, start, time.monotonic() - began, setup=True)
        actual = event.metadata
        position_error = math.sqrt(sum((actual["agent"]["position"][k] - self.episode["start_position"][k]) ** 2 for k in "xyz"))
        horizontal_error = math.hypot(*(actual["agent"]["position"][k] - self.episode["start_position"][k] for k in "xz"))
        height_change = actual["agent"]["position"]["y"] - self.episode["start_position"]["y"]
        yaw_error = abs((actual["agent"]["rotation"]["y"] - self.episode["start_rotation_y"] + 180) % 360 - 180)
        horizon_error = abs(actual["agent"]["cameraHorizon"] - self.episode["start_horizon"])
        missing = sorted({c["object_id"] for stage in self.episode["stage_plan"] for c in stage["allowed_target_candidates"]}
                         - {obj["objectId"] for obj in actual["objects"]})
        receipt = dict(scene_id=self.episode["scene_id"], native_scene_name=actual.get("sceneName"),
                       house_sha256=self.episode["reproducibility"]["house_data_sha256"],
                       actual_start=pose(actual), start_position_error_m=position_error,
                       start_horizontal_error_m=horizontal_error, vertical_settling_m=height_change,
                       strict_start_pose_match=position_error < .02,
                       validation_scope="engineering; native vertical settling is logged, not baseline-validated",
                       start_yaw_error_deg=yaw_error, start_horizon_error_deg=horizon_error,
                       fov_returned=actual.get("fov"), camera_position=actual.get("cameraPosition"),
                       policy_camera_height_m=self.config.camera_height_m,
                       policy_camera_position=(self.controller.last_event.metadata["thirdPartyCameras"][self.camera_index]["position"]
                                               if self.camera_index is not None else actual.get("cameraPosition")),
                       missing_candidate_object_ids=missing, native_object_count=len(actual["objects"]),
                       rgb_shape=list(self.last_rgb.shape), rgb_standard_deviation=float(self.last_rgb.std()),
                       baseline_alignment_confirmed=False, success_scoring="withheld")
        # TeleportFull permits native gravity to settle the default capsule.
        # Accept a small downward-only settle for plumbing checks, but retain
        # the failed exact-pose comparison for the future shared protocol.
        receipt["valid"] = (horizontal_error < .02 and -.1 <= height_change <= .005
                            and yaw_error < .1 and horizon_error < .1 and not missing
                            and abs(actual.get("fov", -1) - self.config.vertical_fov_deg) < .01)
        write_json(self.private / "scene_receipt.json", receipt)
        if not receipt["valid"]:
            raise RuntimeError("Native scene/start/camera validation failed; see private scene receipt")
        return self.observe()

    def observe(self):
        if self.last_rgb is None:
            raise RuntimeError("Session has not initialized")
        from PIL import Image

        self.live.refresh(self)
        buffer = io.BytesIO()
        Image.fromarray(self.last_rgb).save(buffer, format="JPEG", quality=90)
        return dict(**policy_task(self.episode), observation_id=self.observation_id,
                    rgb_jpeg_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
                    action_index=self.actions, remaining_actions=max(0, self.config.max_actions-self.actions),
                    terminal=self.terminal, success_scoring="withheld")

    def step(self, action: str):
        if self.terminal or self.infra_failed:
            raise ValueError("Session is terminal")
        if self.actions >= self.config.max_actions:
            raise ValueError("Engineering action safety cap exhausted")
        commands = {
            "forward": dict(action="MoveAhead", moveMagnitude=self.config.move_magnitude_m),
            "backward": dict(action="MoveBack", moveMagnitude=self.config.move_magnitude_m),
            "left": dict(action="RotateLeft", degrees=self.config.turn_degrees),
            "right": dict(action="RotateRight", degrees=self.config.turn_degrees),
            "look_up": dict(action="LookUp", degrees=self.config.turn_degrees),
            "look_down": dict(action="LookDown", degrees=self.config.turn_degrees),
        }
        if action not in commands:
            raise ValueError("Only physical navigation primitives are exposed")
        self.actions += 1  # Failed collision attempts consume the same budget.
        self.live.event(self, commands[action]["action"], "started")
        began = time.monotonic()
        try:
            event = self.controller.step(**commands[action])
            status = self._save(event, commands[action], time.monotonic() - began)
            self.live.event(self, commands[action]["action"], "finished", ok=status["action_success"])
            if self.actions >= self.config.max_actions:
                self.finish("action_budget_exhausted")
                status["terminal"] = True
        except Exception as exc:
            self.infra_failed = True
            self.live.event(self, commands[action]["action"], "error", ok=False)
            self._journal("errors.jsonl", dict(action_index=self.actions, error=str(exc), action=action))
            raise
        return status

    def claim(self, observation_id: str, description: str, pixel: list, *, view="front"):
        if self.terminal or self.infra_failed or observation_id != self.observation_id:
            raise ValueError("A claim must reference the current, non-terminal observation")
        if not description.strip() or len(description) > 2000:
            raise ValueError("Provide a short description of the resource you claim to have reached")
        if len(pixel) != 2 or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in pixel):
            raise ValueError("Pixel coordinates must be normalized to [0, 1]")
        self._journal("claims.jsonl", dict(observation_id=observation_id, description=description,
                      pixel=pixel, action_index=self.actions, pose=pose(self.controller.last_event.metadata)))
        self.live.point(self, pixel, "ddn_claim_resource", description=description, view=view)
        # No oracle acknowledgement, stage transition or object list is returned.
        return dict(recorded=True, success_scoring="withheld")

    def finish(self, reason="agent_stop"):
        if self.terminal:
            raise ValueError("Session already closed")
        explicit_stop = reason == "agent_stop"
        if explicit_stop and self.actions >= self.config.max_actions:
            raise ValueError("No remaining budget for an explicit STOP")
        if explicit_stop:
            self.actions += 1
        self.terminal = True
        result = dict(episode_id=self.episode["episode_id"], state="infra_failed" if self.infra_failed else "closed_unscored",
                      stop_called=explicit_stop, reason=reason, action_count=self.actions,
                      duration_s=time.monotonic() - self.started, final_observation_id=self.observation_id,
                      success_scoring="withheld", baseline_alignment_confirmed=False)
        write_json(self.output / "status.json", result)
        self.live.finish(self, reason)
        return result

    def local_navigate(self, observation_id: str, pixel: list, policy, max_replans=4, *, goal_rgb=None, publish_goal=True):
        import numpy as np
        from supernav.methods.localnav.adapters import HabitatActionAdapter, HabitatActionAdapterConfig
        from supernav.methods.localnav.policy_client import select_trajectory, waypoint_deltas_to_actions

        if self.terminal or self.infra_failed or observation_id != self.observation_id:
            raise ValueError("Choose a pixel in the current non-terminal observation")
        if len(pixel) != 2 or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in pixel):
            raise ValueError("Invalid goal pixel")
        if not 1 <= max_replans <= 20:
            raise ValueError("max_replans must be in [1, 20]")
        if publish_goal:
            self.live.point(self, pixel, "ddn_local_navigate")
        goal = self.last_rgb.copy() if goal_rgb is None else goal_rgb.copy()
        history = [self.last_rgb.copy()]
        adapter = HabitatActionAdapter(HabitatActionAdapterConfig(
            forward_step_m=self.config.move_magnitude_m, turn_step_degrees=self.config.turn_degrees))
        reason = "replan_limit"
        start_actions = self.actions
        consecutive_done, done_start, last_cycle = 0, 0, None
        for index in range(max_replans):
            if self.terminal:
                return dict(reason="episode_terminal", **self.observe())
            context = history if getattr(policy, "context_mode", "fixed") == "gca" else history[-8:]
            prediction = policy.act(context, goal, pixel)
            arrays = np.asarray(prediction["trajectories"], dtype=float)
            selected = select_trajectory(arrays, prediction.get("scores"))
            if arrays.ndim != 3 or arrays.shape[-1] != 2 or not np.isfinite(arrays).all() or not 0 <= selected < len(arrays):
                raise ValueError("Invalid NoMaD trajectory")
            if not all(math.isfinite(float(prediction[k])) for k in ("done_probability", "confidence")):
                raise ValueError("Non-finite NoMaD confidence/stop output")
            self._journal("localnav.jsonl", dict(replan=index, action_index=self.actions,
                          goal_observation_id=observation_id, goal_pixel=pixel, prediction=prediction))
            if prediction["done_probability"] >= self.config.localnav_stop_probability:
                consecutive_done += 1
                if consecutive_done == 1:
                    done_start = self.actions
                if consecutive_done >= 2 and ((self.actions - start_actions >= 4 and self.actions - done_start >= 3) or last_cycle == 0):
                    reason = "local_policy_stop_not_task_success"
                    break
            else:
                consecutive_done = 0
            if prediction["confidence"] < self.config.localnav_confidence_threshold:
                reason = "low_confidence"
                break
            velocities = waypoint_deltas_to_actions(arrays[selected], .5)
            moved = 0
            for velocity in velocities[:self.config.localnav_execute_actions]:
                for payload in adapter.to_payloads(velocity):
                    rotation = payload["action"].startswith("turn")
                    quantum = self.config.turn_degrees if rotation else self.config.move_magnitude_m
                    count = round(payload["degrees" if rotation else "distance"] / quantum)
                    name = {"turn_left": "left", "turn_right": "right", "move_forward": "forward", "move_backward": "backward"}[payload["action"]]
                    for _ in range(count):
                        if self.terminal or self.actions >= self.config.max_actions:
                            return dict(reason="engineering_action_cap", **self.observe())
                        status = self.step(name)
                        moved += 1
                        history.append(self.last_rgb.copy())
                        if not status["action_success"]:
                            return dict(reason="collision_or_action_rejected", **self.observe())
            last_cycle = moved
            if not moved and consecutive_done == 0:
                reason = "zero_motion"
                break
        return dict(reason=reason, **self.observe())
