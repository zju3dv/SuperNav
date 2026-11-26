"""Four directional brain observations with a front-RGB NoMaD executor."""
import base64
import io
import json
import math
import time

from supernav.backends.ai2thor.native import NativeSession
from supernav.backends.ai2thor.protocol import NativeConfig, vertical_fov

VIEWS = dict(front=0., right=90., back=180., left=-90.)


def configuration():
    return NativeConfig(profile="neednav_fourview_unscored", camera_height_m=1.25,
                        vertical_fov_deg=vertical_fov(90., 640, 480))


class FourViewSession(NativeSession):
    observation_views = tuple(VIEWS)

    def __init__(self, *args, clock=time.monotonic, **kwargs):
        self.clock = clock
        self.deadline = None
        self.side_indices = {}
        self.view_frames = {}
        self.surround_id = None
        super().__init__(*args, **kwargs)
        self.started = self.clock()

    def _publish_live(self, metadata, *, setup=False):
        # Invalidate every direction before the observer sees the new capture.
        self.view_frames = {"front": self.last_rgb.copy()}
        self.surround_id = None
        super()._publish_live(metadata, setup=setup)

    def begin(self):
        if self.deadline is not None or self.terminal:
            raise ValueError("Episode already begun or terminal")
        self.started = self.clock()
        self.deadline = self.started + 3600.
        return dict(started=True, wall_budget_s=3600, max_execution_units=500)

    def expired(self):
        if self.deadline is not None and self.clock() >= self.deadline and not self.terminal:
            self.finish("wall_timeout")
        return self.terminal

    def step(self, action):
        if self.deadline is None:
            raise ValueError("Episode has not begun")
        if self.expired():
            return dict(terminal=True, action_success=False, reason="episode_terminal")
        before = json.loads(json.dumps(self.controller.last_event.metadata["agent"]))
        result = super().step(action)
        after = self.controller.last_event.metadata["agent"]
        if action in ("left", "right"):
            expected = -self.config.turn_degrees if action == "left" else self.config.turn_degrees
            error = abs((after["rotation"]["y"]-before["rotation"]["y"]-expected+180)%360-180)
            if not result["action_success"] or error > .1:
                raise RuntimeError("Physical yaw differs from requested primitive")
        if action in ("forward", "backward"):
            distance = math.hypot(*(after["position"][k]-before["position"][k] for k in ("x","z")))
            if distance > self.config.move_magnitude_m+.02:
                raise RuntimeError("Physical translation exceeds requested primitive")
        return result

    def finish(self, reason="agent_stop"):
        if self.terminal:
            return json.loads((self.output/"status.json").read_text())
        if reason == "agent_stop" and self.deadline is not None and self.clock() >= self.deadline:
            reason = "wall_timeout"
        if self.last_rgb is not None and not self.infra_failed:
            self._surround()
        if reason == "action_budget_exhausted":
            reason = "execution_budget_exhausted"
        result = super().finish(reason)
        result.update(execution_units=float(self.actions),
                      duration_s=self.clock()-self.started,
                      observation_views=list(VIEWS), horizontal_fov_deg=90.)
        from supernav.evaluation.demand_driven.dataset import write_json
        write_json(self.output/"status.json", result)
        return result

    def _surround(self):
        if self.surround_id == self.observation_id:
            return
        import numpy as np
        from PIL import Image
        agent = json.loads(json.dumps(self.controller.last_event.metadata["agent"]))
        for name, delta in VIEWS.items():
            if name == "front":
                continue
            position = dict(x=agent["position"]["x"], y=self.floor_y+1.25, z=agent["position"]["z"])
            rotation = dict(x=agent["cameraHorizon"], y=(agent["rotation"]["y"]+delta)%360, z=0)
            payload = dict(position=position, rotation=rotation, fieldOfView=self.config.vertical_fov_deg)
            if name not in self.side_indices:
                self.side_indices[name] = len(self.controller.last_event.metadata.get("thirdPartyCameras") or [])
                payload["action"] = "AddThirdPartyCamera"
            else:
                payload.update(action="UpdateThirdPartyCamera", thirdPartyCameraId=self.side_indices[name])
            event = self.controller.step(**payload, raise_for_failure=True)
            actual = event.metadata["thirdPartyCameras"][self.side_indices[name]]
            after = event.metadata["agent"]
            if any(abs(actual["position"][k]-position[k])>.001 for k in "xyz"):
                raise RuntimeError("Surround camera mount mismatch")
            if any(abs((actual["rotation"][k]-rotation[k]+180)%360-180)>.01 for k in "xyz"):
                raise RuntimeError("Surround camera orientation mismatch")
            if abs(actual["fieldOfView"]-self.config.vertical_fov_deg)>.01:
                raise RuntimeError("Surround camera FOV mismatch")
            if any(abs(after["position"][k]-agent["position"][k])>.001 for k in "xyz") or any(
                abs((after["rotation"][k]-agent["rotation"][k]+180)%360-180)>.01 for k in "xyz"):
                raise RuntimeError("Surround rendering moved the physical agent")
            frame = np.asarray(event.third_party_camera_frames[self.side_indices[name]])[:,:,:3].copy()
            if frame.shape != (480,640,3) or frame.std()<1:
                raise RuntimeError("Invalid surround RGB")
            self.view_frames[name] = frame
            Image.fromarray(frame).save(self.output/"frames"/(self.observation_id+"_"+name+".jpg"), quality=90)
            self._journal("surround_camera_sync.jsonl",dict(view=name, observation_id=self.observation_id,
                          requested=payload, returned=actual, physical_pose=agent,
                          action_index=self.actions, budget_cost=0))
        self.surround_id = self.observation_id

    def observe(self):
        self.expired()
        result = super().observe()
        result.pop("rgb_jpeg_base64")
        self._surround()
        self.live.refresh(self)
        from PIL import Image
        images = {}
        for name in VIEWS:
            buffer = io.BytesIO()
            Image.fromarray(self.view_frames[name]).save(buffer, format="JPEG", quality=90)
            images[name] = base64.b64encode(buffer.getvalue()).decode("ascii")
        return dict(result, images=images, views=list(VIEWS),
                    view_refs={name:self.observation_id+":"+name for name in VIEWS},
                    remaining_units=max(0,500-self.actions), execution_units=float(self.actions))

    def local_navigate(self, observation_id, pixel, policy, max_replans=4, *, view="front"):
        if view not in VIEWS or observation_id != self.observation_id or self.terminal or self.infra_failed:
            raise ValueError("Select a view and pixel from the latest non-terminal observation")
        if len(pixel)!=2 or any(not math.isfinite(float(x)) or not 0<=float(x)<=1 for x in pixel):
            raise ValueError("Invalid goal pixel")
        if not 1 <= max_replans <= 20:
            raise ValueError("max_replans must be in [1,20]")
        if self.deadline is None:
            raise ValueError("Episode has not begun")
        if self.expired():
            return dict(reason="episode_terminal", **self.observe())
        self._surround()
        self.live.point(self, pixel, "ddn_local_navigate", view=view)
        reference = self.view_frames[view].copy()
        before = self.actions
        delta = VIEWS[view]
        for _ in range(round(abs(delta)/self.config.turn_degrees)):
            if self.expired():
                return dict(reason="episode_terminal", **self.observe())
            self.step("right" if delta>0 else "left")
        self._journal("view_alignment.jsonl",dict(source_observation_id=observation_id,
                      source_view=view, goal_pixel=pixel, yaw_degrees=delta,
                      physical_actions=self.actions-before, observation_id=self.observation_id))
        if self.expired():
            return dict(reason="episode_terminal", **self.observe())
        result = super().local_navigate(self.observation_id, pixel, policy, max_replans,
                                       goal_rgb=reference, publish_goal=False)
        self.expired()
        result["terminal"] = self.terminal
        return result

    def claim_view(self, observation_id, description, pixel, view):
        if view not in VIEWS:
            raise ValueError("Unknown claim view")
        if self.expired():
            raise ValueError("Session is terminal")
        if observation_id != self.observation_id or self.infra_failed:
            raise ValueError("A claim must reference the current, non-terminal observation")
        self._surround()
        result = self.claim(observation_id, description, pixel, view=view)
        self._journal("claim_views.jsonl",dict(observation_id=observation_id, view=view,
                      description=description, pixel=pixel, action_index=self.actions))
        return result
