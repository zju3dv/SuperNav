"""torch Dataset over LocalNavTrainingSample records (point-conditioned hops).

Conditioning is produced by the SAME shared preprocess functions the serving
backend uses (resize → marker at policy resolution → ImageNet normalize), so
training and inference can never diverge. Frames are stored clean on disk;
the marker (and its jitter augmentation) is rendered here on the fly —
switching the conditioning form for the M3 ablation does not require
regenerating data.
"""

from __future__ import annotations

import os
import random
from typing import List, Sequence, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from supernav.methods.localnav.dataset import LocalNavTrainingSample

from supernav.methods.localnav_policy.model import ACTION_HORIZON, DEFAULT_WAYPOINT_SCALE_M
from supernav.methods.localnav_policy.preprocess import goal_pair_tensor, stack_context


def _load_samples(
    samples: Union[str, os.PathLike, Sequence[LocalNavTrainingSample]],
) -> List[LocalNavTrainingSample]:
    if isinstance(samples, (str, os.PathLike)):
        from supernav.methods.localnav.dataset import load_samples_jsonl

        return load_samples_jsonl(samples)
    return list(samples)


class LocalNavTorchDataset(Dataset):
    def __init__(
        self,
        samples: Union[str, os.PathLike, Sequence[LocalNavTrainingSample]],
        root: "str | os.PathLike",
        *,
        context_size: int = 4,
        horizon: int = ACTION_HORIZON,
        waypoint_scale: float = DEFAULT_WAYPOINT_SCALE_M,
        point_jitter_px: float = 2.0,
        brightness_jitter: float = 0.15,
        sim2real_aug: float = 0.0,
        augment: bool = True,
        stop_k: int = 2,
        zero_motion_stop: bool = False,
        image_size: int = 96,
        conditioning: str = "marker",
        context_mode: str = "fixed",
        memory_budget: int = 12,
        memory_recent_tail: int = 0,
        memory_anchor: bool = True,
        memory_mid_span: int = 0,
        memory_mid_count: int = 0,
        hindsight_action_weight: float = 1.0,
    ) -> None:
        self._samples = _load_samples(samples)
        self._root = os.fspath(root)
        self._context_size = int(context_size)
        self._context_mode = str(context_mode)
        self._memory_budget = int(memory_budget)
        self._memory_recent_tail = int(memory_recent_tail)
        self._memory_anchor = bool(memory_anchor)
        self._memory_mid_span = int(memory_mid_span)
        self._memory_mid_count = int(memory_mid_count)
        # Hindsight target points lack depth-based occlusion checks.
        # A zero weight excludes them from action loss while retaining stop/distance labels.
        self._hindsight_action_weight = float(hindsight_action_weight)
        self._horizon = int(horizon)
        self._waypoint_scale = float(waypoint_scale)
        self._point_jitter_px = float(point_jitter_px)
        self._brightness_jitter = float(brightness_jitter)
        self._sim2real_aug = max(0.0, float(sim2real_aug))
        self._augment = bool(augment)
        self._stop_k = int(stop_k)
        self._zero_motion_stop = bool(zero_motion_stop)
        self._image_size = int(image_size)
        self._conditioning = str(conditioning)

    def __len__(self) -> int:
        return len(self._samples)

    def _load_rgb(self, relative_path: str, brightness: float) -> np.ndarray:
        with Image.open(os.path.join(self._root, relative_path)) as image:
            array = np.asarray(image.convert("RGB"), dtype=np.float32)
        if brightness != 1.0:
            array = np.clip(array * brightness, 0.0, 255.0)
        if self._augment and self._sim2real_aug > 0.0:
            array = self._sim2real_photometric(array)
        return array.astype(np.uint8)

    def _sim2real_photometric(self, array: np.ndarray) -> np.ndarray:
        """Apply independent exposure, gamma, noise and blur to each frame."""
        strength = self._sim2real_aug
        gain = 1.0 + random.uniform(-0.35, 0.35) * strength
        gamma = 1.0 + random.uniform(-0.25, 0.25) * strength
        out = np.clip(array * gain, 0.0, 255.0)
        if abs(gamma - 1.0) > 1e-3:
            out = 255.0 * np.power(np.clip(out / 255.0, 0.0, 1.0), 1.0 / max(gamma, 1e-3))
        sigma = 6.0 * strength
        if sigma > 0.1:
            out = out + np.random.normal(0.0, sigma, size=out.shape).astype(np.float32)
        if random.random() < 0.25 * strength:
            # cheap 3x3 box blur stands in for motion blur; frames are captured
            # standing still on purpose, so this is the residual case
            kernel = np.ones((3, 3), dtype=np.float32) / 9.0
            padded = np.pad(out, ((1, 1), (1, 1), (0, 0)), mode="edge")
            blurred = np.zeros_like(out)
            for dy in range(3):
                for dx in range(3):
                    blurred += padded[dy:dy + out.shape[0], dx:dx + out.shape[1]] * kernel[dy, dx]
            out = blurred
        return np.clip(out, 0.0, 255.0)

    def _deltas(self, sample: LocalNavTrainingSample) -> np.ndarray:
        deltas = sample.metadata.get("waypoint_deltas")
        if deltas is None:
            raise KeyError(
                f"sample {sample.sample_id} lacks metadata['waypoint_deltas'] — "
                "regenerate the dataset with waypoint_deltas metadata"
            )
        array = np.asarray(deltas, dtype=np.float32).reshape(-1, 2)
        if array.shape[0] < self._horizon:  # zero-pad (stop semantics)
            pad = np.zeros((self._horizon - array.shape[0], 2), dtype=np.float32)
            array = np.concatenate([array, pad], axis=0)
        return array[: self._horizon]

    def __getitem__(self, index: int) -> dict:
        sample = self._samples[index]
        brightness = (
            1.0 + random.uniform(-self._brightness_jitter, self._brightness_jitter)
            if self._augment
            else 1.0
        )

        gca_ages = None
        if self._context_mode == "gca":
            context_paths, gca_ages = self._gca_paths(sample)
            context_frames = [
                self._load_rgb(path, brightness) for path in context_paths
            ]
            context = stack_context(
                context_frames, context_size=len(context_frames),
                image_size=self._image_size,
            )
            return self._pack_item(
                sample, context, context_frames, brightness,
                frame_ages=gca_ages,
            )
        context_paths = list(sample.current_image_paths)[-self._context_size :]
        if len(context_paths) < self._context_size and context_paths:
            # Datasets store only 4 context paths, but expert frames on disk
            # follow the deterministic frames/%05d.jpg naming — derive the
            # earlier indices for longer context windows (M3 memory knob).
            # On-policy rollout streams don't follow the pattern; for them
            # stack_context's repeat-oldest padding applies as before.
            first = context_paths[0]
            directory, _, name = first.rpartition("/")
            stem, dot, ext = name.partition(".")
            if stem.isdigit():
                width = len(stem)
                index = int(stem)
                floor_index = max(0, int(sample.goal_frame_index))
                extra = []
                need = self._context_size - len(context_paths)
                for step in range(1, need + 1):
                    candidate = index - step
                    if candidate < floor_index:
                        break
                    extra.append(
                        f"{directory}/{candidate:0{width}d}{dot}{ext}"
                    )
                context_paths = list(reversed(extra)) + context_paths
        context_frames = [self._load_rgb(path, brightness) for path in context_paths]
        context = stack_context(
            context_frames, context_size=self._context_size,
            image_size=self._image_size,
        )
        return self._pack_item(sample, context, context_frames, brightness)

    def _gca_paths(self, sample):
        """Variable-length memory paths + ages (primitive units) for one
        sample. Expert episodes follow frames/%05d.jpg; correction episodes
        follow frames/r%04d_%d.jpg (one snapshot per replan ≈ 2 primitives,
        so their ages are scaled ×2). Missing files degrade gracefully by
        dropping that index (anchor/current always kept if loadable)."""
        import re as _re

        from supernav.methods.localnav_policy.memory_schedule import memory_schedule

        cur_path = list(sample.current_image_paths)[-1]
        directory, _, name = cur_path.rpartition("/")
        stem, dot, ext = name.partition(".")
        start = max(0, int(sample.goal_frame_index))
        cur = int(sample.current_frame_index)
        age_scale = 1.0
        fmt = None
        if stem.isdigit():
            width = len(stem)
            fmt = lambda i: f"{directory}/{i:0{width}d}{dot}{ext}"  # noqa: E731
        else:
            match = _re.match(r"r(\d+)_(\d+)$", stem)
            if match:
                width = len(match.group(1))
                k_last = match.group(2)
                fmt = (
                    lambda i: f"{directory}/r{i:0{width}d}_{k_last}{dot}{ext}"
                )  # noqa: E731
                age_scale = 2.0
                start = 0  # dagger replan indexing starts at 0
        if fmt is None or cur <= start:
            return [cur_path], [0.0]
        indices = memory_schedule(
            start, cur, self._memory_budget, self._memory_recent_tail,
            include_anchor=self._memory_anchor,
            mid_span=self._memory_mid_span,
            mid_count=self._memory_mid_count,
        )
        paths, ages = [], []
        for idx in indices:
            if idx == cur:
                paths.append(cur_path)
                ages.append(0.0)
                continue
            candidate = fmt(idx)
            full = os.path.join(self._root, candidate)
            if os.path.exists(full):
                paths.append(candidate)
                ages.append(float(cur - idx) * age_scale)
                continue
            if idx == indices[0]:
                # anchor missing (label-skipped replan in correction
                # episodes): walk forward to the first existing frame so the
                # sequence still opens with the earliest available view.
                for probe in range(idx + 1, cur):
                    alt = fmt(probe)
                    if os.path.exists(os.path.join(self._root, alt)):
                        paths.append(alt)
                        ages.append(float(cur - probe) * age_scale)
                        break
        if not paths or paths[-1] != cur_path:
            paths.append(cur_path)
            ages.append(0.0)
        return paths, ages

    def _pack_item(
        self, sample, context, context_frames, brightness, frame_ages=None
    ):
        goal_rgb = self._load_rgb(sample.goal_image_path, brightness)
        point = np.asarray(sample.point.as_tuple(), dtype=np.float64)
        if self._augment and self._point_jitter_px > 0:
            width = height = self._image_size
            point = point + np.array(
                [
                    random.uniform(-self._point_jitter_px, self._point_jitter_px)
                    / (width - 1),
                    random.uniform(-self._point_jitter_px, self._point_jitter_px)
                    / (height - 1),
                ]
            )
            point = np.clip(point, 0.0, 1.0)
        goal_pair = goal_pair_tensor(
            context_frames[-1], goal_rgb, goal_point=tuple(point),
            image_size=self._image_size, conditioning=self._conditioning,
        )

        actions = np.clip(
            self._deltas(sample) / self._waypoint_scale, -1.0, 1.0
        ).astype(np.float32)
        distance = float(sample.target_frame_index - sample.current_frame_index)
        stop_label = 1.0 if distance <= self._stop_k else 0.0
        if self._zero_motion_stop and float(np.abs(actions).max()) == 0.0:
            stop_label = 1.0
        return {
            "context": context,
            "goal_pair": goal_pair,
            "actions": torch.from_numpy(actions),
            # Raw remaining-primitive count stays available for diagnostics
            # even though nothing regresses on it anymore.
            "distance": torch.tensor(distance, dtype=torch.float32),
            "stop": torch.tensor(stop_label, dtype=torch.float32),
            # The SAME jittered point the marker was rendered with — coord
            # embedding and marker must never disagree about where the goal is.
            "goal_point": torch.tensor(
                [float(point[0]), float(point[1])], dtype=torch.float32
            ),
            # Unknown tracking labels use (-1, -1, -1) so batching keeps a uniform shape.
            "point_track": torch.tensor(
                (sample.metadata or {}).get("_point_track", (-1.0, -1.0, -1.0)),
                dtype=torch.float32,
            ),
            # RL-R1 per-sample eps-loss weight (1.0 unless advantage-annotated).
            "weight": torch.tensor(
                float((sample.metadata or {}).get("_rl_weight", 1.0))
                * (
                    self._hindsight_action_weight
                    if (sample.metadata or {}).get("hindsight")
                    else 1.0
                ),
                dtype=torch.float32
            ),
            **(
                {"frame_ages": torch.tensor(frame_ages, dtype=torch.float32)}
                if frame_ages is not None
                else {}
            ),
        }
