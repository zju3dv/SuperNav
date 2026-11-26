"""Torch-backed NoMaD policy backend for the localnav service."""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Sequence

import numpy as np

from supernav.methods.localnav_policy.backends import PolicyBackend, score_trajectories, score_trajectories_consensus


class NomadBackend(PolicyBackend):
    name = "nomad"
    model = "nomad-point-v1 (efficientnet-b0 + 4L4H transformer + stop head + unet1d)"

    def __init__(
        self,
        checkpoint: "str | None" = None,
        device: "str | None" = None,
        waypoint_scale_m: float = 0.25,
        guidance_scale: "float | None" = None,
        selection: str = "consensus",
    ) -> None:
        import os as _os_threads

        import torch

        from supernav.methods.localnav_policy.model import NomadPolicy

        # >96px per-sample tensor ops cross torch's parallel_for grain; on
        # many-core hosts the default thread pool livelocks preprocessing
        # (measured 2063ms→26ms per decision at 160px on a 244-core box).
        # Same fix as the training dataloader; override via env if needed.
        torch.set_num_threads(
            int(_os_threads.environ.get("NAV_LOCALNAV_TORCH_THREADS", "8"))
        )

        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint = None
        self._waypoint_scale_m = float(waypoint_scale_m)
        if selection not in ("consensus", "bearing"):
            raise ValueError(f"selection must be consensus|bearing, got {selection!r}")
        self._selection = selection

        state = None
        train_config: Dict[str, Any] = {}
        if checkpoint:
            state = torch.load(checkpoint, map_location=self.device)
            if isinstance(state, dict):
                train_config = dict(state.get("train_config") or {})
        # The checkpoint dictates the architecture variant — a coord-embed
        # model has an extra token and a wider positional embedding, so the
        # net must be built to match before load_state_dict.
        self._use_coord_embed = bool(train_config.get("use_coord_embed", False))
        self._use_dist_head = bool(train_config.get("use_dist_head", False))
        self._use_point_head = bool(train_config.get("use_point_head", False))
        # The checkpoint is the single source of truth so the
        # serving path self-configures (train/serve must never diverge).
        self._encoder = str(train_config.get("encoder", "effnet-scratch"))
        self._image_size = int(train_config.get("image_size", 96))
        self._head_type = str(train_config.get("head_type", "ddpm"))
        self._conditioning = str(train_config.get("conditioning", "marker"))
        self._flow_infer_steps = int(train_config.get("flow_infer_steps", 4))
        self._context_size = int(train_config.get("context_size", 4))
        ckpt_scale = train_config.get("waypoint_scale")
        if ckpt_scale is None:
            ckpt_scale = state.get("waypoint_scale") if isinstance(state, dict) else None
        if ckpt_scale is not None:
            self._waypoint_scale_m = float(ckpt_scale)
        self.context_mode = str(train_config.get("context_mode", "fixed"))
        self._configure_memory(train_config)
        # Serving-side token cache (gca): frames are encoded ONCE when they
        # enter memory; replans fuse cached tokens (52-frame episodes would
        # otherwise re-encode everything every 0.5m). Reset on new episode
        # (detected by history length shrinking).
        self._gca_cache: "dict[int, object]" = {}
        self._gca_prev_len = 0
        self._gca_goal_token = None
        self._gca_goal_token_null = None
        trained_with_cfg = float(train_config.get("cfg_drop_prob", 0.0) or 0.0) > 0.0
        if guidance_scale is None:
            guidance_scale = 2.0 if trained_with_cfg else 1.0
        self._guidance_scale = float(guidance_scale)
        if self._guidance_scale != 1.0 and not trained_with_cfg:
            print(
                "[localnav] WARNING: guidance_scale set but checkpoint was "
                "trained without condition dropout — the unconditional branch "
                "is untrained; forcing guidance_scale=1.0",
                file=sys.stderr,
                flush=True,
            )
            self._guidance_scale = 1.0

        self._policy = NomadPolicy(
            use_coord_embed=self._use_coord_embed,
            use_dist_head=self._use_dist_head,
            use_point_head=self._use_point_head,
            encoder=self._encoder,
            image_size=self._image_size,
            head_type=self._head_type,
            context_size=self._context_size,
            context_mode=self.context_mode,
            memory_budget=self._memory_budget,
            # Use the checkpoint action horizon to keep serving tensor shapes aligned.
            action_horizon=int(train_config.get("action_horizon", 8)),
        )
        self._policy = self._policy.to(self.device).eval()
        self._distance_head = None
        if state is not None:
            weights = state
            if isinstance(state, dict) and "state_dict" in state:
                weights = state["state_dict"]
            has_stop = any(k.startswith("stop_head.") for k in weights)
            has_dist = any(k.startswith("dist_head.") for k in weights)
            if has_dist and not has_stop:
                dist_weights = {
                    k[len("dist_head."):]: v
                    for k, v in weights.items()
                    if k.startswith("dist_head.")
                }
                hidden, embed = dist_weights["0.weight"].shape
                self._distance_head = torch.nn.Sequential(
                    torch.nn.Linear(embed, hidden),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden, 1),
                ).to(self.device).eval()
                self._distance_head.load_state_dict(dist_weights)
                print(
                    "[localnav] legacy dist-head checkpoint detected — serving "
                    "done_probability = clip(1 - d/3) for protocol-paired eval",
                    file=sys.stderr,
                    flush=True,
                )
            missing, unexpected = self._policy.load_state_dict(weights, strict=False)
            print(
                f"[localnav] loaded checkpoint {checkpoint} "
                f"(missing={len(missing)}, unexpected={len(unexpected)}, "
                f"coord_embed={self._use_coord_embed}, "
                f"guidance={self._guidance_scale})",
                file=sys.stderr,
                flush=True,
            )
            if missing or unexpected:
                print(
                    f"[localnav] state_dict mismatch — missing={sorted(missing)[:4]} "
                    f"unexpected={sorted(unexpected)[:4]} (showing ≤4 each); a "
                    "dist-head-era checkpoint cannot drive the stop head",
                    file=sys.stderr,
                    flush=True,
                )
            self.checkpoint = checkpoint
        else:
            print(
                "[localnav] WARNING: nomad backend running with RANDOM weights "
                "(no NAV_LOCALNAV_CKPT) — plumbing smoke only, not navigation.",
                file=sys.stderr,
                flush=True,
            )

        # Optional selection-level collision critic (decision-layer RL).
        import os as _os

        self._collision_critic = None
        self._critic_weight = float(
            _os.environ.get("NAV_LOCALNAV_CRITIC_WEIGHT", "1.0")
        )
        critic_path = _os.environ.get("NAV_LOCALNAV_COLLISION_CRITIC")
        if critic_path:
            payload = torch.load(critic_path, map_location=self.device)
            cond_dim = int(payload.get("cond_dim", 256))
            horizon = int(payload.get("horizon", 8))
            net = torch.nn.Sequential(
                torch.nn.Linear(cond_dim + horizon * 2, 256), torch.nn.ReLU(),
                torch.nn.Linear(256, 64), torch.nn.ReLU(),
                torch.nn.Linear(64, 1),
            )
            state_dict = {
                k[len("net."):]: v for k, v in payload["state_dict"].items()
            }
            net.load_state_dict(state_dict)
            self._collision_critic = net.to(self.device).eval()
            print(
                f"[localnav] collision critic armed ({critic_path}, "
                f"weight={self._critic_weight}, "
                f"val_auc={payload.get('val_auc')})",
                file=sys.stderr,
                flush=True,
            )

    def _configure_memory(self, train_config) -> None:
        """Configure memory from checkpoint metadata."""
        get = train_config.get
        self._memory_budget = int(get("memory_budget", 12))
        self._memory_recent_tail = int(get("memory_recent_tail", 0))
        self._memory_anchor = bool(get("memory_anchor", True))
        self._memory_mid_span = int(get("memory_mid_span", 0))
        self._memory_mid_count = int(get("memory_mid_count", 0))

    def _encode_gca(self, context_rgbs, goal_batch):
        """(cond, uncond) via the serving token cache. context_rgbs is the
        FULL per-primitive history since hop start; list position == primitive
        index, so the schedule and ages match the training distribution."""
        import torch

        from supernav.methods.localnav_policy.memory_schedule import memory_schedule
        from supernav.methods.localnav_policy.preprocess import stack_context

        import os as _os

        n = len(context_rgbs)
        if n <= self._gca_prev_len:  # new episode: history restarted
            self._gca_cache.clear()
        self._gca_prev_len = n
        indices = memory_schedule(
            0, n - 1, self._memory_budget, self._memory_recent_tail,
            include_anchor=self._memory_anchor,
            mid_span=self._memory_mid_span,
            mid_count=self._memory_mid_count,
        )
        # NAV_GCA_SURGERY overrides inference frame selection or ages for diagnostics.
        surgery = _os.environ.get("NAV_GCA_SURGERY", "")
        if surgery == "recent4":
            indices = list(range(max(0, n - 4), n))
        elif surgery == "anchor_only":
            indices = [0, n - 1] if n > 1 else [0]
        policy = self._policy
        for idx in indices:
            if idx in self._gca_cache:
                continue
            frame = stack_context(
                [context_rgbs[idx]], context_size=1,
                image_size=self._image_size,
            ).to(self.device)
            if policy.dino is not None:
                token = policy.dino.encode_obs(frame)
            else:
                token = policy.obs_encoder(frame)
            self._gca_cache[idx] = token
        obs_tokens = torch.stack(
            [self._gca_cache[idx] for idx in indices], dim=1
        )
        ages = torch.tensor(
            [[float(n - 1 - idx) for idx in indices]], device=self.device
        )
        if surgery == "zero_ages":
            ages = ages * 0.0
        elif surgery == "clip_ages32":
            ages = ages.clamp(max=32.0)
        # The goal PAIR is [current frame ‖ marked start frame] — its first
        # half changes every replan, so the goal token must be re-encoded
        # per call (one encoder pass, cheap). Caching it froze the pair at
        # frame 0 and broke closed-loop navigation while validation stayed
        # healthy (training-style inputs) — the classic train/serve split.
        if policy.dino is not None:
            goal_token = policy.dino.encode_goal_pair(goal_batch).unsqueeze(1)
        else:
            goal_token = policy.goal_encoder(goal_batch).unsqueeze(1)
        cond = policy.fuse_gca(obs_tokens, goal_token, ages)
        uncond = None
        if self._guidance_scale != 1.0:
            null = policy.null_goal_token.expand(1, -1, -1)
            uncond = policy.fuse_gca(obs_tokens, null, ages)
        return cond, uncond

    def act(
        self,
        context_rgbs: List[np.ndarray],
        goal_rgb: np.ndarray,
        goal_point: Sequence[float],
        num_samples: int,
        denoise_steps: int,
    ) -> Dict[str, Any]:
        torch = self._torch
        from supernav.methods.localnav_policy.preprocess import goal_pair_tensor, stack_context

        context = stack_context(
            context_rgbs, context_size=self._policy.context_size,
            image_size=self._image_size,
        )
        # Conditioning form and resolution follow the ckpt's train_config —
        # marker (drawn at policy resolution), crop-zoom, or clean+coord.
        goal_pair = goal_pair_tensor(
            np.asarray(context_rgbs[-1]), np.asarray(goal_rgb),
            goal_point=goal_point,
            image_size=self._image_size,
            conditioning=self._conditioning,
        )
        with torch.no_grad():
            context_batch = context.unsqueeze(0).to(self.device)
            goal_batch = goal_pair.unsqueeze(0).to(self.device)
            point_batch = None
            if self._use_coord_embed:
                point_batch = torch.tensor(
                    [[float(goal_point[0]), float(goal_point[1])]],
                    device=self.device,
                )
            if self.context_mode == "gca":
                cond, uncond_gca = self._encode_gca(context_rgbs, goal_batch)
            else:
                cond = self._policy.encode(
                    context_batch, goal_batch, goal_point=point_batch
                )
            temporal_distance = None
            stop_probability_raw = None
            if self._distance_head is not None:
                temporal_distance = float(
                    torch.relu(self._distance_head(cond).squeeze(-1)).item()
                )
                stop_probability = float(
                    max(0.0, min(1.0, 1.0 - temporal_distance / 3.0))
                )
            else:
                stop_probability = float(
                    torch.sigmoid(self._policy.predict_stop_logit(cond)).item()
                )
                if self._use_dist_head:
                    # Dual-head variant: auxiliary regression served as
                    # telemetry only — never gates the stop decision.
                    temporal_distance = float(
                        torch.relu(self._policy.predict_distance(cond)).item()
                    )
                    veto = float(os.environ.get(
                        "NAV_LOCALNAV_DIST_VETO_STEPS", "0") or 0)
                    if veto > 0 and temporal_distance > veto:
                        stop_probability_raw = stop_probability
                        stop_probability = 0.0
            uncond = None
            if self._guidance_scale != 1.0:
                if self.context_mode == "gca":
                    uncond = uncond_gca
                else:
                    uncond = self._policy.encode(
                        context_batch,
                        goal_batch,
                        goal_point=point_batch,
                        drop_goal=torch.ones(
                            1, dtype=torch.bool, device=self.device
                        ),
                    )
            trajectories = (
                self._policy.sample_actions(
                    cond,
                    num_samples=max(1, int(num_samples)),
                    denoise_steps=(
                        self._flow_infer_steps
                        if self._head_type == "flow"
                        else int(denoise_steps)
                    ),
                    waypoint_scale=self._waypoint_scale_m,
                    uncond=uncond,
                    guidance_scale=self._guidance_scale,
                )
                .cpu()
                .numpy()
            )

        if self._selection == "consensus":
            scores = score_trajectories_consensus(trajectories)
        else:  # The hop-start bearing becomes stale after rotation.
            scores = score_trajectories(trajectories, goal_point)
        if self._collision_critic is not None:
            # Selection-level bandit: subtract predicted per-candidate
            # collision probability without changing the diffusion chain.
            with torch.no_grad():
                cond_rep = cond.expand(trajectories.shape[0], -1)
                traj_t = torch.tensor(
                    trajectories, dtype=torch.float32, device=self.device
                )
                p_collide = torch.sigmoid(
                    self._collision_critic(
                        torch.cat(
                            [cond_rep, traj_t.reshape(traj_t.shape[0], -1)],
                            dim=-1,
                        )
                    ).squeeze(-1)
                ).cpu().numpy()
            scores = scores - self._critic_weight * p_collide
        endpoints = trajectories.sum(axis=1)
        spread = float(np.std(endpoints, axis=0).mean())
        confidence = float(np.clip(1.0 / (1.0 + spread), 0.0, 1.0))
        return {
            "trajectories": trajectories,
            "scores": scores,
            "selected": int(np.argmax(scores)),
            "temporal_distance": temporal_distance,
            "done_probability": stop_probability,
            # Preserve the stop probability before the distance veto for telemetry.
            "done_probability_raw": stop_probability_raw,
            "confidence": confidence,
        }
