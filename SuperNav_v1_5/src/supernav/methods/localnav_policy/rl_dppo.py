#!/usr/bin/env python
"""DPPO-style online fine-tuning of the diffusion head."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np


import torch  # noqa: E402

from supernav.methods.localnav_policy.backends import (  # noqa: E402
    score_trajectories_consensus,
)
from supernav.methods.localnav_policy.model import (  # noqa: E402
    ACTION_DIM,
    DEFAULT_WAYPOINT_SCALE_M,
    NomadPolicy,
)
from supernav.methods.localnav_policy.preprocess import (  # noqa: E402
    goal_pair_tensor,
    stack_context,
)
from supernav.methods.localnav_policy.train import _git_commit  # noqa: E402

GEO_BINS = (0.0, 1.0, 2.0, 4.0, 8.0, float("inf"))


def _guided_eps(policy, x_t, t: int, cond, uncond, guidance: float):
    """CFG-extrapolated eps — the deployment-time policy's noise estimate."""
    timestep = torch.tensor(t, device=x_t.device)
    if uncond is None or guidance == 1.0:
        return policy.predict_noise(x_t, timestep, cond)
    both = policy.predict_noise(
        torch.cat([x_t, x_t], dim=0), timestep, torch.cat([cond, uncond], dim=0)
    )
    eps_c, eps_u = both.chunk(2, dim=0)
    return eps_u + guidance * (eps_c - eps_u)


def _posterior(policy, x_t, t: int, cond, uncond, guidance: float):
    """Mean/variance of x_t → x_{t-1}, replicating scheduler.step exactly."""
    scheduler = policy.scheduler
    device = x_t.device
    eps = _guided_eps(policy, x_t, t, cond, uncond, guidance)
    alpha_t = scheduler.alphas[t].to(device)
    alpha_bar_t = scheduler.alphas_cumprod[t].to(device)
    alpha_bar_prev = (
        scheduler.alphas_cumprod[t - 1].to(device)
        if t > 0
        else torch.tensor(1.0, device=device)
    )
    beta_t = scheduler.betas[t].to(device)
    x0 = (x_t - (1.0 - alpha_bar_t).sqrt() * eps) / alpha_bar_t.sqrt()
    x0 = x0.clamp(-1.0, 1.0)
    mean = (
        alpha_bar_prev.sqrt() * beta_t / (1.0 - alpha_bar_t) * x0
        + alpha_t.sqrt() * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * x_t
    )
    var = ((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * beta_t).clamp(min=1e-10)
    return mean, var


def chain_logprob_batch(policy, cond, uncond, guidance, chains) -> "torch.Tensor":
    """Log-likelihood of recorded denoise chains under current UNet weights.

    cond: [B, D]; uncond: [B, D] or None; chains: [B, K+1, H, 2] latents in
    scheduler order (pure noise first, x_0 last). Sums log N(x_{t-1}; mu, var)
    over the stochastic steps (t = K-1 … 1); the final t=0 step is the
    deterministic posterior mean and carries no probability mass. → [B].
    """
    timesteps = policy.scheduler.timesteps.tolist()
    total = torch.zeros(chains.shape[0], device=chains.device)
    for index, t in enumerate(timesteps):
        if t == 0:
            break
        x_t = chains[:, index]
        x_prev = chains[:, index + 1]
        mean, var = _posterior(policy, x_t, t, cond, uncond, guidance)
        logp = -0.5 * (((x_prev - mean) ** 2) / var + math.log(2 * math.pi) + var.log())
        total = total + logp.sum(dim=(1, 2))
    return total


@torch.no_grad()
def sample_with_chain(policy, cond, uncond, guidance, num_samples: int):
    """Ancestral sampling that records every latent (normalized space)."""
    device = cond.device
    cond_batch = cond.expand(num_samples, -1)
    uncond_batch = uncond.expand(num_samples, -1) if uncond is not None else None
    x = torch.randn((num_samples, policy.action_horizon, ACTION_DIM), device=device)
    latents = [x.clone()]
    for t in policy.scheduler.timesteps.tolist():
        eps = _guided_eps(policy, x, t, cond_batch, uncond_batch, guidance)
        x = policy.scheduler.step(eps, t, x)
        latents.append(x.clone())
    return x, torch.stack(latents, dim=1)  # x0 [M, H, 2]; chains [M, K+1, H, 2]


def _bin_of(geo: float) -> int:
    for b in range(len(GEO_BINS) - 1):
        if GEO_BINS[b] <= geo < GEO_BINS[b + 1]:
            return b
    return len(GEO_BINS) - 2


def main() -> int:  # noqa: PLR0915 — pilot script, deliberately linear
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--scene", default="0236_840815")
    parser.add_argument(
        "--scene-dataset-config",
        default="/vepfs-pykaxon/jingyi/dataset/interiorGS-processed/scenes.scene_dataset_config.json",
    )
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rollouts-per-iter", type=int, default=30)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=16)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--min-geo", type=float, default=1.0)
    parser.add_argument("--max-geo", type=float, default=8.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out)
    if (out_dir / "ckpt_latest.pt").exists():
        raise RuntimeError(f"{out_dir} already holds a ckpt — run dirs are append-only")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_state = torch.load(args.base, map_location="cpu")
    train_config = dict(base_state.get("train_config") or {})
    policy = NomadPolicy(
        use_coord_embed=bool(train_config.get("use_coord_embed", False)),
        use_dist_head=bool(train_config.get("use_dist_head", False)),
    )
    policy.load_state_dict(base_state["state_dict"], strict=True)
    policy = policy.to(device)
    guidance = 2.0 if float(train_config.get("cfg_drop_prob", 0) or 0) > 0 else 1.0
    for name, param in policy.named_parameters():
        param.requires_grad = name.startswith("noise_pred.")
    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad], lr=args.lr
    )
    waypoint_scale = float(base_state.get("waypoint_scale", DEFAULT_WAYPOINT_SCALE_M))

    # ---- habitat harness (mirrors eval_hops setup) ----------------------
    from supernav.methods.localnav_policy.rollout_geometry import _shortest_path_points, resample_polyline
    from supernav.methods.localnav_policy.rollout_geometry import (
        _agent_pose,
        _capture,
        _geodesic,
        _quat_xyzw_from_yaw,
        _sample_hop_endpoints,
    )
    from supernav.methods.localnav_policy.rollout_geometry import CameraIntrinsics
    from supernav.methods.localnav_policy.rollout_geometry import select_hop_target

    from supernav.backends.habitat.loader import prepare_simulator_import
    prepare_simulator_import()
    from supernav.methods.localnav.navigation import navigate_with_localnav
    from supernav.backends.habitat.adapter import HabitatAdapter

    adapter = HabitatAdapter()
    init = adapter.handle_request(
        {
            "request_id": "dppo-init",
            "action": "init_scene",
            "payload": {
                "scene": args.scene,
                "scene_dataset_config_file": args.scene_dataset_config,
                "sensor": {
                    "width": 512, "height": 512, "hfov": 90, "sensor_height": 1.5,
                    "color_sensor": True, "depth_sensor": True,
                },
            },
        }
    )
    if not init.get("ok"):
        print(f"[dppo] init failed: {init}", file=sys.stderr)
        return 1
    session_id = init["result"]["session_id"]
    session = adapter.navigation_session(session_id)
    pathfinder = adapter.navigation_pathfinder(session)
    camera = CameraIntrinsics(512, 512, 90.0)
    rng = random.Random(args.seed + 1000)

    log_path = out_dir / "train_log.jsonl"
    started = time.time()

    for iteration in range(args.iterations):
        # ---------------- collection (behavior = current weights) --------
        policy.eval()
        episodes = []
        capture_seq = 0
        while len(episodes) < args.rollouts_per_iter:
            endpoints = _sample_hop_endpoints(pathfinder, rng, args.min_geo, args.max_geo)
            if endpoints is None:
                break
            start, goal, _ = endpoints
            yaw0 = rng.uniform(-math.pi, math.pi)
            adapter.set_agent_state(
                session_id,
                {"position": start.tolist(), "rotation": _quat_xyzw_from_yaw(yaw0)},
            )
            rgb0, depth0 = _capture(adapter, session)
            path = _shortest_path_points(pathfinder, start, goal)
            if path is None:
                continue
            waypoints = resample_polyline(path[0], spacing=0.25)
            selected = select_hop_target(
                waypoints, np.full(len(waypoints), yaw0), camera, 1.5, depth0
            )
            if selected is None:
                continue
            target_index, point = selected
            target_world = waypoints[target_index]
            capture_seq += 1
            ref = f"dppo:{iteration}:{capture_seq}"
            session.last_visual_image_refs = {
                ref: {"rgb": rgb0, "capture_seq": capture_seq, "hfov": 90.0}
            }
            session.latest_visual_capture_seq = capture_seq

            decisions = []

            class _Client:
                def healthz(self):
                    return {"ok": True, "backend": "dppo-collect"}

                def act(self, context_rgbs, goal_rgb, goal_point,
                        num_samples=16, denoise_steps=10):
                    context = stack_context(
                        context_rgbs, context_size=policy.context_size
                    )
                    goal_pair = goal_pair_tensor(
                        np.asarray(context_rgbs[-1]), np.asarray(goal_rgb),
                        goal_point=goal_point,
                    )
                    point_tensor = (
                        torch.tensor([list(goal_point)], dtype=torch.float32,
                                     device=device)
                        if policy.use_coord_embed
                        else None
                    )
                    with torch.no_grad():
                        cond = policy.encode(
                            context.unsqueeze(0).to(device),
                            goal_pair.unsqueeze(0).to(device),
                            goal_point=point_tensor,
                        )
                        uncond = (
                            policy.encode(
                                context.unsqueeze(0).to(device),
                                goal_pair.unsqueeze(0).to(device),
                                goal_point=point_tensor,
                                drop_goal=torch.ones(
                                    1, dtype=torch.bool, device=device
                                ),
                            )
                            if guidance != 1.0
                            else None
                        )
                        x0, chains = sample_with_chain(
                            policy, cond, uncond, guidance, args.num_samples
                        )
                        stop_prob = float(
                            torch.sigmoid(policy.predict_stop_logit(cond)).item()
                        )
                    trajectories = (x0.cpu().numpy() * waypoint_scale)
                    scores = score_trajectories_consensus(trajectories)
                    sel = int(np.argmax(scores))
                    position, _ = _agent_pose(adapter, session)
                    geo_now = _geodesic(pathfinder, position, target_world)
                    if not math.isfinite(geo_now):
                        geo_now = math.hypot(
                            position[0] - target_world[0],
                            position[2] - target_world[2],
                        )
                    decisions.append(
                        {
                            "cond": cond.squeeze(0).cpu(),
                            "uncond": None if uncond is None
                            else uncond.squeeze(0).cpu(),
                            "chain": chains[sel].cpu(),
                            "geo": float(geo_now),
                        }
                    )
                    spread = float(np.std(trajectories.sum(axis=1), axis=0).mean())
                    return {
                        "trajectories": trajectories,
                        "scores": scores,
                        "selected": sel,
                        "temporal_distance": None,
                        "done_probability": stop_prob,
                        "confidence": float(np.clip(1.0 / (1.0 + spread), 0.0, 1.0)),
                    }

            result = navigate_with_localnav(
                adapter, session_id,
                {"image_ref": ref, "point": list(point),
                 "max_steps": args.max_steps},
                policy_client=_Client(),
            )
            final_pos, _ = _agent_pose(adapter, session)
            final_geo = _geodesic(pathfinder, final_pos, target_world)
            if not math.isfinite(final_geo):
                final_geo = math.hypot(
                    final_pos[0] - target_world[0], final_pos[2] - target_world[2]
                )
            episodes.append(
                {"decisions": decisions, "final_geo": float(final_geo),
                 "status": result.get("status")}
            )

        # ---------------- returns + geo-binned baseline -------------------
        flat = []
        for episode in episodes:
            decisions = episode["decisions"]
            if not decisions:
                continue
            geos = [d["geo"] for d in decisions] + [episode["final_geo"]]
            rewards = [geos[i] - geos[i + 1] for i in range(len(decisions))]
            terminal = (
                2.0 if geos[-1] <= 1.0 else (1.0 if geos[-1] <= 2.0 else -0.5)
            )
            if episode["status"] == "reached" and geos[-1] > 2.0:
                terminal -= 2.0  # false arrival: don't reinforce that chain
            rewards[-1] += terminal
            acc = 0.0
            for index in range(len(decisions) - 1, -1, -1):
                acc = rewards[index] + args.gamma * acc
                decisions[index]["ret"] = acc
            flat.extend(decisions)
        if not flat:
            print("[dppo] empty iteration, skipping", flush=True)
            continue
        returns_by_bin = {}
        for decision in flat:
            returns_by_bin.setdefault(_bin_of(decision["geo"]), []).append(
                decision["ret"]
            )
        baseline = {b: float(np.mean(v)) for b, v in returns_by_bin.items()}
        for decision in flat:
            decision["adv"] = decision["ret"] - baseline[_bin_of(decision["geo"])]
        adv_std = float(np.std([d["adv"] for d in flat])) or 1.0

        # ---------------- behavior logprobs (before ANY update) -----------
        policy.eval()
        with torch.no_grad():
            for lo in range(0, len(flat), args.minibatch):
                chunk = flat[lo : lo + args.minibatch]
                cond = torch.stack([d["cond"] for d in chunk]).to(device)
                uncond = (
                    torch.stack([d["uncond"] for d in chunk]).to(device)
                    if chunk[0]["uncond"] is not None
                    else None
                )
                chains = torch.stack([d["chain"] for d in chunk]).to(device)
                logp = chain_logprob_batch(policy, cond, uncond, guidance, chains)
                for decision, value in zip(chunk, logp.cpu().tolist()):
                    decision["logp_old"] = value

        # ---------------- PPO update --------------------------------------
        policy.train()
        steps_per_chain = max(1, len(policy.scheduler.timesteps) - 1)
        clipped_count, update_count, loss_sum = 0, 0, 0.0
        for _ in range(args.ppo_epochs):
            random.shuffle(flat)
            for lo in range(0, len(flat), args.minibatch):
                chunk = flat[lo : lo + args.minibatch]
                cond = torch.stack([d["cond"] for d in chunk]).to(device)
                uncond = (
                    torch.stack([d["uncond"] for d in chunk]).to(device)
                    if chunk[0]["uncond"] is not None
                    else None
                )
                chains = torch.stack([d["chain"] for d in chunk]).to(device)
                logp_old = torch.tensor(
                    [d["logp_old"] for d in chunk], device=device
                )
                advantage = torch.tensor(
                    [d["adv"] / adv_std for d in chunk], device=device
                )
                logp_new = chain_logprob_batch(
                    policy, cond, uncond, guidance, chains
                )
                ratio = torch.exp((logp_new - logp_old) / steps_per_chain)
                unclipped = ratio * advantage
                clipped = (
                    torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * advantage
                )
                loss = -torch.min(unclipped, clipped).mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in policy.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                update_count += 1
                loss_sum += float(loss)
                clipped_count += int(
                    ((ratio < 1 - args.clip) | (ratio > 1 + args.clip)).sum()
                )

        record = {
            "iter": iteration,
            "rollouts": len(episodes),
            "decisions": len(flat),
            "mean_return": round(float(np.mean([d["ret"] for d in flat])), 3),
            "sr1m": round(
                float(np.mean([e["final_geo"] <= 1.0 for e in episodes])), 3
            ),
            "mean_final_geo": round(
                float(np.mean([e["final_geo"] for e in episodes])), 3
            ),
            "loss": round(loss_sum / max(1, update_count), 5),
            "clip_frac": round(
                clipped_count / max(1, update_count * args.minibatch), 3
            ),
            "elapsed_s": round(time.time() - started, 1),
        }
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"[dppo] {record}", flush=True)

        # serving-contract ckpt after every iteration (crash-safe progress)
        tuned = {
            k: v.detach().cpu()
            for k, v in policy.state_dict().items()
            if k.startswith("noise_pred.")
        }
        out_state = dict(base_state)
        for slot in ("state_dict", "raw_state_dict"):
            merged = dict(base_state[slot])
            merged.update(tuned)
            out_state[slot] = merged
        out_state["train_config"] = {
            **train_config,
            "rl_dppo": {
                "base": args.base, "iterations": iteration + 1, "lr": args.lr,
                "clip": args.clip, "gamma": args.gamma,
                "rollouts_per_iter": args.rollouts_per_iter,
            },
        }
        torch.save(out_state, out_dir / "ckpt_latest.pt")

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "stage": "RL-R3 DPPO pilot",
                "base_ckpt": args.base,
                "git_commit": _git_commit(),
                "args": vars(args),
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        )
    )
    adapter.close_all()
    print("[dppo] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
