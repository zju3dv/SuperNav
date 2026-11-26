#!/usr/bin/env python
"""Synchronous DPPO with parallel rollouts."""

from __future__ import annotations

import argparse
import json
import math
import os
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
    DEFAULT_WAYPOINT_SCALE_M,
    NomadPolicy,
)
from supernav.methods.localnav_policy.preprocess import (  # noqa: E402
    goal_pair_tensor,
    stack_context,
)
from supernav.methods.localnav_policy.rl_dppo import (  # noqa: E402
    _bin_of,
    chain_logprob_batch,
    sample_with_chain,
)
from supernav.methods.localnav_policy.rl_stop_finetune import (  # noqa: E402
    stop_go_rewards,
)
from supernav.methods.localnav_policy.train import _git_commit  # noqa: E402


def _wait_for(path: Path, timeout_s: float, poll_s: float = 5.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(poll_s)
    return path.exists()


def build_policy(state: dict, device) -> "tuple[NomadPolicy, float]":
    config = dict(state.get("train_config") or {})
    policy = NomadPolicy(
        use_coord_embed=bool(config.get("use_coord_embed", False)),
        use_dist_head=bool(config.get("use_dist_head", False)),
    )
    policy.load_state_dict(state["state_dict"], strict=True)
    guidance = 2.0 if float(config.get("cfg_drop_prob", 0) or 0) > 0 else 1.0
    return policy.to(device), guidance


def collect_episodes(
    policy, guidance, adapter, session_id, session, pathfinder, camera,
    rng, args, device, waypoint_scale, tag: str,
):
    """R episodes through the real navigate loop; returns pilot-format list."""
    from supernav.methods.localnav_policy.rollout_geometry import _shortest_path_points, resample_polyline
    from supernav.methods.localnav_policy.rollout_geometry import (
        _agent_pose,
        _capture,
        _geodesic,
        _quat_xyzw_from_yaw,
        _sample_hop_endpoints,
    )
    from supernav.methods.localnav_policy.rollout_geometry import select_hop_target

    from supernav.methods.localnav.navigation import navigate_with_localnav

    episodes = []
    capture_seq = 0
    while len(episodes) < args.rollouts_per_worker:
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
        ref = f"{tag}:{capture_seq}"
        session.last_visual_image_refs = {
            ref: {"rgb": rgb0, "capture_seq": capture_seq, "hfov": 90.0}
        }
        session.latest_visual_capture_seq = capture_seq
        decisions = []

        class _Client:
            def healthz(self):
                return {"ok": True, "backend": "dppo-sync"}

            def act(self, context_rgbs, goal_rgb, goal_point,
                    num_samples=16, denoise_steps=10):
                context = stack_context(context_rgbs, context_size=policy.context_size)
                goal_pair = goal_pair_tensor(
                    np.asarray(context_rgbs[-1]), np.asarray(goal_rgb),
                    goal_point=goal_point,
                )
                point_tensor = (
                    torch.tensor([list(goal_point)], dtype=torch.float32, device=device)
                    if policy.use_coord_embed else None
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
                            drop_goal=torch.ones(1, dtype=torch.bool, device=device),
                        )
                        if guidance != 1.0 else None
                    )
                    x0, chains = sample_with_chain(
                        policy, cond, uncond, guidance, args.num_samples
                    )
                    stop_prob = float(
                        torch.sigmoid(policy.predict_stop_logit(cond)).item()
                    )
                trajectories = x0.cpu().numpy() * waypoint_scale
                scores = score_trajectories_consensus(trajectories)
                sel = int(np.argmax(scores))
                position, _ = _agent_pose(adapter, session)
                geo_now = _geodesic(pathfinder, position, target_world)
                if not math.isfinite(geo_now):
                    geo_now = math.hypot(
                        position[0] - target_world[0], position[2] - target_world[2]
                    )
                decisions.append(
                    {
                        "cond": cond.squeeze(0).cpu(),
                        "uncond": None if uncond is None else uncond.squeeze(0).cpu(),
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
            {"image_ref": ref, "point": list(point), "max_steps": args.max_steps},
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
    return episodes


def annotate_returns(episodes, gamma: float) -> list:
    """Discounted returns + terminal shaping; returns flattened decisions."""
    flat = []
    for episode in episodes:
        decisions = episode["decisions"]
        if not decisions:
            continue
        geos = [d["geo"] for d in decisions] + [episode["final_geo"]]
        rewards = [geos[i] - geos[i + 1] for i in range(len(decisions))]
        terminal = 2.0 if geos[-1] <= 1.0 else (1.0 if geos[-1] <= 2.0 else -0.5)
        if episode["status"] == "reached" and geos[-1] > 2.0:
            terminal -= 2.0
        rewards[-1] += terminal
        acc = 0.0
        for index in range(len(decisions) - 1, -1, -1):
            acc = rewards[index] + gamma * acc
            decisions[index]["ret"] = acc
        flat.extend(decisions)
    return flat


def main() -> int:  # noqa: PLR0915
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rank", type=int,
                        default=int(os.environ.get("MLP_ROLE_INDEX", 0)))
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--scene", default="0236_840815")
    parser.add_argument(
        "--scene-dataset-config",
        default="/vepfs-pykaxon/jingyi/dataset/interiorGS-processed/scenes.scene_dataset_config.json",
    )
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--rollouts-per-worker", type=int, default=40)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch", type=int, default=64)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--min-geo", type=float, default=1.0)
    parser.add_argument("--max-geo", type=float, default=8.0)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--wait-shards-s", type=float, default=1500.0)
    parser.add_argument("--wait-ckpt-s", type=float, default=2400.0)
    parser.add_argument("--recal-last-iters", type=int, default=4)
    args = parser.parse_args()

    rank, world = args.rank, args.world
    device = torch.device(args.device)
    torch.manual_seed(args.seed + rank * 7919)
    out_dir = Path(args.out)
    sync_dir = out_dir / "sync"
    if rank == 0:
        if (out_dir / "ckpt_latest.pt").exists():
            raise RuntimeError(f"{out_dir} already holds a ckpt — append-only")
        sync_dir.mkdir(parents=True, exist_ok=True)
    else:
        _wait_for(sync_dir, 600)

    base_state = torch.load(args.base, map_location="cpu")
    waypoint_scale = float(base_state.get("waypoint_scale", DEFAULT_WAYPOINT_SCALE_M))

    # rank 0 keeps one live policy + optimizer across iterations; other
    # ranks rebuild from the published ckpt each iteration.
    policy, guidance = build_policy(base_state, device)
    optimizer = None
    if rank == 0:
        for name, param in policy.named_parameters():
            param.requires_grad = name.startswith("noise_pred.")
        optimizer = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=args.lr
        )

    # ---- habitat session -------------------------------------------------
    from supernav.backends.habitat.loader import prepare_simulator_import
    prepare_simulator_import()
    from supernav.methods.localnav_policy.rollout_geometry import CameraIntrinsics

    from supernav.backends.habitat.adapter import HabitatAdapter

    adapter = HabitatAdapter()
    init = adapter.handle_request(
        {
            "request_id": f"dppo-sync-{rank}",
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
        print(f"[dppo-sync r{rank}] init failed: {init}", file=sys.stderr)
        return 1
    session_id = init["result"]["session_id"]
    session = adapter.navigation_session(session_id)
    pathfinder = adapter.navigation_pathfinder(session)
    camera = CameraIntrinsics(512, 512, 90.0)

    log_path = out_dir / "train_log.jsonl"
    started = time.time()

    def publish(state_like: dict, path: Path) -> None:
        tuned = {
            k: v.detach().cpu()
            for k, v in policy.state_dict().items()
            if k.startswith("noise_pred.")
        }
        out_state = dict(state_like)
        for slot in ("state_dict", "raw_state_dict"):
            merged = dict(base_state[slot])
            merged.update(tuned)
            out_state[slot] = merged
        out_state["train_config"] = {
            **dict(base_state.get("train_config") or {}),
            "rl_dppo_sync": {
                "base": args.base, "world": world,
                "rollouts_per_iter": args.rollouts_per_worker * world,
                "lr": args.lr, "clip": args.clip, "gamma": args.gamma,
            },
        }
        tmp = path.with_suffix(".tmp")
        torch.save(out_state, tmp)
        tmp.rename(path)  # atomic on same fs — readers never see partial files

    for iteration in range(args.iterations):
        # -- get this iteration's weights -----------------------------------
        if iteration > 0 and rank != 0:
            ckpt_path = sync_dir / f"ckpt_iter{iteration}.pt"
            if not _wait_for(ckpt_path, args.wait_ckpt_s):
                print(f"[dppo-sync r{rank}] ckpt wait timeout iter {iteration}",
                      file=sys.stderr)
                return 3
            state = torch.load(ckpt_path, map_location="cpu")
            policy.load_state_dict(state["state_dict"], strict=True)
            policy = policy.to(device)

        # -- collect + behavior logprobs -------------------------------------
        policy.eval()
        rng = random.Random(args.seed + rank * 100_000 + iteration * 1_000)
        episodes = collect_episodes(
            policy, guidance, adapter, session_id, session, pathfinder,
            camera, rng, args, device, waypoint_scale,
            tag=f"sync{iteration}r{rank}",
        )
        flat_local = [d for e in episodes for d in e["decisions"]]
        with torch.no_grad():
            for lo in range(0, len(flat_local), args.minibatch):
                chunk = flat_local[lo : lo + args.minibatch]
                cond = torch.stack([d["cond"] for d in chunk]).to(device)
                uncond = (
                    torch.stack([d["uncond"] for d in chunk]).to(device)
                    if chunk[0]["uncond"] is not None else None
                )
                chains = torch.stack([d["chain"] for d in chunk]).to(device)
                logp = chain_logprob_batch(policy, cond, uncond, guidance, chains)
                for decision, value in zip(chunk, logp.cpu().tolist()):
                    decision["logp_old"] = value
        shard_path = sync_dir / f"iter{iteration}_rank{rank}.pt"
        torch.save({"episodes": episodes}, shard_path)
        (sync_dir / f"iter{iteration}_rank{rank}.done").touch()

        if rank != 0:
            continue

        # -- rank 0: pool shards ---------------------------------------------
        got = []
        deadline = time.time() + args.wait_shards_s
        pending = set(range(world))
        while pending and time.time() < deadline:
            for r in sorted(pending):
                if (sync_dir / f"iter{iteration}_rank{r}.done").exists():
                    pending.discard(r)
            if pending:
                time.sleep(5)
        for r in range(world):
            p = sync_dir / f"iter{iteration}_rank{r}.pt"
            if p.exists():
                got.append(torch.load(p, map_location="cpu"))
        pooled = [e for shard in got for e in shard["episodes"]]
        flat = annotate_returns(pooled, args.gamma)
        if not flat:
            print("[dppo-sync r0] empty iteration", flush=True)
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

        # -- rank 0: PPO update ------------------------------------------------
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
                    if chunk[0]["uncond"] is not None else None
                )
                chains = torch.stack([d["chain"] for d in chunk]).to(device)
                logp_old = torch.tensor([d["logp_old"] for d in chunk], device=device)
                advantage = torch.tensor(
                    [d["adv"] / adv_std for d in chunk], device=device
                )
                logp_new = chain_logprob_batch(policy, cond, uncond, guidance, chains)
                ratio = torch.exp((logp_new - logp_old) / steps_per_chain)
                unclipped = ratio * advantage
                clipped = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * advantage
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
            "workers": len(got),
            "rollouts": len(pooled),
            "decisions": len(flat),
            "mean_return": round(float(np.mean([d["ret"] for d in flat])), 3),
            "sr1m": round(float(np.mean([e["final_geo"] <= 1.0 for e in pooled])), 3),
            "mean_final_geo": round(
                float(np.mean([e["final_geo"] for e in pooled])), 3
            ),
            "loss": round(loss_sum / max(1, update_count), 5),
            "clip_frac": round(
                clipped_count / max(1, update_count * args.minibatch), 3
            ),
            "elapsed_s": round(time.time() - started, 1),
        }
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(f"[dppo-sync] {record}", flush=True)
        publish(dict(base_state), sync_dir / f"ckpt_iter{iteration + 1}.pt")
        publish(dict(base_state), out_dir / "ckpt_latest.pt")

    if rank != 0:
        adapter.close_all()
        print(f"[dppo-sync r{rank}] worker done")
        return 0

    # ---- rank 0 freebie: stop-head bandit recalibration on the NEW policy's
    # own visitation distribution (cond is the stop head's exact input).
    pairs_cond, pairs_geo = [], []
    for iteration in range(max(0, args.iterations - args.recal_last_iters),
                           args.iterations):
        for r in range(world):
            p = sync_dir / f"iter{iteration}_rank{r}.pt"
            if not p.exists():
                continue
            for episode in torch.load(p, map_location="cpu")["episodes"]:
                for decision in episode["decisions"]:
                    pairs_cond.append(decision["cond"])
                    pairs_geo.append(decision["geo"])
    recal_meta = {"pairs": len(pairs_cond)}
    if pairs_cond:
        conds = torch.stack(pairs_cond).to(device)
        geos = torch.tensor(pairs_geo, device=device)
        for name, param in policy.named_parameters():
            param.requires_grad = name.startswith("stop_head.")
        head_opt = torch.optim.AdamW(
            [p for p in policy.parameters() if p.requires_grad], lr=3e-4
        )
        policy.train()
        for epoch in range(30):
            perm = torch.randperm(conds.shape[0], device=device)
            for lo in range(0, conds.shape[0], 512):
                idx = perm[lo : lo + 512]
                pi = torch.sigmoid(policy.predict_stop_logit(conds[idx]))
                r_stop, r_go = stop_go_rewards(geos[idx])
                loss = -(pi * r_stop + (1 - pi) * r_go).mean()
                head_opt.zero_grad(set_to_none=True)
                loss.backward()
                head_opt.step()
        tuned_all = {
            k: v.detach().cpu()
            for k, v in policy.state_dict().items()
            if k.startswith(("noise_pred.", "stop_head."))
        }
        recal_state = dict(base_state)
        for slot in ("state_dict", "raw_state_dict"):
            merged = dict(base_state[slot])
            merged.update(tuned_all)
            recal_state[slot] = merged
        recal_state["train_config"] = {
            **dict(base_state.get("train_config") or {}),
            "rl_dppo_sync": "see ckpt_latest",
            "rl_stop_recal": {"pairs": len(pairs_cond), "epochs": 30},
        }
        torch.save(recal_state, out_dir / "ckpt_stoprecal.pt")
        recal_meta["saved"] = True

    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "stage": "RL-R3 v2 sync DPPO",
                "base_ckpt": args.base,
                "world": world,
                "git_commit": _git_commit(),
                "args": {k: v for k, v in vars(args).items()},
                "stop_recal": recal_meta,
                "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        )
    )
    adapter.close_all()
    print("[dppo-sync] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
